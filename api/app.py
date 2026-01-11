from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os

# -----------------------------
# FASTAPI APP
# -----------------------------
app = FastAPI(title="Healthcare Chatbot API")

# -----------------------------
# PART 1: LOAD ML MODELS (SAFE)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

classifier = joblib.load(os.path.join(MODEL_DIR, "best_svm_model.pkl"))
vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))

# -----------------------------
# REQUEST SCHEMAS
# -----------------------------
class ClassifyRequest(BaseModel):
    text: str

class QueryRequest(BaseModel):
    question: str

# -----------------------------
# HEALTH CHECK
# -----------------------------
@app.get("/")
def health():
    return {"status": "API is running"}

# -----------------------------
# PART 1: CLASSIFICATION (UNCHANGED ✅)
# -----------------------------
@app.post("/classify")
def classify_text(request: ClassifyRequest):
    X = vectorizer.transform([request.text])
    pred = classifier.predict(X)[0]
    disease = label_encoder.inverse_transform([pred])[0]

    return {
        "input_text": request.text,
        "predicted_class_id": int(pred),
        "predicted_disease": disease
    }

# -----------------------------
# PART 2: RAG (LAZY LOADING ✅)
# -----------------------------
def load_rag():
    # 🔴 imports ONLY when needed
    from langchain.schema import Document
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain.chains import RetrievalQA
    from langchain_community.llms import HuggingFacePipeline
    from transformers import pipeline

    docs = [
        Document(page_content="Diabetes causes high blood sugar.", metadata={"source": "Diabetes"}),
        Document(page_content="Hypertension is high blood pressure.", metadata={"source": "Hypertension"}),
        Document(page_content="Asthma affects breathing.", metadata={"source": "Asthma"})
    ]

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    hf_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-large",
        max_new_tokens=256
    )

    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )

    return qa

# -----------------------------
# PART 2 ENDPOINT
# -----------------------------
@app.post("/query")
def query_rag(request: QueryRequest):
    qa = load_rag()
    result = qa(request.question)

    return {
        "question": request.question,
        "answer": result["result"],
        "sources": [doc.metadata for doc in result["source_documents"]]
    }
