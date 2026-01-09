from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os

# ------------------------------
# Paths to models (update according to your system)
# ------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

SVM_MODEL_PATH = os.path.join(MODEL_DIR, "best_svm_model.pkl")
TFIDF_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

# ------------------------------
# Load models
# ------------------------------
best_svm_model = joblib.load(SVM_MODEL_PATH)
tfidf_vectorizer = joblib.load(TFIDF_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)

# ------------------------------
# Initialize FastAPI
# ------------------------------
app = FastAPI(
    title="Healthcare Chatbot API",
    description="API for medical text classification using SVM model",
    version="1.0"
)

# ------------------------------
# Request body schema
# ------------------------------
class TextInput(BaseModel):
    text: str

# ------------------------------
# Test endpoint
# ------------------------------
@app.get("/")
def home():
    return {"message": "Healthcare Chatbot API is running"}

# ------------------------------
# Medical Text Classification Endpoint
# ------------------------------
@app.post("/classify")
def classify_text(data: TextInput):
    try:
        # 1️⃣ Get input text
        text = data.text

        # 2️⃣ Vectorize using TF-IDF
        text_vec = tfidf_vectorizer.transform([text])

        # 3️⃣ Predict using SVM
        pred_class_id = int(best_svm_model.predict(text_vec)[0])

        # 4️⃣ Convert class ID back to disease label
        disease = label_encoder.inverse_transform([pred_class_id])[0]

        # 5️⃣ Return response
        return {
            "input_text": text,
            "predicted_class_id": pred_class_id,
            "predicted_disease": disease
        }

    except Exception as e:
        return {"error": str(e)}
    
    
    # app.py
from fastapi import FastAPI
from pydantic import BaseModel

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

from transformers import pipeline


app = FastAPI(title="Medical RAG - Part 2")

# ------------------ DATA ------------------
medical_docs = [
    {
        "text": "Diabetes causes high blood sugar. Symptoms include frequent urination, thirst, fatigue, and blurred vision.",
        "source": "Diabetes Doc"
    },
    {
        "text": "Hypertension is persistent high blood pressure. Risk factors include obesity and high salt intake.",
        "source": "Hypertension Doc"
    }
]

class QueryRequest(BaseModel):
    question: str

# ------------------ LOAD RAG ------------------
def load_rag():
    docs = [
        Document(page_content=d["text"], metadata={"source": d["source"]})
        for d in medical_docs
    ]

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    hf_pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=200
    )