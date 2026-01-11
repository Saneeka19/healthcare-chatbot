# 🏥 Healthcare Chatbot using NLP & Machine Learning

## 📌 Project Overview
This project implements a **Healthcare Chatbot** that classifies user medical queries into disease categories using **Machine Learning and NLP techniques**, and provides intelligent responses.

The system is built in two parts:
- **Part 1:** Disease Classification using traditional ML models  
- **Part 2:** Medical Question Answering using Retrieval-Augmented Generation (RAG) *(in progress)*

This project is developed as part of an academic assignment and follows industry-level structuring and deployment practices.

---

## 🎯 Objectives
- Classify medical text into disease categories  
- Deploy the trained model using **FastAPI**  
- Enable real-time predictions via REST API  
- Extend the system with explainability and advanced NLP techniques  

---

## 🧠 Part 1: Disease Classification (✅)

### 🔹 Dataset
- Medical text dataset with disease labels  
- Preprocessed using NLP techniques (cleaning, tokenization, vectorization)  

### 🔹 Techniques Used
- TF-IDF Vectorization  
- Linear Support Vector Classifier (LinearSVC)  
- Label Encoding  
- Model serialization using `.pkl` files  

### 🔹 Output
- Disease class ID  
- Disease name prediction

### 🔹 Example API Response
json
{
  "input_text": "Patient has high blood sugar and frequent urination",
  "predicted_class_id": 20,
  "predicted_disease": "urinary tract infection"
}
  
## 🤖****Part 2: Medical Question Answering (RAG) (🚧)****

Uses document retrieval + LLM-based generation
Allows users to ask general medical questions

**Technologies used:**
FAISS vector store
Transformer-based embeddings
HuggingFace pipelines

### Language Model
* HuggingFace google/flan-t5-base
* Used for answer generation based on retrieved context

### 🚀 FastAPI Endpoint – Query API
🔹 Endpoint
POST /query

🔹 Request Body
{
  "question": "What medicines are used for asthma?"
}

🔹 Response Format
{
  "question": "What medicines are used for asthma?",
  "answer": "Asthma is commonly treated using inhaled corticosteroids and bronchodilators.",
  "sources": [
    { "source": "NIH - Asthma Medications" },
    { "source": "WikiDoc - Asthma" }
  ]
}

### 📊 Output Explanation

* question: User medical query
* answer: Generated response using retrieved medical context
* sources: Documents used to generate the answer (for transparency)

## 🎁 Bonus Task Implemented
* ✅ Model Explainability using LIME
* Local Interpretable Model-agnostic Explanations (LIME)
* Explains which words influenced disease predictions
* Improves trust and transparency in medical predictions
## 🗂️ Project Structure

healthcare-chatbot/
│
├── api/
│   └── app.py                  # FastAPI application
│
├── models/
│   ├── best_svm_model.pkl          # Trained ML model
│   ├── tfidf_vectorizer.pkl          # TF-IDF vectorizer
│   └── label_encoder.pkl       # Label encoder
│
├── notebooks/
│   ├── Medical_Classification.ipynb
│   └── LIME_Explanation.ipynb
│
├── README.md
└── requirements.txt


## 🚀 How to Run the Project
### 1️⃣ Clone the Repository
git clone https://github.com/Saneeka19/healthcare-chatbot.git
cd healthcare-chatbot

### 2️⃣ Create Virtual Environment
conda create -n healthcare_env python=3.10
conda activate healthcare_env

### 3️⃣ Install Dependencies
pip install -r requirements.txt

### 4️⃣ Run FastAPI Server
cd api
uvicorn app:app --reload

### 5️⃣ Access API Documentation

Open browser:
http://127.0.0.1:8000/docs

### 📬 API Endpoints
🔹 Disease Classification

### POST /classify
Request Body
{
  "text": "Patient has high blood sugar and frequent urination"
}


### Response
{
  "predicted_disease": "urinary tract infection"
}

### 📊 Model Performance

* Accuracy and evaluation metrics analyzed in Jupyter notebooks
* Performs well on structured medical symptom descriptions

### ⚠️ Challenges Faced
* Dataset imbalance
* Dependency conflicts (LangChain & Transformers)
* Model version mismatch warnings
* Environment setup for RAG components

### 🔮 Future Improvements
* Complete Part 2 RAG implementation
* Fine-tune BioBERT for medical QA
* Add frontend UI using Streamlit
* Deploy on cloud (AWS / Azure)
* Add monitoring dashboard



## 👩‍💻 Author

### Sanika Keskar
## Healthcare Chatbot — 2026


---




