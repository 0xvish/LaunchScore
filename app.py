from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import re
import os
import torch
import torch.nn as nn
from dotenv import load_dotenv

# LangChain / LLM Imports
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# === Load environment variables ===
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
else:
    raise ValueError("GOOGLE_API_KEY is missing. Add it to your .env file.")

# === Flask app setup ===
app = Flask(__name__)
CORS(app)

# === Load FAISS index ===
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(
    "models/startup_faiss_index",
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# === LLM setup ===
llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash-latest",
    temperature=0.5
)

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a startup venture analyst with deep knowledge of market trends, startup funding, and investor behavior.

You are provided with descriptions and funding data of similar startups:

{context}

A new startup idea has been proposed:

"{question}"

Your tasks:
1. Analyze the idea and compare it with the retrieved startups.
2. Evaluate market demand, originality, competition, and funding trends.
3. Predict its likelihood of success on a scale of 0 to 10.
4. Justify the score with a brief analysis including potential risks and advantages.

Format your response like this:

Success Score: <score>/10  
Reasoning:  
- <Insight 1>  
- <Insight 2>  
- ...
"""
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt}
)

# === Load Neural Network model and preprocessing ===
class StartupSuccessNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# Load vocabularies and scaler
sector_vocab = joblib.load("models/sector_vocab.pkl")
hq_vocab = joblib.load("models/hq_vocab.pkl")
scaler = joblib.load("models/nn_scaler.pkl")

# Load trained model
nn_model = StartupSuccessNN(input_dim=4)
nn_model.load_state_dict(torch.load("models/startup_nn.pt"))
nn_model.eval()

# === Inference route ===
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # --- Input validation ---
    idea = data.get('idea')
    sector = data.get('sector')
    stage = data.get('stage')  # Not used in NN, could be used for LLM insight
    hq = data.get('headquarter')
    founded = data.get('founded', 2020)  # Optional, default 2020
    amount = data.get('amount', 1000000)  # Optional, default 1M

    if not all([idea, sector, hq]):
        return jsonify({'error': 'Missing required fields: idea, sector, headquarter'}), 400

    # --- LLM inference ---
    llm_response = qa_chain.run(idea)
    match = re.search(r'(\d+(?:\.\d+)?)\/10', llm_response)
    llm_score = float(match.group(1)) if match else 0.0
    cleaned_llm_response = '\n'.join(llm_response.strip().split('\n')[1:]).strip()

    # --- Neural Network inference ---
    try:
        sector_idx = sector_vocab.get(sector, 0)
        hq_idx = hq_vocab.get(hq, 0)
        X = np.array([[sector_idx, hq_idx, founded, amount]])
        X[:, 2:] = scaler.transform(X[:, 2:])  # normalize year and amount
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            prob = nn_model(X_tensor).item()
            ml_score = prob * 10
    except Exception as e:
        return jsonify({'error': f"Neural network prediction failed: {str(e)}"}), 500

    # --- Final score (blended) ---
    final_score = 0.5 * llm_score + 0.5 * ml_score

    return jsonify({
        'llm_score': round(llm_score, 2),
        'ml_score': round(ml_score, 2),
        'final_score': round(final_score, 2),
        'llm_analysis': cleaned_llm_response
    })

# === Frontend route ===
@app.route('/')
def index():
    return render_template('index.html')

# === Run the app ===
if __name__ == '__main__':
    app.run(debug=True)
