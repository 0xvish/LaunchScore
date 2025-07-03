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
CORS(app, origins=["http://localhost:3000", "https://*.vercel.app"])  # Allow frontend origins

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
🚀 **Startup Success Oracle** here! Let me analyze this venture with startup wisdom and market intelligence.

**Similar Startups for Reference:**
{context}

**New Startup Analysis:**
{question}

**Your Mission:**
1. 📊 **Compare** with similar startups from the database
2. 🎯 **Evaluate** market potential, competition, and timing  
3. 🔮 **Predict** success likelihood (0-10 scale)
4. 💭 **Highlight** specific contextual factors the algorithm likely missed

**Response Format:**
Success Score: X/10

**Key Insights:**
- 💪 **Strengths:** [What works for this specific venture]
- ⚠️ **Risks:** [Real challenges in this market/location/timing]
- 🎯 **Market:** [Opportunity assessment with local context]
- � **Algorithmic Blind Spots:** [Specific factors like local competition, cultural nuances, timing advantages/disadvantages that data models typically miss]

Keep it practical and market-focused! 🎯
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

    # --- LLM inference with ML context ---
    formatted_question = f"""
💡 **Idea:** {idea}
🏢 **Sector:** {sector} | 📈 **Stage:** {stage} | 🏙️ **HQ:** {hq}
📅 **Founded:** {founded} | 💰 **Funding:** ₹{amount:,}

🤖 **ML Model Prediction:** {round(ml_score, 2)}/10

Analyze this startup comprehensively. Consider how location-specific advantages/disadvantages, sector trends, and timing factors might create blind spots in algorithmic predictions. Focus on real-world context the model likely missed.
"""
    
    llm_response = qa_chain.run(formatted_question)
    
    # Extract LLM score and clean response
    match = re.search(r'(\d+(?:\.\d+)?)\/10', llm_response)
    llm_score = float(match.group(1)) if match else 0.0
    cleaned_llm_response = '\n'.join(llm_response.strip().split('\n')[1:]).strip()
    
    # If cleaned response is too short, use full response
    if len(cleaned_llm_response) < 20:
        cleaned_llm_response = llm_response.strip()

    # --- Final score (blended) ---
    final_score = 0.5 * llm_score + 0.5 * ml_score

    return jsonify({
        'llm_score': round(llm_score, 2),
        'ml_score': round(ml_score, 2),
        'final_score': round(final_score, 2),
        'llm_analysis': cleaned_llm_response
    })

# === Health check route ===
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'service': 'launchscore-backend'}), 200

# === Frontend route ===
# @app.route('/')
# def index():
#     return render_template('index.html')
# Frontend now served by Next.js app; HTML route disabled

# === Run the app ===
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
