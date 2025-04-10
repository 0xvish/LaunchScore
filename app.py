from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd
import re
import os
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

# === Load ML pipeline ===
ml_model = joblib.load("models/ml_pipeline.pkl")  # Contains preprocessing + model

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

# === Inference route ===
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # --- Input validation ---
    idea = data.get('idea')
    sector = data.get('sector')
    stage = data.get('stage')
    hq = data.get('headquarter')

    if not all([idea, sector, stage, hq]):
        return jsonify({'error': 'Missing required fields: idea, sector, stage, headquarter'}), 400

    # --- LLM inference ---
    llm_response = qa_chain.run(idea)
    match = re.search(r'(\d+(?:\.\d+)?)\/10', llm_response)
    llm_score = float(match.group(1)) if match else 0.0

    cleaned_llm_response = '\n'.join(llm_response.strip().split('\n')[1:]).strip()

    # --- ML inference ---
    input_df = pd.DataFrame([{
        'Sector': sector,
        'Stage': stage,
        'HeadQuarter': hq
    }])

    try:
        # Use full pipeline (handles encoding + prediction)
        ml_score = ml_model.predict_proba(input_df)[0][1] * 10  # Scale 0–10
    except Exception as e:
        return jsonify({'error': f"ML model prediction failed: {str(e)}"}), 500

    # --- Final blended score ---
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
