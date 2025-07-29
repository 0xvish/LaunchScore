"""
Model loading and initialization functions for LaunchScore
"""
import joblib
import torch
import torch.nn as nn
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from config import MODEL_PATHS, LLM_CONFIG


class StartupSuccessNN(nn.Module):
    """Neural Network model for startup success prediction"""
    
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


def create_llm_prompt():
    """Create the prompt template for LLM analysis"""
    return PromptTemplate(
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
- 🧠 **Algorithmic Blind Spots:** [Specific factors like local competition, cultural nuances, timing advantages/disadvantages that data models typically miss]

Keep it practical and market-focused! 🎯
"""
    )


@st.cache_resource
def load_models():
    """Load all models and components"""
    try:
        # Load FAISS index
        embedding_model = HuggingFaceEmbeddings(model_name=LLM_CONFIG["embedding_model"])
        vectorstore = FAISS.load_local(
            MODEL_PATHS["faiss_index"],
            embeddings=embedding_model,
            allow_dangerous_deserialization=True
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        # LLM setup
        llm = ChatGoogleGenerativeAI(
            model=LLM_CONFIG["model"],
            temperature=LLM_CONFIG["temperature"]
        )

        # Create QA chain
        prompt = create_llm_prompt()
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt}
        )

        # Load vocabularies and scaler
        sector_vocab = joblib.load(MODEL_PATHS["sector_vocab"])
        hq_vocab = joblib.load(MODEL_PATHS["hq_vocab"])
        
        # Try to load stage_vocab, if not exists create from config
        try:
            stage_vocab = joblib.load(MODEL_PATHS["stage_vocab"])
        except FileNotFoundError:
            # Create stage vocabulary from config if file doesn't exist
            from config import DEVELOPMENT_STAGES
            stage_vocab = {stage: idx for idx, stage in enumerate(DEVELOPMENT_STAGES)}
            st.warning("⚠️ stage_vocab.pkl not found. Using config-based stage mapping.")
        
        scaler = joblib.load(MODEL_PATHS["nn_scaler"])

        # Load trained neural network model
        nn_model = StartupSuccessNN(input_dim=4)
        nn_model.load_state_dict(torch.load(MODEL_PATHS["startup_nn"]))
        nn_model.eval()

        return qa_chain, nn_model, sector_vocab, hq_vocab, stage_vocab, scaler
    
    except Exception as e:
        st.error(f"❌ Failed to load models: {str(e)}")
        raise e
