"""
Prediction utilities for LaunchScore application
"""
import numpy as np
import torch
import re
import streamlit as st

from config import SCORE_THRESHOLDS


def get_score_class(score):
    """Determine CSS class based on score"""
    if score >= SCORE_THRESHOLDS["high"]:
        return "success-high"
    elif score >= SCORE_THRESHOLDS["medium"]:
        return "success-medium"
    else:
        return "success-low"


def predict_startup_success(idea, sector, stage, hq, founded, amount, qa_chain, nn_model, sector_vocab, hq_vocab, stage_vocab, scaler):
    """Main prediction function"""
    # Input validation
    if not all([idea.strip(), sector, stage, hq]):
        st.error("❌ Please fill in all required fields: Idea, Sector, Stage, and Headquarters")
        return None
    
    # Validate inputs against vocabularies
    if sector not in sector_vocab:
        st.error(f"❌ Invalid sector: '{sector}'. Please select from the dropdown.")
        return None
    
    if hq not in hq_vocab:
        st.error(f"❌ Invalid headquarters: '{hq}'. Please select from the dropdown.")
        return None
        
    if stage not in stage_vocab:
        st.error(f"❌ Invalid stage: '{stage}'. Please select from the dropdown.")
        return None
    
    # Neural Network inference
    try:
        sector_idx = sector_vocab.get(sector, 0)
        hq_idx = hq_vocab.get(hq, 0)
        
        # For the original 4-feature model, we'll use: [sector_idx, hq_idx, founded, amount]
        # The stage information will be used in the LLM analysis but not the neural network
        X = np.array([[sector_idx, hq_idx, founded, amount]])
        
        # Normalize numerical features (founded year and amount) - last 2 columns
        X[:, 2:] = scaler.transform(X[:, 2:])
        
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            prob = nn_model(X_tensor).item()
            ml_score = prob * 10
    except Exception as e:
        st.error(f"Neural network prediction failed: {str(e)}")
        st.error(f"Debug info - Sector idx: {sector_vocab.get(sector, 'N/A')}, HQ idx: {hq_vocab.get(hq, 'N/A')}")
        return None

    # LLM inference with ML context
    formatted_question = f"""
💡 **Idea:** {idea}
🏢 **Sector:** {sector} | 📈 **Stage:** {stage} | 🏙️ **HQ:** {hq}
📅 **Founded:** {founded} | 💰 **Funding:** ₹{amount:,}

🤖 **ML Model Prediction:** {round(ml_score, 2)}/10

Analyze this startup comprehensively. Consider how location-specific advantages/disadvantages, sector trends, and timing factors might create blind spots in algorithmic predictions. Focus on real-world context the model likely missed.
"""
    
    with st.spinner("🤖 Analyzing with AI..."):
        llm_response = qa_chain.run(formatted_question)
    
    # Extract LLM score and clean response
    match = re.search(r'(\d+(?:\.\d+)?)\/10', llm_response)
    llm_score = float(match.group(1)) if match else 0.0
    cleaned_llm_response = '\n'.join(llm_response.strip().split('\n')[1:]).strip()
    
    # If cleaned response is too short, use full response
    if len(cleaned_llm_response) < 20:
        cleaned_llm_response = llm_response.strip()

    # Final score (blended)
    final_score = 0.5 * llm_score + 0.5 * ml_score

    return {
        'llm_score': round(llm_score, 2),
        'ml_score': round(ml_score, 2),
        'final_score': round(final_score, 2),
        'llm_analysis': cleaned_llm_response
    }
