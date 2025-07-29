import streamlit as st
import os

# Local imports
from config import GOOGLE_API_KEY, APP_CONFIG
from styles import CUSTOM_CSS, HEADER_HTML, WELCOME_MESSAGE
from models import load_models
from utils import predict_startup_success
from components import create_sidebar_inputs, display_score_cards, display_analysis, display_score_breakdown

# === Environment setup ===
if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
else:
    st.error("GOOGLE_API_KEY is missing. Add it to your .env file.")
    st.stop()

# === Configure Streamlit page ===
st.set_page_config(**APP_CONFIG)

# === Apply custom CSS ===
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# === Main App ===
def main():
    # Header
    st.markdown(HEADER_HTML, unsafe_allow_html=True)

    # Load models
    try:
        with st.spinner("🔄 Loading AI models..."):
            qa_chain, nn_model, sector_vocab, hq_vocab, stage_vocab, scaler = load_models()
        st.success("✅ Models loaded successfully!")
    except Exception as e:
        st.error(f"❌ Failed to load models: {str(e)}")
        st.stop()

    # Create sidebar inputs
    idea, sector, stage, hq, founded, amount, predict_button = create_sidebar_inputs()

    # Main content area
    if predict_button:
        if not idea or not sector or not hq or not stage:
            st.error("❌ Please fill in all required fields: Idea, Sector, Stage, and Headquarters")
        else:
            with st.spinner("🔮 Analyzing your startup..."):
                result = predict_startup_success(
                    idea, sector, stage, hq, founded, amount,
                    qa_chain, nn_model, sector_vocab, hq_vocab, stage_vocab, scaler
                )
            
            if result:
                # Display results
                display_score_cards(result)
                display_analysis(result)
                display_score_breakdown(result)
    else:
        # Welcome message
        st.markdown(WELCOME_MESSAGE)


if __name__ == "__main__":
    main()
