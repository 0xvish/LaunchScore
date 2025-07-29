"""
UI components for LaunchScore application
"""
import streamlit as st
from config import DEVELOPMENT_STAGES
from utils import get_score_class


"""
UI components for LaunchScore application
"""
import streamlit as st
from config import DEVELOPMENT_STAGES, SECTORS, HEADQUARTERS
from utils import get_score_class


def create_sidebar_inputs():
    """Create sidebar input form for startup details"""
    st.sidebar.header("📝 Startup Details")
    
    with st.sidebar:
        # Input fields
        idea = st.text_area(
            "💡 Startup Idea",
            placeholder="Describe your startup idea in detail...",
            height=100,
            help="Provide a comprehensive description of your startup idea"
        )
        
        sector = st.selectbox(
            "🏢 Sector",
            options=[''] + sorted(SECTORS),  # Add empty option for validation
            help="Choose the primary sector that best matches your startup",
            format_func=lambda x: "Select a sector..." if x == '' else x
        )
        
        stage = st.selectbox(
            "📈 Funding Stage",
            options=[''] + sorted(DEVELOPMENT_STAGES),  # Add empty option for validation
            help="Current funding stage of your startup",
            format_func=lambda x: "Select a stage..." if x == '' else x
        )
        
        hq = st.selectbox(
            "🏙️ Headquarters",
            options=[''] + sorted(HEADQUARTERS),  # Add empty option for validation
            help="Primary location/city for your startup headquarters",
            format_func=lambda x: "Select a location..." if x == '' else x
        )
        
        col1, col2 = st.columns(2)
        with col1:
            founded = st.number_input(
                "📅 Founded Year",
                min_value=2000,
                max_value=2030,
                value=2023,
                step=1,
                help="Year when the startup was founded"
            )
        
        with col2:
            amount = st.number_input(
                "💰 Funding Amount (₹)",
                min_value=100000,
                max_value=10000000000,  # 10 billion max
                value=1000000,
                step=100000,
                format="%d",
                help="Total funding amount received in Indian Rupees"
            )
        
        # Validation warnings
        validation_errors = []
        if not idea.strip():
            validation_errors.append("💡 Startup Idea is required")
        if not sector:
            validation_errors.append("🏢 Sector is required")
        if not stage:
            validation_errors.append("📈 Funding Stage is required")
        if not hq:
            validation_errors.append("🏙️ Headquarters is required")
            
        if validation_errors:
            st.warning("⚠️ Please complete all required fields:")
            for error in validation_errors:
                st.write(f"• {error}")
        
        predict_button = st.button(
            "🚀 Predict Success", 
            type="primary", 
            use_container_width=True,
            disabled=bool(validation_errors)  # Disable if validation errors exist
        )

    return idea, sector, stage, hq, founded, amount, predict_button


def display_score_cards(result):
    """Display score cards with results"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="score-card {get_score_class(result['ml_score'])}">
            <h3>🤖 ML Score</h3>
            <h2>{result['ml_score']}/10</h2>
            <p>Neural Network Prediction</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="score-card {get_score_class(result['llm_score'])}">
            <h3>🧠 AI Analysis Score</h3>
            <h2>{result['llm_score']}/10</h2>
            <p>Market Intelligence</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="score-card {get_score_class(result['final_score'])}">
            <h3>🎯 Final Score</h3>
            <h2>{result['final_score']}/10</h2>
            <p>Combined Prediction</p>
        </div>
        """, unsafe_allow_html=True)


def display_analysis(result):
    """Display AI analysis section"""
    st.markdown("---")
    st.markdown("## 🧠 AI Market Analysis")
    st.markdown(result['llm_analysis'])


def display_score_breakdown(result):
    """Display score breakdown with progress bars"""
    st.markdown("---")
    st.markdown("### 📊 Score Breakdown")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Machine Learning Score", f"{result['ml_score']}/10")
        st.progress(result['ml_score'] / 10)
    
    with col2:
        st.metric("AI Analysis Score", f"{result['llm_score']}/10")
        st.progress(result['llm_score'] / 10)
    
    st.metric("Final Combined Score", f"{result['final_score']}/10")
    st.progress(result['final_score'] / 10)
