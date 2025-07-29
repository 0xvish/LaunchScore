"""
Custom CSS styles for the LaunchScore application
"""

CUSTOM_CSS = """
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    .score-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 1px solid #e0e0e0;
    }
    .score-card h3 {
        color: #2c3e50;
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
        font-weight: 600;
    }
    .score-card h2 {
        color: #2c3e50;
        margin: 0.5rem 0;
        font-size: 2.5rem;
        font-weight: bold;
    }
    .score-card p {
        color: #6c757d;
        margin-top: 0.5rem;
        font-size: 0.9rem;
    }
    .success-high { 
        border-left-color: #28a745;
        background: linear-gradient(135deg, #ffffff 0%, #f8fff9 100%);
    }
    .success-high h2 { color: #28a745; }
    .success-medium { 
        border-left-color: #ffc107;
        background: linear-gradient(135deg, #ffffff 0%, #fffef8 100%);
    }
    .success-medium h2 { color: #e67e22; }
    .success-low { 
        border-left-color: #dc3545;
        background: linear-gradient(135deg, #ffffff 0%, #fff8f8 100%);
    }
    .success-low h2 { color: #dc3545; }
</style>
"""

HEADER_HTML = """
<div class="main-header">
    <h1>🚀 LaunchScore</h1>
    <h3>AI-Powered Startup Success Predictor</h3>
    <p>Combining Machine Learning & Market Intelligence</p>
</div>
"""

WELCOME_MESSAGE = """
## 🎯 How it works

**LaunchScore** combines two powerful AI systems to evaluate your startup:

1. **🤖 Neural Network Model** - Analyzes numerical patterns from successful startups
2. **🧠 Market Intelligence AI** - Provides contextual analysis using startup database

### 📝 Get Started
Fill in your startup details in the sidebar and click "Predict Success" to get:
- Comprehensive success score (0-10)
- Market analysis and insights
- Strengths and risk assessment
- Algorithmic blind spots identification

### 🔍 What makes this different?
- **Real startup data** from comprehensive database
- **Contextual analysis** considering market timing and location
- **Blind spot detection** highlighting factors algorithms typically miss
"""
