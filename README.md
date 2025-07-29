# 🚀 LaunchScore – Startup Success Predictor

**LaunchScore** is a hybrid AI-powered platform that predicts the likelihood of success for a startup idea using both:

- A **machine learning model** trained on Indian startup funding data.
- A **language model (LLM)** using LangChain with Gemini API and FAISS vector search for contextual analysis.

Users can enter their startup idea, sector, funding stage, location, funding amount, and founded year to get an overall success prediction along with insightful feedback.

---

## 🧠 Features

- ✅ Neural Network-based ML model trained on real-world startup funding data.
- ✅ LLM-powered qualitative analysis using Gemini API and LangChain.
- ✅ FAISS-powered retrieval from a startup dataset for enhanced context.
- ✅ Modern Streamlit web application with responsive UI.
- ✅ Modular codebase for easy maintenance and scaling.
- ✅ One-click deployment ready.

---

## 📁 Project Structure

```
LaunchScore/
├── streamlit_app.py            # Main Streamlit application
├── config.py                   # Configuration and constants
├── models.py                   # ML model loading and initialization
├── utils.py                    # Prediction utilities
├── components.py               # UI components
├── styles.py                   # CSS styling and layouts
├── models/
│   ├── ml_pipeline.pkl         # Trained ML pipeline (legacy)
│   ├── ml_model.pkl            # ML model weights (legacy)
│   ├── startup_nn.pt           # Neural network model
│   ├── *_vocab.pkl             # Vocabulary encoders
│   └── startup_faiss_index/    # FAISS vector store
├── docker-compose.yml          # Multi-container setup (optional)
├── Dockerfile                  # Container configuration (optional)
├── .env                        # Environment variables
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 🛠️ Development Setup

### Prerequisites

- Python 3.8+ installed
- Git installed
- Google API Key for Gemini API

### 1. 📦 Clone the Repository

```bash
git clone https://github.com/your-username/launchscore.git
cd launchscore
```

### 2. 🌐 Environment Configuration

Create a `.env` file in the root directory:

```env
GOOGLE_API_KEY=your-google-api-key-here
```

Get your Google API Key from [Google AI Studio](https://makersuite.google.com/app).

### 3. 🐍 Backend Setup

```bash
# Create and activate virtual environment
python -m venv venv

# On Windows (WSL/PowerShell)
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run Streamlit application
streamlit run streamlit_app.py
```

The Streamlit application will be available at [http://localhost:8501](http://localhost:8501)

### 4. 🧪 Development Testing

- **Streamlit App**: `http://localhost:8501`
- **Health Check**: Navigate to the app and ensure models load successfully

## 🚀 Production Deployment

### Option 1: Streamlit Cloud (Recommended)

1. Push your code to GitHub
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your GitHub repository
4. Add your `GOOGLE_API_KEY` in the secrets section
5. Deploy with one click

### Option 2: VPS/Cloud Deployment

#### Ubuntu Server Setup

```bash
# 1. Update system
sudo apt update && sudo apt upgrade -y

# 2. Install Python and pip
sudo apt install python3 python3-pip python3-venv git -y

# 3. Clone and setup
git clone https://github.com/your-username/launchscore.git
cd launchscore

# 4. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 5. Install dependencies
pip install -r requirements.txt

# 6. Setup environment
echo "GOOGLE_API_KEY=your-google-api-key" > .env

# 7. Run with screen (for persistent session)
sudo apt install screen -y
screen -S launchscore
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0

# Detach with Ctrl+A, D
# Reattach with: screen -r launchscore
```

### 🔧 Environment Variables

#### Production (.env)

```env
GOOGLE_API_KEY=your-google-api-key
```

#### Streamlit Cloud Secrets

Add in Streamlit Cloud dashboard under "Secrets":

```toml
GOOGLE_API_KEY = "your-google-api-key"
```

### 📊 Production Monitoring

````bash
# Check application status (if using screen)
screen -r launchscore

# View logs
tail -f ~/.streamlit/logs/streamlit.log

# Restart application
# Kill screen session and start new one
```---

## 🔧 Troubleshooting

### Common Issues

#### Application Issues

**Problem**: `ModuleNotFoundError` for ML libraries

```bash
# Solution: Reinstall requirements in virtual environment
pip install --upgrade -r requirements.txt
````

**Problem**: FAISS index not found

```bash
# Solution: Ensure models directory is complete
ls -la models/
# Re-download missing model files if needed
```

**Problem**: Google API quota exceeded

```bash
# Solution: Check API usage in Google Cloud Console
# Generate new API key if needed
```

**Problem**: Streamlit app won't start

```bash
# Solution: Check if port 8501 is available
lsof -i :8501
# Kill process using the port if needed
kill -9 <PID>
```

### Performance Optimization

- Use Streamlit's caching decorators (@st.cache_resource, @st.cache_data)
- Monitor memory usage for ML models
- Consider using Streamlit Cloud for automatic scaling

### Logs and Debugging

```bash
# Streamlit logs (local)
tail -f ~/.streamlit/logs/streamlit.log

# System resource usage
top
htop
```

---

## 🧪 Testing the Application

### Development Testing

Open your browser and go to:

- **Streamlit App**: `http://localhost:8501`

### Production Testing

For deployed applications:

- **Streamlit Cloud**: Your deployed app URL from Streamlit Cloud
- **VPS**: `http://your-server:8501`

### Sample Test Data

Fill in the following example data:

- ✍️ **Startup Idea** – "AI-powered personal finance management app that uses machine learning to provide personalized investment recommendations"
- 🏷️ **Sector** – FinTech
- 💰 **Funding Stage** – Seed
- 📍 **Headquarter** – Bangalore
- 📅 **Founded Year** – 2024
- 💵 **Funding Amount (in ₹Rupees)** – 5000000

Click **Predict** to view:

- The ML & LLM scores
- A combined score out of 10
- Insights and reasoning from the LLM

---

## 📊 How It Works

- **ML Score**: Computed from structured inputs using a neural network trained on a labeled dataset.
- **LLM Score**: Generated by LangChain + Gemini API using contextual data from similar startups retrieved by FAISS.
- **Final Score**: Average of the ML and LLM scores (scaled to 10).

---

## 📚 Tech Stack

- **Frontend & Backend**: Streamlit (Python web framework)
- **LLM Integration**: LangChain + Gemini API (via Google GenerativeAI)
- **ML Models**: PyTorch Neural Networks + Scikit-learn preprocessing
- **Vector Search**: FAISS
- **Embeddings**: HuggingFace `all-MiniLM-L6-v2`
- **Deployment**: Streamlit Cloud or VPS

---

## 📌 Requirements

- Python 3.8+
- Google API Key (for Gemini API)
- Internet connection (for LLM inference)

---

## 🤝 Contributing

Pull requests, feedback, and feature suggestions are welcome!  
Feel free to fork this repo and build upon it.

---

## 🧾 License

This project is for academic and demonstration purposes and is released under the [MIT License](LICENSE).

---

## 💡 Credits

Built by [Vishvam Moliya](https://vishvam.dev) as part of a university project using open-source tools and free-tier APIs.
