# 🚀 LaunchScore – Startup Success Predictor

**LaunchScore** is a hybrid AI-powered platform that predicts the likelihood of success for a startup idea using both:

- A **machine learning model** trained on Indian startup funding data.
- A **language model (LLM)** using LangChain with Gemini Pro and FAISS vector search for contextual analysis.

Users can enter their startup idea, sector, funding stage, location, funding amount, and founded year to get an overall success prediction along with insightful feedback.

---

## 🧠 Features

- ✅ Neural Network-based ML model trained on real-world startup funding data.
- ✅ LLM-powered qualitative analysis using Gemini Pro and LangChain.
- ✅ FAISS-powered retrieval from a startup dataset for enhanced context.
- ✅ Fully responsive frontend (TailwindCSS).
- ✅ Flask API backend.
- ✅ One-click deployment ready.

---

## 📁 Project Structure

```
LaunchScore/
├── app.py                      # Flask server with LLM + ML integration
├── startup_success_2_0.py      # Enhanced ML predictor
├── models/
│   ├── ml_pipeline.pkl         # Trained ML pipeline
│   ├── ml_model.pkl            # ML model weights
│   ├── startup_nn.pt           # Neural network model
│   ├── *_vocab.pkl             # Vocabulary encoders
│   └── startup_faiss_index/    # FAISS vector store
├── frontend/                   # Next.js frontend application
│   ├── src/
│   │   └── app/
│   │       ├── page.tsx        # Main prediction interface
│   │       ├── layout.tsx      # App layout
│   │       └── globals.css     # Tailwind styles
│   ├── package.json            # Frontend dependencies
│   └── next.config.ts          # Next.js configuration
├── templates/
│   └── index.html              # Backup HTML template
├── docker-compose.yml          # Multi-container setup
├── Dockerfile                  # Backend container config
├── nginx.conf                  # Reverse proxy config
├── deploy.sh                   # Automated deployment script
├── .env                        # Environment variables
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 🛠️ Development Setup

### Prerequisites

- Python 3.8+ installed
- Node.js 18+ and npm/bun installed (for frontend)
- Git installed
- Google API Key for Gemini Pro

### 1. 📦 Clone the Repository

```bash
git clone https://github.com/your-username/launchscore.git
cd launchscore
```

### 2. � Environment Configuration

Create a `.env` file in the root directory:

```env
GOOGLE_API_KEY=your-google-api-key-here
FLASK_ENV=development
FLASK_DEBUG=true
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

# Install backend dependencies
pip install -r requirements.txt

# Run backend development server
python app.py
```

The backend API will be available at [http://localhost:5000](http://localhost:5000)

### 4. ⚛️ Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies (using bun - faster alternative to npm)
bun install

# Start development server
bun dev
```

The frontend will be available at [http://localhost:3000](http://localhost:3000)

### 5. 🧪 Development Testing

- **Backend Health Check**: `http://localhost:5000/health`
- **API Endpoint**: `http://localhost:5000/predict` (POST)
- **Frontend**: `http://localhost:3000`

## 🚀 Production Deployment

### Option 1: Docker Deployment (Recommended)

#### Docker Prerequisites

- Docker and Docker Compose installed
- Server with at least 2GB RAM
- Domain/subdomain (optional)

#### Quick Deploy with Script

```bash
# Make deploy script executable
chmod +x deploy.sh

# Run deployment
./deploy.sh
```

#### Manual Docker Deployment

```bash
# 1. Create .env file with production settings
cp .env.example .env
nano .env  # Add your GOOGLE_API_KEY

# 2. Build and start services
docker-compose up -d --build

# 3. Check service status
docker-compose ps

# 4. View logs
docker-compose logs -f
```

The API will be available at `http://your-server:5000`

### Option 2: Frontend Deployment

#### Vercel Deployment (Next.js)

```bash
# Install Vercel CLI
npm i -g vercel

# Navigate to frontend directory
cd frontend

# Deploy to Vercel
vercel

# Set environment variables in Vercel dashboard:
# NEXT_PUBLIC_API_URL=https://your-backend-url.com
```

#### Manual Frontend Build

```bash
cd frontend

# Build for production
bun build

# Start production server
bun start
```

### Option 3: VPS/Cloud Deployment

#### Ubuntu Server Setup

```bash
# 1. Update system
sudo apt update && sudo apt upgrade -y

# 2. Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# 3. Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# 4. Clone and deploy
git clone https://github.com/your-username/launchscore.git
cd launchscore
chmod +x deploy.sh
./deploy.sh
```

#### Nginx Reverse Proxy (Optional)

If deploying on port 80/443:

```bash
# Install Nginx
sudo apt install nginx

# Configure reverse proxy
sudo nano /etc/nginx/sites-available/launchscore

# Add configuration:
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}

# Enable site
sudo ln -s /etc/nginx/sites-available/launchscore /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### 🔧 Environment Variables

#### Backend (.env)

```env
GOOGLE_API_KEY=your-google-api-key
FLASK_ENV=production
FLASK_DEBUG=false
```

#### Frontend (Vercel/Netlify)

```env
NEXT_PUBLIC_API_URL=https://your-backend-domain.com
```

### 📊 Production Monitoring

```bash
# Check container status
docker-compose ps

# View logs
docker-compose logs -f app
docker-compose logs -f nginx

# Restart services
docker-compose restart

# Update deployment
git pull
docker-compose up -d --build
```

---

## 🔧 Troubleshooting

### Common Issues

#### Backend Issues

**Problem**: `ModuleNotFoundError` for ML libraries

```bash
# Solution: Reinstall requirements in virtual environment
pip install --upgrade -r requirements.txt
```

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

#### Frontend Issues

**Problem**: API connection refused

```bash
# Solution: Check backend is running and CORS is configured
curl http://localhost:5000/health
```

**Problem**: Build failures in Next.js

```bash
# Solution: Clear cache and reinstall
rm -rf .next node_modules
bun install
bun dev
```

#### Docker Issues

**Problem**: Port already in use

```bash
# Solution: Stop conflicting services
sudo lsof -i :5000
docker-compose down
```

**Problem**: Out of disk space

```bash
# Solution: Clean Docker cache
docker system prune -a
```

### Performance Optimization

- Use production builds for frontend (`bun build`)
- Enable Docker layer caching
- Configure appropriate worker processes in Gunicorn
- Monitor memory usage for ML models

### Logs and Debugging

```bash
# Backend logs
docker-compose logs -f app

# Nginx logs
docker-compose logs -f nginx

# System resource usage
docker stats

# Container shell access
docker exec -it launchscore-backend bash
```

---

## 🧪 Testing the Application

### Development Testing

Open your browser and go to:

- **Frontend**: `http://localhost:3000` (Next.js dev server)
- **Backend API**: `http://localhost:5000` (Flask dev server)
- **Health Check**: `http://localhost:5000/health`

### Production Testing

For deployed applications:

- **Frontend**: Your deployed frontend URL (Vercel/Netlify)
- **Backend API**: Your deployed backend URL
- **Health Check**: `https://your-backend-domain.com/health`

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
- **LLM Score**: Generated by LangChain + Gemini Pro using contextual data from similar startups retrieved by FAISS.
- **Final Score**: Average of the ML and LLM scores (scaled to 10).

---

## 📚 Tech Stack

- **Backend**: Flask + LangChain + Gemini Pro (via Google GenerativeAI)
- **Frontend**: Tailwind CSS, HTML, JavaScript
- **ML**: Scikit-learn MLPClassifier with pipeline
- **Vector Search**: FAISS
- **Embeddings**: HuggingFace `all-MiniLM-L6-v2`

---

## 📌 Requirements

- Python 3.8+
- Google API Key (for Gemini)
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
