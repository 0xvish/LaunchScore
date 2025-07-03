# LaunchScore Deployment Guide

## Architecture Overview

- **Frontend**: Next.js application (deployed on Vercel/Netlify)
- **Backend**: Flask API with ML/LLM capabilities (deployed on VPS/Cloud)

## 📋 Prerequisites

### For Backend Deployment

- VPS/Cloud server (Ubuntu 20.04+ recommended)
- Docker & Docker Compose installed
- Domain/subdomain for backend API (optional but recommended)
- Google API key for Gemini LLM

### For Frontend Deployment

- Vercel/Netlify account
- GitHub repository

## 🚀 Backend Deployment

### Step 1: Server Setup

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Logout and login to apply docker group changes
```

### Step 2: Deploy Backend

```bash
# Clone your repository
git clone <your-repo-url>
cd fnn_startup_success

# Create environment file
cp .env.example .env
nano .env
```

### Step 3: Configure Environment Variables

Create `.env` file in backend directory:

```env
# Google API Configuration
GOOGLE_API_KEY=your_google_api_key_here

# Flask Configuration
FLASK_ENV=production
FLASK_DEBUG=False

# Optional: Add other configurations
PYTHONUNBUFFERED=1
```

### Step 4: Build and Deploy

```bash
# Build and start services
docker-compose up -d

# Check if services are running
docker-compose ps

# View logs
docker-compose logs -f

# Test the API
curl http://localhost:5000/health
```

### Step 5: Configure Firewall (Optional)

```bash
# Allow HTTP traffic
sudo ufw allow 5000/tcp
sudo ufw enable
```

## 🌐 Frontend Deployment

### Step 1: Prepare Frontend for Deployment

Update `frontend/.env.local` for production:

```env
# For production deployment
NEXT_PUBLIC_API_URL=https://your-backend-domain.com
# OR if using IP
NEXT_PUBLIC_API_URL=http://your-server-ip:5000
```

### Step 2: Deploy to Vercel

```bash
# Install Vercel CLI
npm i -g vercel

# Navigate to frontend directory
cd frontend

# Deploy
vercel

# Follow the prompts:
# - Link to existing project or create new
# - Set build command: npm run build
# - Set output directory: .next
# - Set install command: npm install
```

### Step 3: Configure Vercel Environment Variables

In Vercel Dashboard:

1. Go to your project settings
2. Navigate to "Environment Variables"
3. Add:
   ```
   NEXT_PUBLIC_API_URL = https://your-backend-domain.com
   ```

### Alternative: Deploy to Netlify

```bash
# Build the project
npm run build

# Deploy to Netlify (drag and drop .next folder)
# OR use Netlify CLI
npm install -g netlify-cli
netlify deploy --prod --dir=.next
```

## 🔧 Environment Configuration

### Backend Environment Variables

```env
# Required
GOOGLE_API_KEY=your_google_api_key_here

# Optional
FLASK_ENV=production
FLASK_DEBUG=False
CORS_ORIGINS=https://your-frontend-domain.vercel.app,https://your-custom-domain.com
```

### Frontend Environment Variables

```env
# Production
NEXT_PUBLIC_API_URL=https://api.yourdomain.com

# Development
NEXT_PUBLIC_API_URL=http://localhost:5000
```

## 🔒 Security Considerations

### Backend Security

1. **Update CORS origins** in `app.py`:

```python
CORS(app, origins=[
    "https://your-frontend-domain.vercel.app",
    "https://your-custom-domain.com"
])
```

2. **Use environment variables** for sensitive data
3. **Enable firewall** on your server
4. **Use HTTPS** (add SSL certificate)

### Frontend Security

1. **Never expose backend URLs** in client-side code
2. **Use environment variables** for API endpoints
3. **Enable HTTPS** on your domain

## 📊 Monitoring & Maintenance

### Backend Monitoring

```bash
# Check container status
docker-compose ps

# View logs
docker-compose logs app
docker-compose logs nginx

# Restart services
docker-compose restart

# Update deployment
git pull
docker-compose down
docker-compose up -d --build
```

### Frontend Monitoring

- Use Vercel Analytics
- Monitor API response times
- Set up error tracking (Sentry)

## 🔄 CI/CD Pipeline (Optional)

### GitHub Actions for Backend

Create `.github/workflows/deploy-backend.yml`:

```yaml
name: Deploy Backend

on:
  push:
    branches: [main]
    paths: ["!frontend/**"]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Deploy to server
        uses: appleboy/ssh-action@v0.1.5
        with:
          host: ${{ secrets.HOST }}
          username: ${{ secrets.USERNAME }}
          key: ${{ secrets.KEY }}
          script: |
            cd /path/to/your/app
            git pull
            docker-compose down
            docker-compose up -d --build
```

### Vercel Auto-Deploy

Vercel automatically deploys on git push to main branch when connected to GitHub.

## 🧪 Testing Deployment

### Test Backend

```bash
# Health check
curl https://your-backend-domain.com/health

# Test prediction endpoint
curl -X POST https://your-backend-domain.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "idea": "AI-powered food delivery",
    "sector": "FoodTech",
    "stage": "Seed",
    "headquarter": "Bengaluru",
    "founded": 2023,
    "amount": 5000000
  }'
```

### Test Frontend

1. Visit your frontend URL
2. Fill out the form
3. Submit and verify prediction results
4. Check browser console for any CORS errors

## 🚨 Troubleshooting

### Common Backend Issues

1. **CORS errors**: Update CORS origins in app.py
2. **Model loading errors**: Ensure all model files are present
3. **Memory issues**: Increase server resources
4. **Port conflicts**: Change port mapping in docker-compose.yml

### Common Frontend Issues

1. **API connection failed**: Check NEXT_PUBLIC_API_URL
2. **Build errors**: Verify all dependencies are installed
3. **Environment variables not working**: Ensure they start with NEXT*PUBLIC*

## 📱 Quick Deployment Commands

### Backend Deployment

```bash
git clone <repo>
cd fnn_startup_success
cp .env.example .env
# Edit .env with your keys
docker-compose up -d
```

### Frontend Deployment

```bash
cd frontend
vercel
# Follow prompts and add environment variables
```

## 🔗 Final URLs

- **Frontend**: `https://your-app.vercel.app`
- **Backend API**: `https://your-backend-domain.com`
- **Health Check**: `https://your-backend-domain.com/health`
- **Prediction API**: `https://your-backend-domain.com/predict`
