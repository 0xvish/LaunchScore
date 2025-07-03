#!/bin/bash

# LaunchScore Backend Deployment Script

echo "🚀 Starting LaunchScore Backend Deployment..."

# Check if .env file exists
if [ ! -f .env ]; then
    echo "❌ .env file not found!"
    echo "📝 Please copy .env.example to .env and configure your settings:"
    echo "   cp .env.example .env"
    echo "   nano .env"
    exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed!"
    echo "📦 Install Docker first: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed!"
    echo "📦 Install Docker Compose first"
    exit 1
fi

echo "✅ Prerequisites check passed"

# Stop existing containers
echo "🛑 Stopping existing containers..."
docker-compose down

# Build and start services
echo "🔨 Building and starting services..."
docker-compose up -d --build

# Wait for services to start
echo "⏳ Waiting for services to start..."
sleep 10

# Check if services are running
echo "🔍 Checking service status..."
docker-compose ps

# Test health endpoint
echo "🩺 Testing health endpoint..."
sleep 5
curl -f http://localhost:5000/health || echo "⚠️ Health check failed - service might still be starting"

echo ""
echo "✅ Deployment complete!"
echo "🌐 Backend API available at: http://localhost:5000"
echo "🩺 Health check: http://localhost:5000/health"
echo "📊 Prediction endpoint: http://localhost:5000/predict"
echo ""
echo "📝 To view logs: docker-compose logs -f"
echo "🛑 To stop: docker-compose down"
