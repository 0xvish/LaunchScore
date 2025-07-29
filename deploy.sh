#!/bin/bash

# LaunchScore Streamlit Deployment Script
# This script automates the deployment process for the LaunchScore application

set -e  # Exit on any error

echo "🚀 Starting LaunchScore Deployment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if .env file exists
if [ ! -f .env ]; then
    print_warning ".env file not found. Creating template..."
    echo "GOOGLE_API_KEY=your-google-api-key-here" > .env
    print_error "Please edit .env file and add your Google API key before proceeding"
    exit 1
fi

# Check if GOOGLE_API_KEY is set
if ! grep -q "GOOGLE_API_KEY=" .env || grep -q "your-google-api-key-here" .env; then
    print_error "Please set a valid GOOGLE_API_KEY in your .env file"
    exit 1
fi

# Check if models directory exists
if [ ! -d "models" ]; then
    print_error "Models directory not found. Please ensure your trained models are in the 'models/' directory"
    exit 1
fi

# Check for required model files
required_files=("models/sector_vocab.pkl" "models/hq_vocab.pkl" "models/nn_scaler.pkl" "models/startup_nn.pt")
missing_files=()

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -ne 0 ]; then
    print_error "Missing required model files:"
    for file in "${missing_files[@]}"; do
        echo "  - $file"
    done
    exit 1
fi

# Choose deployment method
echo ""
echo "Choose deployment method:"
echo "1) Local development server"
echo "2) Docker deployment"
echo "3) Production server setup"
read -p "Enter your choice (1-3): " choice

case $choice in
    1)
        print_status "Setting up local development environment..."
        
        # Check if virtual environment exists
        if [ ! -d "venv" ]; then
            print_status "Creating virtual environment..."
            python3 -m venv venv
        fi
        
        print_status "Activating virtual environment and installing dependencies..."
        source venv/bin/activate
        pip install --upgrade pip
        pip install -r requirements.txt
        
        print_success "Setup complete! Starting Streamlit development server..."
        print_status "Access your app at: http://localhost:8501"
        streamlit run streamlit_app.py
        ;;
        
    2)
        print_status "Setting up Docker deployment..."
        
        # Check if Docker is installed
        if ! command -v docker &> /dev/null; then
            print_error "Docker is not installed. Please install Docker first."
            exit 1
        fi
        
        # Check if  Compose is installed
        if ! command -v docker-compose &> /dev/null; then
            print_error "Docker Compose is not installed. Please install Docker Compose first."
            exit 1
        fi
        
        print_status "Building and starting Docker containers..."
        docker-compose down 2>/dev/null || true
        docker-compose up -d --build
        
        print_success "Docker deployment complete!"
        print_status "Access your app at: http://localhost:8501"
        print_status "View logs with: docker-compose logs -f"
        ;;
        
    3)
        print_status "Setting up production server..."
        
        # Install system dependencies
        print_status "Installing system dependencies..."
        sudo apt update
        sudo apt install -y python3 python3-pip python3-venv nginx screen
        
        # Setup application
        if [ ! -d "venv" ]; then
            print_status "Creating virtual environment..."
            python3 -m venv venv
        fi
        
        print_status "Installing Python dependencies..."
        source venv/bin/activate
        pip install --upgrade pip
        pip install -r requirements.txt
        
        # Setup systemd service
        print_status "Creating systemd service..."
        sudo tee /etc/systemd/system/launchscore.service > /dev/null <<EOF
[Unit]
Description=LaunchScore Streamlit App
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
Environment=PATH=$(pwd)/venv/bin
ExecStart=$(pwd)/venv/bin/streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
Restart=always

[Install]
WantedBy=multi-user.target
EOF

        # Start and enable service
        sudo systemctl daemon-reload
        sudo systemctl enable launchscore
        sudo systemctl start launchscore
        
        print_success "Production setup complete!"
        print_status "Service status: sudo systemctl status launchscore"
        print_status "View logs: sudo journalctl -u launchscore -f"
        print_status "Access your app at: http://your-server-ip:8501"
        ;;
        
    *)
        print_error "Invalid choice. Please run the script again."
        exit 1
        ;;
esac

print_success "🎉 LaunchScore deployment completed successfully!"
