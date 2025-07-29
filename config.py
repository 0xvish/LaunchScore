"""
Configuration settings for the LaunchScore application
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Model paths
MODEL_PATHS = {
    "faiss_index": "models/startup_faiss_index",
    "sector_vocab": "models/sector_vocab.pkl",
    "hq_vocab": "models/hq_vocab.pkl",
    "stage_vocab": "models/stage_vocab.pkl",
    "nn_scaler": "models/nn_scaler.pkl",
    "startup_nn": "models/startup_nn.pt"
}

# LLM Configuration
LLM_CONFIG = {
    "model": "models/gemini-1.5-flash-latest",
    "temperature": 0.5,
    "embedding_model": "all-MiniLM-L6-v2"
}

# App Configuration
APP_CONFIG = {
    "page_title": "🚀 LaunchScore - Startup Success Predictor",
    "page_icon": "🚀",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Development stages (matching trained model)
DEVELOPMENT_STAGES = [
    'Bridge', 'Debt', 'PE', 'Pre-seed', 'Pre-series A', 'Pre-series A1', 
    'Pre-series B', 'Seed', 'Seed+', 'Series A', 'Series A+', 'Series A2', 
    'Series B', 'Series B3', 'Series C', 'Series D', 'Series D1', 'Series E', 
    'Series F', 'Series F2', 'Series G', 'Series H', 'Series I'
]

# Sectors (matching trained model)
SECTORS = [
    'AI Chatbot', 'AI company', 'AI startup', 'Advertisement', 'Aeorspace', 
    'AgriTech', 'Analytics', 'Apparel & Fashion', 'Automation', 'Automotive', 
    'Aviation', 'B2B', 'B2B E-commerce', 'B2B Ecommerce', 'B2B Manufacturing', 
    'B2B Travel', 'B2B service', 'B2B startup', 'Banking', 'Beverages', 
    'BioTechnology', 'Biotechnology', 'Blockchain', 'Blockchain startup', 
    'Blogging', 'Business Supplies & Equipment', 'Cannabis startup', 
    'Capital Markets', 'Celebrity Engagement', 'Clothing', 'Cloud kitchen', 
    'Co-working', 'Community', 'Community platform', 'Computer & Network Security', 
    'Computer Software', 'Computer software', 'Construction', 'Consulting', 
    'Consumer Goods', 'Consumer Services', 'Content publishing', 'Cosmetics', 
    'Crypto', 'D2C', 'D2C Business', 'Dating', 'Deep Tech', 'Deeptech', 
    'Delivery service', 'Design', 'Drone', 'E-commerce', 'E-learning', 
    'EV startup', 'EdTech', 'Education', 'Education Management', 'Entertainment', 
    'Equity Management', 'Farming', 'Fashion', 'Fashion & Lifestyle', 'FemTech', 
    'Femtech', 'FinTech', 'Finance', 'Financial Services', 'Fishery', 'Fitness', 
    'Food', 'Food & Beverages', 'Food and Beverages', 'Foootwear', 'Furniture', 
    'Gaming', 'HR Tech', 'Hauz Khas', 'Health', 'Health, Wellness & Fitness', 
    'HealthCare', 'HealthTech', 'Healthcare', 'Healthtech', 'Heathcare', 
    'Helathcare', 'Higher Education', 'Hospital & Health Care', 'Hospitality', 
    'Human Resources', 'IT', 'IT startup', 'Industrial Automation', 
    'Information Services', 'Information Technology', 'Information Technology & Services', 
    'Innovation Management', 'Insurance', 'InsureTech', 'Insuretech', 
    'Interior Design', 'Internet', 'Job discovery platform', 'Logistics', 
    'Logistics & Supply Chain', 'MLOps platform', 'Management Consulting', 
    'Manchester, Greater Manchester', 'Manufacturing startup', 'MarTech', 
    'Maritime', 'Marketing', 'Marketing & Advertising', 'Matrimony', 
    'Mechanical & Industrial Engineering', 'Media', 'Merchandise', 'Milk startup', 
    'Mobile Games', 'Mobility', 'Music', 'NFT', 'OTT', 'Online Media', 'Pet care', 
    'Product studio', 'Professional Training & Coaching', 'Real Estate', 
    'Recruitment', 'Renewable Energy', 'Rental', 'Retail', 'SaaS', 'SaaS startup', 
    'Sales and Distribution', 'Skill development', 'Social Media', 'Social commerce', 
    'Social network', 'Social platform', 'Software', 'Software Startup', 
    'Software company', 'Solar', 'SpaceTech', 'Spiritual', 'Sports startup', 
    'SportsTech', 'Supply chain platform', 'TaaS startup', 'Tech Startup', 
    'Tech startup', 'Telecommunications', 'Textiles', 'Tourism', 'Trading platform', 
    'Translation & Localization', 'Transportation', 'Vehicle repair startup', 
    'Veterinary', 'Video communication', 'Wholesale', 'sports'
]

# Headquarters (matching trained model)
HEADQUARTERS = [
    'Ahmadabad', 'Ahmedabad', 'Andheri', 'Bangalore', 'Chandigarh', 'Chennai', 
    'Cochin', 'Faridabad, Haryana', 'Food & Beverages', 'Gandhinagar', 
    'Ghaziabad', 'Goa', 'Gujarat', 'Gurgaon', 'Gurugram', 'Haryana', 
    'Hyderabad', 'Indore', 'Information Technology & Services', 'Jaipur', 
    'Jodhpur', 'Kochi', 'Kolkata', 'Mumbai', 'Nagpur', 'New Delhi', 'Noida', 
    'Orissia', 'Panaji', 'Patna', 'Powai', 'Pune', 'Ranchi', 'Silvassa', 
    'Small Towns, Andhra Pradesh', 'Surat', 'Telangana', 'Telugana', 'Thane', 
    'The Nilgiris', 'Trivandrum'
]

# Score thresholds
SCORE_THRESHOLDS = {
    "high": 7,
    "medium": 4
}
