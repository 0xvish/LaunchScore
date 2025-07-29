# LaunchScore - Dataset Integration Update

## 🔄 Recent Changes

The application has been updated to match the exact dataset structure used by the trained neural network model. Here are the key changes:

### ✅ Updated Input Fields

1. **Sector Dropdown**: Now contains all 145+ sectors from the training dataset including:

   - AI/Tech: `AI Chatbot`, `AI company`, `AI startup`, `Tech Startup`
   - Finance: `FinTech`, `Banking`, `Financial Services`, `InsureTech`
   - Healthcare: `HealthTech`, `Healthcare`, `BioTechnology`, `FemTech`
   - E-commerce: `E-commerce`, `B2B E-commerce`, `D2C`, `Social commerce`
   - And many more...

2. **Stage Dropdown**: Now uses actual funding stages from the dataset:

   - Early: `Pre-seed`, `Seed`, `Seed+`
   - Growth: `Series A`, `Series B`, `Series C`, `Series D`
   - Late: `Series E`, `Series F`, `Series G`, `Series H`, `Series I`
   - Other: `Bridge`, `Debt`, `PE`

3. **Headquarters Dropdown**: Contains all Indian cities from the dataset:
   - Major cities: `Bangalore`, `Mumbai`, `Delhi`, `Pune`, `Hyderabad`
   - Tier-2 cities: `Ahmedabad`, `Chennai`, `Kolkata`, `Jaipur`
   - And 40+ other locations

### 🔧 Technical Improvements

1. **Data Validation**: Added comprehensive validation to ensure inputs match the training data
2. **Error Handling**: Better error messages for invalid inputs
3. **UI Enhancements**:
   - Sorted dropdown options for better UX
   - Validation warnings in real-time
   - Disabled predict button until all fields are filled
4. **Neural Network Input**: Updated to handle the correct feature format expected by the trained model

### 📁 File Structure

The application is now modularized into:

- `config.py`: All configuration and dropdown values
- `models.py`: Neural network and LLM model loading
- `utils.py`: Prediction logic with proper validation
- `components.py`: UI components with data validation
- `styles.py`: CSS and styling
- `streamlit_app.py`: Main application (clean and concise)

### 🚀 How to Run

1. Ensure all model files are in the `models/` directory:

   - `sector_vocab.pkl`
   - `hq_vocab.pkl`
   - `stage_vocab.pkl`
   - `nn_scaler.pkl`
   - `startup_nn.pt`
   - `startup_faiss_index/` (directory)

2. Set up your `.env` file with `GOOGLE_API_KEY`

3. Run: `streamlit run streamlit_app.py`

### ⚠️ Important Notes

- The neural network model expects inputs in a specific order: `[sector_idx, stage_idx, hq_idx, founded_year, funding_amount]`
- All categorical inputs are validated against the training vocabularies
- The scaler is applied only to numerical features (founded year and funding amount)
- Input validation prevents invalid data from reaching the model

This update ensures the frontend perfectly matches the backend model's expectations, providing more accurate predictions!
