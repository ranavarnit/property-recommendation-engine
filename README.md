# 🏠 Property Recommendation Engine

A Streamlit-based real estate recommendation system that helps users find properties in Bengaluru based on budget, location, and BHK preferences.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)

## 🚀 Features

- **Smart Filtering**: Filter by budget (±20% range), location, and BHK count
- **Rich Visualizations**: 
  - Price distribution histograms
  - Location-based pie charts
  - Price vs. square footage scatter plots
  - Price per sqft analysis
  - Top locations by average price
- **Price Trends Table**: Location-wise price statistics (mean, median, min, max)
- **Summary Metrics**: Quick stats cards showing properties found, average price, size, and baths
- **One-Click Reset**: Reset all filters to defaults instantly

## 📊 Dataset

Uses the **Bengaluru House Data** dataset containing 13,320 property records with:

**Source**: [Kaggle - Bengaluru House Price Data](https://www.kaggle.com/datasets/ameythakur20/bangalore-house-prices) citeweb_search:17#1

> **Note**: The dataset is not included in this repository due to size. Please download it from Kaggle and place it as `Bengaluru_House_Data.csv` in the project root.
- Location, area type, size (BHK/Bedroom)
- Total square footage, bathrooms, balconies
- Price in Lakhs

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/property-recommendation-engine.git
   cd property-recommendation-engine
   ```

2. **Create virtual environment (recommended)**
   ```bash
   python -m venv venv

   # On Windows
   venv\Scripts\activate

   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**
   ```bash
   streamlit run app.py
   ```

The app will open at `http://localhost:8501`

## 📁 Project Structure

```
property-recommendation-engine/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── LICENSE               # MIT License
├── Bengaluru_House_Data.csv  # Dataset (not included, download separately)
└── assets/
    └── screenshot.png     # App screenshot
```

## 🎯 How to Use

1. **Set Your Budget**: Use the slider to set your budget in ₹ Lakhs
2. **Choose Location**: Select a preferred area or keep "Any" for all locations
3. **Select BHK**: Pick 1-43 BHK options or "Any" for all sizes
4. **Set Recommendations**: Choose how many properties to display (1-10)
5. **View Results**: Browse recommended properties with detailed metrics
6. **Analyze Charts**: Explore price distributions, location breakdowns, and price analysis
7. **Reset Anytime**: Click "🔄 Reset Filters" to start over

## 🧠 How It Works

### Recommendation Algorithm
1. **Data Preprocessing**: 
   - Extracts numeric BHK from strings (e.g., "2 BHK" → 2)
   - Handles range values in square footage (e.g., "2100-2850" → average)
   - Fills missing values with median/mode
   - Encodes categorical variables

2. **Similarity Calculation**:
   - Standardizes features using StandardScaler
   - Computes cosine similarity between properties
   - Ranks by price proximity to budget

3. **Filtering**:
   - Budget filter: ±20% of selected budget
   - Location filter: Exact match or "Any"
   - BHK filter: Exact match or "Any"

## 📸 Screenshots

*Add your screenshots to the `assets/` folder*

## 📝 Requirements

- streamlit >= 1.28.0
- pandas >= 1.5.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- plotly >= 5.18.0

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset: Bengaluru House Data from Kaggle
- Built with [Streamlit](https://streamlit.io/)
- Visualizations powered by [Plotly](https://plotly.com/)

## 📧 Contact

Your Name - [@yourtwitter](https://twitter.com/yourtwitter) - email@example.com

Project Link: [https://github.com/yourusername/property-recommendation-engine](https://github.com/yourusername/property-recommendation-engine)
