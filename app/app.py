"""
Main Streamlit Application
Entry point for the Olive Oil Export Forecasting System.
"""

import streamlit as st
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def main():
    """Main application entry point."""

    # Page configuration
    st.set_page_config(
        page_title="Olive Oil Export Forecasting",
        page_icon="🫒",
        layout="wide"
    )

    # Custom CSS for better design
    st.markdown("""
    <style>
        .main-title {
            font-size: 2.5em;
            color: #2E7D32;
            text-align: center;
            font-weight: bold;
            margin-bottom: 20px;
        }
        .subtitle {
            font-size: 1.2em;
            color: #388E3C;
            text-align: center;
            margin-bottom: 30px;
        }
        .sidebar-title {
            font-size: 1.5em;
            color: #1B5E20;
            font-weight: bold;
        }
        .metric-card {
            background-color: #2E7D32;
            color: white;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            text-align: center;
        }
        }
    </style>
    """, unsafe_allow_html=True)

    # Main title
    st.markdown('<div class="main-title">🫒 Olive Oil Export Forecasting System</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Use AI to predict export volumes with high accuracy</div>', unsafe_allow_html=True)

    # Sidebar
    st.sidebar.markdown('<div class="sidebar-title">📊 Navigation</div>', unsafe_allow_html=True)

    # Navigation
    page = st.sidebar.radio(
        "Choose page:",
        ["🏠 Home", "📈 Analytics & Prediction Dashboard"],
        index=0
    )

    # About section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About the App")
    st.sidebar.markdown("""
    **What it does:** Predicts olive oil export volumes using machine learning.

    **Data:** 7,560 records from 42 countries (2010-2024)

    **Model:** RandomForest Regressor with 26 features

    **Built with:** Python, Streamlit, scikit-learn, pandas, plotly

    **For beginners:** Simple code with clear comments and easy-to-use interface.
    """)

    # Page content
    if page == "🏠 Home":
        show_home_page()
    elif page == "📈 Analytics & Prediction Dashboard":
        from pages import export_dashboard
        export_dashboard.show()


def show_home_page():
    """Display the home page."""

    st.markdown("### Welcome to the Olive Oil Export Forecasting System! 🌿")
    st.markdown("""
    This system helps you:
    - 📊 **Analyze historical data** for olive oil exports
    - 🔮 **Predict export volumes** with high accuracy
    - 📈 **View trends and statistics** interactively

    **How to use it:**
    1. Go to "Analytics & Prediction Dashboard"
    2. Choose filters to display data
    3. Enter values for instant predictions
    """)

    # Key metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card">📊 <b>Data</b><br>7,560 records</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">🌍 <b>Countries</b><br>42 countries</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">🤖 <b>Accuracy</b><br>R² = 0.90</div>', unsafe_allow_html=True)


def show_about_page():
    """Display the about page."""
    
    st.markdown("""
    ## About This System
    
    ### 🎯 What This System Does
    
    This ML system predicts **olive oil export volumes** using production data, market prices, 
    and historical patterns. It uses a RandomForest regression model trained on 7,500+ records 
    from 42 countries spanning 2010-2024.
    
    ### 📦 Project Structure
    
    ```
    project/
    ├── data/
    │   ├── raw/              # Raw CSV datasets
    │   └── forecasts/        # Saved batch forecasts
    ├── models/               # Trained ML models
    ├── src/                  # Python modules
    │   ├── data_loader.py
    │   ├── preprocess.py
    │   ├── model_train.py
    │   ├── forecast_realtime.py
    │   ├── forecast_batch.py
    │   └── utils.py
    ├── app/                  # Streamlit application
    │   ├── app.py
    │   └── pages/
    └── notebooks/            # Jupyter notebooks
    ```
    
    ### 🛠️ Technologies Used
    
    - **Python 3.8+**: Core programming language
    - **Streamlit**: Web application framework
    - **Prophet**: Facebook's time series forecasting library
    - **scikit-learn**: Machine learning library (RandomForest)
    - **Pandas**: Data manipulation
    - **Plotly**: Interactive visualizations
    
    ### 📊 Features
    
    - ✅ Real-time export predictions with instant feedback
    - ✅ Batch forecasting for multiple periods
    - ✅ Interactive visualizations with export trends
    - ✅ RandomForest model with 26 engineered features
    - ✅ Export ratio and revenue calculations
    - ✅ CSV export functionality
    - ✅ Production-ready code structure
    - ✅ Docker deployment ready
    
    ### 🔜 Future Enhancements
    
    - Add more ML models (LSTM, XGBoost)
    - Model performance comparison dashboard
    - Automated model retraining pipeline
    - API endpoints for external integration
    - Docker containerization
    - Cloud deployment scripts
    
    ### 📝 Version
    
    **Version:** 1.0.0  
    **Last Updated:** November 2025
    """)


if __name__ == "__main__":
    main()
