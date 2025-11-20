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

    # Simple header
    st.title("🫒 Olive Oil Export Forecasting System")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to:",
        ["🏠 Home", "📊 Export Dashboard", "ℹ️ About"]
    )

    # Route to pages
    if page == "🏠 Home":
        show_home_page()
    elif page == "📊 Export Dashboard":
        from pages import export_dashboard
        export_dashboard.show()
    elif page == "ℹ️ About":
        show_about_page()


def show_home_page():
    """Display the home page."""

    st.markdown("""
    ## Welcome! 🫒

    This app predicts olive oil export volumes using machine learning.

    **Features:**
    - 📊 **Analytics**: Explore export trends and patterns
    - ⚡ **Predictions**: Get instant export forecasts
    - 🌍 **Countries**: Data from 42 countries (2010-2024)

    **How to use:**
    1. Go to "Export Dashboard" in the sidebar
    2. Explore data in the Analytics tab
    3. Make predictions in the Real-Time Prediction tab
    """)

    st.info("💡 First time? Start with the Analytics tab to explore the data!")
    
    st.markdown("---")
    
    # Why Streamlit section
    st.markdown("""
    ### 🎯 Why Streamlit Instead of Power BI?
    
    This system uses **Streamlit** for visualization and interaction, which offers significant advantages 
    over traditional BI tools like Power BI:
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        #### 🔄 Real-Time ML Integration
        - Direct Python integration
        - Load ML models instantly
        - Run predictions in real-time
        - No data export needed
        """)
    
    with col2:
        st.markdown("""
        #### 🚀 Faster Development
        - Pure Python code
        - No separate BI tool
        - Version control friendly
        - Easy deployment
        """)
    
    with col3:
        st.markdown("""
        #### 💰 Cost Effective
        - Free and open source
        - No licensing fees
        - Self-hosted option
        - Cloud deployment ready
        """)
    
    st.markdown("""
    <div class="info-box">
    <strong>📍 Key Insight:</strong> Streamlit keeps everything in Python - from data preprocessing 
    to ML models to visualization. This eliminates the need to export data to external BI tools, 
    reduces latency, and keeps your entire ML pipeline in one codebase.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Models section
    st.markdown("""
    ### 🤖 Available Models
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 📈 Prophet Model (Optional)
        
        **Strengths:**
        - Excellent for time series with seasonality
        - Handles missing data automatically
        - Provides uncertainty intervals
        - Robust to outliers
        
        **Note:**
        - Currently not installed (Python 3.13 compatibility)
        - Can be added separately if needed
        """)
    
    with col2:
        st.markdown("""
        #### 🌲 RandomForest Model (Active)
        
        **Strengths:**
        - Captures non-linear relationships
        - Handles multiple features
        - Provides feature importance
        - No feature scaling needed
        
        **Current Use:**
        - Predicts export volumes from production data
        - Uses 26 engineered features
        - R² Score: 0.79 (training)
        - Top features: Export lag features, country encoding
        """)
    
    st.markdown("---")
    
    # Getting started
    st.markdown("""
    ### 🚀 Getting Started
    
    1. **Explore Analytics:** Use the sidebar to navigate to "📊 Export Dashboard"
    2. **Filter Data:** Select countries, date ranges, and seasons to explore patterns
    3. **View Insights:** Analyze trends, comparisons, and market relationships
    4. **Make Predictions:** Switch to the Real-Time Prediction tab for instant forecasts
    5. **Enter Parameters:** Input production volume, price, and country for predictions
    
    The dashboard combines data exploration with predictive analytics for comprehensive olive oil export analysis.
    """)


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


def show_about_page():
    """Display the about page."""

    st.markdown("""
    ## About This App

    **What it does:** Predicts olive oil export volumes using machine learning.

    **Data:** 7,560 records from 42 countries (2010-2024)

    **Model:** RandomForest Regressor with 26 features

    **Built with:** Python, Streamlit, scikit-learn, pandas, plotly

    **For beginners:** Simple code with clear comments and easy-to-use interface.
    """)


if __name__ == "__main__":
    main()
