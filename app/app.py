"""
Main Streamlit Application
Entry point for the Olive Oil Production Forecasting System.
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
        page_title="Olive Oil Forecasting System",
        page_icon="🫒",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #2E7D32;
            text-align: center;
            padding: 1rem;
            background: linear-gradient(90deg, #f0f2f6 0%, #ffffff 100%);
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        .info-box {
            background-color: #E8F5E9;
            padding: 1rem;
            border-radius: 5px;
            border-left: 5px solid #2E7D32;
            margin: 1rem 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="main-header">🫒 Olive Oil Production Forecasting System</div>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    st.sidebar.markdown("---")
    
    # Navigation options
    page = st.sidebar.radio(
        "Select Page:",
        [
            "🏠 Home",
            "⚡ Real-Time Forecast",
            "📊 Batch Forecast Viewer",
            "ℹ️ About"
        ]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### 📌 Quick Guide
    
    **Real-Time Forecast:**
    - Instant predictions
    - User inputs features
    - Interactive results
    
    **Batch Forecast:**
    - Generate full-year forecasts
    - View saved predictions
    - Compare scenarios
    """)
    
    # Route to pages
    if page == "🏠 Home":
        show_home_page()
    elif page == "⚡ Real-Time Forecast":
        # Import and show real-time forecast page
        from pages import real_time_forecast
        real_time_forecast.show()
    elif page == "📊 Batch Forecast Viewer":
        # Import and show batch forecast viewer
        from pages import batch_forecast_viewer
        batch_forecast_viewer.show()
    elif page == "ℹ️ About":
        show_about_page()


def show_home_page():
    """Display the home page."""
    
    st.markdown("""
    ## Welcome to the Olive Oil Forecasting System 🫒
    
    This production-ready machine learning system provides **real-time** and **batch forecasting** 
    capabilities for olive oil production analysis.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### ⚡ Real-Time Forecasting
        
        **What it does:**
        - Generates instant predictions based on user inputs
        - Provides immediate feedback (< 1 second)
        - Interactive visualizations
        - What-if scenario analysis
        
        **Best for:**
        - Quick decision making
        - Interactive exploration
        - User-driven analysis
        - Dashboard applications
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Batch Forecasting
        
        **What it does:**
        - Generates forecasts for extended periods
        - Processes thousands of predictions
        - Saves results to CSV files
        - Automated reporting
        
        **Best for:**
        - Monthly/quarterly planning
        - Long-term strategy
        - Stakeholder reports
        - Scheduled updates
        """)
    
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
        #### 📈 Prophet Model
        
        **Strengths:**
        - Excellent for time series with seasonality
        - Handles missing data automatically
        - Provides uncertainty intervals
        - Robust to outliers
        
        **Best for:**
        - Long-term trend forecasting
        - Seasonal pattern analysis
        - Uncertainty quantification
        """)
    
    with col2:
        st.markdown("""
        #### 🌲 RandomForest Model
        
        **Strengths:**
        - Captures non-linear relationships
        - Handles multiple features
        - Provides feature importance
        - No feature scaling needed
        
        **Best for:**
        - Multi-feature predictions
        - Complex pattern recognition
        - Feature analysis
        """)
    
    st.markdown("---")
    
    # Getting started
    st.markdown("""
    ### 🚀 Getting Started
    
    1. **Real-Time Forecast:** Use the sidebar to navigate to "⚡ Real-Time Forecast"
    2. **Select Model:** Choose between Prophet or RandomForest
    3. **Input Features:** Enter country, date, and other features
    4. **View Results:** Get instant predictions with visualizations
    
    For batch forecasting, navigate to "📊 Batch Forecast Viewer" to see saved forecasts.
    """)


def show_about_page():
    """Display the about page."""
    
    st.markdown("""
    ## About This System
    
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
    
    - ✅ Real-time predictions with instant feedback
    - ✅ Batch forecasting for multiple periods
    - ✅ Interactive visualizations
    - ✅ Multiple model support (Prophet, RandomForest)
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
