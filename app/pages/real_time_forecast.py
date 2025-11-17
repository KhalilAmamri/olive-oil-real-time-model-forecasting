"""
Real-Time Forecast Page
Provides interactive interface for generating instant predictions.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.forecast_realtime import run_realtime_forecast, generate_forecast_range
from src.utils import format_number, get_available_countries


def show():
    """Display the real-time forecast page."""
    
    st.title("⚡ Real-Time Forecast")
    
    st.markdown("""
    Generate **instant predictions** by providing input features. Results appear immediately 
    with interactive visualizations.
    
    ---
    """)
    
    # Model selection
    col1, col2 = st.columns([1, 2])
    
    with col1:
        model_type = st.selectbox(
            "🤖 Select Model",
            ["Prophet", "RandomForest"],
            help="Choose the forecasting model to use"
        )
    
    with col2:
        st.markdown(f"""
        <div style="background-color: #E3F2FD; padding: 1rem; border-radius: 5px; margin-top: 0.5rem;">
        <strong>ℹ️ {model_type} Model:</strong> {'Time series focused with seasonality' if model_type == 'Prophet' else 'Feature-based with complex patterns'}
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Check if models exist
    models_dir = project_root / "models"
    model_filename = f"{model_type.lower()}_model.pkl"
    model_path = models_dir / model_filename
    
    if not model_path.exists():
        st.error(f"""
        ⚠️ **Model not found!**
        
        The {model_type} model has not been trained yet. Please train the model first by running:
        
        ```python
        # In a Jupyter notebook or Python script:
        from src.data_loader import load_raw_data
        from src.preprocess import preprocess_data, prepare_prophet_data, prepare_ml_data
        from src.model_train import train_prophet_model, train_rf_model, save_model
        
        # Load and preprocess data
        df = load_raw_data()
        df_processed = preprocess_data(df)
        
        # Train model
        model = train_{model_type.lower()}_model(...)
        save_model(model, "{model_filename}")
        ```
        
        Or use the training notebooks in the `notebooks/` directory.
        """)
        return
    
    # Input form
    st.subheader("📝 Input Features")
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["🎯 Simple Input", "🔧 Advanced Input"])
    
    with tab1:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            country = st.selectbox(
                "Country",
                ["Italy", "Spain", "Greece", "Turkey", "Tunisia", "Portugal", "Morocco", "France"],
                help="Select the country for prediction"
            )
        
        with col2:
            forecast_date = st.date_input(
                "Forecast Date",
                value=datetime.now() + timedelta(days=30),
                min_value=datetime(2024, 1, 1),
                max_value=datetime(2030, 12, 31),
                help="Select the date for prediction"
            )
        
        with col3:
            if model_type == "RandomForest":
                export_tons = st.number_input(
                    "Export (Tons)",
                    value=25000,
                    min_value=0,
                    max_value=100000,
                    step=1000,
                    help="Expected export volume"
                )
        
        if model_type == "RandomForest":
            col1, col2, col3 = st.columns(3)
            
            with col1:
                usd_price = st.number_input(
                    "USD Price",
                    value=12.5,
                    min_value=5.0,
                    max_value=20.0,
                    step=0.1,
                    help="Expected price per unit"
                )
            
            with col2:
                season = st.selectbox(
                    "Season",
                    ["Spring", "Summer", "Fall", "Winter"],
                    help="Season of the year"
                )
    
    with tab2:
        st.markdown("**Advanced features (for RandomForest model)**")
        
        if model_type == "RandomForest":
            col1, col2, col3 = st.columns(3)
            
            with col1:
                lag_1 = st.number_input("Lag 1", value=250000, step=10000)
                lag_7 = st.number_input("Lag 7", value=245000, step=10000)
            
            with col2:
                lag_30 = st.number_input("Lag 30", value=240000, step=10000)
                rolling_mean = st.number_input("Rolling Mean 7", value=248000, step=10000)
            
            with col3:
                rolling_std = st.number_input("Rolling Std 7", value=15000, step=1000)
        else:
            st.info("Advanced features are only used for RandomForest model")
    
    st.markdown("---")
    
    # Prediction button
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        predict_button = st.button("🚀 Generate Prediction", type="primary", use_container_width=True)
    
    # Generate prediction
    if predict_button:
        with st.spinner("🔮 Generating prediction..."):
            
            # Prepare features
            features = {}
            if model_type == "RandomForest":
                features = {
                    'Export_Tons': export_tons,
                    'USD_Price': usd_price,
                }
                
                # Add advanced features if provided
                if 'lag_1' in locals():
                    features.update({
                        'Production_Tons_lag_1': lag_1,
                        'Production_Tons_lag_7': lag_7,
                        'Production_Tons_lag_30': lag_30,
                        'Production_Tons_rolling_mean_7': rolling_mean,
                        'Production_Tons_rolling_std_7': rolling_std,
                    })
            
            # Run prediction
            result = run_realtime_forecast(
                model_type=model_type.lower(),
                country=country,
                date=forecast_date.strftime('%Y-%m-%d'),
                features=features
            )
            
            # Display results
            if result['success']:
                st.success("✅ Prediction completed successfully!")
                
                # Main prediction display
                st.markdown("---")
                st.subheader("📊 Prediction Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label="Predicted Production",
                        value=f"{format_number(result['prediction'])} tons",
                        delta=None
                    )
                
                with col2:
                    if 'lower_bound' in result:
                        st.metric(
                            label="Lower Bound (95%)",
                            value=f"{format_number(result['lower_bound'])} tons"
                        )
                
                with col3:
                    if 'upper_bound' in result:
                        st.metric(
                            label="Upper Bound (95%)",
                            value=f"{format_number(result['upper_bound'])} tons"
                        )
                
                # Visualization
                st.markdown("---")
                st.subheader("📈 Forecast Visualization")
                
                # Create gauge chart
                fig = create_gauge_chart(result['prediction'])
                st.plotly_chart(fig, use_container_width=True)
                
                # Time series context (show nearby dates)
                try:
                    st.markdown("---")
                    st.subheader("📅 Forecast Context (30 days)")
                    
                    start_date = (forecast_date - timedelta(days=15)).strftime('%Y-%m-%d')
                    end_date = (forecast_date + timedelta(days=15)).strftime('%Y-%m-%d')
                    
                    context_df = generate_forecast_range(
                        model_type=model_type.lower(),
                        start_date=start_date,
                        end_date=end_date,
                        country=country,
                        freq='D'
                    )
                    
                    if not context_df.empty:
                        fig_timeline = create_timeline_chart(context_df, forecast_date)
                        st.plotly_chart(fig_timeline, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not generate context chart: {str(e)}")
                
                # Model details
                with st.expander("🔍 Prediction Details"):
                    st.json(result)
            
            else:
                st.error(f"❌ Prediction failed: {result.get('error', 'Unknown error')}")


def create_gauge_chart(value: float, max_value: float = 500000) -> go.Figure:
    """Create a gauge chart for the prediction."""
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Production (tons)", 'font': {'size': 24}},
        delta={'reference': max_value * 0.5},
        gauge={
            'axis': {'range': [None, max_value], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#2E7D32"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, max_value * 0.33], 'color': '#FFCDD2'},
                {'range': [max_value * 0.33, max_value * 0.66], 'color': '#FFF9C4'},
                {'range': [max_value * 0.66, max_value], 'color': '#C8E6C9'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_value * 0.9
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig


def create_timeline_chart(df: pd.DataFrame, highlight_date: datetime) -> go.Figure:
    """Create timeline chart with highlighted prediction date."""
    
    fig = go.Figure()
    
    # Add line for all predictions
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Prediction'],
        mode='lines',
        name='Forecast',
        line=dict(color='#1976D2', width=2)
    ))
    
    # Highlight the selected date
    highlight_df = df[df['Date'] == pd.to_datetime(highlight_date)]
    if not highlight_df.empty:
        fig.add_trace(go.Scatter(
            x=highlight_df['Date'],
            y=highlight_df['Prediction'],
            mode='markers',
            name='Selected Date',
            marker=dict(size=15, color='#D32F2F', symbol='star')
        ))
    
    fig.update_layout(
        title="Production Forecast Timeline",
        xaxis_title="Date",
        yaxis_title="Production (tons)",
        hovermode='x unified',
        height=400,
        showlegend=True
    )
    
    return fig


# Additional helper info
def show_model_info():
    """Show information about the models."""
    
    st.markdown("""
    ### 🤖 Model Information
    
    #### Prophet Model
    - **Type**: Time Series Forecasting
    - **Inputs**: Date (automatically extracts seasonality)
    - **Outputs**: Prediction with confidence intervals
    - **Best for**: Long-term trends with seasonal patterns
    
    #### RandomForest Model
    - **Type**: Ensemble Machine Learning
    - **Inputs**: Multiple features (date, price, exports, etc.)
    - **Outputs**: Single point prediction
    - **Best for**: Complex feature interactions
    """)
