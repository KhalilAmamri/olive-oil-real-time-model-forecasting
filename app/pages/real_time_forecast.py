"""
Real-Time Olive Oil Production Forecasting Page
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.forecast import predict_export
from src.data_loader import load_olive_oil_data, get_available_countries


def show():
    st.title("⚡ Real-Time Olive Oil Export Forecast")
    st.markdown("Get instant export volume predictions.")
    
    model_path = project_root / "models" / "rf_olive_oil_model.pkl"
    if not model_path.exists():
        st.error("Model not found! Please train first:")
        st.code("python scripts/train_models.py")
        return
    
    try:
        df = load_olive_oil_data()
        countries = get_available_countries(df)
        avg_export = df['Export_Tons'].mean()
        avg_price = df['USD_Price'].mean()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Forecast Parameters")
        
        forecast_date = st.date_input(
            "Forecast Date",
            value=datetime.now() + timedelta(days=30),
            min_value=datetime(2024, 1, 1),
            max_value=datetime(2030, 12, 31)
        )
        
        country = st.selectbox(
            "Country",
            options=countries,
            index=countries.index("Italy") if "Italy" in countries else 0
        )
        
        production_tons = st.number_input(
            "Expected Production Volume (tons)",
            min_value=0.0,
            max_value=500000.0,
            value=float(df['Production_Tons'].mean()),
            step=1000.0
        )
    
    with col2:
        st.subheader("Market Conditions")
        
        usd_price = st.number_input(
            "USD Price per Ton",
            min_value=0.0,
            max_value=50.0,
            value=float(avg_price),
            step=0.5
        )
        
        use_lags = st.checkbox("Use historical data", value=False)
        
        if use_lags:
            lag_1 = st.number_input("Export 1 period ago (tons)", value=20000.0)
            lag_7 = st.number_input("Export 7 periods ago (tons)", value=19500.0)
        else:
            lag_1 = None
            lag_7 = None
    
    if st.button("Generate Forecast", type="primary"):
        with st.spinner("Generating..."):
            try:
                result = predict_export(
                    model_name="rf_olive_oil_model.pkl",
                    date=datetime.combine(forecast_date, datetime.min.time()),
                    country=country,
                    production_tons=production_tons,
                    usd_price=usd_price,
                    lag_1=lag_1,
                    lag_7=lag_7
                )
                
                st.success("Forecast Generated!")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Predicted Export",
                        f"{result['predicted_export_tons']:,.0f} tons"
                    )
                
                with col2:
                    export_ratio = (result['predicted_export_tons'] / production_tons) * 100
                    st.metric("Export Ratio", f"{export_ratio:.1f}%")
                
                with col3:
                    revenue = result['predicted_export_tons'] * usd_price
                    st.metric("Est. Revenue", f"${revenue/1e6:.2f}M")
                
                fig = go.Figure()
                fig.add_trace(go.Indicator(
                    mode="gauge+number",
                    value=result['predicted_export_tons'],
                    title={'text': "Export Volume (tons)"},
                    gauge={
                        'axis': {'range': [None, 100000]},
                        'bar': {'color': "#1976D2"},
                        'steps': [
                            {'range': [0, 30000], 'color': "#FFF9C4"},
                            {'range': [30000, 60000], 'color': "#BBDEFB"},
                            {'range': [60000, 100000], 'color': "#64B5F6"}
                        ]
                    }
                ))
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error: {e}")


if __name__ == "__main__":
    show()
