"""
Unified Olive Oil Export Dashboard
Combines real-time prediction with analytics dashboard.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.forecast import predict_export
from src.data_loader import load_olive_oil_data, get_available_countries


def show():
    st.title("📈 Olive Oil Export Analytics & Prediction Dashboard")
    st.markdown("### Choose filters to display historical data, then enter values for instant predictions")

    # Check if model exists
    model_path = project_root / "models" / "rf_olive_oil_model.pkl"
    if not model_path.exists():
        st.error("Model not found! Please train first:")
        st.code("python scripts/train_models.py")
        return

    # Load data
    try:
        df = load_olive_oil_data()
        countries = get_available_countries(df)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    # Create tabs for dashboard and prediction
    tab1, tab2 = st.tabs(["📊 Analytics Dashboard", "⚡ Real-Time Prediction"])

    with tab1:
        show_analytics_dashboard(df, countries)

    with tab2:
        show_real_time_prediction(df, countries)


def show_analytics_dashboard(df, countries):
    """Display analytics dashboard with filters and visualizations."""
    st.header("Export Analytics Dashboard")

    # Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        selected_countries = st.multiselect(
            "Select Countries",
            options=countries,
            default=["Italy", "Spain", "Greece"] if all(c in countries for c in ["Italy", "Spain", "Greece"]) else countries[:3],
            help="Choose countries to analyze"
        )

    with col2:
        date_range = st.date_input(
            "Date Range",
            value=(datetime(2020, 1, 1), datetime.now()),
            help="Select date range for analysis"
        )

    with col3:
        seasons = ["All", "Spring", "Summer", "Autumn", "Winter"]
        selected_season = st.selectbox("Season", seasons, help="Filter by olive oil season")

    # Filter data
    filtered_df = df.copy()
    if selected_countries:
        filtered_df = filtered_df[filtered_df['Country'].isin(selected_countries)]

    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = filtered_df[
            (filtered_df['Date'] >= pd.Timestamp(start_date)) &
            (filtered_df['Date'] <= pd.Timestamp(end_date))
        ]

    if selected_season != "All":
        season_map = {
            "Spring": [3, 4, 5],
            "Summer": [6, 7, 8],
            "Autumn": [9, 10, 11],
            "Winter": [12, 1, 2]
        }
        months = season_map[selected_season]
        filtered_df = filtered_df[filtered_df['Date'].dt.month.isin(months)]

    if filtered_df.empty:
        st.warning("No data available for selected filters.")
        return

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Records", f"{len(filtered_df):,}")

    with col2:
        avg_export = filtered_df['Export_Tons'].mean()
        st.metric("Avg Export (tons)", f"{avg_export:,.0f}")

    with col3:
        avg_price = filtered_df['USD_Price'].mean()
        st.metric("Avg Price ($/ton)", f"{avg_price:.2f}")

    with col4:
        total_export = filtered_df['Export_Tons'].sum() / 1e6
        st.metric("Total Export (M tons)", f"{total_export:.2f}")

    # Visualizations
    st.subheader("Export Trends")

    # Time series chart
    fig_ts = px.line(
        filtered_df,
        x='Date',
        y='Export_Tons',
        color='Country',
        title="Export Volume Over Time",
        labels={'Export_Tons': 'Export (tons)', 'Date': 'Date'}
    )
    fig_ts.update_layout(height=400)
    st.plotly_chart(fig_ts, use_container_width=True)

    # Country comparison
    if len(selected_countries) > 1:
        col1, col2 = st.columns(2)

        with col1:
            fig_bar = px.bar(
                filtered_df.groupby('Country')['Export_Tons'].mean().reset_index(),
                x='Country',
                y='Export_Tons',
                title="Average Export by Country",
                labels={'Export_Tons': 'Avg Export (tons)'}
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            fig_box = px.box(
                filtered_df,
                x='Country',
                y='Export_Tons',
                title="Export Distribution by Country",
                labels={'Export_Tons': 'Export (tons)'}
            )
            st.plotly_chart(fig_box, use_container_width=True)

    # Price vs Export scatter
    fig_scatter = px.scatter(
        filtered_df,
        x='USD_Price',
        y='Export_Tons',
        color='Country',
        title="Price vs Export Volume",
        labels={'USD_Price': 'Price ($/ton)', 'Export_Tons': 'Export (tons)'}
    )
    st.plotly_chart(fig_scatter, use_container_width=True)


def show_real_time_prediction(df, countries):
    """Display real-time prediction interface."""
    st.header("Real-Time Export Prediction")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📝 Input Parameters")

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
            step=1000.0,
            help="Enter expected olive oil production volume"
        )

    with col2:
        st.subheader("💰 Market Conditions")

        usd_price = st.number_input(
            "USD Price per Ton",
            min_value=0.0,
            max_value=50.0,
            value=float(df['USD_Price'].mean()),
            step=0.5,
            help="Current market price per ton"
        )

        use_historical = st.checkbox("Use Historical Export Data", value=False)

        lag_1 = None
        lag_7 = None

        if use_historical:
            # Get recent data for the selected country
            country_data = df[df['Country'] == country].sort_values('Date')

            if not country_data.empty:
                recent_export = country_data['Export_Tons'].iloc[-1]
                lag_1 = st.number_input(
                    "Recent Export (tons)",
                    value=float(recent_export),
                    help="Most recent export volume"
                )

                if len(country_data) >= 7:
                    week_ago_export = country_data['Export_Tons'].iloc[-7]
                    lag_7 = st.number_input(
                        "Export 7 Days Ago (tons)",
                        value=float(week_ago_export),
                        help="Export volume 7 days ago"
                    )
            else:
                st.warning("No historical data for selected country. Using defaults.")

    # Prediction button
    if st.button("🔮 Generate Prediction", type="primary", use_container_width=True):
        with st.spinner("Generating prediction..."):
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

                # Display results
                st.success("✅ Prediction Generated!")

                # Metrics
                col1, col2, col3, col4 = st.columns(4)

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

                with col4:
                    profit_margin = (usd_price * 0.3)  # Assuming 30% margin
                    profit = result['predicted_export_tons'] * profit_margin
                    st.metric("Est. Profit", f"${profit/1e6:.2f}M")

                # Visualization
                st.subheader("📊 Prediction Visualization")

                fig = go.Figure()

                # Gauge chart
                fig.add_trace(go.Indicator(
                    mode="gauge+number",
                    value=result['predicted_export_tons'],
                    title={'text': "Predicted Export Volume (tons)"},
                    gauge={
                        'axis': {'range': [0, max(100000, result['predicted_export_tons'] * 1.2)]},
                        'bar': {'color': "#2E7D32"},
                        'steps': [
                            {'range': [0, result['predicted_export_tons'] * 0.5], 'color': "#FFF9C4"},
                            {'range': [result['predicted_export_tons'] * 0.5, result['predicted_export_tons'] * 0.8], 'color': "#BBDEFB"},
                            {'range': [result['predicted_export_tons'] * 0.8, result['predicted_export_tons'] * 1.2], 'color': "#64B5F6"}
                        ]
                    }
                ))

                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

                # Comparison with historical data
                country_data = df[df['Country'] == country]
                if not country_data.empty:
                    st.subheader("📈 Historical Comparison")

                    # Get last 12 months
                    last_year = country_data[country_data['Date'] >= (datetime.now() - timedelta(days=365))]

                    if not last_year.empty:
                        avg_historical = last_year['Export_Tons'].mean()

                        fig_comp = go.Figure()

                        fig_comp.add_trace(go.Bar(
                            x=['Historical Avg', 'Prediction'],
                            y=[avg_historical, result['predicted_export_tons']],
                            marker_color=['#1976D2', '#2E7D32']
                        ))

                        fig_comp.update_layout(
                            title=f"Export Comparison for {country}",
                            yaxis_title="Export Volume (tons)"
                        )

                        st.plotly_chart(fig_comp, use_container_width=True)

            except Exception as e:
                st.error(f"❌ Error generating prediction: {e}")
                st.info("Please check your inputs and try again.")


if __name__ == "__main__":
    show()