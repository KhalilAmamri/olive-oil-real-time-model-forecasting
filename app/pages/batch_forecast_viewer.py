"""
Batch Forecast Viewer Page
View and analyze saved batch forecasts with interactive visualizations.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.forecast_batch import load_batch_forecast, generate_forecast_summary, run_batch_forecast
from src.utils import format_number, aggregate_by_period


def show():
    """Display the batch forecast viewer page."""
    
    st.title("📊 Batch Forecast Viewer")
    
    st.markdown("""
    View and analyze **batch forecasts** - large-scale predictions generated for extended periods.
    
    Batch forecasts are ideal for:
    - **Strategic Planning**: Forecast entire quarters or years
    - **Report Generation**: Create automated forecast reports
    - **Stakeholder Sharing**: Export CSV files for distribution
    - **Trend Analysis**: Analyze long-term patterns
    
    ---
    """)
    
    # Tabs for different actions
    tab1, tab2 = st.tabs(["📂 View Saved Forecasts", "🚀 Generate New Forecast"])
    
    with tab1:
        show_saved_forecasts()
    
    with tab2:
        show_forecast_generator()


def show_saved_forecasts():
    """Display saved batch forecasts."""
    
    st.subheader("📂 Saved Forecasts")
    
    # Get list of forecast files
    forecasts_dir = project_root / "data" / "forecasts"
    
    if not forecasts_dir.exists() or not list(forecasts_dir.glob("*.csv")):
        st.warning("""
        ⚠️ No batch forecasts found.
        
        Generate a new batch forecast using the "🚀 Generate New Forecast" tab, 
        or run the batch forecasting script:
        
        ```python
        from src.forecast_batch import run_batch_forecast
        
        df = run_batch_forecast(
            model_type="prophet",
            start_date="2026-01-01",
            periods=365,
            freq="D"
        )
        ```
        """)
        return
    
    # List available forecasts
    forecast_files = sorted(forecasts_dir.glob("*.csv"), key=lambda x: x.stat().st_mtime, reverse=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_file = st.selectbox(
            "Select Forecast File",
            options=[f.name for f in forecast_files],
            help="Choose a forecast file to visualize"
        )
    
    with col2:
        st.markdown(f"""
        <div style="background-color: #E8F5E9; padding: 1rem; border-radius: 5px; margin-top: 1.7rem;">
        <strong>📁 Total Files:</strong> {len(forecast_files)}
        </div>
        """, unsafe_allow_html=True)
    
    if selected_file:
        try:
            # Load forecast
            with st.spinner("Loading forecast..."):
                df = load_batch_forecast(selected_file)
            
            st.success(f"✅ Loaded {len(df)} predictions")
            
            # Display summary statistics
            st.markdown("---")
            st.subheader("📈 Forecast Summary")
            
            summary = generate_forecast_summary(df)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Predictions", format_number(summary['total_predictions'], 0))
            
            with col2:
                st.metric("Mean Production", f"{format_number(summary['production_stats']['mean'])} tons")
            
            with col3:
                st.metric("Min Production", f"{format_number(summary['production_stats']['min'])} tons")
            
            with col4:
                st.metric("Max Production", f"{format_number(summary['production_stats']['max'])} tons")
            
            # Date range
            st.markdown(f"""
            **📅 Date Range:** {summary['date_range']['start']} to {summary['date_range']['end']}
            """)
            
            # Visualization options
            st.markdown("---")
            st.subheader("📊 Visualizations")
            
            # Create tabs for different charts
            chart_tab1, chart_tab2, chart_tab3 = st.tabs(["📈 Time Series", "📊 Distribution", "🗺️ By Country"])
            
            with chart_tab1:
                # Time series plot
                fig_timeline = create_timeline_plot(df)
                st.plotly_chart(fig_timeline, use_container_width=True)
                
                # Aggregation options
                col1, col2 = st.columns([1, 3])
                with col1:
                    agg_period = st.selectbox(
                        "Aggregate by",
                        ["Daily", "Weekly", "Monthly", "Quarterly"],
                        index=2
                    )
                
                period_map = {"Daily": "D", "Weekly": "W", "Monthly": "M", "Quarterly": "Q"}
                
                if agg_period != "Daily":
                    agg_df = aggregate_by_period(df, period_map[agg_period], 'Predicted_Production_Tons')
                    fig_agg = create_aggregated_plot(agg_df, agg_period)
                    st.plotly_chart(fig_agg, use_container_width=True)
            
            with chart_tab2:
                # Distribution plot
                fig_dist = create_distribution_plot(df)
                st.plotly_chart(fig_dist, use_container_width=True)
                
                # Statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Median", f"{format_number(summary['production_stats']['median'])} tons")
                with col2:
                    st.metric("Std Dev", f"{format_number(summary['production_stats']['std'])} tons")
                with col3:
                    st.metric("Range", f"{format_number(summary['production_stats']['max'] - summary['production_stats']['min'])} tons")
            
            with chart_tab3:
                if 'Country' in df.columns:
                    # By country analysis
                    fig_country = create_country_comparison(df)
                    st.plotly_chart(fig_country, use_container_width=True)
                    
                    # Country table
                    country_stats = df.groupby('Country')['Predicted_Production_Tons'].agg(['mean', 'min', 'max', 'count'])
                    country_stats.columns = ['Average', 'Minimum', 'Maximum', 'Count']
                    country_stats = country_stats.sort_values('Average', ascending=False)
                    st.dataframe(country_stats.style.format({
                        'Average': '{:,.0f}',
                        'Minimum': '{:,.0f}',
                        'Maximum': '{:,.0f}'
                    }), use_container_width=True)
                else:
                    st.info("Country-level breakdown not available in this forecast.")
            
            # Data table
            st.markdown("---")
            st.subheader("📋 Forecast Data")
            
            # Filters
            col1, col2 = st.columns(2)
            
            with col1:
                if 'Country' in df.columns:
                    countries = ['All'] + sorted(df['Country'].unique().tolist())
                    selected_country = st.selectbox("Filter by Country", countries)
                else:
                    selected_country = 'All'
            
            with col2:
                show_rows = st.slider("Number of rows to display", 10, 100, 20)
            
            # Apply filters
            display_df = df.copy()
            if selected_country != 'All' and 'Country' in df.columns:
                display_df = display_df[display_df['Country'] == selected_country]
            
            # Display table
            st.dataframe(
                display_df.head(show_rows).style.format({
                    'Predicted_Production_Tons': '{:,.0f}',
                    'Lower_Bound': '{:,.0f}' if 'Lower_Bound' in display_df.columns else '{}',
                    'Upper_Bound': '{:,.0f}' if 'Upper_Bound' in display_df.columns else '{}'
                }),
                use_container_width=True
            )
            
            # Download button
            st.markdown("---")
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Full Forecast (CSV)",
                data=csv,
                file_name=f"forecast_export_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        except Exception as e:
            st.error(f"❌ Error loading forecast: {str(e)}")


def show_forecast_generator():
    """Interface for generating new batch forecasts."""
    
    st.subheader("🚀 Generate New Batch Forecast")
    
    st.markdown("""
    Create a new batch forecast for future periods. This will generate predictions for multiple 
    dates and save them to a CSV file.
    """)
    
    # Check if models exist
    models_dir = project_root / "models"
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_type = st.selectbox(
            "Select Model",
            ["Prophet", "RandomForest"],
            help="Choose the model to use for forecasting"
        )
    
    with col2:
        freq = st.selectbox(
            "Frequency",
            ["Daily", "Weekly", "Monthly"],
            help="Time frequency for predictions"
        )
    
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime(2026, 1, 1),
            min_value=datetime(2024, 1, 1),
            max_value=datetime(2030, 12, 31)
        )
    
    with col2:
        periods = st.number_input(
            "Number of Periods",
            value=365,
            min_value=1,
            max_value=1000,
            step=1,
            help="Number of time periods to forecast"
        )
    
    # Model-specific options
    if model_type == "RandomForest":
        st.markdown("**Countries to Forecast:**")
        countries = st.multiselect(
            "Select Countries",
            ["Italy", "Spain", "Greece", "Turkey", "Tunisia", "Portugal", "Morocco", "France"],
            default=["Italy", "Spain", "Greece"]
        )
    
    # Summary
    st.markdown("---")
    st.markdown(f"""
    **Forecast Configuration:**
    - Model: {model_type}
    - Start Date: {start_date}
    - Periods: {periods}
    - Frequency: {freq}
    - Total Predictions: {periods if model_type == 'Prophet' else periods * len(countries) if model_type == 'RandomForest' and 'countries' in locals() else periods}
    """)
    
    # Generate button
    if st.button("🚀 Generate Forecast", type="primary"):
        
        # Check model exists
        model_path = models_dir / f"{model_type.lower()}_model.pkl"
        if not model_path.exists():
            st.error(f"❌ {model_type} model not found. Please train the model first.")
            return
        
        with st.spinner(f"Generating {periods} predictions..."):
            try:
                freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "M"}
                
                kwargs = {
                    'model_type': model_type.lower(),
                    'start_date': start_date.strftime('%Y-%m-%d'),
                    'periods': periods,
                    'freq': freq_map[freq],
                    'save_results': True
                }
                
                if model_type == "RandomForest" and 'countries' in locals():
                    kwargs['countries'] = countries
                
                df = run_batch_forecast(**kwargs)
                
                st.success(f"✅ Successfully generated {len(df)} predictions!")
                
                # Show preview
                st.markdown("### 📊 Preview")
                st.dataframe(df.head(10), use_container_width=True)
                
                st.info("💡 Switch to 'View Saved Forecasts' tab to visualize the complete forecast.")
            
            except Exception as e:
                st.error(f"❌ Error generating forecast: {str(e)}")
                st.exception(e)


def create_timeline_plot(df: pd.DataFrame) -> go.Figure:
    """Create timeline plot."""
    
    fig = px.line(
        df,
        x='Date',
        y='Predicted_Production_Tons',
        title='Production Forecast Over Time',
        labels={'Predicted_Production_Tons': 'Production (tons)', 'Date': 'Date'}
    )
    
    # Add confidence intervals if available
    if 'Lower_Bound' in df.columns and 'Upper_Bound' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['Upper_Bound'],
            fill=None,
            mode='lines',
            line=dict(color='lightblue', width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['Lower_Bound'],
            fill='tonexty',
            mode='lines',
            line=dict(color='lightblue', width=0),
            name='Confidence Interval'
        ))
    
    fig.update_layout(height=500, hovermode='x unified')
    
    return fig


def create_aggregated_plot(df: pd.DataFrame, period: str) -> go.Figure:
    """Create aggregated timeline plot."""
    
    fig = px.bar(
        df,
        x='Date',
        y='Production_Tons',
        title=f'{period} Production Forecast',
        labels={'Production_Tons': 'Production (tons)', 'Date': period}
    )
    
    fig.update_layout(height=400)
    
    return fig


def create_distribution_plot(df: pd.DataFrame) -> go.Figure:
    """Create distribution histogram."""
    
    fig = px.histogram(
        df,
        x='Predicted_Production_Tons',
        nbins=50,
        title='Production Distribution',
        labels={'Predicted_Production_Tons': 'Production (tons)'}
    )
    
    fig.update_layout(height=400)
    
    return fig


def create_country_comparison(df: pd.DataFrame) -> go.Figure:
    """Create country comparison chart."""
    
    country_avg = df.groupby('Country')['Predicted_Production_Tons'].mean().sort_values(ascending=False)
    
    fig = px.bar(
        x=country_avg.index,
        y=country_avg.values,
        title='Average Production by Country',
        labels={'x': 'Country', 'y': 'Average Production (tons)'}
    )
    
    fig.update_layout(height=400)
    
    return fig
