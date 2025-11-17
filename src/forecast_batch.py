"""
Batch Forecasting Module
Handles generating forecasts for extended periods and saving to CSV.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.model_train import load_model


def run_batch_forecast(
    model_type: str,
    start_date: str,
    periods: int,
    freq: str = 'D',
    countries: List[str] = None,
    save_results: bool = True
) -> pd.DataFrame:
    """
    Generate batch forecasts for multiple periods and optionally save to CSV.
    
    Batch forecasting means:
    - Generate predictions for many future dates at once (e.g., full year)
    - Process large volumes efficiently
    - Save results to file for later analysis
    - Share predictions with stakeholders
    - Schedule regular forecast updates
    
    This is ideal for:
    - Monthly/quarterly business planning
    - Long-term strategic decisions
    - Report generation
    - Automated forecast pipelines
    
    Key differences from real-time:
    - Real-time: One prediction on-demand, instant response
    - Batch: Thousands of predictions, run periodically, saved to file
    
    Args:
        model_type: "prophet" or "rf"
        start_date: Start date for forecasting (YYYY-MM-DD)
        periods: Number of periods to forecast
        freq: Frequency ('D'=daily, 'W'=weekly, 'M'=monthly)
        countries: List of countries to forecast for
        save_results: Whether to save results to CSV
    
    Returns:
        DataFrame with batch forecast results
    """
    print(f"\n{'='*60}")
    print(f"Starting Batch Forecast")
    print(f"{'='*60}")
    print(f"Model: {model_type.upper()}")
    print(f"Start Date: {start_date}")
    print(f"Periods: {periods} ({freq})")
    print(f"{'='*60}\n")
    
    try:
        if model_type.lower() == "prophet":
            results_df = batch_forecast_prophet(start_date, periods, freq)
        elif model_type.lower() == "rf":
            results_df = batch_forecast_rf(start_date, periods, freq, countries)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Add metadata columns
        results_df['Model'] = model_type.upper()
        results_df['Generated_At'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Save to CSV if requested
        if save_results:
            save_path = save_forecast_results(results_df, model_type)
            print(f"\n✓ Batch forecast completed!")
            print(f"✓ Results saved to: {save_path}")
            print(f"✓ Total predictions: {len(results_df)}")
        
        return results_df
    
    except Exception as e:
        print(f"✗ Error in batch forecasting: {str(e)}")
        raise


def batch_forecast_prophet(start_date: str, periods: int, freq: str = 'D') -> pd.DataFrame:
    """
    Generate batch forecast using Prophet model.
    
    Args:
        start_date: Start date for forecast
        periods: Number of periods
        freq: Frequency
    
    Returns:
        DataFrame with forecast results
    """
    # Load Prophet model
    print("Loading Prophet model...")
    model = load_model("prophet_model.pkl")
    
    # Create future dates
    future = model.make_future_dataframe(periods=periods, freq=freq)
    
    # Filter to only future dates (after start_date)
    future = future[future['ds'] >= pd.to_datetime(start_date)]
    
    print(f"Generating predictions for {len(future)} periods...")
    
    # Generate forecast
    forecast = model.predict(future)
    
    # Prepare results DataFrame
    results_df = pd.DataFrame({
        'Date': forecast['ds'],
        'Predicted_Production_Tons': forecast['yhat'],
        'Lower_Bound': forecast['yhat_lower'],
        'Upper_Bound': forecast['yhat_upper'],
        'Trend': forecast['trend'],
        'Uncertainty': forecast['yhat_upper'] - forecast['yhat_lower']
    })
    
    return results_df


def batch_forecast_rf(
    start_date: str,
    periods: int,
    freq: str = 'D',
    countries: List[str] = None
) -> pd.DataFrame:
    """
    Generate batch forecast using RandomForest model.
    
    Args:
        start_date: Start date for forecast
        periods: Number of periods
        freq: Frequency
        countries: List of countries to forecast for
    
    Returns:
        DataFrame with forecast results
    """
    # Load RF model
    print("Loading RandomForest model...")
    model = load_model("rf_model.pkl")
    
    # Default countries if not provided
    if countries is None:
        countries = ["Italy", "Spain", "Greece", "Turkey", "Tunisia"]
    
    # Generate date range
    date_range = pd.date_range(start=start_date, periods=periods, freq=freq)
    
    all_predictions = []
    
    print(f"Generating predictions for {len(countries)} countries...")
    
    for country in countries:
        for date in date_range:
            # Prepare features for this date
            features = prepare_batch_features(date, country)
            
            # Generate prediction
            prediction = model.predict(features)[0]
            
            all_predictions.append({
                'Date': date,
                'Country': country,
                'Predicted_Production_Tons': prediction
            })
    
    results_df = pd.DataFrame(all_predictions)
    
    return results_df


def prepare_batch_features(date: pd.Timestamp, country: str) -> pd.DataFrame:
    """
    Prepare features for batch forecasting.
    
    Args:
        date: Date for prediction
        country: Country name
    
    Returns:
        DataFrame with features
    """
    # Default/average values for features (in production, use historical averages)
    feature_dict = {
        'Month': date.month,
        'Year': date.year,
        'DayOfWeek': date.dayofweek,
        'DayOfMonth': date.day,
        'WeekOfYear': date.isocalendar().week,
        'Quarter': date.quarter,
        'MonthSin': np.sin(2 * np.pi * date.month / 12),
        'MonthCos': np.cos(2 * np.pi * date.month / 12),
        'Export_Tons': 25000,  # Average value
        'USD_Price': 12.5,     # Average value
        # Add more features as needed
    }
    
    return pd.DataFrame([feature_dict])


def save_forecast_results(df: pd.DataFrame, model_type: str) -> str:
    """
    Save forecast results to CSV file.
    
    Args:
        df: DataFrame with forecast results
        model_type: Type of model used
    
    Returns:
        Path to saved file
    """
    project_root = Path(__file__).parent.parent
    forecasts_dir = project_root / "data" / "forecasts"
    forecasts_dir.mkdir(exist_ok=True)
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"batch_forecast_{model_type}_{timestamp}.csv"
    filepath = forecasts_dir / filename
    
    # Also save as "latest" for easy access
    latest_filepath = forecasts_dir / f"batch_forecast_latest.csv"
    
    # Save files
    df.to_csv(filepath, index=False)
    df.to_csv(latest_filepath, index=False)
    
    return str(filepath)


def load_batch_forecast(filename: str = "batch_forecast_latest.csv") -> pd.DataFrame:
    """
    Load a previously saved batch forecast.
    
    Args:
        filename: Name of the forecast file
    
    Returns:
        DataFrame with forecast results
    """
    project_root = Path(__file__).parent.parent
    filepath = project_root / "data" / "forecasts" / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"Forecast file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    
    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    return df


def compare_forecasts(forecast_files: List[str]) -> pd.DataFrame:
    """
    Compare multiple forecast files.
    
    Args:
        forecast_files: List of forecast CSV filenames
    
    Returns:
        Combined DataFrame for comparison
    """
    all_forecasts = []
    
    for filename in forecast_files:
        df = load_batch_forecast(filename)
        df['Source_File'] = filename
        all_forecasts.append(df)
    
    combined_df = pd.concat(all_forecasts, ignore_index=True)
    
    return combined_df


def generate_forecast_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate summary statistics for a forecast.
    
    Args:
        df: Forecast DataFrame
    
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        'total_predictions': len(df),
        'date_range': {
            'start': df['Date'].min().strftime('%Y-%m-%d'),
            'end': df['Date'].max().strftime('%Y-%m-%d')
        },
        'production_stats': {
            'mean': df['Predicted_Production_Tons'].mean(),
            'median': df['Predicted_Production_Tons'].median(),
            'min': df['Predicted_Production_Tons'].min(),
            'max': df['Predicted_Production_Tons'].max(),
            'std': df['Predicted_Production_Tons'].std()
        }
    }
    
    if 'Country' in df.columns:
        summary['countries'] = df['Country'].nunique()
        summary['by_country'] = df.groupby('Country')['Predicted_Production_Tons'].mean().to_dict()
    
    return summary
