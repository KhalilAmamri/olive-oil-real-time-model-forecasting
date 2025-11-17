"""
Real-time Forecasting Module
Handles single prediction requests with user inputs.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.model_train import load_model


def run_realtime_forecast(
    model_type: str,
    country: str,
    date: str,
    features: Dict[str, float]
) -> Dict[str, Any]:
    """
    Generate a single real-time prediction.
    
    Real-time forecasting means:
    - User provides input features via UI
    - Model generates prediction instantly (< 1 second)
    - Result displayed immediately with visualization
    - No batch processing or file storage needed
    
    This is ideal for:
    - Interactive dashboards
    - What-if scenario analysis
    - User-driven exploration
    - Immediate business decisions
    
    Args:
        model_type: "prophet" or "rf"
        country: Country name
        date: Date string (YYYY-MM-DD)
        features: Dictionary of feature values
    
    Returns:
        Dictionary containing prediction and metadata
    """
    try:
        if model_type.lower() == "prophet":
            result = predict_with_prophet(date, features)
        elif model_type.lower() == "rf":
            result = predict_with_rf(country, date, features)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return result
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'prediction': None
        }


def predict_with_prophet(date: str, features: Dict[str, float] = None) -> Dict[str, Any]:
    """
    Generate prediction using Prophet model.
    
    Args:
        date: Date for prediction (YYYY-MM-DD)
        features: Additional features (Prophet mainly uses date)
    
    Returns:
        Prediction results
    """
    # Load Prophet model
    model = load_model("prophet_model.pkl")
    
    # Prepare future dataframe
    future_df = pd.DataFrame({
        'ds': [pd.to_datetime(date)]
    })
    
    # Generate forecast
    forecast = model.predict(future_df)
    
    prediction = forecast['yhat'].values[0]
    lower_bound = forecast['yhat_lower'].values[0]
    upper_bound = forecast['yhat_upper'].values[0]
    
    result = {
        'success': True,
        'model_type': 'Prophet',
        'date': date,
        'prediction': float(prediction),
        'lower_bound': float(lower_bound),
        'upper_bound': float(upper_bound),
        'confidence_interval': f"[{lower_bound:,.0f}, {upper_bound:,.0f}]",
        'features_used': ['Date', 'Seasonality', 'Trend']
    }
    
    return result


def predict_with_rf(country: str, date: str, features: Dict[str, float]) -> Dict[str, Any]:
    """
    Generate prediction using RandomForest model.
    
    Args:
        country: Country name
        date: Date for prediction (YYYY-MM-DD)
        features: Dictionary of feature values
    
    Returns:
        Prediction results
    """
    # Load RF model
    model = load_model("rf_model.pkl")
    
    # Prepare features
    feature_df = prepare_rf_features(country, date, features)
    
    # Generate prediction
    prediction = model.predict(feature_df)[0]
    
    # Get feature importance for explanation
    feature_names = feature_df.columns.tolist()
    
    result = {
        'success': True,
        'model_type': 'RandomForest',
        'country': country,
        'date': date,
        'prediction': float(prediction),
        'features_used': feature_names,
        'feature_values': features
    }
    
    return result


def prepare_rf_features(country: str, date: str, features: Dict[str, float]) -> pd.DataFrame:
    """
    Prepare feature DataFrame for RandomForest prediction.
    
    Args:
        country: Country name
        date: Date string
        features: Feature dictionary
    
    Returns:
        DataFrame with all required features
    """
    date_obj = pd.to_datetime(date)
    
    # Create base features
    feature_dict = {
        'Month': date_obj.month,
        'Year': date_obj.year,
        'DayOfWeek': date_obj.dayofweek,
        'DayOfMonth': date_obj.day,
        'WeekOfYear': date_obj.isocalendar().week,
        'Quarter': date_obj.quarter,
        'MonthSin': np.sin(2 * np.pi * date_obj.month / 12),
        'MonthCos': np.cos(2 * np.pi * date_obj.month / 12),
    }
    
    # Add user-provided features
    feature_dict.update(features)
    
    # Create DataFrame
    df = pd.DataFrame([feature_dict])
    
    return df


def generate_forecast_range(
    model_type: str,
    start_date: str,
    end_date: str,
    country: str = None,
    freq: str = 'D'
) -> pd.DataFrame:
    """
    Generate forecasts for a date range (useful for visualization).
    
    Args:
        model_type: "prophet" or "rf"
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        country: Country filter
        freq: Frequency ('D' for daily, 'W' for weekly, 'M' for monthly)
    
    Returns:
        DataFrame with predictions for each date
    """
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    
    predictions = []
    
    for date in date_range:
        date_str = date.strftime('%Y-%m-%d')
        
        # Default features (can be enhanced)
        default_features = {
            'Export_Tons': 25000,
            'USD_Price': 12.0
        }
        
        result = run_realtime_forecast(model_type, country or "Italy", date_str, default_features)
        
        if result['success']:
            predictions.append({
                'Date': date,
                'Prediction': result['prediction']
            })
    
    return pd.DataFrame(predictions)


def calculate_prediction_confidence(prediction: float, historical_std: float) -> Dict[str, float]:
    """
    Calculate confidence intervals for prediction.
    
    Args:
        prediction: Point prediction
        historical_std: Historical standard deviation
    
    Returns:
        Dictionary with confidence bounds
    """
    # 95% confidence interval (±1.96 std)
    margin = 1.96 * historical_std
    
    return {
        'prediction': prediction,
        'lower_95': prediction - margin,
        'upper_95': prediction + margin,
        'margin': margin
    }
