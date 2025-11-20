"""
Olive Oil Export Forecasting Module - Simplified Version
Handles real-time forecasting for trained models.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import joblib


def load_model(model_name: str) -> Any:
    """Load a trained model from models/ directory."""
    project_root = Path(__file__).parent.parent
    model_path = project_root / "models" / model_name
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")


def predict_export(model_name: str,
                   date: datetime,
                   country: str,
                   production_tons: float,
                   usd_price: float,
                   lag_1: float = None,
                   lag_7: float = None) -> Dict[str, Any]:
    """
    Predict olive oil export volume for given inputs.
    
    Args:
        model_name: Name of the model file (e.g., 'rf_olive_oil_model.pkl')
        date: Forecast date
        country: Country name
        production_tons: Production volume in tons
        usd_price: Price per ton in USD
        lag_1: Export lag 1 period (if available)
        lag_7: Export lag 7 periods (if available)
    
    Returns:
        Dictionary with prediction and metadata
    """
    model = load_model(model_name)
    
    # Load feature names to ensure correct column order
    project_root = Path(__file__).parent.parent
    feature_names_path = project_root / "models" / "feature_names.pkl"
    
    if not feature_names_path.exists():
        raise FileNotFoundError(f"Feature names file not found: {feature_names_path}")
    
    try:
        feature_names = joblib.load(feature_names_path)
    except Exception as e:
        raise Exception(f"Error loading feature names: {str(e)}")
    
    # Create feature dictionary
    features = prepare_forecast_features(
        date=date,
        country=country,
        production_tons=production_tons,
        usd_price=usd_price,
        lag_1=lag_1,
        lag_7=lag_7
    )
    
    # Convert to DataFrame and ensure correct column order
    feature_df = pd.DataFrame([features])
    feature_df = feature_df[feature_names]  # Reorder columns to match training
    
    # Make prediction
    prediction = model.predict(feature_df)[0]
    
    result = {
        'predicted_export_tons': max(0, prediction),  # Ensure non-negative
        'date': date,
        'country': country,
        'production_tons': production_tons,
        'usd_price': usd_price,
        'model_used': model_name
    }
    
    return result


def prepare_forecast_features(date: datetime,
                              country: str,
                              production_tons: float,
                              usd_price: float,
                              lag_1: float = None,
                              lag_7: float = None) -> Dict[str, float]:
    """
    Prepare features for forecasting matching training pipeline.
    
    This must match the exact feature set used during training.
    """
    # Time features
    month = date.month
    year = date.year
    day_of_week = date.weekday()
    day_of_month = date.day
    day_of_year = date.timetuple().tm_yday
    week_of_year = date.isocalendar()[1]
    quarter = (month - 1) // 3 + 1
    
    # Boolean features
    is_weekend = 1 if day_of_week >= 5 else 0
    is_month_start = 1 if day_of_month == 1 else 0
    is_month_end = 1 if (date + pd.Timedelta(days=1)).day == 1 else 0
    
    # Cyclical encoding
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    
    # Days since a reference date (e.g., 2010-01-01)
    reference_date = datetime(2010, 1, 1)
    days_since_start = (date - reference_date).days
    
    # Country encoding (simple mapping for common countries)
    country_map = {
        'Tunisia': 0, 'Italy': 1, 'Spain': 2, 'Greece': 3, 'Turkey': 4,
        'Morocco': 5, 'Portugal': 6, 'Syria': 7, 'Algeria': 8, 'Jordan': 9,
        'Lebanon': 10, 'Egypt': 11, 'Libya': 12, 'Israel': 13, 'Croatia': 14,
        'France': 15, 'Slovenia': 16, 'Albania': 17, 'Montenegro': 18, 'Bosnia and Herzegovina': 19,
        'Serbia': 20, 'North Macedonia': 21, 'Bulgaria': 22, 'Romania': 23, 'Ukraine': 24,
        'Russia': 25, 'Belarus': 26, 'Poland': 27, 'Czech Republic': 28, 'Slovakia': 29,
        'Hungary': 30, 'Austria': 31, 'Germany': 32, 'Netherlands': 33, 'Belgium': 34,
        'Luxembourg': 35, 'Switzerland': 36, 'United Kingdom': 37, 'Ireland': 38, 'Denmark': 39,
        'Sweden': 40, 'Norway': 41
    }
    country_encoded = country_map.get(country, 0)  # Default to 0 if not found
    
    # Build feature dict
    features = {
        'Production_Tons': production_tons,
        'USD_Price': usd_price,
        'Month': month,
        'Year': year,
        'DayOfWeek': day_of_week,
        'DayOfMonth': day_of_month,
        'DayOfYear': day_of_year,
        'WeekOfYear': week_of_year,
        'Quarter': quarter,
        'IsWeekend': is_weekend,
        'IsMonthStart': is_month_start,
        'IsMonthEnd': is_month_end,
        'MonthSin': month_sin,
        'MonthCos': month_cos,
        'DaysSinceStart': days_since_start,
        'Country_Encoded': country_encoded
    }
    
    # Add lag features if provided
    if lag_1 is not None:
        features['Export_Tons_lag_1'] = lag_1
        features['Export_Tons_rolling_mean_7'] = lag_1  # Simplified
        features['Export_Tons_rolling_std_7'] = lag_1 * 0.1  # Simplified
    else:
        # Use reasonable defaults based on historical averages
        features['Export_Tons_lag_1'] = 20000
        features['Export_Tons_rolling_mean_7'] = 20000
        features['Export_Tons_rolling_std_7'] = 2000
    
    if lag_7 is not None:
        features['Export_Tons_lag_7'] = lag_7
    else:
        features['Export_Tons_lag_7'] = 20000
    
    # Add other lag features with defaults
    features['Export_Tons_lag_14'] = features.get('Export_Tons_lag_7', 20000)
    features['Export_Tons_lag_30'] = features.get('Export_Tons_lag_7', 20000)
    features['Export_Tons_rolling_mean_30'] = features.get('Export_Tons_rolling_mean_7', 20000)
    features['Export_Tons_rolling_std_30'] = features.get('Export_Tons_rolling_std_7', 2000)
    features['Production_Tons_lag_1'] = production_tons
    features['Production_Tons_lag_7'] = production_tons
    
    return features
