"""
Utility Functions
Common helper functions used across the project.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import json


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def get_data_dir() -> Path:
    """Get the data directory."""
    return get_project_root() / "data"


def get_models_dir() -> Path:
    """Get the models directory."""
    return get_project_root() / "models"


def format_number(num: float, decimals: int = 2) -> str:
    """
    Format number with thousands separator.
    
    Args:
        num: Number to format
        decimals: Number of decimal places
    
    Returns:
        Formatted string
    """
    return f"{num:,.{decimals}f}"


def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """
    Calculate percentage change between two values.
    
    Args:
        old_value: Original value
        new_value: New value
    
    Returns:
        Percentage change
    """
    if old_value == 0:
        return 0.0
    return ((new_value - old_value) / old_value) * 100


def get_season(month: int) -> str:
    """
    Get season from month number.
    
    Args:
        month: Month number (1-12)
    
    Returns:
        Season name
    """
    if month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    elif month in [9, 10, 11]:
        return "Fall"
    else:
        return "Winter"


def get_available_countries(df: pd.DataFrame) -> List[str]:
    """
    Get list of unique countries from dataset.
    
    Args:
        df: DataFrame with Country column
    
    Returns:
        Sorted list of countries
    """
    if 'Country' in df.columns:
        return sorted(df['Country'].unique().tolist())
    return []


def filter_by_date_range(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Filter DataFrame by date range.
    
    Args:
        df: Input DataFrame with Date column
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
    
    Returns:
        Filtered DataFrame
    """
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    
    mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
    return df[mask]


def aggregate_by_period(df: pd.DataFrame, period: str = 'M', value_col: str = 'Production_Tons') -> pd.DataFrame:
    """
    Aggregate data by time period.
    
    Args:
        df: Input DataFrame with Date column
        period: Aggregation period ('D', 'W', 'M', 'Q', 'Y')
        value_col: Column to aggregate
    
    Returns:
        Aggregated DataFrame
    """
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    
    aggregated = df[value_col].resample(period).sum().reset_index()
    
    return aggregated


def detect_outliers(df: pd.DataFrame, column: str, threshold: float = 3.0) -> pd.DataFrame:
    """
    Detect outliers using Z-score method.
    
    Args:
        df: Input DataFrame
        column: Column to check for outliers
        threshold: Z-score threshold
    
    Returns:
        DataFrame with outlier indicator column
    """
    df = df.copy()
    
    mean = df[column].mean()
    std = df[column].std()
    
    df['z_score'] = (df[column] - mean) / std
    df['is_outlier'] = np.abs(df['z_score']) > threshold
    
    return df


def save_config(config: Dict[str, Any], filename: str = "config.json") -> None:
    """
    Save configuration to JSON file.
    
    Args:
        config: Configuration dictionary
        filename: Output filename
    """
    config_path = get_project_root() / filename
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"OK: Config saved to: {config_path}")


def load_config(filename: str = "config.json") -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        filename: Config filename
    
    Returns:
        Configuration dictionary
    """
    config_path = get_project_root() / filename
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config


def create_date_features_dict(date_str: str) -> Dict[str, Any]:
    """
    Create dictionary of date features from date string.
    
    Args:
        date_str: Date string (YYYY-MM-DD)
    
    Returns:
        Dictionary of date features
    """
    date = pd.to_datetime(date_str)
    
    return {
        'year': date.year,
        'month': date.month,
        'day': date.day,
        'dayofweek': date.dayofweek,
        'quarter': date.quarter,
        'weekofyear': date.isocalendar().week,
        'season': get_season(date.month)
    }


def validate_date_string(date_str: str) -> bool:
    """
    Validate date string format.
    
    Args:
        date_str: Date string to validate
    
    Returns:
        True if valid, False otherwise
    """
    try:
        pd.to_datetime(date_str)
        return True
    except:
        return False


def get_model_info() -> Dict[str, str]:
    """
    Get information about available models.
    
    Returns:
        Dictionary with model information
    """
    models_dir = get_models_dir()
    
    info = {
        'models_directory': str(models_dir),
        'available_models': []
    }
    
    if models_dir.exists():
        info['available_models'] = [f.name for f in models_dir.glob('*.pkl')]
    
    return info


def print_section_header(title: str, width: int = 60) -> None:
    """
    Print formatted section header.
    
    Args:
        title: Section title
        width: Width of header
    """
    print(f"\n{'='*width}")
    print(f"{title:^{width}}")
    print(f"{'='*width}\n")


def calculate_trend(series: pd.Series, window: int = 7) -> str:
    """
    Calculate trend direction from a time series.
    
    Args:
        series: Time series data
        window: Window size for moving average
    
    Returns:
        Trend direction ("Increasing", "Decreasing", "Stable")
    """
    if len(series) < window:
        return "Insufficient Data"
    
    recent_avg = series.tail(window).mean()
    previous_avg = series.iloc[-2*window:-window].mean()
    
    change_pct = calculate_percentage_change(previous_avg, recent_avg)
    
    if change_pct > 5:
        return "Increasing"
    elif change_pct < -5:
        return "Decreasing"
    else:
        return "Stable"
