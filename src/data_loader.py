"""
Data Loader Module
Handles loading raw data from CSV files.
"""

import pandas as pd
from pathlib import Path
from typing import Optional


def load_raw_data(file_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load raw olive oil dataset from CSV file.
    
    Args:
        file_path: Path to the CSV file. If None, uses default path.
    
    Returns:
        DataFrame containing the raw data.
    """
    if file_path is None:
        # Default path relative to project root
        project_root = Path(__file__).parent.parent
        file_path = project_root / "data" / "raw" / "tunisia_olive_oil_dataset.csv"
    
    try:
        df = pd.read_csv(file_path)
        print(f"✓ Successfully loaded {len(df)} records from {file_path}")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found at: {file_path}")
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")


def get_data_info(df: pd.DataFrame) -> dict:
    """
    Get basic information about the dataset.
    
    Args:
        df: Input DataFrame
    
    Returns:
        Dictionary containing dataset statistics
    """
    info = {
        "shape": df.shape,
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "date_range": {
            "min": df['Date'].min() if 'Date' in df.columns else None,
            "max": df['Date'].max() if 'Date' in df.columns else None
        },
        "countries": df['Country'].nunique() if 'Country' in df.columns else None,
        "numeric_summary": df.describe().to_dict()
    }
    return info


def load_forecast_results(forecast_name: str = "batch_forecast_latest.csv") -> pd.DataFrame:
    """
    Load previously generated forecast results.
    
    Args:
        forecast_name: Name of the forecast CSV file
    
    Returns:
        DataFrame containing forecast results
    """
    project_root = Path(__file__).parent.parent
    forecast_path = project_root / "data" / "forecasts" / forecast_name
    
    try:
        df = pd.read_csv(forecast_path)
        print(f"✓ Successfully loaded forecast from {forecast_path}")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Forecast file not found at: {forecast_path}")
    except Exception as e:
        raise Exception(f"Error loading forecast: {str(e)}")
