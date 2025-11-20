"""
Olive Oil Data Loader Module
Handles loading and basic exploration of the Tunisia olive oil production dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List


def load_olive_oil_data(file_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load the Tunisia olive oil production dataset.
    
    Dataset columns:
    - Date: Production date
    - Country: Country name
    - Production_Tons: Olive oil production in metric tons
    - Export_Tons: Export volume in metric tons
    - USD_Price: Price in USD per ton
    - Month, Year: Extracted time features
    - Season: Seasonal indicator
    
    Args:
        file_path: Path to CSV. Defaults to data/raw/tunisia_olive_oil_dataset.csv
    
    Returns:
        DataFrame with olive oil production data
    """
    if file_path is None:
        project_root = Path(__file__).parent.parent
        file_path = project_root / "data" / "raw" / "tunisia_olive_oil_dataset.csv"
    
    try:
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        print(f"Loaded {len(df):,} olive oil production records")
        print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
        if df['Country'].nunique() > 5:
            print(f"Countries: {df['Country'].nunique()} ({', '.join(df['Country'].unique()[:5])}...)")
        else:
            print(f"Countries: {', '.join(df['Country'].unique())}")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at: {file_path}")
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")


def get_dataset_summary(df: pd.DataFrame) -> Dict:
    """
    Generate comprehensive summary statistics for the olive oil dataset.
    
    Args:
        df: Olive oil DataFrame
    
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        "total_records": len(df),
        "date_range": {
            "start": df['Date'].min(),
            "end": df['Date'].max(),
            "years": sorted(df['Year'].unique().tolist()) if 'Year' in df.columns else []
        },
        "countries": {
            "count": df['Country'].nunique(),
            "list": sorted(df['Country'].unique().tolist())
        },
        "production": {
            "total_tons": df['Production_Tons'].sum(),
            "avg_tons": df['Production_Tons'].mean(),
            "min_tons": df['Production_Tons'].min(),
            "max_tons": df['Production_Tons'].max()
        },
        "exports": {
            "total_tons": df['Export_Tons'].sum(),
            "avg_tons": df['Export_Tons'].mean()
        },
        "pricing": {
            "avg_usd_per_ton": df['USD_Price'].mean(),
            "min_usd_per_ton": df['USD_Price'].min(),
            "max_usd_per_ton": df['USD_Price'].max()
        },
        "missing_values": df.isnull().sum().to_dict()
    }
    return summary


def get_country_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate production statistics by country.
    
    Args:
        df: Olive oil DataFrame
    
    Returns:
        DataFrame with country-level statistics
    """
    stats = df.groupby('Country').agg({
        'Production_Tons': ['count', 'sum', 'mean', 'std'],
        'Export_Tons': ['sum', 'mean'],
        'USD_Price': 'mean'
    }).round(2)
    
    stats.columns = ['Records', 'Total_Production', 'Avg_Production', 'Std_Production',
                     'Total_Exports', 'Avg_Exports', 'Avg_Price_USD']
    return stats.sort_values('Total_Production', ascending=False)


def get_available_countries(df: pd.DataFrame) -> List[str]:
    """Get sorted list of countries in the dataset."""
    return sorted(df['Country'].unique().tolist())


def get_date_range(df: pd.DataFrame) -> tuple:
    """Get min and max dates from the dataset."""
    return df['Date'].min(), df['Date'].max()


def load_forecast_results(forecast_name: str = "batch_forecast_latest.csv") -> pd.DataFrame:
    """
    Load saved batch forecast results.
    
    Args:
        forecast_name: CSV filename in data/forecasts/
    
    Returns:
        DataFrame with forecast data
    """
    project_root = Path(__file__).parent.parent
    forecast_path = project_root / "data" / "forecasts" / forecast_name
    
    try:
        df = pd.read_csv(forecast_path)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        print(f"Loaded forecast: {forecast_name} ({len(df)} records)")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Forecast not found: {forecast_path}")
    except Exception as e:
        raise Exception(f"Error loading forecast: {str(e)}")
