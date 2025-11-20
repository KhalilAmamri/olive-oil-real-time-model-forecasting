"""
Olive Oil Data Preprocessing Module - Simplified for Beginners
Handles cleaning, feature engineering, and preparation for forecasting models.

This module takes raw olive oil data and prepares it for machine learning:
1. Fixes missing data
2. Creates useful features from dates (month, day, etc.)
3. Adds lag features (previous export values)
4. Converts country names to numbers
5. Prepares data for training
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler


def preprocess_olive_oil_data(df: pd.DataFrame, scale_features: bool = False) -> pd.DataFrame:
    """
    Main preprocessing function - transforms raw data into ML-ready format.

    Think of this as a cooking recipe:
    - Raw ingredients (data) → Clean ingredients → Add flavors (features) → Ready to cook (ML)

    Args:
        df: Raw olive oil data from CSV
        scale_features: Whether to normalize numbers (usually not needed for RandomForest)

    Returns:
        Clean DataFrame ready for training ML models
    """
    df = df.copy()  # Don't modify original data

    # Ensure Date is in datetime format (for date operations)
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'])

    # Sort by date (important for time series)
    df = df.sort_values('Date').reset_index(drop=True)

    # Step 1: Fix missing values
    df = handle_missing_values(df)

    # Step 2: Create time features (month, day, etc.)
    df = create_time_features(df)

    # Step 3: Create lag features (previous export values)
    df = create_lag_features(df, target_col='Export_Tons')

    # Step 4: Convert countries to numbers
    df = encode_country(df)

    # Step 5: Optional scaling (not needed for RandomForest)
    if scale_features:
        df = scale_numerical_features(df)

    # Remove rows with missing values after creating lags
    initial_len = len(df)
    df = df.dropna()
    print(f"Preprocessing complete: {len(df):,} records (removed {initial_len - len(df)} due to lag features)")

    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the olive oil dataset.
    
    Strategy:
    - Numerical: Forward fill then backward fill
    - Categorical: Fill with mode
    """
    df = df.copy()
    
    # Numeric columns
    numeric_cols = ['Production_Tons', 'Export_Tons', 'USD_Price']
    for col in numeric_cols:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
    
    # Categorical columns
    if 'Country' in df.columns and df['Country'].isnull().any():
        df['Country'] = df['Country'].fillna(df['Country'].mode()[0])
    
    if 'Season' in df.columns and df['Season'].isnull().any():
        df['Season'] = df['Season'].fillna(df['Season'].mode()[0])
    
    return df


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract useful information from dates for machine learning.

    Dates contain hidden patterns that help predictions:
    - Month: Seasonal patterns (harvest time)
    - Day of week: Weekly cycles
    - Quarter: Business quarters
    - Weekend flag: Different behavior on weekends

    Args:
        df: DataFrame with Date column

    Returns:
        DataFrame with new time-based columns
    """
    df = df.copy()

    # Basic time components
    df['DayOfWeek'] = df['Date'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['DayOfMonth'] = df['Date'].dt.day       # 1-31
    df['DayOfYear'] = df['Date'].dt.dayofyear  # 1-365
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)  # 1-52
    df['Quarter'] = df['Date'].dt.quarter      # 1-4

    # Boolean features (True/False converted to 1/0)
    df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)  # Saturday/Sunday
    df['IsMonthStart'] = df['Date'].dt.is_month_start.astype(int)  # First day of month
    df['IsMonthEnd'] = df['Date'].dt.is_month_end.astype(int)      # Last day of month

    # Cyclical encoding for Month (makes December close to January)
    df['MonthSin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['MonthCos'] = np.cos(2 * np.pi * df['Month'] / 12)

    # Days since dataset start (trend feature)
    min_date = df['Date'].min()
    df['DaysSinceStart'] = (df['Date'] - min_date).dt.days

    return df


def create_lag_features(df: pd.DataFrame, target_col: str = 'Export_Tons', 
                        lags: list = [1, 7, 14, 30]) -> pd.DataFrame:
    """
    Create lag features - use past values to predict future.

    Lag features help the model learn patterns:
    - lag_1: Yesterday's export (short-term memory)
    - lag_7: Last week's export (weekly patterns)
    - lag_30: Last month's export (monthly trends)

    Args:
        df: DataFrame (must be sorted by date!)
        target_col: Column to create lags for ('Export_Tons')
        lags: List of lag periods in days/records

    Returns:
        DataFrame with lag columns added
    """
    df = df.copy()

    if target_col not in df.columns:
        return df  # No target column, skip

    # Create lag features (shift by N periods)
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)

    # Rolling statistics (moving averages)
    df[f'{target_col}_rolling_mean_7'] = df[target_col].rolling(window=7, min_periods=1).mean()
    df[f'{target_col}_rolling_std_7'] = df[target_col].rolling(window=7, min_periods=1).std()
    df[f'{target_col}_rolling_mean_30'] = df[target_col].rolling(window=30, min_periods=1).mean()
    df[f'{target_col}_rolling_std_30'] = df[target_col].rolling(window=30, min_periods=1).std()

    # Similar features for production (as input features)
    if 'Production_Tons' in df.columns:
        for lag in [1, 7]:
            df[f'Production_Tons_lag_{lag}'] = df['Production_Tons'].shift(lag)

    return df


def encode_country(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert country names to numbers for machine learning.

    ML models need numbers, not text. We assign numbers based on frequency:
    - Most common country = 0
    - Second most common = 1
    - etc.

    This preserves the "importance" of countries.

    Args:
        df: DataFrame with Country column

    Returns:
        DataFrame with Country_Encoded column
    """
    df = df.copy()

    if 'Country' in df.columns:
        # Count how many times each country appears
        country_counts = df['Country'].value_counts()

        # Create mapping: country name -> number (0, 1, 2, ...)
        country_map = {country: idx for idx, country in enumerate(country_counts.index)}
        df['Country_Encoded'] = df['Country'].map(country_map)

    return df


def scale_numerical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scale numerical features using StandardScaler.
    
    Only scales feature columns, not target (Production_Tons).
    """
    df = df.copy()
    
    # Columns to scale (exclude Date, Country, Season, and target)
    exclude_cols = ['Date', 'Country', 'Season', 'Export_Tons']
    scale_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                  if col not in exclude_cols]
    
    if scale_cols:
        scaler = StandardScaler()
        df[scale_cols] = scaler.fit_transform(df[scale_cols])
    
    return df


def prepare_prophet_data(df: pd.DataFrame, 
                         target_col: str = 'Production_Tons',
                         country: Optional[str] = None) -> pd.DataFrame:
    """
    Prepare data in Prophet's required format (ds, y columns).
    
    Prophet expects:
    - ds: Date column
    - y: Target variable
    
    Args:
        df: Preprocessed DataFrame
        target_col: Target column name
        country: Optional country filter for single-country forecasting
    
    Returns:
        DataFrame formatted for Prophet
    """
    df = df.copy()
    
    # Filter by country if specified
    if country and 'Country' in df.columns:
        df = df[df['Country'] == country]
        print(f"Filtered to {country}: {len(df)} records")
    
    # Create Prophet format
    prophet_df = pd.DataFrame({
        'ds': df['Date'],
        'y': df[target_col]
    })
    
    # Prophet works better with aggregated data (daily/weekly/monthly)
    # Aggregate to monthly for olive oil production
    prophet_df = prophet_df.set_index('ds').resample('M').agg({'y': 'sum'}).reset_index()
    print(f"Prophet data prepared: {len(prophet_df)} monthly records")
    
    return prophet_df


def prepare_ml_data(df: pd.DataFrame, 
                    target_col: str = 'Production_Tons',
                    test_size: float = 0.2,
                    country: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare data for ML models (RandomForest, etc.) with train/test split.
    
    Args:
        df: Preprocessed DataFrame
        target_col: Target column name
        test_size: Proportion for test set
        country: Optional country filter
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    df = df.copy()
    
    # Filter by country if specified
    if country and 'Country' in df.columns:
        df = df[df['Country'] == country]
        print(f"Filtered to {country}: {len(df)} records")
    
    # Define feature columns (exclude non-features)
    exclude_cols = ['Date', 'Country', 'Season', target_col]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Remove rows with NaN
    df = df.dropna()
    
    X = df[feature_cols]
    y = df[target_col]
    
    # Time-based split (preserves temporal order)
    split_idx = int(len(df) * (1 - test_size))
    
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    
    print(f"Train set: {len(X_train):,} samples | Test set: {len(X_test):,} samples")
    print(f"Features: {len(feature_cols)} ({', '.join(feature_cols[:5])}...)")
    
    return X_train, X_test, y_train, y_test
