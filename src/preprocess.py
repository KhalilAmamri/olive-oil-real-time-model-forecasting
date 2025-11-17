"""
Data Preprocessing Module
Handles data cleaning, feature engineering, and preparation for modeling.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Complete preprocessing pipeline for olive oil dataset.
    
    Args:
        df: Raw DataFrame
    
    Returns:
        Preprocessed DataFrame ready for modeling
    """
    df = df.copy()
    
    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Sort by date
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Create additional time features
    df = create_time_features(df)
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Create lag features for time series
    df = create_lag_features(df)
    
    # Encode categorical variables
    df = encode_categorical_features(df)
    
    print(f"✓ Preprocessing complete. Shape: {df.shape}")
    
    return df


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features from Date column.
    
    Args:
        df: Input DataFrame with Date column
    
    Returns:
        DataFrame with additional time features
    """
    df = df.copy()
    
    # Extract time components
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['DayOfMonth'] = df['Date'].dt.day
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week
    df['Quarter'] = df['Date'].dt.quarter
    
    # Cyclical encoding for month (already exists but enhance it)
    df['MonthSin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['MonthCos'] = np.cos(2 * np.pi * df['Month'] / 12)
    
    # Days since start (for trend)
    min_date = df['Date'].min()
    df['DaysSinceStart'] = (df['Date'] - min_date).dt.days
    
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with missing values handled
    """
    df = df.copy()
    
    # Fill numeric columns with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    
    # Fill categorical columns with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col != 'Date' and df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    return df


def create_lag_features(df: pd.DataFrame, target_col: str = 'Production_Tons', lags: list = [1, 7, 30]) -> pd.DataFrame:
    """
    Create lag features for time series forecasting.
    
    Args:
        df: Input DataFrame
        target_col: Column to create lags for
        lags: List of lag periods
    
    Returns:
        DataFrame with lag features
    """
    df = df.copy()
    
    if target_col in df.columns:
        for lag in lags:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        
        # Rolling statistics
        df[f'{target_col}_rolling_mean_7'] = df[target_col].rolling(window=7, min_periods=1).mean()
        df[f'{target_col}_rolling_std_7'] = df[target_col].rolling(window=7, min_periods=1).std()
    
    return df


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical features for modeling.
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with encoded categorical features
    """
    df = df.copy()
    
    # Label encoding for Country (preserving original for reference)
    if 'Country' in df.columns:
        df['Country_Encoded'] = pd.factorize(df['Country'])[0]
    
    # One-hot encoding for Season
    if 'Season' in df.columns:
        season_dummies = pd.get_dummies(df['Season'], prefix='Season')
        df = pd.concat([df, season_dummies], axis=1)
    
    return df


def prepare_prophet_data(df: pd.DataFrame, target_col: str = 'Production_Tons') -> pd.DataFrame:
    """
    Prepare data in Prophet's required format (ds, y columns).
    
    Args:
        df: Input DataFrame
        target_col: Target column name
    
    Returns:
        DataFrame formatted for Prophet
    """
    prophet_df = pd.DataFrame({
        'ds': df['Date'],
        'y': df[target_col]
    })
    
    return prophet_df


def prepare_ml_data(df: pd.DataFrame, target_col: str = 'Production_Tons', 
                    test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare data for machine learning models (train/test split).
    
    Args:
        df: Preprocessed DataFrame
        target_col: Target column name
        test_size: Proportion of data for testing
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    df = df.copy()
    
    # Drop non-feature columns
    drop_cols = ['Date', 'Country', 'Season']
    feature_cols = [col for col in df.columns if col not in drop_cols and col != target_col]
    
    # Remove rows with NaN (from lag features)
    df = df.dropna()
    
    # Prepare features and target
    X = df[feature_cols]
    y = df[target_col]
    
    # Time-based split (not random, to preserve temporal order)
    split_idx = int(len(df) * (1 - test_size))
    
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    
    print(f"✓ Train set: {len(X_train)} samples")
    print(f"✓ Test set: {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test
