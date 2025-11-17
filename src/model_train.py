"""
Model Training Module
Handles training of Prophet and RandomForest models.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Prophet import with fallback
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("⚠ Prophet not available. Install with: pip install prophet")


def train_prophet_model(df: pd.DataFrame, 
                        country_filter: str = None,
                        seasonality_mode: str = 'multiplicative') -> Any:
    """
    Train a Prophet model for time series forecasting.
    
    Prophet is excellent for:
    - Handling missing data and outliers automatically
    - Capturing seasonal patterns (yearly, monthly, weekly)
    - Being robust to trend changes
    - Not requiring feature engineering
    
    Args:
        df: DataFrame with 'ds' (date) and 'y' (target) columns
        country_filter: Optional country to filter data
        seasonality_mode: 'additive' or 'multiplicative'
    
    Returns:
        Trained Prophet model
    """
    if not PROPHET_AVAILABLE:
        raise ImportError("Prophet is not installed. Run: pip install prophet")
    
    df = df.copy()
    
    if country_filter:
        if 'Country' in df.columns:
            df = df[df['Country'] == country_filter]
    
    # Prepare Prophet format
    if 'ds' not in df.columns or 'y' not in df.columns:
        if 'Date' in df.columns:
            df = df.rename(columns={'Date': 'ds', 'Production_Tons': 'y'})
    
    # Initialize Prophet model
    model = Prophet(
        seasonality_mode=seasonality_mode,
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.05
    )
    
    # Add custom seasonalities
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    
    print(f"Training Prophet model on {len(df)} samples...")
    model.fit(df[['ds', 'y']])
    
    print("✓ Prophet model trained successfully")
    
    return model


def train_rf_model(X_train: pd.DataFrame, 
                   y_train: pd.Series,
                   X_test: pd.DataFrame = None,
                   y_test: pd.Series = None,
                   **rf_params) -> Dict[str, Any]:
    """
    Train a RandomForest model for production forecasting.
    
    RandomForest is excellent for:
    - Handling non-linear relationships
    - Feature importance analysis
    - Robust to outliers
    - No feature scaling required
    - Capturing complex interactions between features
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features (optional)
        y_test: Test target (optional)
        **rf_params: Additional RandomForest parameters
    
    Returns:
        Dictionary containing model and metrics
    """
    # Default parameters for production use
    default_params = {
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'random_state': 42,
        'n_jobs': -1
    }
    
    # Update with user parameters
    default_params.update(rf_params)
    
    print(f"Training RandomForest with {default_params['n_estimators']} trees...")
    
    # Initialize and train model
    model = RandomForestRegressor(**default_params)
    model.fit(X_train, y_train)
    
    # Calculate metrics
    train_pred = model.predict(X_train)
    train_metrics = calculate_metrics(y_train, train_pred, "Train")
    
    test_metrics = {}
    if X_test is not None and y_test is not None:
        test_pred = model.predict(X_test)
        test_metrics = calculate_metrics(y_test, test_pred, "Test")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("✓ RandomForest model trained successfully")
    print(f"  Top 5 features: {', '.join(feature_importance['feature'].head().tolist())}")
    
    return {
        'model': model,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'feature_importance': feature_importance
    }


def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray, dataset_name: str = "") -> Dict[str, float]:
    """
    Calculate regression metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        dataset_name: Name for printing (e.g., "Train" or "Test")
    
    Returns:
        Dictionary of metrics
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    metrics = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape
    }
    
    if dataset_name:
        print(f"\n{dataset_name} Metrics:")
        print(f"  MAE:  {mae:,.2f}")
        print(f"  RMSE: {rmse:,.2f}")
        print(f"  R²:   {r2:.4f}")
        print(f"  MAPE: {mape:.2f}%")
    
    return metrics


def save_model(model: Any, model_name: str, model_type: str = "prophet") -> str:
    """
    Save trained model to disk.
    
    Args:
        model: Trained model object
        model_name: Name for the saved model file
        model_type: Type of model ("prophet" or "rf")
    
    Returns:
        Path to saved model
    """
    project_root = Path(__file__).parent.parent
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Add extension based on model type
    if not model_name.endswith('.pkl'):
        model_name = f"{model_name}.pkl"
    
    model_path = models_dir / model_name
    
    try:
        joblib.dump(model, model_path)
        print(f"✓ Model saved to: {model_path}")
        return str(model_path)
    except Exception as e:
        raise Exception(f"Error saving model: {str(e)}")


def load_model(model_name: str) -> Any:
    """
    Load a saved model from disk.
    
    Args:
        model_name: Name of the model file
    
    Returns:
        Loaded model object
    """
    project_root = Path(__file__).parent.parent
    model_path = project_root / "models" / model_name
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at: {model_path}")
    
    try:
        model = joblib.load(model_path)
        print(f"✓ Model loaded from: {model_path}")
        return model
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")
