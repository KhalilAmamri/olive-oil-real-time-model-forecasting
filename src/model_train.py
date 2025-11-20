"""
Model Training Module - Simplified
Handles training of RandomForest models for export forecasting.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train_rf_model(X_train: pd.DataFrame,
                   y_train: pd.Series,
                   X_test: pd.DataFrame = None,
                   y_test: pd.Series = None,
                   **rf_params) -> Dict[str, Any]:
    """
    Train a RandomForest model for export forecasting.

    RandomForest is good for:
    - Learning complex patterns
    - Handling different types of data
    - Showing which features are important

    Args:
        X_train: Training features
        y_train: Training target (export values)
        X_test: Test features (optional)
        y_test: Test target (optional)
        **rf_params: RandomForest settings

    Returns:
        Dictionary with model and results
    """
    # Default settings for RandomForest
    default_params = {
        'n_estimators': 200,  # Number of trees
        'max_depth': 20,      # Maximum tree depth
        'random_state': 42    # For reproducible results
    }

    # Use custom settings if provided
    default_params.update(rf_params)

    print(f"Training RandomForest with {default_params['n_estimators']} trees...")

    # Create and train the model
    model = RandomForestRegressor(**default_params)
    model.fit(X_train, y_train)

    # Calculate how well the model performs
    train_pred = model.predict(X_train)
    train_metrics = calculate_metrics(y_train, train_pred, "Train")

    test_metrics = {}
    if X_test is not None and y_test is not None:
        test_pred = model.predict(X_test)
        test_metrics = calculate_metrics(y_test, test_pred, "Test")

    # Show which features are most important
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("Model trained successfully!")
    print(f"Top 5 important features: {', '.join(feature_importance['feature'].head().tolist())}")

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


def save_model(model: Any, model_name: str) -> str:
    """
    Save trained model to disk.

    Args:
        model: Trained model object
        model_name: Name for the saved model file

    Returns:
        Path to saved model
    """
    project_root = Path(__file__).parent.parent
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)

    # Add .pkl extension if not present
    if not model_name.endswith('.pkl'):
        model_name = f"{model_name}.pkl"

    model_path = models_dir / model_name

    try:
        joblib.dump(model, model_path)
        print(f"Model saved to: {model_path}")
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
        print(f"OK: Model loaded from: {model_path}")
        return model
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")
