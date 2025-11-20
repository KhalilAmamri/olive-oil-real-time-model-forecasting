"""
Olive Oil Export Forecasting - Model Training Script

This script trains forecasting models for olive oil exports:
1. Loads the Tunisia olive oil dataset
2. Preprocesses data (feature engineering, lag features, encoding)
3. Trains RandomForest model for export forecasting
4. Optionally trains Prophet model for time series forecasting
5. Saves trained models to models/ directory

Usage:
    python scripts/train_models.py
"""

from pathlib import Path
import sys

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import load_olive_oil_data, get_dataset_summary
from src.preprocess import preprocess_olive_oil_data, prepare_prophet_data, prepare_ml_data
from src.model_train import train_prophet_model, train_rf_model, save_model, PROPHET_AVAILABLE


def main() -> None:
    print("=" * 60)
    print("OLIVE OIL EXPORT FORECASTING - MODEL TRAINING")
    print("=" * 60)
    
    # 1. Load Dataset
    print("\n[1/5] Loading olive oil production dataset...")
    df = load_olive_oil_data()
    
    # Show summary
    summary = get_dataset_summary(df)
    print(f"\nDataset Summary:")
    print(f"  - Total Records: {summary['total_records']:,}")
    print(f"  - Countries: {summary['countries']['count']}")
    print(f"  - Date Range: {summary['date_range']['start'].date()} to {summary['date_range']['end'].date()}")
    print(f"  - Avg Export: {summary['exports']['avg_tons']:,.0f} tons")
    print(f"  - Total Export: {summary['exports']['total_tons']:,.0f} tons")
    
    # 2. Preprocess Data
    print("\n[2/5] Preprocessing data (feature engineering, lag features)...")
    df_processed = preprocess_olive_oil_data(df, scale_features=False)
    
    # 3. Train RandomForest Model
    print("\n[3/5] Training RandomForest Regressor...")
    X_train, X_test, y_train, y_test = prepare_ml_data(df_processed, target_col='Export_Tons')
    
    rf_result = train_rf_model(
        X_train, y_train, X_test, y_test,
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        random_state=42
    )
    
    save_model(rf_result['model'], "rf_olive_oil_model.pkl", model_type="rf")
    
    print(f"\nTop 5 Important Features:")
    for idx, row in rf_result['feature_importance'].head().iterrows():
        print(f"  {idx+1}. {row['feature']}: {row['importance']:.4f}")
    
    # 4. Train Prophet Model (Optional)
    if PROPHET_AVAILABLE:
        print("\n[4/5] Training Prophet time series model...")
        try:
            prophet_df = prepare_prophet_data(df_processed)
            prophet_model = train_prophet_model(prophet_df, seasonality_mode='multiplicative')
            save_model(prophet_model, "prophet_olive_oil_model.pkl", model_type="prophet")
        except Exception as e:
            print(f"Prophet training skipped: {e}")
    else:
        print("\n[4/5] Prophet not installed - skipping Prophet model")
        print("  To install: pip install 'prophet>=1.1.5' --no-build-isolation")
    
    # 5. Summary
    print("\n[5/5] Training Complete!")
    print("=" * 60)
    print("Models saved to models/ directory:")
    print("  - rf_olive_oil_model.pkl (RandomForest)")
    if PROPHET_AVAILABLE:
        print("  - prophet_olive_oil_model.pkl (Prophet)")
    print("\nNext steps:")
    print("  1. Run Streamlit app: streamlit run app/app.py")
    print("  2. Try real-time forecasting with trained models")
    print("  3. Generate batch forecasts for planning")
    print("=" * 60)


if __name__ == "__main__":
    main()
