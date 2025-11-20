#!/usr/bin/env python3
"""
Simple Training Script for Olive Oil Export Forecasting

This script trains a RandomForest model to predict olive oil export volumes.
Run this before using the dashboard.

Usage:
    python train.py
"""

from pathlib import Path
import sys

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data_loader import load_olive_oil_data, get_dataset_summary
from src.preprocess import preprocess_olive_oil_data, prepare_ml_data
from src.model_train import train_rf_model, save_model


def main():
    print("🫒 Training Olive Oil Export Forecasting Model")
    print("=" * 50)

    try:
        # 1. Load data
        print("\n📊 Loading dataset...")
        df = load_olive_oil_data()
        summary = get_dataset_summary(df)
        print(f"✅ Loaded {summary['total_records']:,} records from {summary['countries']['count']} countries")

        # 2. Preprocess data
        print("\n🔧 Preprocessing data...")
        df_processed = preprocess_olive_oil_data(df, scale_features=False)
        print(f"✅ Created {len(df_processed)} processed records with features")

        # 3. Train model
        print("\n🤖 Training RandomForest model...")
        X_train, X_test, y_train, y_test = prepare_ml_data(df_processed, target_col='Export_Tons')

        rf_result = train_rf_model(
            X_train, y_train, X_test, y_test,
            n_estimators=200,
            max_depth=20,
            random_state=42
        )

        # 4. Save model
        print("\n💾 Saving model...")
        save_model(rf_result['model'], "rf_olive_oil_model.pkl")

        # 5. Show results
        print("\n✅ Training Complete!")
        print(f"📈 Training R²: {rf_result['train_metrics']['R2']:.3f}")
        print(f"📈 Test R²: {rf_result['test_metrics']['R2']:.3f}")
        print(f"📊 Test MAE: {rf_result['test_metrics']['MAE']:,.0f} tons")

        print("\n🚀 Ready to use! Run: streamlit run app/app.py")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n🔍 Check that all required files exist and try again.")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())