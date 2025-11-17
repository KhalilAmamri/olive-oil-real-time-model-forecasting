from pathlib import Path
import sys

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import load_raw_data
from src.preprocess import preprocess_data, prepare_prophet_data, prepare_ml_data
from src.model_train import (
    train_prophet_model,
    train_rf_model,
    save_model,
)

# Prophet availability check is inside model_train via try/except; import safely
try:
    from src.model_train import PROPHET_AVAILABLE  # type: ignore
except Exception:
    PROPHET_AVAILABLE = False  # Fallback


def main() -> None:
    print("=== Training Pipeline Start ===")
    # 1) Load
    df = load_raw_data()

    # 2) Preprocess
    dfp = preprocess_data(df)

    # 3) Prophet (optional)
    if PROPHET_AVAILABLE:
        try:
            prophet_df = prepare_prophet_data(dfp)
            prophet_model = train_prophet_model(prophet_df)
            save_model(prophet_model, "prophet_model.pkl", model_type="prophet")
        except Exception as e:
            print(f"Skipping Prophet training: {e}")
    else:
        print("Prophet not installed; skipping Prophet model.")

    # 4) RandomForest
    X_train, X_test, y_train, y_test = prepare_ml_data(dfp)
    rf_result = train_rf_model(X_train, y_train, X_test, y_test)
    save_model(rf_result["model"], "rf_model.pkl", model_type="rf")

    print("=== Training Pipeline Complete ===")


if __name__ == "__main__":
    main()
