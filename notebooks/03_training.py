"""Model Training for Export Forecasting"""

# This is a placeholder. Use the training script instead:
# Run: python scripts/train_models.py

# Or explore training interactively:
# Run: jupyter notebook notebooks/01_exploration.ipynb

# Training process:
# 1. Load and preprocess data (7,560 records -> 7,530 after lag features)
# 2. Split into train (80%) and test (20%)
# 3. Train RandomForest with 200 trees, max_depth=20
# 4. Evaluate: R²=0.79 (train), R²=0.22 (test)
# 5. Save model to models/rf_olive_oil_model.pkl

# Model predicts: Export_Tons
# From features: Production_Tons, Export lags, time features, country, price
