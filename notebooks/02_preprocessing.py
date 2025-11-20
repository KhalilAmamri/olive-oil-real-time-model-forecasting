"""Preprocessing for Export Forecasting - See notebooks/01_exploration.ipynb"""

# This is a placeholder. Use the Jupyter notebook for interactive preprocessing:
# Run: jupyter notebook notebooks/01_exploration.ipynb

# Key preprocessing steps for export forecasting:
# 1. Load tunisia_olive_oil_dataset.csv (7,560 records)
# 2. Create lag features for Export_Tons (lag 1, 7, 14, 30 days)
# 3. Create rolling statistics (mean/std over 7 and 30 day windows)
# 4. Encode country variable (42 countries)
# 5. Create time features (month, year, day of week, etc.)
# 6. Use Production_Tons as input feature

# Target variable: Export_Tons
# Features: 26 total (production, lags, rolling stats, time, country, price)
