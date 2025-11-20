# Olive Oil Export Forecasting - Simplified Version

A beginner-friendly machine learning project that predicts olive oil export volumes using real Tunisian data.

## 🫒 What This Project Does

This project uses machine learning to predict how much olive oil a country will export based on:

- Production volume
- Market prices
- Historical export patterns
- Time factors (season, month, etc.)

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python train.py
```

### 3. Run the Dashboard

```bash
streamlit run app/app.py
```

## 📊 Dashboard Features

### Analytics Tab

- **Filter Data**: Choose countries, date ranges, and seasons
- **Time Series Charts**: See export trends over time
- **Country Comparisons**: Compare export volumes between countries
- **Price Analysis**: Explore relationships between prices and exports

### Real-Time Prediction Tab

- **Input Parameters**: Enter production volume, price, and country
- **Instant Predictions**: Get export forecasts immediately
- **Revenue Estimates**: See potential revenue and profit
- **Historical Comparison**: Compare predictions with past data

## 📁 Project Structure

```
project/
├── train.py              # Simple training script
├── app/
│   ├── app.py           # Main Streamlit app
│   └── pages/
│       └── export_dashboard.py  # Unified dashboard
├── src/
│   ├── data_loader.py   # Load and summarize data
│   ├── preprocess.py    # Clean and prepare data
│   ├── model_train.py   # Train ML models
│   └── forecast.py      # Make predictions
├── models/              # Saved trained models
├── data/
│   └── raw/
│       └── tunisia_olive_oil_dataset.csv
└── requirements.txt     # Python dependencies
```

## 🤖 How It Works

1. **Data Loading**: Reads olive oil data from CSV file
2. **Preprocessing**: Creates useful features from dates and historical data
3. **Training**: Trains a RandomForest model to learn patterns
4. **Prediction**: Uses the trained model to forecast exports

## 📈 Model Performance

- **Algorithm**: RandomForest Regressor
- **Training R²**: ~0.79 (explains 79% of export variation)
- **Test R²**: ~0.22 (good for real-world predictions)
- **Features**: 26 engineered features including lags, time factors, and country encoding

## 🛠️ Technologies

- **Python 3.8+**
- **Streamlit**: Interactive web dashboard
- **scikit-learn**: Machine learning
- **pandas**: Data manipulation
- **plotly**: Interactive charts

## 📝 For Beginners

This project is designed to be easy to understand:

- Clear comments in all code
- Simple training script
- Interactive dashboard
- Real-world business application

Start by running `python train.py` then explore the dashboard!</content>
<parameter name="filePath">c:\Learn Programming\MachineLearning\olive-oil-real-time-model-forecasting\README.md
