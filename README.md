# 🫒 Olive Oil Production Forecasting System

A **production-ready machine learning system** for olive oil production forecasting with real-time predictions and batch forecasting capabilities.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Training Models](#training-models)
  - [Running Streamlit App](#running-streamlit-app)
  - [Batch Forecasting](#batch-forecasting)
- [Models](#models)
- [Why Streamlit vs Power BI](#why-streamlit-vs-power-BI)
- [Future Enhancements](#future-enhancements)

---

## 🎯 Overview

This system provides end-to-end machine learning capabilities for forecasting olive oil production using:

- **Prophet**: Facebook's time series forecasting model for seasonal patterns
- **RandomForest**: Ensemble learning for complex feature interactions

The system supports two forecasting modes:

1. **Real-Time Forecasting**: Instant predictions based on user inputs via Streamlit UI
2. **Batch Forecasting**: Generate forecasts for extended periods and save to CSV

---

## ✨ Features

- ✅ **Modular Architecture**: Clean separation of data loading, preprocessing, training, and forecasting
- ✅ **Interactive Dashboard**: Beautiful Streamlit interface with real-time visualizations
- ✅ **Dual Model Support**: Prophet for time series + RandomForest for multi-feature predictions
- ✅ **Batch Processing**: Generate and save large-scale forecasts
- ✅ **Production Ready**: Structured for Docker deployment and CI/CD pipelines
- ✅ **Comprehensive Notebooks**: Jupyter notebooks for exploration, preprocessing, and training

---

## 📁 Project Structure

```
project/
│
├── data/
│   ├── raw/                    # Raw CSV datasets
│   │   └── tunisia_olive_oil_dataset.csv
│   └── forecasts/              # Saved batch forecasts
│
├── models/                     # Trained ML models (.pkl files)
│   ├── prophet_model.pkl
│   └── rf_model.pkl
│
├── src/                        # Core Python modules
│   ├── data_loader.py          # Data loading utilities
│   ├── preprocess.py           # Data preprocessing & feature engineering
│   ├── model_train.py          # Model training (Prophet, RF)
│   ├── forecast_realtime.py    # Real-time prediction logic
│   ├── forecast_batch.py       # Batch forecasting logic
│   └── utils.py                # Helper functions
│
├── app/                        # Streamlit application
│   ├── app.py                  # Main Streamlit entry point
│   └── pages/
│       ├── real_time_forecast.py    # Real-time forecast page
│       └── batch_forecast_viewer.py # Batch forecast viewer
│
├── notebooks/                  # Jupyter notebooks
│   ├── 01_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_training.ipynb
│
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository

```bash
cd "c:\Learn Programming\MachineLearning\olive-oil-real-time-model-forecasting"
```

### Step 2: Create Virtual Environment

```powershell
# Create virtual environment
python -m venv .venv

# Activate (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Activate (Windows CMD)
.venv\Scripts\activate.bat

# Activate (Linux/Mac)
source .venv/bin/activate
```

### Step 3: Install Dependencies

```powershell
pip install -r requirements.txt
```

---

## 📖 Usage

### Training Models

Before using the forecasting system, you need to train the models:

#### Option 1: Using Python Script

```python
# Run in Python or Jupyter
from src.data_loader import load_raw_data
from src.preprocess import preprocess_data, prepare_prophet_data, prepare_ml_data
from src.model_train import train_prophet_model, train_rf_model, save_model

# 1. Load raw data
df = load_raw_data()
print(f"Loaded {len(df)} records")

# 2. Preprocess data
df_processed = preprocess_data(df)

# 3. Prepare data for Prophet
prophet_df = prepare_prophet_data(df_processed)

# 4. Train Prophet model
prophet_model = train_prophet_model(prophet_df)
save_model(prophet_model, "prophet_model.pkl")

# 5. Prepare data for RandomForest
X_train, X_test, y_train, y_test = prepare_ml_data(df_processed)

# 6. Train RandomForest model
rf_result = train_rf_model(X_train, y_train, X_test, y_test)
save_model(rf_result['model'], "rf_model.pkl")

print("✓ Models trained and saved successfully!")
```

#### Option 2: Using Jupyter Notebooks

```powershell
jupyter notebook notebooks/03_training.ipynb
```

Follow the step-by-step instructions in the notebook.

---

### Running Streamlit App

Once models are trained, launch the interactive dashboard:

```powershell
streamlit run app/app.py
```

The app will open in your browser at `http://localhost:8501`

#### Features:

1. **🏠 Home Page**: Overview and documentation
2. **⚡ Real-Time Forecast**:
   - Select country, date, and features
   - Get instant predictions from both models
   - View interactive visualizations
3. **📊 Batch Forecast Viewer**:
   - View saved batch forecasts
   - Interactive charts and filtering
   - Download forecast CSVs

---

### Batch Forecasting

Generate forecasts for extended periods:

```python
from src.forecast_batch import run_batch_forecast

# Generate 12-month forecast
results_df = run_batch_forecast(
    model_type="prophet",        # or "rf"
    start_date="2026-01-01",
    periods=365,                 # number of days
    freq="D",                    # Daily frequency
    save_results=True
)

print(f"Generated {len(results_df)} predictions")
print(f"Saved to: data/forecasts/")
```

**Batch forecasts are automatically saved to:**

- `data/forecasts/batch_forecast_prophet_YYYYMMDD_HHMMSS.csv`
- `data/forecasts/batch_forecast_latest.csv` (always points to most recent)

---

## 🤖 Models

### Prophet Model

**Type**: Time Series Forecasting  
**Best for**: Capturing seasonal patterns and trends

**How it works**:

- Decomposes time series into trend + seasonality + holidays
- Handles missing data automatically
- Provides uncertainty intervals (confidence bands)

**Training**:

```python
from src.model_train import train_prophet_model

model = train_prophet_model(prophet_data, seasonality_mode='multiplicative')
```

### RandomForest Model

**Type**: Ensemble Machine Learning  
**Best for**: Multi-feature predictions with complex interactions

**How it works**:

- Builds 200 decision trees on bootstrapped samples
- Each tree votes on the prediction
- Handles non-linear relationships automatically
- Provides feature importance

**Training**:

```python
from src.model_train import train_rf_model

rf_result = train_rf_model(X_train, y_train, X_test, y_test, n_estimators=200)
```

---

## 🎯 Real-Time vs Batch Forecasting

### Real-Time Forecasting

**What it is**:

- User provides inputs via UI
- Model generates prediction instantly (< 1 second)
- Result displayed immediately with charts

**Use cases**:

- Interactive what-if analysis
- Quick decision-making
- User-driven exploration
- Dashboard applications

**Example flow**:

```
User Input → Model Inference → Display Result
  (< 1 sec total)
```

### Batch Forecasting

**What it is**:

- Generate forecasts for many periods at once (e.g., full year)
- Process hundreds/thousands of predictions
- Save results to CSV files

**Use cases**:

- Monthly/quarterly business planning
- Long-term strategic decisions
- Automated reporting
- Stakeholder distribution

**Example flow**:

```
Define Parameters → Generate All Predictions → Save to CSV → Share File
  (runs periodically, e.g., weekly)
```

---

## 💡 Why Streamlit vs Power BI?

This project uses **Streamlit** instead of traditional BI tools like Power BI. Here's why:

### Streamlit Advantages

| Feature                   | Streamlit              | Power BI                        |
| ------------------------- | ---------------------- | ------------------------------- |
| **Python Integration**    | ✅ Native              | ❌ Limited (via Python visuals) |
| **ML Model Loading**      | ✅ Direct import       | ❌ Requires gateway/export      |
| **Real-Time Predictions** | ✅ Instant             | ❌ Slow (batch only)            |
| **Version Control**       | ✅ Git-friendly        | ❌ Binary files                 |
| **Deployment**            | ✅ Docker/Cloud easy   | ❌ Complex infrastructure       |
| **Cost**                  | ✅ Free & Open Source  | ❌ Licensing required           |
| **Customization**         | ✅ Full Python control | ❌ Limited to built-in features |
| **CI/CD Integration**     | ✅ Simple              | ❌ Difficult                    |

### Key Insight

Streamlit keeps **everything in Python**:

```
Data → Preprocessing → ML Models → Visualization → User Interaction
        ALL IN ONE CODEBASE
```

No need to:

- Export data to external tools
- Set up complex gateways
- Deal with licensing restrictions
- Maintain separate BI environments

---

## 🔮 Future Enhancements

- [ ] Add LSTM/XGBoost models
- [ ] Model performance comparison dashboard
- [ ] Automated model retraining pipeline
- [ ] REST API endpoints
- [ ] Docker containerization
- [ ] Cloud deployment (AWS/Azure/GCP)
- [ ] A/B testing framework
- [ ] Model versioning with MLflow
- [ ] Real-time monitoring dashboard

---

## 📊 Example Workflows

### Workflow 1: Train and Test Models

```powershell
# 1. Start Jupyter
jupyter notebook

# 2. Run notebooks in order
#    - 01_exploration.ipynb (understand data)
#    - 02_preprocessing.ipynb (prepare features)
#    - 03_training.ipynb (train models)

# 3. Verify models are saved
dir models
```

### Workflow 2: Generate Monthly Reports

```python
from src.forecast_batch import run_batch_forecast
from datetime import datetime

# Generate forecasts for next quarter
results = run_batch_forecast(
    model_type="prophet",
    start_date=datetime(2026, 1, 1),
    periods=90,  # 3 months
    freq="D",
    save_results=True
)

# Share the CSV with stakeholders
print(f"Report saved: data/forecasts/batch_forecast_latest.csv")
```

### Workflow 3: Interactive Analysis

```powershell
# Launch Streamlit
streamlit run app/app.py

# Use the UI to:
# 1. Select different countries
# 2. Try various dates
# 3. Experiment with feature values
# 4. Compare model predictions
```

---

## 📝 License

This project is licensed under the MIT License.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📧 Contact

For questions or support, please open an issue in the repository.

---

**Built with ❤️ using Python, Streamlit, Prophet, and scikit-learn**
