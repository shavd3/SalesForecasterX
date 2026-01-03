# Sales Prediction Model with XAI

A time series sales prediction model using XGBoost with Explainable AI (XAI) capabilities. This project predicts daily sales using past trends and business factors like promotions, with interactive visualizations and model interpretability.

## Features

- **Time Series Forecasting**: Predicts daily sales using historical trends
- **Feature Engineering**: Creates lag features, rolling means, and time-based features
- **XGBoost Model**: High-performance gradient boosting model
- **Explainable AI**: SHAP-based model interpretability
- **Interactive UI**: Streamlit dashboard for exploring predictions and insights
- **What-If Analysis**: Explore how changing business factors affects predictions

## Installation

1. Clone or download this repository

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Verify your setup:
```bash
python setup_check.py
```

4. Set up Kaggle credentials (for dataset download):
   - Create a Kaggle account at https://www.kaggle.com
   - Go to Account settings and create an API token
   - Place `kaggle.json` in `~/.kaggle/` (or configure kagglehub)
   - Alternatively, you can manually download the dataset and place `train.csv` and `store.csv` in a `data/` directory

## Usage

### Quick Start

For a quick overview, run:
```bash
python quick_start.py
```

### 1. Train the Model

Run the main script to train the model and generate predictions:

```bash
python main.py
```

This will:
- Download the Rossmann Store Sales dataset from Kaggle (or use local files if available)
- Process and engineer features
- Train the XGBoost model
- Generate predictions and visualizations
- Save model and data to `output/` directory

**Note:** The script will automatically use local data if `train.csv` and `store.csv` are found in a `data/` directory.

### 2. Launch the UI

Start the Streamlit dashboard:

```bash
streamlit run app.py
```

The dashboard provides:
- **Overview**: Model performance metrics and key insights
- **Predictions**: Detailed prediction analysis
- **Store Analysis**: Store-specific sales trends
- **XAI Insights**: Feature importance and SHAP explanations
- **What-If Analysis**: Explore scenario changes

## Project Structure

```
.
├── main.py                 # Main executable script
├── app.py                  # Streamlit UI application
├── data_processor.py       # Data loading and feature engineering
├── sales_model.py          # Model training and prediction
├── xai_explainer.py        # SHAP-based explainability
├── visualizations.py       # Plotting functions
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── output/                # Generated outputs (created after running main.py)
    ├── sales_model.pkl
    ├── processed_data.pkl
    └── *.png (visualizations)
```

## Model Features

The model uses the following features:

- **Time Features**: day_of_week, week_of_year, month, is_weekend
- **Business Factors**: Promo, SchoolHoliday
- **Lag Features**: sales_lag_1, sales_lag_7, sales_lag_14, sales_lag_28
- **Rolling Means**: rolling_mean_7, rolling_mean_14, rolling_mean_28

## Output Files

After running `main.py`, the following files are generated in `output/`:

- `sales_model.pkl`: Trained model
- `processed_data.pkl`: Processed data for UI
- `store_sales.png`: Store sales visualization
- `sales_rolling_mean.png`: Sales vs rolling means
- `prediction_vs_actual.png`: Prediction accuracy plot
- `error_distribution.png`: Error analysis
- `absolute_error_histogram.png`: Error magnitude distribution
- `promo_effect.png`: Promotion impact analysis
- `feature_importance.png`: SHAP feature importance
- `shap_summary.png`: SHAP summary plot

## Model Performance

The model is evaluated using:
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)

Performance metrics are displayed in the UI and console output.

## XAI Features

The explainability module provides:
- Feature importance rankings
- SHAP summary plots
- Individual prediction explanations (waterfall plots)
- What-if scenario analysis

## Notes

- The dataset is automatically downloaded from Kaggle on first run
- Model training may take a few minutes depending on your system
- The UI requires the model to be trained first (run `main.py`)
- SHAP computations use a sample of test data for performance

## License

This project uses the Rossmann Store Sales dataset from Kaggle. Please refer to the dataset license for usage terms.

