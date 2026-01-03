"""
Configuration file for the sales prediction model.
Modify these settings as needed.
"""

# Dataset configuration
DATASET_NAME = "pratyushakar/rossmann-store-sales"
LOCAL_DATA_DIR = "data"  # Set to None to always download from Kaggle

# Model configuration
MODEL_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "reg:squarederror",
    "random_state": 42
}

# Training configuration
TRAIN_TEST_SPLIT_DATE = "2015-06-01"

# SHAP configuration
SHAP_SAMPLE_SIZE = 100  # Number of samples for SHAP explanations

# Output configuration
OUTPUT_DIR = "output"

# Visualization configuration
PLOT_STYLE = "seaborn-v0_8"
FIGURE_DPI = 150

