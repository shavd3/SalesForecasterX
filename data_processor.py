"""
Data processing module for sales time series model.
Handles data loading, cleaning, and feature engineering.
"""

import os
import pandas as pd
import numpy as np
import kagglehub


class DataProcessor:
    """Handles data loading, cleaning, and feature engineering."""
    
    def __init__(self, dataset_name="pratyushakar/rossmann-store-sales"):
        """
        Initialize the data processor.
        
        Args:
            dataset_name: Kaggle dataset identifier
        """
        self.dataset_name = dataset_name
        self.df = None
        self.processed_df = None
        
    def load_data(self, cache_dir=None, local_data_dir=None):
        """
        Load data from Kaggle dataset or local files.
        
        Args:
            cache_dir: Optional custom cache directory
            local_data_dir: Optional local directory with train.csv and store.csv
            
        Returns:
            Tuple of (train_df, stores_df)
        """
        # Try local files first if provided
        if local_data_dir and os.path.exists(local_data_dir):
            train_path = os.path.join(local_data_dir, "train.csv")
            store_path = os.path.join(local_data_dir, "store.csv")
            
            if os.path.exists(train_path) and os.path.exists(store_path):
                print(f"Loading data from local directory: {local_data_dir}")
                df = pd.read_csv(train_path, parse_dates=["Date"])
                stores = pd.read_csv(store_path)
                df = df.merge(stores, on="Store", how="left")
                self.df = df
                return df, stores
        
        # Download from Kaggle
        try:
            print("Downloading dataset from Kaggle...")
            path = kagglehub.dataset_download(self.dataset_name)
            print(f"Dataset downloaded to: {path}")
        except Exception as e:
            raise Exception(f"Failed to download dataset from Kaggle: {e}\n"
                          f"Please ensure kagglehub is configured with valid credentials.")
        
        # Determine dataset path
        if cache_dir and os.path.exists(cache_dir):
            dataset_path = cache_dir
        else:
            # kagglehub returns the path directly to the dataset directory
            dataset_path = path
            
            # If path doesn't contain CSV files, try to find them
            if not os.path.exists(os.path.join(dataset_path, "train.csv")):
                # Try common cache locations
                possible_paths = [
                    os.path.join(os.path.expanduser("~"), ".cache", "kagglehub", "datasets", 
                               self.dataset_name.replace("/", "-"), "versions"),
                    os.path.join(os.path.expanduser("~"), ".kaggle", "datasets", 
                               self.dataset_name.replace("/", "-")),
                ]
                
                for base_path in possible_paths:
                    if os.path.exists(base_path):
                        versions = [d for d in os.listdir(base_path) 
                                  if os.path.isdir(os.path.join(base_path, d))]
                        if versions:
                            latest_version = sorted(versions)[-1]
                            candidate_path = os.path.join(base_path, latest_version)
                            if os.path.exists(os.path.join(candidate_path, "train.csv")):
                                dataset_path = candidate_path
                                break
        
        # Load CSV files
        train_path = os.path.join(dataset_path, "train.csv")
        store_path = os.path.join(dataset_path, "store.csv")
        
        if not os.path.exists(train_path):
            raise FileNotFoundError(
                f"train.csv not found in {dataset_path}. "
                f"Please check the dataset path or download the dataset manually."
            )
        
        if not os.path.exists(store_path):
            raise FileNotFoundError(
                f"store.csv not found in {dataset_path}. "
                f"Please check the dataset path or download the dataset manually."
            )
        
        print(f"Loading data from: {dataset_path}")
        df = pd.read_csv(train_path, parse_dates=["Date"])
        stores = pd.read_csv(store_path)
        
        # Merge datasets
        df = df.merge(stores, on="Store", how="left")
        
        self.df = df
        return df, stores
    
    def clean_data(self, df=None):
        """
        Clean the dataset by filtering open stores and positive sales.
        
        Args:
            df: DataFrame to clean (uses self.df if None)
            
        Returns:
            Cleaned DataFrame
        """
        if df is None:
            df = self.df.copy()
        
        # Filter for open stores with positive sales
        df = df[df["Open"] == 1].copy()
        df = df[df["Sales"] > 0].copy()
        
        # Handle missing values
        df["StateHoliday"] = df["StateHoliday"].astype(str)
        df.fillna(0, inplace=True)
        
        return df
    
    def create_features(self, df=None):
        """
        Create time-based and lag features.
        
        Args:
            df: DataFrame to process (uses self.df if None)
            
        Returns:
            DataFrame with engineered features
        """
        if df is None:
            df = self.df.copy()
        
        # Time-based features
        df["day_of_week"] = df["Date"].dt.dayofweek
        df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)
        df["month"] = df["Date"].dt.month
        df["year"] = df["Date"].dt.year
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        
        # Sort by Store and Date for lag features
        df = df.sort_values(["Store", "Date"]).copy()
        
        # Lag features
        lags = [1, 7, 14, 28]
        for lag in lags:
            df[f"sales_lag_{lag}"] = df.groupby("Store")["Sales"].shift(lag)
        
        # Encode categorical features
        df["StoreType"] = df["StoreType"].astype("category").cat.codes
        df["Assortment"] = df["Assortment"].astype("category").cat.codes
        
        # Rolling window features
        windows = [7, 14, 28]
        for window in windows:
            df[f"rolling_mean_{window}"] = (
                df.groupby("Store")["Sales"]
                .shift(1)
                .rolling(window)
                .mean()
            )
        
        self.processed_df = df
        return df
    
    def prepare_model_data(self, df=None, split_date="2015-06-01"):
        """
        Prepare data for model training and testing.
        
        Args:
            df: DataFrame to process (uses self.processed_df if None)
            split_date: Date to split train/test sets
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, df_model)
        """
        if df is None:
            df = self.processed_df.copy()
        
        # Define features
        features = [
            "day_of_week", "week_of_year", "month", "is_weekend",
            "Promo", "SchoolHoliday",
            "sales_lag_1", "sales_lag_7", "sales_lag_14", "sales_lag_28",
            "rolling_mean_7", "rolling_mean_14", "rolling_mean_28"
        ]
        
        # Remove rows with missing features
        df_model = df.dropna(subset=features + ["Sales"]).copy()
        
        X = df_model[features]
        y = df_model["Sales"]
        
        # Split by date
        X_train = X[df_model["Date"] < split_date]
        X_test = X[df_model["Date"] >= split_date]
        y_train = y[df_model["Date"] < split_date]
        y_test = y[df_model["Date"] >= split_date]
        
        return X_train, X_test, y_train, y_test, df_model
    
    def get_store_data(self, store_id, df=None):
        """
        Get data for a specific store.
        
        Args:
            store_id: Store ID
            df: DataFrame to filter (uses self.processed_df if None)
            
        Returns:
            Filtered DataFrame for the store
        """
        if df is None:
            df = self.processed_df.copy()
        
        store_df = (
            df[df["Store"] == store_id]
            .sort_values("Date")
            .set_index("Date")
        )
        
        return store_df

