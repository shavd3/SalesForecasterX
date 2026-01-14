"""
Company data processing module.
Handles uploaded Excel/CSV files from companies and prepares them for prediction.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class CompanyDataProcessor:
    """Processes company-uploaded sales data for predictions."""
    
    def __init__(self):
        """Initialize the company data processor."""
        self.required_columns = {
            'date': ['date', 'Date', 'DATE', 'day', 'Day', 'DAY'],
            'sales': ['sales', 'Sales', 'SALES', 'revenue', 'Revenue', 'REVENUE', 
                     'amount', 'Amount', 'AMOUNT', 'value', 'Value', 'VALUE'],
            'store': ['store', 'Store', 'STORE', 'store_id', 'Store_ID', 'STORE_ID', 
                     'location', 'Location', 'LOCATION', 'branch', 'Branch', 'BRANCH']
        }
        self.optional_columns = {
            'promo': ['promo', 'Promo', 'PROMO', 'promotion', 'Promotion', 'PROMOTION',
                     'has_promo', 'Has_Promo', 'HAS_PROMO'],
            'school_holiday': ['school_holiday', 'School_Holiday', 'SCHOOL_HOLIDAY',
                              'holiday', 'Holiday', 'HOLIDAY', 'is_holiday', 'Is_Holiday'],
            'open': ['open', 'Open', 'OPEN', 'is_open', 'Is_Open', 'IS_OPEN']
        }
    
    def detect_column_mapping(self, df):
        """
        Automatically detect column mappings from uploaded data.
        
        Args:
            df: DataFrame with uploaded data
            
        Returns:
            Dictionary mapping standard names to actual column names
        """
        mapping = {}
        df_columns_lower = [col.lower().strip() for col in df.columns]
        
        # Find required columns
        for standard_name, possible_names in self.required_columns.items():
            for col in df.columns:
                if col.lower().strip() in [n.lower() for n in possible_names]:
                    mapping[standard_name] = col
                    break
        
        # Find optional columns
        for standard_name, possible_names in self.optional_columns.items():
            for col in df.columns:
                if col.lower().strip() in [n.lower() for n in possible_names]:
                    mapping[standard_name] = col
                    break
        
        return mapping
    
    def validate_data(self, df, mapping):
        """
        Validate uploaded data has required columns.
        
        Args:
            df: DataFrame to validate
            mapping: Column mapping dictionary
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check required columns
        missing = []
        for req_col in ['date', 'sales', 'store']:
            if req_col not in mapping:
                missing.append(req_col)
        
        if missing:
            return False, f"Missing required columns: {', '.join(missing)}"
        
        # Check data types
        try:
            # Try to parse date
            pd.to_datetime(df[mapping['date']].iloc[0])
        except:
            return False, f"Date column '{mapping['date']}' cannot be parsed as dates"
        
        # Check sales is numeric
        try:
            pd.to_numeric(df[mapping['sales']])
        except:
            return False, f"Sales column '{mapping['sales']}' must contain numeric values"
        
        return True, None
    
    def process_uploaded_data(self, df, mapping=None):
        """
        Process uploaded company data into standard format.
        
        Args:
            df: DataFrame with uploaded data
            mapping: Optional column mapping (auto-detected if None)
            
        Returns:
            Processed DataFrame in standard format
        """
        # Auto-detect mapping if not provided
        if mapping is None:
            mapping = self.detect_column_mapping(df)
        
        # Validate
        is_valid, error = self.validate_data(df, mapping)
        if not is_valid:
            raise ValueError(error)
        
        # Create standardized dataframe
        processed = pd.DataFrame()
        
        # Map columns
        processed['Date'] = pd.to_datetime(df[mapping['date']])
        processed['Sales'] = pd.to_numeric(df[mapping['sales']], errors='coerce')
        processed['Store'] = df[mapping['store']].astype(str)
        
        # Handle optional columns
        if 'promo' in mapping:
            promo_col = df[mapping['promo']]
            # Convert to 0/1 if needed
            if promo_col.dtype == 'object' or promo_col.dtype == 'bool':
                processed['Promo'] = (promo_col.astype(str).str.lower().isin(
                    ['true', '1', 'yes', 'y', 'with', 'with promotion', 'promo'])).astype(int)
            else:
                processed['Promo'] = (promo_col > 0).astype(int)
        else:
            processed['Promo'] = 0
        
        if 'school_holiday' in mapping:
            holiday_col = df[mapping['school_holiday']]
            if holiday_col.dtype == 'object' or holiday_col.dtype == 'bool':
                processed['SchoolHoliday'] = (holiday_col.astype(str).str.lower().isin(
                    ['true', '1', 'yes', 'y'])).astype(int)
            else:
                processed['SchoolHoliday'] = (holiday_col > 0).astype(int)
        else:
            processed['SchoolHoliday'] = 0
        
        if 'open' in mapping:
            open_col = df[mapping['open']]
            if open_col.dtype == 'object' or open_col.dtype == 'bool':
                processed['Open'] = (open_col.astype(str).str.lower().isin(
                    ['true', '1', 'yes', 'y', 'open'])).astype(int)
            else:
                processed['Open'] = (open_col > 0).astype(int)
        else:
            # Assume all stores are open if not specified
            processed['Open'] = 1
        
        # Add default store features (can be enhanced later)
        processed['StoreType'] = 0
        processed['Assortment'] = 0
        processed['StateHoliday'] = '0'
        
        # Sort by Store and Date
        processed = processed.sort_values(['Store', 'Date']).reset_index(drop=True)
        
        return processed
    
    def create_features_for_prediction(self, df, model_features):
        """
        Create features needed for model prediction from processed data.
        
        Args:
            df: Processed DataFrame
            model_features: List of feature names the model expects
            
        Returns:
            DataFrame with features ready for prediction
        """
        # Make a copy
        df_feat = df.copy()
        
        # Time-based features
        df_feat['day_of_week'] = df_feat['Date'].dt.dayofweek
        df_feat['week_of_year'] = df_feat['Date'].dt.isocalendar().week.astype(int)
        df_feat['month'] = df_feat['Date'].dt.month
        df_feat['year'] = df_feat['Date'].dt.year
        df_feat['is_weekend'] = df_feat['day_of_week'].isin([5, 6]).astype(int)
        
        # Sort by Store and Date for lag features
        df_feat = df_feat.sort_values(['Store', 'Date']).reset_index(drop=True)
        
        # Lag features (using available historical data)
        lags = [1, 7, 14, 28]
        for lag in lags:
            df_feat[f'sales_lag_{lag}'] = df_feat.groupby('Store')['Sales'].shift(lag)
        
        # Rolling window features
        windows = [7, 14, 28]
        for window in windows:
            df_feat[f'rolling_mean_{window}'] = (
                df_feat.groupby('Store')['Sales']
                .shift(1)
                .rolling(window, min_periods=1)
                .mean()
            )
        
        # Encode categorical features
        df_feat['StoreType'] = df_feat['StoreType'].astype('category').cat.codes
        df_feat['Assortment'] = df_feat['Assortment'].astype('category').cat.codes
        
        # Select only features needed by model
        available_features = [f for f in model_features if f in df_feat.columns]
        missing_features = [f for f in model_features if f not in df_feat.columns]
        
        # Fill missing features with defaults
        for feat in missing_features:
            if 'lag' in feat or 'rolling' in feat:
                df_feat[feat] = df_feat['Sales'].mean()  # Use mean sales as default
            else:
                df_feat[feat] = 0
        
        # Ensure all model features are present
        feature_df = df_feat[model_features].copy()
        
        # Fill NaN values (using forward fill, backward fill, then zero)
        feature_df = feature_df.ffill().bfill().fillna(0)
        
        return feature_df, df_feat

