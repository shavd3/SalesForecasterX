"""
Sales prediction model module.
Handles model training, prediction, and evaluation.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import pickle
import os


class SalesPredictor:
    """XGBoost-based sales prediction model."""
    
    def __init__(self, model_params=None):
        """
        Initialize the sales predictor.
        
        Args:
            model_params: Dictionary of XGBoost parameters
        """
        if model_params is None:
            model_params = {
                "n_estimators": 300,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "objective": "reg:squarederror",
                "random_state": 42
            }
        
        self.model_params = model_params
        self.model = xgb.XGBRegressor(**model_params)
        self.is_trained = False
        self.feature_names = None
        
    def train(self, X_train, y_train):
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training target
        """
        self.model.fit(X_train, y_train)
        self.feature_names = X_train.columns.tolist()
        self.is_trained = True
    
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X: Features to predict on
            
        Returns:
            Array of predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary with RMSE and MAE metrics
        """
        y_pred = self.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        return {
            "rmse": rmse,
            "mae": mae,
            "predictions": y_pred,
            "actual": y_test.values
        }
    
    def what_if_analysis(self, X_sample, feature_name, new_value):
        """
        Perform what-if analysis by changing a feature value.
        
        Args:
            X_sample: Sample feature row (DataFrame with one row)
            feature_name: Name of feature to change
            new_value: New value for the feature
            
        Returns:
            Dictionary with original and new predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before what-if analysis")
        
        original_pred = self.predict(X_sample)[0]
        
        what_if = X_sample.copy()
        what_if[feature_name] = new_value
        new_pred = self.predict(what_if)[0]
        
        return {
            "original_prediction": original_pred,
            "new_prediction": new_pred,
            "difference": new_pred - original_pred,
            "feature": feature_name,
            "original_value": X_sample[feature_name].iloc[0],
            "new_value": new_value
        }
    
    def save_model(self, filepath):
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_names': self.feature_names,
                'params': self.model_params
            }, f)
    
    def load_model(self, filepath):
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to load the model from
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.feature_names = data['feature_names']
            self.model_params = data['params']
            self.is_trained = True

