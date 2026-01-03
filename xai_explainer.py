"""
Explainable AI module using SHAP.
Provides model interpretability and feature importance analysis.
"""

import shap
import numpy as np
import pandas as pd


class XAIExplainer:
    """SHAP-based explainer for model interpretability."""
    
    def __init__(self, model, X_background=None):
        """
        Initialize the XAI explainer.
        
        Args:
            model: Trained XGBoost model
            X_background: Background dataset for SHAP (uses sample if None)
        """
        self.model = model
        self.explainer = None
        self.shap_values = None
        self.X_background = X_background
        
    def create_explainer(self, X_background=None, max_samples=100):
        """
        Create SHAP explainer.
        
        Args:
            X_background: Background dataset (uses provided or samples from X_background)
            max_samples: Maximum samples for background dataset
        """
        if X_background is None:
            X_background = self.X_background
        
        if X_background is not None and len(X_background) > max_samples:
            # Sample for faster computation
            X_background = X_background.sample(n=max_samples, random_state=42)
        
        self.explainer = shap.Explainer(self.model, X_background)
        self.X_background = X_background
    
    def explain(self, X):
        """
        Generate SHAP values for given data.
        
        Args:
            X: Features to explain
            
        Returns:
            SHAP values object
        """
        if self.explainer is None:
            self.create_explainer(X)
        
        self.shap_values = self.explainer(X)
        return self.shap_values
    
    def get_feature_importance(self, shap_values=None):
        """
        Get feature importance from SHAP values.
        
        Args:
            shap_values: SHAP values (uses self.shap_values if None)
            
        Returns:
            DataFrame with feature importance
        """
        if shap_values is None:
            if self.shap_values is None:
                raise ValueError("SHAP values not computed. Call explain() first.")
            shap_values = self.shap_values
        
        shap_importance = np.abs(shap_values.values).mean(axis=0)
        
        importance_df = pd.DataFrame({
            "Feature": shap_values.feature_names,
            "Impact": shap_importance
        }).sort_values("Impact", ascending=False)
        
        return importance_df
    
    def get_instance_explanation(self, instance_idx, shap_values=None):
        """
        Get explanation for a specific instance.
        
        Args:
            instance_idx: Index of instance to explain
            shap_values: SHAP values (uses self.shap_values if None)
            
        Returns:
            SHAP values for the instance
        """
        if shap_values is None:
            if self.shap_values is None:
                raise ValueError("SHAP values not computed. Call explain() first.")
            shap_values = self.shap_values
        
        return shap_values[instance_idx]

