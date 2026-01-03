"""
Visualization module for sales predictions and model insights.
Supports both matplotlib (for static images) and Plotly (for interactive UI).
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import shap
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class SalesVisualizer:
    """Creates visualizations for sales data and model results."""
    
    def __init__(self, style="seaborn-v0_8"):
        """
        Initialize the visualizer.
        
        Args:
            style: Matplotlib style to use
        """
        plt.style.use(style)
        sns.set_palette("husl")
    
    def plot_store_sales(self, store_df, store_id, figsize=(14, 5)):
        """
        Plot daily sales for a specific store.
        
        Args:
            store_df: DataFrame with Date index and Sales column
            store_id: Store ID for title
            figsize: Figure size
            
        Returns:
            matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        store_df["Sales"].plot(ax=ax, title=f"Store {store_id} – Daily Sales")
        ax.set_ylabel("Sales")
        ax.set_xlabel("Date")
        
        plt.tight_layout()
        return fig
    
    def plot_sales_vs_rolling_mean(self, store_df, store_id, figsize=(14, 5)):
        """
        Plot sales vs rolling means.
        
        Args:
            store_df: DataFrame with Date index and Sales/rolling_mean columns
            store_id: Store ID for title
            figsize: Figure size
            
        Returns:
            matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(store_df.index, store_df["Sales"], label="Actual Sales", alpha=0.7)
        
        if "rolling_mean_7" in store_df.columns:
            ax.plot(store_df.index, store_df["rolling_mean_7"], 
                   label="7-day Rolling Mean", linewidth=2)
        
        if "rolling_mean_28" in store_df.columns:
            ax.plot(store_df.index, store_df["rolling_mean_28"], 
                   label="28-day Rolling Mean", linewidth=2)
        
        ax.set_title(f"Sales vs Rolling Mean – Store {store_id}")
        ax.set_ylabel("Sales")
        ax.set_xlabel("Date")
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_prediction_vs_actual(self, y_actual, y_pred, sample_size=None, 
                                  figsize=(14, 5), title="Prediction vs Actual"):
        """
        Plot predicted vs actual sales.
        
        Args:
            y_actual: Actual sales values
            y_pred: Predicted sales values
            sample_size: Number of samples to plot (None for all)
            figsize: Figure size
            title: Plot title
            
        Returns:
            matplotlib figure
        """
        if sample_size is not None:
            y_actual = y_actual[:sample_size]
            y_pred = y_pred[:sample_size]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(y_actual, label="Actual", linewidth=2)
        ax.plot(y_pred, label="Predicted", linewidth=2)
        ax.set_title(title)
        ax.set_ylabel("Sales")
        ax.set_xlabel("Days")
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_error_distribution(self, errors, figsize=(10, 5)):
        """
        Plot distribution of prediction errors.
        
        Args:
            errors: Prediction errors (actual - predicted)
            figsize: Figure size
            
        Returns:
            matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.histplot(errors, bins=50, kde=True, ax=ax)
        ax.set_title("Prediction Error Distribution")
        ax.set_xlabel("Error (Actual - Predicted)")
        ax.set_ylabel("Frequency")
        
        plt.tight_layout()
        return fig
    
    def plot_absolute_error_histogram(self, errors, figsize=(10, 5)):
        """
        Plot histogram of absolute errors.
        
        Args:
            errors: Prediction errors
            figsize: Figure size
            
        Returns:
            matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.hist(np.abs(errors), bins=40)
        ax.set_title("How Big Are the Prediction Errors?")
        ax.set_xlabel("Absolute Error (Units)")
        ax.set_ylabel("Number of Days")
        
        plt.tight_layout()
        return fig
    
    def plot_promo_effect(self, df_model, figsize=(6, 4)):
        """
        Plot average sales with vs without promotion.
        
        Args:
            df_model: DataFrame with Promo and Sales columns
            figsize: Figure size
            
        Returns:
            matplotlib figure
        """
        promo_effect = df_model.groupby("Promo")["Sales"].mean()
        
        fig, ax = plt.subplots(figsize=figsize)
        promo_effect.plot(kind="bar", ax=ax, rot=0)
        ax.set_title("Average Sales: Promotion vs No Promotion")
        ax.set_ylabel("Average Sales")
        ax.set_xlabel("Promotion")
        ax.set_xticklabels(["No Promo", "Promo"])
        
        plt.tight_layout()
        return fig
    
    def plot_feature_importance(self, importance_df, figsize=(8, 6)):
        """
        Plot feature importance from SHAP values.
        
        Args:
            importance_df: DataFrame with Feature and Impact columns
            figsize: Figure size
            
        Returns:
            matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Sort by impact for horizontal bar chart
        importance_df_sorted = importance_df.sort_values("Impact", ascending=True)
        
        ax.barh(importance_df_sorted["Feature"], importance_df_sorted["Impact"])
        ax.set_title("Main Factors That Influence Sales Predictions")
        ax.set_xlabel("Average Impact on Prediction")
        
        plt.tight_layout()
        return fig
    
    def plot_shap_summary(self, shap_values, X, max_display=10, figsize=(10, 8)):
        """
        Plot SHAP summary plot.
        
        Args:
            shap_values: SHAP values object
            X: Feature matrix
            max_display: Maximum features to display
            figsize: Figure size
            
        Returns:
            matplotlib figure
        """
        fig = plt.figure(figsize=figsize)
        shap.summary_plot(shap_values, X, max_display=max_display, show=False)
        plt.tight_layout()
        return fig
    
    def plot_shap_waterfall(self, shap_values_instance, max_display=10, figsize=(10, 6)):
        """
        Plot SHAP waterfall plot for a single instance.
        
        Args:
            shap_values_instance: SHAP values for a single instance
            max_display: Maximum features to display
            figsize: Figure size
            
        Returns:
            matplotlib figure
        """
        fig = plt.figure(figsize=figsize)
        shap.plots.waterfall(shap_values_instance, max_display=max_display, show=False)
        plt.tight_layout()
        return fig
    
    def plot_trends_comparison(self, store_df, figsize=(16, 6)):
        """
        Plot daily sales vs short-term and long-term trends.
        
        Args:
            store_df: DataFrame with Date index and Sales/rolling_mean columns
            figsize: Figure size
            
        Returns:
            matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(store_df.index, store_df["Sales"], 
               label="Daily Sales", alpha=0.4)
        
        if "rolling_mean_7" in store_df.columns:
            ax.plot(store_df.index, store_df["rolling_mean_7"],
                   label="7-day Average", linewidth=2)
        
        if "rolling_mean_28" in store_df.columns:
            ax.plot(store_df.index, store_df["rolling_mean_28"],
                   label="28-day Average", linewidth=3)
        
        ax.set_title("Daily Sales vs Short-term and Long-term Trends")
        ax.set_ylabel("Sales")
        ax.set_xlabel("Date")
        ax.legend()
        
        plt.tight_layout()
        return fig


class PlotlyVisualizer:
    """Creates interactive Plotly visualizations for Streamlit UI."""
    
    def __init__(self):
        """Initialize the Plotly visualizer."""
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is not installed. Install it with: pip install plotly")
        
        # Default color scheme
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'tertiary': '#2ca02c',
            'background': '#f7f7f7'
        }
    
    def plot_store_sales(self, store_df, store_id):
        """Plot daily sales for a specific store (Plotly)."""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=store_df.index,
            y=store_df["Sales"],
            mode='lines',
            name='Daily Sales',
            line=dict(color=self.colors['primary'], width=2),
            hovertemplate='<b>Date:</b> %{x}<br><b>Sales:</b> %{y:,.0f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f"Store {store_id} – Daily Sales",
            xaxis_title="Date",
            yaxis_title="Sales",
            hovermode='x unified',
            template='plotly_white',
            height=400,
            showlegend=True
        )
        
        return fig
    
    def plot_sales_vs_rolling_mean(self, store_df, store_id):
        """Plot sales vs rolling means (Plotly)."""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=store_df.index,
            y=store_df["Sales"],
            mode='lines',
            name='Actual Sales',
            line=dict(color=self.colors['primary'], width=1.5),
            opacity=0.7,
            hovertemplate='<b>Date:</b> %{x}<br><b>Sales:</b> %{y:,.0f}<extra></extra>'
        ))
        
        if "rolling_mean_7" in store_df.columns:
            fig.add_trace(go.Scatter(
                x=store_df.index,
                y=store_df["rolling_mean_7"],
                mode='lines',
                name='7-day Rolling Mean',
                line=dict(color=self.colors['secondary'], width=2.5),
                hovertemplate='<b>Date:</b> %{x}<br><b>7-day Avg:</b> %{y:,.0f}<extra></extra>'
            ))
        
        if "rolling_mean_28" in store_df.columns:
            fig.add_trace(go.Scatter(
                x=store_df.index,
                y=store_df["rolling_mean_28"],
                mode='lines',
                name='28-day Rolling Mean',
                line=dict(color=self.colors['tertiary'], width=3),
                hovertemplate='<b>Date:</b> %{x}<br><b>28-day Avg:</b> %{y:,.0f}<extra></extra>'
            ))
        
        fig.update_layout(
            title=f"Sales vs Rolling Mean – Store {store_id}",
            xaxis_title="Date",
            yaxis_title="Sales",
            hovermode='x unified',
            template='plotly_white',
            height=400,
            showlegend=True
        )
        
        return fig
    
    def plot_prediction_vs_actual(self, y_actual, y_pred, sample_size=None, 
                                  title="Prediction vs Actual"):
        """Plot predicted vs actual sales (Plotly)."""
        if sample_size is not None:
            y_actual = y_actual[:sample_size]
            y_pred = y_pred[:sample_size]
        
        days = np.arange(len(y_actual))
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=days,
            y=y_actual,
            mode='lines',
            name='Actual',
            line=dict(color=self.colors['primary'], width=2.5),
            hovertemplate='<b>Day:</b> %{x}<br><b>Actual Sales:</b> %{y:,.0f}<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=days,
            y=y_pred,
            mode='lines',
            name='Predicted',
            line=dict(color=self.colors['secondary'], width=2.5, dash='dash'),
            hovertemplate='<b>Day:</b> %{x}<br><b>Predicted Sales:</b> %{y:,.0f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Days",
            yaxis_title="Sales",
            hovermode='x unified',
            template='plotly_white',
            height=400,
            showlegend=True
        )
        
        return fig
    
    def plot_error_distribution(self, errors):
        """Plot distribution of prediction errors (Plotly)."""
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=errors,
            nbinsx=50,
            name='Error Distribution',
            marker_color=self.colors['primary'],
            opacity=0.7,
            hovertemplate='<b>Error:</b> %{x:.2f}<br><b>Frequency:</b> %{y}<extra></extra>'
        ))
        
        # Add KDE-like smoothing
        try:
            from scipy import stats
            if len(errors) > 10:
                x_kde = np.linspace(errors.min(), errors.max(), 200)
                y_kde = stats.gaussian_kde(errors)(x_kde)
                y_kde = y_kde * len(errors) * (errors.max() - errors.min()) / 50
                
                fig.add_trace(go.Scatter(
                    x=x_kde,
                    y=y_kde,
                    mode='lines',
                    name='Density',
                    line=dict(color=self.colors['secondary'], width=2),
                    hovertemplate='<b>Error:</b> %{x:.2f}<br><b>Density:</b> %{y:.2f}<extra></extra>'
                ))
        except:
            pass  # Skip KDE if scipy not available
        
        fig.update_layout(
            title="Prediction Error Distribution",
            xaxis_title="Error (Actual - Predicted)",
            yaxis_title="Frequency",
            hovermode='x unified',
            template='plotly_white',
            height=400,
            showlegend=True,
            bargap=0.1
        )
        
        return fig
    
    def plot_absolute_error_histogram(self, errors):
        """Plot histogram of absolute errors (Plotly)."""
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=np.abs(errors),
            nbinsx=40,
            name='Absolute Errors',
            marker_color=self.colors['primary'],
            opacity=0.8,
            hovertemplate='<b>Absolute Error:</b> %{x:.2f}<br><b>Number of Days:</b> %{y}<extra></extra>'
        ))
        
        fig.update_layout(
            title="How Big Are the Prediction Errors?",
            xaxis_title="Absolute Error (Units)",
            yaxis_title="Number of Days",
            hovermode='x unified',
            template='plotly_white',
            height=400,
            showlegend=False,
            bargap=0.1
        )
        
        return fig
    
    def plot_promo_effect(self, df_model):
        """Plot average sales with vs without promotion (Plotly)."""
        promo_effect = df_model.groupby("Promo")["Sales"].mean()
        
        fig = go.Figure()
        
        colors_bar = [self.colors['secondary'] if i == 0 else self.colors['primary'] 
                     for i in range(len(promo_effect))]
        
        fig.add_trace(go.Bar(
            x=['No Promo', 'Promo'],
            y=promo_effect.values,
            marker_color=colors_bar,
            text=[f'{val:,.0f}' for val in promo_effect.values],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br><b>Average Sales:</b> %{y:,.0f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Average Sales: Promotion vs No Promotion",
            xaxis_title="Promotion",
            yaxis_title="Average Sales",
            template='plotly_white',
            height=400,
            showlegend=False
        )
        
        return fig
    
    def plot_feature_importance(self, importance_df):
        """Plot feature importance from SHAP values (Plotly)."""
        importance_df_sorted = importance_df.sort_values("Impact", ascending=True)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=importance_df_sorted["Feature"],
            x=importance_df_sorted["Impact"],
            orientation='h',
            marker_color=self.colors['primary'],
            text=[f'{val:.2f}' for val in importance_df_sorted["Impact"]],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br><b>Impact:</b> %{x:.4f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Main Factors That Influence Sales Predictions",
            xaxis_title="Average Impact on Prediction",
            yaxis_title="Feature",
            template='plotly_white',
            height=max(400, len(importance_df) * 30),
            showlegend=False
        )
        
        return fig
    
    def plot_trends_comparison(self, store_df):
        """Plot daily sales vs short-term and long-term trends (Plotly)."""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=store_df.index,
            y=store_df["Sales"],
            mode='lines',
            name='Daily Sales',
            line=dict(color=self.colors['primary'], width=1),
            opacity=0.4,
            hovertemplate='<b>Date:</b> %{x}<br><b>Sales:</b> %{y:,.0f}<extra></extra>'
        ))
        
        if "rolling_mean_7" in store_df.columns:
            fig.add_trace(go.Scatter(
                x=store_df.index,
                y=store_df["rolling_mean_7"],
                mode='lines',
                name='7-day Average',
                line=dict(color=self.colors['secondary'], width=2.5),
                hovertemplate='<b>Date:</b> %{x}<br><b>7-day Avg:</b> %{y:,.0f}<extra></extra>'
            ))
        
        if "rolling_mean_28" in store_df.columns:
            fig.add_trace(go.Scatter(
                x=store_df.index,
                y=store_df["rolling_mean_28"],
                mode='lines',
                name='28-day Average',
                line=dict(color=self.colors['tertiary'], width=3.5),
                hovertemplate='<b>Date:</b> %{x}<br><b>28-day Avg:</b> %{y:,.0f}<extra></extra>'
            ))
        
        fig.update_layout(
            title="Daily Sales vs Short-term and Long-term Trends",
            xaxis_title="Date",
            yaxis_title="Sales",
            hovermode='x unified',
            template='plotly_white',
            height=450,
            showlegend=True
        )
        
        return fig
    
    def plot_shap_summary(self, shap_values, X, max_display=10):
        """Plot SHAP summary plot. Returns image bytes for Streamlit display."""
        # For SHAP plots, we'll use matplotlib and return image bytes
        import matplotlib.pyplot as plt
        import io
        
        fig = plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X, max_display=max_display, show=False)
        plt.tight_layout()
        
        # Convert matplotlib figure to image bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        return buf
    
    def plot_shap_waterfall(self, shap_values_instance, max_display=10):
        """Plot SHAP waterfall plot. Returns image bytes for Streamlit display."""
        import matplotlib.pyplot as plt
        import io
        
        fig = plt.figure(figsize=(10, 6))
        shap.plots.waterfall(shap_values_instance, max_display=max_display, show=False)
        plt.tight_layout()
        
        # Convert matplotlib figure to image bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        return buf

