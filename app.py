"""
Streamlit UI for Sales Prediction Model.
Interactive interface to explore predictions, visualizations, and XAI insights.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from data_processor import DataProcessor
from sales_model import SalesPredictor
from xai_explainer import XAIExplainer
from visualizations import PlotlyVisualizer
import config
import shap


# Page configuration
st.set_page_config(
    page_title="Sales Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_model_and_data():
    """Load trained model and processed data."""
    try:
        # Load model
        model = SalesPredictor()
        model_path = os.path.join(config.OUTPUT_DIR, "sales_model.pkl")
        if os.path.exists(model_path):
            model.load_model(model_path)
        else:
            return None, None, f"Model not found at {model_path}. Please run main.py first."
        
        # Load processed data
        data_path = os.path.join(config.OUTPUT_DIR, "processed_data.pkl")
        if os.path.exists(data_path):
            with open(data_path, "rb") as f:
                data = pickle.load(f)
            return model, data, None
        else:
            return model, None, f"Processed data not found at {data_path}. Please run main.py first."
    except Exception as e:
        return None, None, f"Error loading data: {str(e)}"


def main():
    """Main Streamlit app."""
    st.markdown('<h1 class="main-header">📊 Sales Prediction Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Load model and data
    model, data, error = load_model_and_data()
    
    if error:
        st.error(error)
        st.info("""
        **To get started:**
        1. Run `python main.py` to train the model and generate predictions
        2. This will create the necessary output files
        3. Refresh this page to see the dashboard
        """)
        return
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Overview", "Predictions", "Store Analysis", "XAI Insights", "What-If Analysis"]
    )
    
    if page == "Overview":
        show_overview(data)
    elif page == "Predictions":
        show_predictions(data)
    elif page == "Store Analysis":
        show_store_analysis(data)
    elif page == "XAI Insights":
        show_xai_insights(model, data)
    elif page == "What-If Analysis":
        show_what_if_analysis(model, data)


def show_overview(data):
    """Show overview dashboard."""
    st.header("📈 Model Overview")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("RMSE", f"{data['results']['rmse']:.2f}")
    
    with col2:
        st.metric("MAE", f"{data['results']['mae']:.2f}")
    
    with col3:
        st.metric("Test Samples", len(data['y_test']))
    
    with col4:
        avg_sales = data['df_model']['Sales'].mean()
        st.metric("Avg Sales", f"{avg_sales:.0f}")
    
    # Key visualizations
    st.subheader("Key Insights")
    
    visualizer = PlotlyVisualizer()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Prediction vs Actual")
        fig = visualizer.plot_prediction_vs_actual(
            data['results']['actual'], 
            data['results']['predictions'],
            sample_size=300
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Promotion Effect")
        fig = visualizer.plot_promo_effect(data['df_model'])
        st.plotly_chart(fig, use_container_width=True)
    
    # Error analysis
    st.subheader("Error Analysis")
    errors = data['results']['actual'] - data['results']['predictions']
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = visualizer.plot_error_distribution(errors)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = visualizer.plot_absolute_error_histogram(errors)
        st.plotly_chart(fig, use_container_width=True)


def show_predictions(data):
    """Show prediction analysis."""
    st.header("🔮 Predictions Analysis")
    
    # Sample size selector
    sample_size = st.slider("Number of days to display", 50, 500, 300)
    
    # Plot predictions
    visualizer = PlotlyVisualizer()
    fig = visualizer.plot_prediction_vs_actual(
        data['results']['actual'],
        data['results']['predictions'],
        sample_size=sample_size,
        title=f"Actual vs Predicted Sales (First {sample_size} Days)"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Prediction statistics
    st.subheader("Prediction Statistics")
    
    pred_df = pd.DataFrame({
        'Actual': data['results']['actual'],
        'Predicted': data['results']['predictions'],
        'Error': data['results']['actual'] - data['results']['predictions'],
        'Abs_Error': np.abs(data['results']['actual'] - data['results']['predictions'])
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(pred_df.describe(), use_container_width=True)
    
    with col2:
        st.write("**Error Percentiles:**")
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        error_percentiles = np.percentile(pred_df['Abs_Error'], percentiles)
        percentile_df = pd.DataFrame({
            'Percentile': [f"{p}%" for p in percentiles],
            'Absolute Error': error_percentiles
        })
        st.dataframe(percentile_df, use_container_width=True, hide_index=True)


def show_store_analysis(data):
    """Show store-specific analysis."""
    st.header("🏪 Store Analysis")
    
    # Store selector
    available_stores = sorted(data['df_processed']['Store'].unique())
    selected_store = st.selectbox("Select Store", available_stores)
    
    # Get store data
    processor = DataProcessor()
    processor.processed_df = data['df_processed']
    store_df = processor.get_store_data(selected_store)
    
    # Store metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Avg Daily Sales", f"{store_df['Sales'].mean():.0f}")
    
    with col2:
        st.metric("Max Daily Sales", f"{store_df['Sales'].max():.0f}")
    
    with col3:
        st.metric("Min Daily Sales", f"{store_df['Sales'].min():.0f}")
    
    with col4:
        promo_days = data['df_processed'][data['df_processed']['Store'] == selected_store]['Promo'].sum()
        st.metric("Promo Days", f"{promo_days}")
    
    # Visualizations
    visualizer = PlotlyVisualizer()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Daily Sales Trend")
        fig = visualizer.plot_store_sales(store_df, selected_store)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Sales vs Rolling Means")
        fig = visualizer.plot_sales_vs_rolling_mean(store_df, selected_store)
        st.plotly_chart(fig, use_container_width=True)
    
    # Trends comparison
    st.subheader("Trend Analysis")
    fig = visualizer.plot_trends_comparison(store_df)
    st.plotly_chart(fig, use_container_width=True)


def show_xai_insights(model, data):
    """Show XAI insights."""
    st.header("🔍 Explainable AI Insights")
    
    st.markdown("""
    This section provides insights into how the model makes predictions using SHAP (SHapley Additive exPlanations).
    """)
    
    # Check if SHAP values exist
    if 'shap_values' not in data or data['shap_values'] is None:
        st.warning("SHAP values not found. Please run main.py to generate explanations.")
        return
    
    # Feature importance
    st.subheader("Feature Importance")
    
    explainer = XAIExplainer(model.model)
    explainer.shap_values = data['shap_values']
    
    importance_df = explainer.get_feature_importance()
    
    visualizer = PlotlyVisualizer()
    fig = visualizer.plot_feature_importance(importance_df)
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance table
    st.dataframe(importance_df, use_container_width=True, hide_index=True)
    
    # SHAP summary plot
    st.subheader("SHAP Summary Plot")
    st.markdown("""
    This plot shows how each feature affects the model's output. 
    Red indicates higher feature values, blue indicates lower values.
    """)
    
    img_bytes = visualizer.plot_shap_summary(
        data['shap_values'], 
        data['X_test_sample'],
        max_display=10
    )
    st.image(img_bytes, use_container_width=True)
    
    # Individual instance explanation
    st.subheader("Individual Prediction Explanation")
    
    max_idx = min(99, len(data.get('X_test_sample', pd.DataFrame())) - 1)
    if max_idx < 0:
        st.warning("No test samples available for explanation.")
        return
    
    instance_idx = st.slider(
        "Select instance to explain",
        0, 
        max_idx,
        0
    )
    
    shap_instance = explainer.get_instance_explanation(instance_idx)
    
    # Show prediction
    instance_features = data['X_test_sample'].iloc[[instance_idx]]
    prediction = model.predict(instance_features)[0]
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predicted Sales", f"{prediction:.0f}")
    with col2:
        try:
            if 'X_test' in data and instance_idx < len(data['X_test']):
                actual_idx = data['X_test'].index.get_loc(data['X_test_sample'].index[instance_idx])
                if actual_idx < len(data['y_test']):
                    actual = data['y_test'].iloc[actual_idx]
                    st.metric("Actual Sales", f"{actual:.0f}")
                    st.metric("Error", f"{abs(prediction - actual):.0f}")
        except (KeyError, IndexError):
            pass
    
    # Waterfall plot
    img_bytes = visualizer.plot_shap_waterfall(shap_instance, max_display=10)
    st.image(img_bytes, use_container_width=True)
    
    # Feature values for this instance
    st.subheader("Feature Values")
    st.dataframe(instance_features.T, use_container_width=True)


def show_what_if_analysis(model, data):
    """Show what-if analysis."""
    st.header("🎯 What-If Analysis")
    
    st.markdown("""
    Explore how changing business factors (like promotions) would affect sales predictions.
    """)
    
    # Select instance
    max_idx = min(99, len(data.get('X_test_sample', pd.DataFrame())) - 1)
    if max_idx < 0:
        st.warning("No test samples available for what-if analysis.")
        return
    
    instance_idx = st.slider(
        "Select instance for analysis",
        0,
        max_idx,
        min(10, max_idx)
    )
    
    instance_features = data['X_test_sample'].iloc[[instance_idx]].copy()
    
    # Current prediction
    current_pred = model.predict(instance_features)[0]
    
    st.subheader("Current Scenario")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Predicted Sales", f"{current_pred:.0f}")
    with col2:
        promo_status = "Yes" if instance_features['Promo'].iloc[0] == 1 else "No"
        st.metric("Has Promotion", promo_status)
    with col3:
        st.metric("Day of Week", int(instance_features['day_of_week'].iloc[0]))
    
    # What-if scenarios
    st.subheader("What-If Scenarios")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Scenario 1: Toggle Promotion**")
        new_promo = st.selectbox("Promotion Status", [0, 1], 
                                index=int(instance_features['Promo'].iloc[0]))
        
        if new_promo != instance_features['Promo'].iloc[0]:
            what_if_result = model.what_if_analysis(instance_features, "Promo", new_promo)
            
            st.metric("New Prediction", f"{what_if_result['new_prediction']:.0f}")
            st.metric("Change", f"{what_if_result['difference']:+.0f}")
            
            if what_if_result['difference'] > 0:
                st.success(f"✅ Adding promotion increases sales by {what_if_result['difference']:.0f}")
            else:
                st.warning(f"⚠️ Removing promotion decreases sales by {abs(what_if_result['difference']):.0f}")
    
    with col2:
        st.write("**Scenario 2: Change Day of Week**")
        new_dow = st.selectbox("Day of Week", 
                              [0, 1, 2, 3, 4, 5, 6],
                              index=int(instance_features['day_of_week'].iloc[0]))
        
        if new_dow != instance_features['day_of_week'].iloc[0]:
            what_if_result = model.what_if_analysis(instance_features, "day_of_week", new_dow)
            
            st.metric("New Prediction", f"{what_if_result['new_prediction']:.0f}")
            st.metric("Change", f"{what_if_result['difference']:+.0f}")
    
    # Feature values
    st.subheader("Current Feature Values")
    feature_display = instance_features.T.copy()
    feature_display.columns = ['Value']
    st.dataframe(feature_display, use_container_width=True)


if __name__ == "__main__":
    main()

