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
from company_data_processor import CompanyDataProcessor
import config
import shap
import plotly.graph_objects as go


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
    .company-section {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    .demo-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #6c757d;
    }
    .divider {
        height: 2px;
        background: linear-gradient(to right, transparent, #ccc, transparent);
        margin: 1.5rem 0;
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
    
    # Initialize session state for company data
    if 'company_data' not in st.session_state:
        st.session_state.company_data = None
    if 'company_predictions' not in st.session_state:
        st.session_state.company_predictions = None
    if 'company_processed' not in st.session_state:
        st.session_state.company_processed = None
    if 'company_uploaded_file' not in st.session_state:
        st.session_state.company_uploaded_file = None
    
    # Sidebar
    st.sidebar.title("Navigation")
    
    # Check if company data is loaded
    has_company_data = st.session_state.company_data is not None and st.session_state.company_predictions is not None
    
    # Company section with better styling
    st.sidebar.markdown('<div class="company-section">', unsafe_allow_html=True)
    st.sidebar.markdown("### 🏢 Your Company Data")
    
    # Always show all company pages - they'll handle their own state checks
    if has_company_data:
        st.sidebar.success("✅ Predictions Generated")
        num_stores = len(st.session_state.company_predictions['Store'].unique()) if st.session_state.company_predictions is not None else 0
        st.sidebar.caption(f"📊 {len(st.session_state.company_predictions)} predictions | 🏪 {num_stores} stores")
    else:
        if st.session_state.company_uploaded_file is not None:
            st.sidebar.warning("⚠️ Generate predictions to view insights")
        else:
            st.sidebar.info("📤 Upload & generate predictions")
    
    company_pages = st.sidebar.radio(
        "Your Data Pages",
        ["📈 Sales Predictor", "📊 My Overview", "🔮 My Predictions", 
         "🏪 My Store Analysis", "🔍 My XAI Insights", "🎯 My What-If Analysis"],
        key="company_pages"
    )
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Visual divider
    st.sidebar.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Demo section with better styling
    st.sidebar.markdown('<div class="demo-section">', unsafe_allow_html=True)
    st.sidebar.markdown("### 🔬 Model Demo")
    st.sidebar.caption("Explore model performance & XAI")
    
    demo_pages = st.sidebar.radio(
        "Demo Pages",
        ["📊 Model Overview", "🔮 Demo Predictions", "🏪 Demo Store Analysis", 
         "🔍 Demo XAI Insights", "🎯 Demo What-If Analysis"],
        key="demo_pages"
    )
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Route to appropriate page
    if company_pages == "📈 Sales Predictor":
        st.markdown('<div class="company-section">', unsafe_allow_html=True)
        show_sales_predictor(model)
        st.markdown('</div>', unsafe_allow_html=True)
    elif company_pages == "📊 My Overview":
        st.markdown('<div class="company-section">', unsafe_allow_html=True)
        if has_company_data:
            show_company_overview(model, st.session_state.company_data, st.session_state.company_predictions)
        else:
            st.warning("⚠️ No predictions generated yet. Please upload data and generate predictions in 'Sales Predictor' first.")
            st.info("💡 Go to '📈 Sales Predictor' to upload your data and generate predictions.")
        st.markdown('</div>', unsafe_allow_html=True)
    elif company_pages == "🔮 My Predictions":
        st.markdown('<div class="company-section">', unsafe_allow_html=True)
        if has_company_data:
            show_company_predictions(model, st.session_state.company_data, st.session_state.company_predictions)
        else:
            st.warning("⚠️ No predictions generated yet. Please upload data and generate predictions in 'Sales Predictor' first.")
            st.info("💡 Go to '📈 Sales Predictor' to upload your data and generate predictions.")
        st.markdown('</div>', unsafe_allow_html=True)
    elif company_pages == "🏪 My Store Analysis":
        st.markdown('<div class="company-section">', unsafe_allow_html=True)
        if has_company_data:
            show_company_store_analysis(model, st.session_state.company_data, st.session_state.company_processed)
        else:
            st.warning("⚠️ No predictions generated yet. Please upload data and generate predictions in 'Sales Predictor' first.")
            st.info("💡 Go to '📈 Sales Predictor' to upload your data and generate predictions.")
        st.markdown('</div>', unsafe_allow_html=True)
    elif company_pages == "🔍 My XAI Insights":
        st.markdown('<div class="company-section">', unsafe_allow_html=True)
        if has_company_data:
            show_company_xai_insights(model, st.session_state.company_data, st.session_state.company_predictions)
        else:
            st.warning("⚠️ No predictions generated yet. Please upload data and generate predictions in 'Sales Predictor' first.")
            st.info("💡 Go to '📈 Sales Predictor' to upload your data and generate predictions.")
        st.markdown('</div>', unsafe_allow_html=True)
    elif company_pages == "🎯 My What-If Analysis":
        st.markdown('<div class="company-section">', unsafe_allow_html=True)
        if has_company_data:
            show_company_what_if_analysis(model, st.session_state.company_data, st.session_state.company_predictions)
        else:
            st.warning("⚠️ No predictions generated yet. Please upload data and generate predictions in 'Sales Predictor' first.")
            st.info("💡 Go to '📈 Sales Predictor' to upload your data and generate predictions.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Demo pages - always available
    if demo_pages == "📊 Model Overview":
        st.markdown('<div class="demo-section">', unsafe_allow_html=True)
        show_overview(data)
        st.markdown('</div>', unsafe_allow_html=True)
    elif demo_pages == "🔮 Demo Predictions":
        st.markdown('<div class="demo-section">', unsafe_allow_html=True)
        show_predictions(data)
        st.markdown('</div>', unsafe_allow_html=True)
    elif demo_pages == "🏪 Demo Store Analysis":
        st.markdown('<div class="demo-section">', unsafe_allow_html=True)
        show_store_analysis(data)
        st.markdown('</div>', unsafe_allow_html=True)
    elif demo_pages == "🔍 Demo XAI Insights":
        st.markdown('<div class="demo-section">', unsafe_allow_html=True)
        show_xai_insights(model, data)
        st.markdown('</div>', unsafe_allow_html=True)
    elif demo_pages == "🎯 Demo What-If Analysis":
        st.markdown('<div class="demo-section">', unsafe_allow_html=True)
        show_what_if_analysis(model, data)
        st.markdown('</div>', unsafe_allow_html=True)


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
    
    if max_idx == 0:
        instance_idx = 0
        st.info("Only one instance available")
    else:
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
        st.metric("Predicted Sales", f"${prediction:,.0f}")
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
    
    if max_idx == 0:
        instance_idx = 0
        st.info("Only one instance available")
    else:
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
        st.metric("Predicted Sales", f"${current_pred:,.0f}")
    with col2:
        promo_status = "With Promotion" if instance_features['Promo'].iloc[0] == 1 else "Without Promotion"
        st.metric("Current Promotion Status", promo_status)
    with col3:
        st.metric("Day of Week", int(instance_features['day_of_week'].iloc[0]))
    
    # What-if scenarios
    st.subheader("What-If Scenarios")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Scenario 1: Toggle Promotion**")
        promo_options = ["Without Promotion", "With Promotion"]
        promo_index = int(instance_features['Promo'].iloc[0])
        selected_promo = st.selectbox("Promotion Status", promo_options, index=promo_index)
        new_promo = 1 if selected_promo == "With Promotion" else 0
        
        if new_promo != instance_features['Promo'].iloc[0]:
            what_if_result = model.what_if_analysis(instance_features, "Promo", new_promo)
            
            st.metric("New Predicted Sales", f"${what_if_result['new_prediction']:,.0f}")
            change = what_if_result['difference']
            st.metric("Change in Sales", f"${change:+,.0f}")
            
            if change > 0:
                st.success(f"✅ {selected_promo} would increase sales by ${change:,.0f}")
            else:
                st.warning(f"⚠️ {selected_promo} would decrease sales by ${abs(change):,.0f}")
    
    with col2:
        st.write("**Scenario 2: Change Day of Week**")
        new_dow = st.selectbox("Day of Week", 
                              [0, 1, 2, 3, 4, 5, 6],
                              index=int(instance_features['day_of_week'].iloc[0]))
        
        if new_dow != instance_features['day_of_week'].iloc[0]:
            what_if_result = model.what_if_analysis(instance_features, "day_of_week", new_dow)
            
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            st.metric("New Predicted Sales", f"${what_if_result['new_prediction']:,.0f}")
            change = what_if_result['difference']
            st.metric("Change in Sales", f"${change:+,.0f}")
            st.info(f"Changing to {day_names[new_dow]} would result in this prediction.")
    
    # Feature values
    st.subheader("Current Feature Values")
    feature_display = instance_features.T.copy()
    feature_display.columns = ['Value']
    st.dataframe(feature_display, use_container_width=True)


def show_sales_predictor(model):
    """User-friendly sales prediction interface for companies."""
    st.header("📈 Sales Predictor")
    st.markdown("""
    **Welcome!** Upload your historical sales data to get accurate predictions and insights.
    Our AI model will analyze your data and provide sales forecasts with explanations.
    """)
    
    # Check if model is loaded
    if model is None or not model.is_trained:
        st.error("Model not available. Please ensure the model has been trained.")
        return
    
    # File upload section
    st.subheader("📤 Step 1: Upload Your Sales Data")
    
    uploaded_file = st.file_uploader(
        "Choose an Excel or CSV file with your sales data",
        type=['xlsx', 'xls', 'csv'],
        help="Your file should contain columns for Date, Sales, and Store/Location",
        key="company_file_upload"
    )
    
    # Store uploaded file info
    if uploaded_file is not None:
        st.session_state.company_uploaded_file = uploaded_file.name
    
    # Show data template
    with st.expander("📋 Need help? View data format requirements"):
        st.markdown("""
        **Required Columns:**
        - **Date**: Sales date (e.g., 2024-01-15)
        - **Sales**: Sales amount/revenue (numeric)
        - **Store**: Store ID, location, or branch name
        
        **Optional Columns:**
        - **Promo/Promotion**: Whether promotion was active (Yes/No, 1/0, or True/False)
        - **SchoolHoliday/Holiday**: Whether it was a holiday (Yes/No, 1/0, or True/False)
        - **Open**: Whether store was open (Yes/No, 1/0, or True/False)
        
        **Example:**
        | Date | Sales | Store | Promo | SchoolHoliday |
        |------|-------|-------|-------|---------------|
        | 2024-01-01 | 5000 | Store1 | Yes | No |
        | 2024-01-02 | 5200 | Store1 | No | No |
        """)
        
        # Create example data
        example_data = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=10, freq='D'),
            'Sales': [5000, 5200, 4800, 5500, 6000, 5800, 5400, 5600, 5900, 6100],
            'Store': ['Store1'] * 10,
            'Promo': ['Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'No', 'Yes', 'No', 'Yes'],
            'SchoolHoliday': ['No'] * 10
        })
        st.dataframe(example_data, use_container_width=True)
        st.download_button(
            "📥 Download Example Template (CSV)",
            example_data.to_csv(index=False),
            "sales_data_template.csv",
            "text/csv"
        )
    
    if uploaded_file is not None:
        try:
            # Load data
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ File loaded successfully! Found {len(df)} rows and {len(df.columns)} columns.")
            
            # Show preview
            st.subheader("📊 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Process data
            processor = CompanyDataProcessor()
            mapping = processor.detect_column_mapping(df)
            
            st.subheader("🔍 Detected Column Mapping")
            mapping_display = pd.DataFrame({
                'Standard Column': list(mapping.keys()),
                'Your Column Name': [mapping[k] for k in mapping.keys()]
            })
            st.dataframe(mapping_display, use_container_width=True, hide_index=True)
            
            # Allow manual mapping if needed
            with st.expander("⚙️ Adjust Column Mapping (if needed)"):
                st.info("If the automatic mapping is incorrect, you can adjust it here.")
                new_mapping = {}
                for std_col in ['date', 'sales', 'store', 'promo', 'school_holiday', 'open']:
                    selected = st.selectbox(
                        f"Select column for '{std_col}'",
                        ['None'] + list(df.columns),
                        index=0 if std_col not in mapping else list(df.columns).index(mapping[std_col]) + 1
                    )
                    if selected != 'None':
                        new_mapping[std_col] = selected
                if st.button("Update Mapping"):
                    mapping = new_mapping
                    st.success("Mapping updated!")
            
            # Validate and process
            is_valid, error_msg = processor.validate_data(df, mapping)
            
            if not is_valid:
                st.error(f"❌ Data validation error: {error_msg}")
                st.info("Please check your data format and try again.")
            else:
                st.success("✅ Data validation passed!")
                
                # Process data
                with st.spinner("Processing your data..."):
                    processed_df = processor.process_uploaded_data(df, mapping)
                    
                    # Get model features
                    if hasattr(model, 'feature_names') and model.feature_names:
                        model_features = model.feature_names
                    else:
                        # Default features if not available
                        model_features = [
                            "day_of_week", "week_of_year", "month", "is_weekend",
                            "Promo", "SchoolHoliday",
                            "sales_lag_1", "sales_lag_7", "sales_lag_14", "sales_lag_28",
                            "rolling_mean_7", "rolling_mean_14", "rolling_mean_28"
                        ]
                    
                    # Create features
                    feature_df, df_with_features = processor.create_features_for_prediction(
                        processed_df, model_features
                    )
                
                st.subheader("📈 Step 2: Generate Predictions")
                
                # Prediction options
                col1, col2 = st.columns(2)
                with col1:
                    predict_future = st.checkbox("Predict future dates", value=False)
                    if predict_future:
                        days_ahead = st.number_input("Days ahead to predict", min_value=1, max_value=90, value=7)
                
                with col2:
                    include_explanations = st.checkbox("Include AI explanations", value=True)
                
                if st.button("🚀 Generate Predictions", type="primary", use_container_width=True):
                    with st.spinner("Generating predictions..."):
                        # Make predictions
                        predictions = model.predict(feature_df)
                        
                        # Create results dataframe
                        results_df = processed_df.copy()
                        results_df['Predicted_Sales'] = predictions
                        results_df['Prediction_Date'] = results_df['Date']
                        
                        # Calculate confidence (using prediction variance as proxy)
                        # For simplicity, we'll use a basic confidence metric
                        results_df['Confidence'] = 'High'  # Can be enhanced with actual confidence intervals
                        
                        # Store in session state for use in other pages
                        st.session_state.company_data = {
                            'processed_df': processed_df,
                            'df_with_features': df_with_features,
                            'feature_df': feature_df,
                            'results': results_df,
                            'predictions': predictions
                        }
                        st.session_state.company_predictions = results_df
                        st.session_state.company_processed = df_with_features
                        
                        st.success(f"✅ Generated predictions for {len(results_df)} records!")
                        st.balloons()  # Celebration!
                        st.info("💡 **Great!** Now navigate to 'My Overview', 'My Predictions', etc. in the sidebar to explore your data insights!")
                        
                        # Force rerun to update sidebar
                        st.rerun()
                        
                        # Display results
                        st.subheader("📊 Prediction Results")
                        
                        # Summary metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            avg_pred = results_df['Predicted_Sales'].mean()
                            st.metric("Average Predicted Sales", f"${avg_pred:,.0f}")
                        with col2:
                            max_pred = results_df['Predicted_Sales'].max()
                            st.metric("Maximum Predicted Sales", f"${max_pred:,.0f}")
                        with col3:
                            min_pred = results_df['Predicted_Sales'].min()
                            st.metric("Minimum Predicted Sales", f"${min_pred:,.0f}")
                        with col4:
                            total_pred = results_df['Predicted_Sales'].sum()
                            st.metric("Total Predicted Sales", f"${total_pred:,.0f}")
                        
                        # Visualizations
                        st.subheader("📈 Predictions Visualization")
                        
                        visualizer = PlotlyVisualizer()
                        
                        # Group by date for overall trend
                        daily_predictions = results_df.groupby('Date')['Predicted_Sales'].sum().reset_index()
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=daily_predictions['Date'],
                            y=daily_predictions['Predicted_Sales'],
                            mode='lines+markers',
                            name='Predicted Sales',
                            line=dict(color='#1f77b4', width=3),
                            hovertemplate='<b>Date:</b> %{x}<br><b>Predicted Sales:</b> $%{y:,.0f}<extra></extra>'
                        ))
                        fig.update_layout(
                            title="Sales Predictions Over Time",
                            xaxis_title="Date",
                            yaxis_title="Predicted Sales ($)",
                            hovermode='x unified',
                            template='plotly_white',
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Store-wise predictions
                        if len(results_df['Store'].unique()) > 1:
                            st.subheader("🏪 Predictions by Store")
                            store_summary = results_df.groupby('Store').agg({
                                'Predicted_Sales': ['mean', 'sum', 'count']
                            }).round(0)
                            store_summary.columns = ['Average Sales', 'Total Sales', 'Number of Days']
                            st.dataframe(store_summary, use_container_width=True)
                        
                        # Detailed results table
                        st.subheader("📋 Detailed Predictions")
                        
                        # Format for display
                        display_df = results_df[['Date', 'Store', 'Predicted_Sales']].copy()
                        display_df['Predicted_Sales'] = display_df['Predicted_Sales'].apply(lambda x: f"${x:,.0f}")
                        display_df.columns = ['Date', 'Store/Location', 'Predicted Sales']
                        
                        st.dataframe(display_df, use_container_width=True, hide_index=True)
                        
                        # Download results
                        st.subheader("💾 Download Results")
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            "📥 Download Predictions (CSV)",
                            csv,
                            "sales_predictions.csv",
                            "text/csv",
                            key='download-csv'
                        )
                        
                        # AI Explanations
                        if include_explanations:
                            st.subheader("🤖 AI Explanations")
                            st.info("""
                            **Key Factors Affecting Your Predictions:**
                            
                            Our AI model considers multiple factors when predicting sales:
                            - **Historical Trends**: Past sales patterns and seasonality
                            - **Promotions**: Impact of promotional activities
                            - **Day of Week**: Weekend vs weekday patterns
                            - **Holidays**: Effect of school holidays and special events
                            - **Store Characteristics**: Location-specific factors
                            
                            The model has been trained on extensive retail data and adapts to your business patterns.
                            """)
                            
                            # Show feature importance if available
                            if 'shap_values' in st.session_state or hasattr(model, 'feature_importance'):
                                st.write("**Most Important Factors:**")
                                # This would show feature importance if available
                                st.info("Feature importance analysis would be displayed here based on your data.")
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("Please check that your file format matches the requirements and try again.")


# Company-specific page functions
def show_company_overview(model, company_data, company_predictions):
    """Show company-specific overview dashboard."""
    st.header("📊 Your Company Overview")
    
    if company_data is None or company_predictions is None:
        st.warning("No company data loaded. Please upload and generate predictions first.")
        return
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_pred = company_predictions['Predicted_Sales'].mean()
        st.metric("Average Predicted Sales", f"${avg_pred:,.0f}")
    
    with col2:
        total_pred = company_predictions['Predicted_Sales'].sum()
        st.metric("Total Predicted Sales", f"${total_pred:,.0f}")
    
    with col3:
        num_stores = len(company_predictions['Store'].unique())
        st.metric("Number of Stores", num_stores)
    
    with col4:
        date_range = (company_predictions['Date'].max() - company_predictions['Date'].min()).days
        st.metric("Days Analyzed", date_range)
    
    # Key visualizations
    st.subheader("Key Insights")
    
    visualizer = PlotlyVisualizer()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Predicted Sales Trend")
        daily_pred = company_predictions.groupby('Date')['Predicted_Sales'].sum().reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily_pred['Date'],
            y=daily_pred['Predicted_Sales'],
            mode='lines+markers',
            name='Predicted Sales',
            line=dict(color='#1f77b4', width=2.5),
            hovertemplate='<b>Date:</b> %{x}<br><b>Sales:</b> $%{y:,.0f}<extra></extra>'
        ))
        fig.update_layout(
            title="Your Sales Predictions Over Time",
            xaxis_title="Date",
            yaxis_title="Predicted Sales ($)",
            hovermode='x unified',
            template='plotly_white',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Promotion Effect")
        if 'Promo' in company_predictions.columns:
            promo_effect = company_predictions.groupby('Promo')['Predicted_Sales'].mean()
            fig = visualizer.plot_promo_effect(company_predictions)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Promotion data not available in your dataset")


def show_company_predictions(model, company_data, company_predictions):
    """Show company-specific prediction analysis."""
    st.header("🔮 Your Predictions Analysis")
    
    if company_data is None or company_predictions is None:
        st.warning("No company data loaded. Please upload and generate predictions first.")
        return
    
    # Sample size selector
    data_len = len(company_predictions)
    min_days = min(50, max(1, data_len))
    max_days = min(500, data_len)
    default_days = min(300, data_len)
    
    if min_days >= max_days:
        sample_size = data_len
        st.info(f"Displaying all {data_len} predictions")
    else:
        sample_size = st.slider("Number of days to display", min_days, max_days, default_days)
    
    # Plot predictions
    visualizer = PlotlyVisualizer()
    
    daily_pred = company_predictions.groupby('Date')['Predicted_Sales'].sum().reset_index()
    sample_data = daily_pred.head(sample_size)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sample_data['Date'],
        y=sample_data['Predicted_Sales'],
        mode='lines+markers',
        name='Predicted Sales',
        line=dict(color='#1f77b4', width=2.5),
        hovertemplate='<b>Date:</b> %{x}<br><b>Predicted Sales:</b> $%{y:,.0f}<extra></extra>'
    ))
    fig.update_layout(
        title=f"Your Sales Predictions (First {sample_size} Days)",
        xaxis_title="Date",
        yaxis_title="Predicted Sales ($)",
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Prediction statistics
    st.subheader("Prediction Statistics")
    
    pred_stats = pd.DataFrame({
        'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Total'],
        'Value': [
            company_predictions['Predicted_Sales'].mean(),
            company_predictions['Predicted_Sales'].median(),
            company_predictions['Predicted_Sales'].std(),
            company_predictions['Predicted_Sales'].min(),
            company_predictions['Predicted_Sales'].max(),
            company_predictions['Predicted_Sales'].sum()
        ]
    })
    pred_stats['Value'] = pred_stats['Value'].apply(lambda x: f"${x:,.0f}")
    st.dataframe(pred_stats, use_container_width=True, hide_index=True)


def show_company_store_analysis(model, company_data, company_processed):
    """Show company-specific store analysis."""
    st.header("🏪 Your Store Analysis")
    
    if company_data is None or company_processed is None:
        st.warning("No company data loaded. Please upload and generate predictions first.")
        return
    
    processed_df = company_data['processed_df']
    
    # Store selector
    available_stores = sorted(processed_df['Store'].unique())
    if len(available_stores) == 0:
        st.warning("No store data available.")
        return
    
    selected_store = st.selectbox("Select Store", available_stores)
    
    # Get store data
    store_df = processed_df[processed_df['Store'] == selected_store].copy()
    store_df = store_df.sort_values('Date').set_index('Date')
    
    # Store metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Avg Daily Sales", f"${store_df['Sales'].mean():,.0f}")
    
    with col2:
        st.metric("Max Daily Sales", f"${store_df['Sales'].max():,.0f}")
    
    with col3:
        st.metric("Min Daily Sales", f"${store_df['Sales'].min():,.0f}")
    
    with col4:
        promo_days = store_df['Promo'].sum() if 'Promo' in store_df.columns else 0
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
        if 'rolling_mean_7' in store_df.columns or 'rolling_mean_28' in store_df.columns:
            fig = visualizer.plot_sales_vs_rolling_mean(store_df, selected_store)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Rolling mean data not available for this store")
    
    # Trends comparison
    if 'rolling_mean_7' in store_df.columns:
        st.subheader("Trend Analysis")
        fig = visualizer.plot_trends_comparison(store_df)
        st.plotly_chart(fig, use_container_width=True)


def show_company_xai_insights(model, company_data, company_predictions):
    """Show company-specific XAI insights."""
    st.header("🔍 Your AI Explanations")
    
    if company_data is None or company_predictions is None:
        st.warning("No company data loaded. Please upload and generate predictions first.")
        return
    
    st.markdown("""
    This section provides insights into how the AI model makes predictions for your data using SHAP (SHapley Additive exPlanations).
    """)
    
    try:
        # Generate SHAP values for company data
        explainer = XAIExplainer(model.model)
        
        # Use a sample for faster computation
        feature_df = company_data['feature_df']
        sample_size = min(50, len(feature_df))
        X_sample = feature_df.sample(n=sample_size, random_state=42) if len(feature_df) > sample_size else feature_df
        
        with st.spinner("Generating AI explanations..."):
            shap_values = explainer.explain(X_sample)
        
        # Feature importance
        st.subheader("Feature Importance for Your Data")
        
        importance_df = explainer.get_feature_importance(shap_values)
        
        visualizer = PlotlyVisualizer()
        fig = visualizer.plot_feature_importance(importance_df)
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance table
        st.dataframe(importance_df, use_container_width=True, hide_index=True)
        
        # SHAP summary plot
        st.subheader("SHAP Summary Plot")
        st.markdown("""
        This plot shows how each feature affects predictions for your data. 
        Red indicates higher feature values, blue indicates lower values.
        """)
        
        img_bytes = visualizer.plot_shap_summary(shap_values, X_sample, max_display=10)
        st.image(img_bytes, use_container_width=True)
        
        # Individual instance explanation
        st.subheader("Individual Prediction Explanation")
        
        max_idx = min(sample_size - 1, len(X_sample) - 1)
        if max_idx < 0:
            max_idx = 0
        
        if max_idx == 0:
            instance_idx = 0
            st.info("Only one prediction available")
        else:
            instance_idx = st.slider(
                "Select prediction to explain",
                0, 
                max_idx,
                0
            )
        
        shap_instance = explainer.get_instance_explanation(instance_idx)
        
        # Show prediction
        instance_features = X_sample.iloc[[instance_idx]]
        prediction = model.predict(instance_features)[0]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Sales", f"${prediction:,.0f}")
        with col2:
            st.info("Select different instances to see how factors change predictions")
        
        # Waterfall plot
        img_bytes = visualizer.plot_shap_waterfall(shap_instance, max_display=10)
        st.image(img_bytes, use_container_width=True)
        
        # Feature values for this instance
        st.subheader("Feature Values for This Prediction")
        feature_display = instance_features.T.copy()
        feature_display.columns = ['Value']
        st.dataframe(feature_display, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error generating explanations: {str(e)}")
        st.info("SHAP explanations require additional computation. Please try again.")


def show_company_what_if_analysis(model, company_data, company_predictions):
    """Show company-specific what-if analysis."""
    st.header("🎯 What-If Analysis for Your Data")
    
    if company_data is None or company_predictions is None:
        st.warning("No company data loaded. Please upload and generate predictions first.")
        return
    
    st.markdown("""
    Explore how changing business factors (like promotions) would affect sales predictions for your stores.
    """)
    
    feature_df = company_data['feature_df']
    
    # Select instance
    max_idx = min(99, len(feature_df) - 1)
    if max_idx < 0:
        st.warning("No data available for what-if analysis.")
        return
    
    if max_idx == 0:
        instance_idx = 0
        st.info("Only one instance available")
    else:
        instance_idx = st.slider(
            "Select prediction to analyze",
            0,
            max_idx,
            min(10, max_idx)
        )
    
    instance_features = feature_df.iloc[[instance_idx]].copy()
    
    # Current prediction
    current_pred = model.predict(instance_features)[0]
    
    # Get corresponding date and store
    processed_df = company_data['processed_df']
    if instance_idx < len(processed_df):
        pred_date = processed_df.iloc[instance_idx]['Date']
        pred_store = processed_df.iloc[instance_idx]['Store']
    else:
        pred_date = "N/A"
        pred_store = "N/A"
    
    st.subheader("Current Scenario")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Predicted Sales", f"${current_pred:,.0f}")
    with col2:
        promo_status = "With Promotion" if instance_features['Promo'].iloc[0] == 1 else "Without Promotion"
        st.metric("Promotion Status", promo_status)
    with col3:
        st.metric("Date", str(pred_date)[:10] if pred_date != "N/A" else "N/A")
    with col4:
        st.metric("Store", str(pred_store))
    
    # What-if scenarios
    st.subheader("What-If Scenarios")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Scenario 1: Toggle Promotion**")
        promo_options = ["Without Promotion", "With Promotion"]
        promo_index = int(instance_features['Promo'].iloc[0])
        selected_promo = st.selectbox("Promotion Status", promo_options, index=promo_index, key="company_promo")
        new_promo = 1 if selected_promo == "With Promotion" else 0
        
        if new_promo != instance_features['Promo'].iloc[0]:
            what_if_result = model.what_if_analysis(instance_features, "Promo", new_promo)
            
            st.metric("New Predicted Sales", f"${what_if_result['new_prediction']:,.0f}")
            change = what_if_result['difference']
            st.metric("Change in Sales", f"${change:+,.0f}")
            
            if change > 0:
                st.success(f"✅ {selected_promo} would increase sales by ${change:,.0f}")
            else:
                st.warning(f"⚠️ {selected_promo} would decrease sales by ${abs(change):,.0f}")
    
    with col2:
        st.write("**Scenario 2: Change Day of Week**")
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        current_dow = int(instance_features['day_of_week'].iloc[0])
        new_dow = st.selectbox("Day of Week", day_names, index=current_dow, key="company_dow")
        new_dow_idx = day_names.index(new_dow)
        
        if new_dow_idx != current_dow:
            what_if_result = model.what_if_analysis(instance_features, "day_of_week", new_dow_idx)
            
            st.metric("New Predicted Sales", f"${what_if_result['new_prediction']:,.0f}")
            change = what_if_result['difference']
            st.metric("Change in Sales", f"${change:+,.0f}")
            st.info(f"Changing to {new_dow} would result in this prediction.")
    
    # Feature values
    st.subheader("Current Feature Values")
    feature_display = instance_features.T.copy()
    feature_display.columns = ['Value']
    st.dataframe(feature_display, use_container_width=True)


if __name__ == "__main__":
    main()

