"""
Main executable script for sales prediction model.
This script can be run directly to train the model and generate predictions.
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

from data_processor import DataProcessor
from sales_model import SalesPredictor
from xai_explainer import XAIExplainer
from visualizations import SalesVisualizer
import config
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt


def main():
    """Main execution function."""
    print("=" * 60)
    print("Sales Prediction Model - Training and Evaluation")
    print("=" * 60)
    
    # Initialize components
    print("\n[1/6] Initializing data processor...")
    processor = DataProcessor(dataset_name=config.DATASET_NAME)
    
    # Load data
    print("\n[2/6] Loading data...")
    try:
        # Check if local data directory exists
        local_data_dir = None
        if config.LOCAL_DATA_DIR and os.path.exists(config.LOCAL_DATA_DIR):
            if os.path.exists(os.path.join(config.LOCAL_DATA_DIR, "train.csv")):
                local_data_dir = config.LOCAL_DATA_DIR
                print(f"Found local data directory: {local_data_dir}")
        
        df, stores = processor.load_data(local_data_dir=local_data_dir)
        print(f"Loaded {len(df)} records")
    except Exception as e:
        print(f"\n[ERROR] Error loading data: {e}")
        print("\nTroubleshooting:")
        print("  1. Ensure kagglehub is configured with valid credentials")
        print("  2. Or place train.csv and store.csv in a 'data/' directory")
        print("  3. Run 'python setup_check.py' to verify your setup")
        return
    
    # Clean and process data
    print("\n[3/6] Cleaning and engineering features...")
    df_clean = processor.clean_data(df)
    df_processed = processor.create_features(df_clean)
    print(f"Processed {len(df_processed)} records with features")
    
    # Prepare model data
    print("\n[4/6] Preparing training and test sets...")
    X_train, X_test, y_train, y_test, df_model = processor.prepare_model_data(
        df_processed, split_date=config.TRAIN_TEST_SPLIT_DATE
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Train model
    print("\n[5/6] Training XGBoost model...")
    model = SalesPredictor(model_params=config.MODEL_PARAMS)
    model.train(X_train, y_train)
    print("Model training completed!")
    
    # Evaluate model
    print("\n[6/6] Evaluating model...")
    results = model.evaluate(X_test, y_test)
    print(f"\nModel Performance:")
    print(f"  RMSE: {results['rmse']:.2f}")
    print(f"  MAE:  {results['mae']:.2f}")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    visualizer = SalesVisualizer(style=config.PLOT_STYLE)
    
    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Plot 1: Store sales example
    store_df = processor.get_store_data(1, df_processed)
    fig = visualizer.plot_store_sales(store_df, 1)
    plt.savefig(os.path.join(config.OUTPUT_DIR, "store_sales.png"), 
                dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: {config.OUTPUT_DIR}/store_sales.png")
    
    # Plot 2: Sales vs rolling mean
    fig = visualizer.plot_sales_vs_rolling_mean(store_df, 1)
    plt.savefig(os.path.join(config.OUTPUT_DIR, "sales_rolling_mean.png"), 
                dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: {config.OUTPUT_DIR}/sales_rolling_mean.png")
    
    # Plot 3: Prediction vs actual
    fig = visualizer.plot_prediction_vs_actual(
        results['actual'], results['predictions'], 
        sample_size=300, title="Actual vs Predicted Sales (Sample Period)"
    )
    plt.savefig(os.path.join(config.OUTPUT_DIR, "prediction_vs_actual.png"), 
                dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: {config.OUTPUT_DIR}/prediction_vs_actual.png")
    
    # Plot 4: Error distribution
    errors = results['actual'] - results['predictions']
    fig = visualizer.plot_error_distribution(errors)
    plt.savefig(os.path.join(config.OUTPUT_DIR, "error_distribution.png"), 
                dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: {config.OUTPUT_DIR}/error_distribution.png")
    
    # Plot 5: Absolute error histogram
    fig = visualizer.plot_absolute_error_histogram(errors)
    plt.savefig(os.path.join(config.OUTPUT_DIR, "absolute_error_histogram.png"), 
                dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: {config.OUTPUT_DIR}/absolute_error_histogram.png")
    
    # Plot 6: Promo effect
    fig = visualizer.plot_promo_effect(df_model)
    plt.savefig(os.path.join(config.OUTPUT_DIR, "promo_effect.png"), 
                dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: {config.OUTPUT_DIR}/promo_effect.png")
    
    # XAI Analysis
    print("\nGenerating XAI explanations...")
    explainer = XAIExplainer(model.model)
    
    # Use a sample for faster computation
    sample_size = min(config.SHAP_SAMPLE_SIZE, len(X_test))
    X_test_sample = X_test.sample(n=sample_size, random_state=42)
    shap_values = explainer.explain(X_test_sample)
    
    # Feature importance
    importance_df = explainer.get_feature_importance(shap_values)
    fig = visualizer.plot_feature_importance(importance_df)
    plt.savefig(os.path.join(config.OUTPUT_DIR, "feature_importance.png"), 
                dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: {config.OUTPUT_DIR}/feature_importance.png")
    
    # SHAP summary plot
    fig = visualizer.plot_shap_summary(shap_values, X_test_sample, max_display=10)
    plt.savefig(os.path.join(config.OUTPUT_DIR, "shap_summary.png"), 
                dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: {config.OUTPUT_DIR}/shap_summary.png")
    
    # Save model
    model_path = os.path.join(config.OUTPUT_DIR, "sales_model.pkl")
    model.save_model(model_path)
    print(f"\n  [OK] Saved: {model_path}")
    
    # Save processed data for UI
    import pickle
    data_path = os.path.join(config.OUTPUT_DIR, "processed_data.pkl")
    with open(data_path, "wb") as f:
        pickle.dump({
            'df_processed': df_processed,
            'df_model': df_model,
            'X_test': X_test,
            'y_test': y_test,
            'results': results,
            'shap_values': shap_values,
            'X_test_sample': X_test_sample
        }, f)
    print(f"  [OK] Saved: {data_path}")
    
    print("\n" + "=" * 60)
    print("All tasks completed successfully!")
    print("=" * 60)
    print("\nTo launch the UI, run: streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()

