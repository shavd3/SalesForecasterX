"""
Quick start script with minimal setup.
This script provides a simplified way to test the system.
"""

import os
import sys


def main():
    """Quick start guide."""
    print("=" * 60)
    print("Sales Prediction Model - Quick Start Guide")
    print("=" * 60)
    
    print("\n📋 Step 1: Install Dependencies")
    print("   Run: pip install -r requirements.txt")
    
    print("\n🔍 Step 2: Verify Setup")
    print("   Run: python setup_check.py")
    
    print("\n📥 Step 3: Download Dataset")
    print("   The dataset will be automatically downloaded when you run main.py")
    print("   Make sure you have Kaggle credentials configured:")
    print("   - Create a Kaggle account at https://www.kaggle.com")
    print("   - Go to Account > API and create a token")
    print("   - Place kaggle.json in ~/.kaggle/ (or configure kagglehub)")
    
    print("\n🚀 Step 4: Train the Model")
    print("   Run: python main.py")
    print("   This will:")
    print("   - Download the dataset (if not already downloaded)")
    print("   - Process and engineer features")
    print("   - Train the XGBoost model")
    print("   - Generate predictions and visualizations")
    print("   - Save outputs to output/ directory")
    
    print("\n🎨 Step 5: Launch the UI")
    print("   Run: streamlit run app.py")
    print("   The dashboard will open in your browser")
    
    print("\n📁 Alternative: Use Local Data")
    print("   If you already have the dataset files (train.csv, store.csv):")
    print("   - Place them in a directory (e.g., 'data/')")
    print("   - Modify main.py to use: processor.load_data(local_data_dir='data/')")
    
    print("\n" + "=" * 60)
    print("For more details, see README.md")
    print("=" * 60)


if __name__ == "__main__":
    main()

