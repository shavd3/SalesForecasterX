"""
Setup verification script.
Checks if all dependencies are installed and configuration is correct.
"""

import sys
import importlib


def check_import(module_name, package_name=None):
    """Check if a module can be imported."""
    if package_name is None:
        package_name = module_name
    
    try:
        importlib.import_module(module_name)
        return True, None
    except ImportError as e:
        return False, str(e)


def main():
    """Run setup checks."""
    print("=" * 60)
    print("Setup Verification")
    print("=" * 60)
    
    # Required packages
    required_packages = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("sklearn", "scikit-learn"),
        ("xgboost", "xgboost"),
        ("shap", "shap"),
        ("kagglehub", "kagglehub"),
        ("streamlit", "streamlit"),
    ]
    
    print("\n[1/3] Checking Python packages...")
    all_ok = True
    missing = []
    
    for module, package in required_packages:
        ok, error = check_import(module, package)
        if ok:
            print(f"  ✓ {package}")
        else:
            print(f"  ✗ {package} - MISSING")
            missing.append(package)
            all_ok = False
    
    if not all_ok:
        print(f"\n⚠ Missing packages: {', '.join(missing)}")
        print("Install them with: pip install -r requirements.txt")
    else:
        print("\n✓ All packages installed!")
    
    # Check project modules
    print("\n[2/3] Checking project modules...")
    project_modules = [
        "data_processor",
        "sales_model",
        "xai_explainer",
        "visualizations",
    ]
    
    all_modules_ok = True
    for module in project_modules:
        ok, error = check_import(module)
        if ok:
            print(f"  ✓ {module}.py")
        else:
            print(f"  ✗ {module}.py - {error}")
            all_modules_ok = False
    
    if not all_modules_ok:
        print("\n⚠ Some project modules are missing or have errors.")
    else:
        print("\n✓ All project modules found!")
    
    # Check Kaggle configuration
    print("\n[3/3] Checking Kaggle configuration...")
    try:
        import kagglehub
        # Try to get user info (this will fail if not configured)
        print("  ✓ kagglehub installed")
        print("  ℹ Note: Kaggle credentials will be checked when downloading dataset")
    except Exception as e:
        print(f"  ✗ kagglehub error: {e}")
    
    print("\n" + "=" * 60)
    if all_ok and all_modules_ok:
        print("✓ Setup verification complete! You're ready to run the project.")
        print("\nNext steps:")
        print("  1. Run: python main.py (to train the model)")
        print("  2. Run: streamlit run app.py (to launch the UI)")
    else:
        print("⚠ Please fix the issues above before proceeding.")
    print("=" * 60)


if __name__ == "__main__":
    main()

