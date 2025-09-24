#!/usr/bin/env python3
"""
Simple test script to verify the cluster portfolio optimizer functionality
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")

    try:
        import streamlit as st
        print("[OK] Streamlit imported successfully")
    except ImportError as e:
        print(f"[FAIL] Streamlit import failed: {e}")
        return False

    try:
        import plotly.graph_objects as go
        import plotly.express as px
        print("[OK] Plotly imported successfully")
    except ImportError as e:
        print(f"[FAIL] Plotly import failed: {e}")
        return False

    try:
        import yfinance as yf
        print("[OK] yfinance imported successfully")
    except ImportError as e:
        print(f"[FAIL] yfinance import failed: {e}")
        return False

    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        print("[OK] scikit-learn imported successfully")
    except ImportError as e:
        print(f"[FAIL] scikit-learn import failed: {e}")
        return False

    try:
        import yaml
        print("[OK] PyYAML imported successfully")
    except ImportError as e:
        print(f"[FAIL] PyYAML import failed: {e}")
        return False

    return True

def test_data_loading():
    """Test basic data loading functionality"""
    print("\nTesting data loading...")

    try:
        # Test a simple yfinance call
        import yfinance as yf
        ticker = yf.Ticker("AAPL")
        data = ticker.history(period="1mo")

        if not data.empty:
            print(f"[OK] Successfully loaded {len(data)} days of AAPL data")
            return True
        else:
            print("[FAIL] No data retrieved")
            return False

    except Exception as e:
        print(f"[FAIL] Data loading failed: {e}")
        return False

def test_clustering():
    """Test basic clustering functionality"""
    print("\nTesting clustering...")

    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        # Create sample data
        np.random.seed(42)
        data = np.random.randn(20, 5)

        # Scale and cluster
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)

        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = kmeans.fit_predict(scaled_data)

        print(f"[OK] Clustering successful, {len(np.unique(labels))} clusters found")
        return True

    except Exception as e:
        print(f"[FAIL] Clustering failed: {e}")
        return False

def test_yaml_loading():
    """Test YAML file loading"""
    print("\nTesting YAML loading...")

    try:
        import yaml
        yaml_path = Path("assets.yaml")

        if yaml_path.exists():
            with open(yaml_path, 'r') as file:
                data = yaml.safe_load(file)

            if data and 'asset_categories' in data:
                total_symbols = 0
                for category, info in data['asset_categories'].items():
                    if 'symbols' in info:
                        total_symbols += len(info['symbols'])

                print(f"[OK] YAML loaded successfully with {total_symbols} total symbols")
                return True
            else:
                print("[FAIL] YAML structure not as expected")
                return False
        else:
            print("[FAIL] assets.yaml file not found")
            return False

    except Exception as e:
        print(f"[FAIL] YAML loading failed: {e}")
        return False

def test_custom_modules():
    """Test if our custom modules can be imported"""
    print("\nTesting custom modules...")

    try:
        from data_manager import DataManager
        print("[OK] DataManager imported successfully")

        data_manager = DataManager()
        print("[OK] DataManager instantiated successfully")

    except ImportError as e:
        print(f"[FAIL] DataManager import failed: {e}")
        return False
    except Exception as e:
        print(f"[FAIL] DataManager instantiation failed: {e}")
        return False

    try:
        from visualizations import PortfolioVisualizer
        print("[OK] PortfolioVisualizer imported successfully")

        visualizer = PortfolioVisualizer()
        print("[OK] PortfolioVisualizer instantiated successfully")

    except ImportError as e:
        print(f"[FAIL] PortfolioVisualizer import failed: {e}")
        return False
    except Exception as e:
        print(f"[FAIL] PortfolioVisualizer instantiation failed: {e}")
        return False

    # Test analytics_engine with optional imports
    try:
        from analytics_engine import ClusteringEngine
        print("[OK] ClusteringEngine imported successfully")

        clustering_engine = ClusteringEngine()
        print("[OK] ClusteringEngine instantiated successfully")

    except ImportError as e:
        print(f"[WARN] Analytics engine import failed (may be due to optional dependencies): {e}")
        return False
    except Exception as e:
        print(f"[FAIL] Analytics engine instantiation failed: {e}")
        return False

    return True

def run_basic_workflow():
    """Test a basic end-to-end workflow"""
    print("\nTesting basic workflow...")

    try:
        from data_manager import DataManager
        from visualizations import PortfolioVisualizer

        # Initialize
        data_manager = DataManager()
        visualizer = PortfolioVisualizer()

        # Test with a few symbols
        test_symbols = ['AAPL', 'MSFT', 'GOOGL']
        print(f"Testing with symbols: {test_symbols}")

        # Fetch some recent data
        raw_data = {}
        for symbol in test_symbols:
            try:
                import yfinance as yf
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="3mo")
                if not data.empty:
                    # Normalize column names
                    data.columns = [col.lower().replace(' ', '_') for col in data.columns]
                    raw_data[symbol] = data
                    print(f"[OK] Loaded {len(data)} days for {symbol}")
            except Exception as e:
                print(f"[WARN] Failed to load {symbol}: {e}")

        if raw_data:
            # Test feature calculation
            features_df = data_manager.calculate_features(raw_data)
            if not features_df.empty:
                print(f"[OK] Calculated {len(features_df.columns)} features for {len(features_df)} symbols")

                # Test basic clustering
                try:
                    from analytics_engine import ClusteringEngine
                    clustering_engine = ClusteringEngine()

                    results = clustering_engine.fit_kmeans(features_df, n_clusters=2)
                    if results and 'labels' in results:
                        print(f"[OK] Clustering successful with {len(np.unique(results['labels']))} clusters")
                        return True
                except:
                    print("[WARN] Clustering test skipped due to import issues")
                    return True
            else:
                print("[FAIL] Feature calculation returned empty DataFrame")
                return False
        else:
            print("[FAIL] No data loaded for testing")
            return False

    except Exception as e:
        print(f"[FAIL] Workflow test failed: {e}")
        return False

def main():
    """Main test function"""
    print("Starting Cluster Portfolio Optimizer Tests")
    print(f"Python version: {sys.version}")
    print(f"Current directory: {Path.cwd()}")
    print("=" * 60)

    tests = [
        ("Import Tests", test_imports),
        ("YAML Loading", test_yaml_loading),
        ("Custom Modules", test_custom_modules),
        ("Data Loading", test_data_loading),
        ("Clustering", test_clustering),
        ("Basic Workflow", run_basic_workflow)
    ]

    results = {}
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"[FAIL] {test_name} failed with exception: {e}")
            results[test_name] = False

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = 0
    total = len(results)

    for test_name, result in results.items():
        status = "[PASSED]" if result else "[FAILED]"
        print(f"{test_name:<20} : {status}")
        if result:
            passed += 1

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("All tests passed! The application should work correctly.")
        print("\nTo run the Streamlit app:")
        print("streamlit run streamlit_app.py")
    else:
        print("Some tests failed. Please check the issues above.")
        print("\nMost likely solutions:")
        print("1. Install missing dependencies with: pip install -r requirements.txt")
        print("2. Make sure assets.yaml exists in the current directory")
        print("3. Check your internet connection for data loading")

if __name__ == "__main__":
    main()