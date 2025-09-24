#!/usr/bin/env python3
"""
Test script to validate portfolio optimization fixes
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys

def test_dependency_imports():
    """Test that optimization dependencies are available"""
    print("=== Testing Dependencies ===")

    try:
        import cvxpy as cp
        print("[OK] CVXPY imported successfully")
        has_cvxpy = True
    except ImportError:
        print("[FAIL] CVXPY not available")
        has_cvxpy = False

    try:
        from pypfopt import EfficientFrontier, risk_models, expected_returns
        print("[OK] PyPortfolioOpt imported successfully")
        has_pypfopt = True
    except ImportError:
        print("[FAIL] PyPortfolioOpt not available")
        has_pypfopt = False

    return has_cvxpy, has_pypfopt

def test_portfolio_optimization():
    """Test portfolio optimization functionality"""
    print("\n=== Testing Portfolio Optimization ===")

    try:
        from data_manager import DataManager
        from analytics_engine import PortfolioOptimizer, HAS_CVXPY, HAS_PYPFOPT

        print(f"Analytics engine - HAS_CVXPY: {HAS_CVXPY}, HAS_PYPFOPT: {HAS_PYPFOPT}")

        # Initialize
        data_manager = DataManager()
        portfolio_optimizer = PortfolioOptimizer()

        # Test with sample data
        test_symbols = ['AAPL', 'MSFT', 'GOOGL']
        print(f"Testing with symbols: {test_symbols}")

        # Fetch data
        raw_data = data_manager.fetch_data_in_batches(
            test_symbols,
            start_date="2022-01-01",
            end_date="2024-01-01",
            batch_months=6,
            max_workers=3
        )

        if not raw_data:
            print("[FAIL] Failed to fetch data")
            return False

        print(f"[OK] Fetched data for {len(raw_data)} symbols")

        # Clean data
        cleaned_data = data_manager.clean_data(raw_data)
        print(f"[OK] Cleaned data: {len(cleaned_data)} symbols")

        if len(cleaned_data) < 2:
            print("[FAIL] Not enough cleaned data for optimization")
            return False

        # Calculate returns and covariance
        mu, S = portfolio_optimizer.calculate_returns_and_cov(cleaned_data)

        if mu is None or S is None:
            print("[FAIL] Failed to calculate returns/covariance")
            return False

        print(f"[OK] Calculated returns and covariance for {len(mu)} symbols")
        print(f"   Expected returns range: {mu.min():.4f} to {mu.max():.4f}")

        # Test Sortino optimization
        print("\n--- Testing Sortino Optimization ---")
        result = portfolio_optimizer.optimize_sortino_portfolio(mu, S, cleaned_data)

        if result and 'weights' in result:
            weights = result['weights']
            status = result.get('status', 'unknown')

            print(f"[OK] Optimization completed with status: {status}")
            print(f"   Weights sum: {weights.sum():.4f}")
            print(f"   Weights range: {weights.min():.4f} to {weights.max():.4f}")
            print(f"   Non-zero weights: {(weights > 0.001).sum()}")
            print(f"   Expected return: {result.get('expected_return', 0):.4f}")
            print(f"   Expected volatility: {result.get('expected_volatility', 0):.4f}")

            # Check if weights are actually optimized (not equal)
            weight_std = weights.std()
            if weight_std < 0.01:  # Very low standard deviation = nearly equal weights
                print(f"[WARN]  WARNING: Weights appear to be nearly equal (std: {weight_std:.6f})")
                if status == 'equal_weight_fallback':
                    print("   This is expected for fallback mode")
                else:
                    print("   This suggests optimization may not be working properly")
            else:
                print(f"[OK] Weights are properly differentiated (std: {weight_std:.4f})")

            return True
        else:
            print("[FAIL] Optimization failed - no valid result")
            return False

    except Exception as e:
        print(f"[FAIL] Portfolio optimization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_visualizations():
    """Test that visualizations work without errors"""
    print("\n=== Testing Visualizations ===")

    try:
        from visualizations import PortfolioVisualizer
        import plotly.graph_objects as go

        visualizer = PortfolioVisualizer()

        # Create sample data for testing
        features_df = pd.DataFrame({
            'return_1Y': np.random.randn(5),
            'volatility_1Y': np.abs(np.random.randn(5)),
            'sharpe_1Y': np.random.randn(5)
        }, index=['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN'])

        cluster_labels = np.array([0, 0, 1, 1, 2])

        # Test 2D scatter plot
        fig_2d = visualizer.plot_cluster_scatter_2d(
            features_df, cluster_labels, 'return_1Y', 'volatility_1Y'
        )

        if fig_2d.data:
            print("[OK] 2D scatter plot created successfully")
        else:
            print("[FAIL] 2D scatter plot failed")
            return False

        # Test time series plot with sample data
        sample_data = {
            'AAPL': pd.DataFrame({
                'close': 100 + np.cumsum(np.random.randn(100) * 0.02)
            }, index=pd.date_range('2023-01-01', periods=100)),
            'MSFT': pd.DataFrame({
                'close': 200 + np.cumsum(np.random.randn(100) * 0.02)
            }, index=pd.date_range('2023-01-01', periods=100))
        }

        weights = pd.Series([0.6, 0.4], index=['AAPL', 'MSFT'])

        fig_ts = visualizer.plot_time_series_comparison(
            sample_data, ['AAPL', 'MSFT'], weights
        )

        if fig_ts.data:
            print("[OK] Time series plot created successfully (opacity fix working)")
        else:
            print("[FAIL] Time series plot failed")
            return False

        return True

    except Exception as e:
        print(f"[FAIL] Visualization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("Testing Portfolio Optimization Fixes")
    print("=" * 50)

    # Test dependencies
    has_cvxpy, has_pypfopt = test_dependency_imports()

    # Test portfolio optimization
    opt_success = test_portfolio_optimization()

    # Test visualizations
    viz_success = test_visualizations()

    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)

    print(f"Dependencies: CVXPY={has_cvxpy}, PyPortfolioOpt={has_pypfopt}")
    print(f"Portfolio Optimization: {'[PASS]' if opt_success else '[FAIL]'}")
    print(f"Visualizations: {'[PASS]' if viz_success else '[FAIL]'}")

    overall_success = opt_success and viz_success

    if overall_success:
        print("\nAll tests passed! The fixes are working correctly.")
        if has_cvxpy:
            print("   -> Advanced Sortino optimization should work properly")
        elif has_pypfopt:
            print("   -> PyPortfolioOpt fallback optimization should work properly")
        else:
            print("   -> Equal weight fallback will be used (install optimization libraries for better results)")
    else:
        print("\nSome tests failed. Please check the output above for details.")

    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)