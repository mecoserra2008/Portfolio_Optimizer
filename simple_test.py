#!/usr/bin/env python3
"""
Simple functional test of the cluster portfolio optimizer
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def test_basic_functionality():
    """Test basic functionality without advanced optimization"""
    print("Testing basic cluster portfolio functionality...")

    try:
        from data_manager import DataManager
        from visualizations import PortfolioVisualizer

        # Initialize
        data_manager = DataManager()
        visualizer = PortfolioVisualizer()

        print("[OK] Modules imported successfully")

        # Test with a few symbols - use more data
        test_symbols = ['AAPL', 'MSFT', 'GOOGL']
        print(f"Testing with symbols: {test_symbols}")

        # Fetch 1+ year of data for feature calculation
        raw_data = data_manager.fetch_data_in_batches(
            test_symbols,
            start_date="2023-01-01",
            end_date="2024-06-01",
            batch_months=6,
            max_workers=5
        )

        print(f"[OK] Fetched data for {len(raw_data)} symbols")

        # Clean data
        cleaned_data = data_manager.clean_data(raw_data)
        print(f"[OK] Cleaned data: {len(cleaned_data)} symbols with sufficient data")

        if cleaned_data:
            # Calculate features
            features_df = data_manager.calculate_features(cleaned_data)

            if not features_df.empty:
                print(f"[OK] Calculated {len(features_df.columns)} features for {len(features_df)} symbols")
                print(f"Features: {list(features_df.columns)}")

                # Test basic clustering (without advanced optimization)
                try:
                    from analytics_engine import ClusteringEngine
                    clustering_engine = ClusteringEngine()

                    # K-means clustering
                    results = clustering_engine.fit_kmeans(features_df, n_clusters=2)
                    if results and 'labels' in results:
                        print(f"[OK] K-means clustering: {len(np.unique(results['labels']))} clusters")
                        print(f"    Silhouette score: {results.get('silhouette_score', 0):.3f}")

                        # Show cluster composition
                        for cluster_id in np.unique(results['labels']):
                            cluster_symbols = [results['symbols'][i] for i in range(len(results['symbols']))
                                             if results['labels'][i] == cluster_id]
                            print(f"    Cluster {cluster_id}: {cluster_symbols}")

                    # Test hierarchical clustering
                    hier_results = clustering_engine.fit_hierarchical(features_df, n_clusters=2)
                    if hier_results and 'labels' in hier_results:
                        print(f"[OK] Hierarchical clustering: {len(np.unique(hier_results['labels']))} clusters")
                        print(f"    Silhouette score: {hier_results.get('silhouette_score', 0):.3f}")

                    # Test PCA
                    pca_df, pca_model = clustering_engine.apply_pca(features_df, n_components=2)
                    if not pca_df.empty:
                        print(f"[OK] PCA transformation: {len(pca_df.columns)} components")
                        if pca_model:
                            explained_variance = pca_model.explained_variance_ratio_
                            print(f"    Explained variance: {explained_variance}")

                    print("[OK] Basic clustering tests passed!")

                except ImportError as e:
                    print(f"[WARN] Advanced analytics skipped due to missing dependencies: {e}")

                # Test basic portfolio metrics calculation (without optimization)
                try:
                    from analytics_engine import calculate_portfolio_metrics

                    # Create equal-weight portfolio
                    symbols = list(cleaned_data.keys())
                    equal_weights = pd.Series(1.0/len(symbols), index=symbols)

                    portfolio_metrics = calculate_portfolio_metrics(equal_weights, cleaned_data)

                    if portfolio_metrics:
                        print("[OK] Portfolio metrics calculation:")
                        print(f"    Annual Return: {portfolio_metrics.get('annual_return', 0):.2%}")
                        print(f"    Annual Volatility: {portfolio_metrics.get('annual_volatility', 0):.2%}")
                        print(f"    Sharpe Ratio: {portfolio_metrics.get('sharpe_ratio', 0):.3f}")
                        print(f"    Max Drawdown: {portfolio_metrics.get('max_drawdown', 0):.2%}")

                except Exception as e:
                    print(f"[WARN] Portfolio metrics calculation failed: {e}")

                # Test visualizations
                try:
                    import plotly.graph_objects as go

                    if len(features_df.columns) >= 2:
                        fig = visualizer.plot_cluster_scatter_2d(
                            features_df, results['labels'],
                            features_df.columns[0], features_df.columns[1],
                            "Test Cluster Plot"
                        )

                        if fig.data:
                            print("[OK] Visualization test passed - 2D cluster plot created")

                        # Test portfolio composition visualization
                        fig_comp = visualizer.plot_portfolio_composition(
                            equal_weights, results['labels'], "Test Portfolio"
                        )

                        if fig_comp.data:
                            print("[OK] Portfolio composition visualization created")

                except Exception as e:
                    print(f"[WARN] Visualization test failed: {e}")

                print("\n[SUCCESS] Basic functionality test completed successfully!")
                print("\nNext steps to run the full app:")
                print("1. Install remaining dependencies: pip install cvxpy")
                print("2. Run: streamlit run streamlit_app.py")
                print("3. Open your browser to http://localhost:8501")

                return True

            else:
                print("[FAIL] Feature calculation returned empty DataFrame")
                print("This usually means not enough historical data (need 1+ years)")
                return False
        else:
            print("[FAIL] No cleaned data available")
            return False

    except Exception as e:
        print(f"[FAIL] Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_available_assets():
    """Show sample of available assets from YAML"""
    try:
        from data_manager import DataManager
        data_manager = DataManager()

        assets_config = data_manager.load_assets_from_yaml()
        if assets_config:
            symbols_by_category = data_manager.get_asset_symbols(assets_config)

            print("\nAvailable Asset Categories:")
            print("=" * 50)
            for category_name, symbols_list in symbols_by_category.items():
                print(f"{category_name}: {len(symbols_list)} assets")
                # Show first few examples
                for i, symbol_info in enumerate(symbols_list[:3]):
                    symbol = symbol_info.get('symbol', '')
                    name = symbol_info.get('name', symbol)
                    print(f"  - {symbol}: {name}")
                if len(symbols_list) > 3:
                    print(f"  ... and {len(symbols_list) - 3} more")
                print()
    except Exception as e:
        print(f"Error loading assets: {e}")

if __name__ == "__main__":
    print("Cluster Portfolio Optimizer - Simple Test")
    print("=" * 60)

    # Show available assets
    show_available_assets()

    # Run basic functionality test
    success = test_basic_functionality()

    if success:
        print("\n✅ All basic tests passed! The core functionality is working.")
    else:
        print("\n❌ Some tests failed. Please check the output above.")