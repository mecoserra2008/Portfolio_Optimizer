import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

# Import custom modules
from data_manager import DataManager
from analytics_engine import (ClusteringEngine, PortfolioOptimizer, BayesianOptimizer,
                             calculate_portfolio_metrics, HAS_CVXPY)
from visualizations import PortfolioVisualizer

# Page configuration
st.set_page_config(
    page_title="Advanced Cluster Portfolio Optimizer",
    page_icon="📈",
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
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 0.5rem;
        padding: 1rem;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 16px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'selected_symbols' not in st.session_state:
    st.session_state.selected_symbols = []
if 'portfolio_data' not in st.session_state:
    st.session_state.portfolio_data = {}
if 'cluster_results' not in st.session_state:
    st.session_state.cluster_results = {}
if 'portfolio_weights' not in st.session_state:
    st.session_state.portfolio_weights = pd.Series()

@st.cache_data
def load_data_manager():
    return DataManager()

@st.cache_data
def load_visualizer():
    return PortfolioVisualizer()

# Initialize managers
data_manager = load_data_manager()
visualizer = load_visualizer()

# Main title
st.markdown('<h1 class="main-header">Cluster Portfolio Optimizer</h1>',
           unsafe_allow_html=True)

# Show cvxpy status
if not HAS_CVXPY:
    st.warning("⚠️ **Optional Enhancement Available:** Install `cvxpy` for advanced portfolio optimization features (Sortino optimization, CVaR optimization). Current version uses PyPortfolioOpt fallbacks.")
    with st.expander("Installation Instructions"):
        st.code("pip install cvxpy", language="bash")
        st.info("After installation, restart the application to enable advanced optimization features.")

# Sidebar for main controls
with st.sidebar:
    st.header("⚙️ Configuration")

    # Data loading section
    st.subheader("📊 Data Settings")

    start_date = st.date_input(
        "Start Date",
        value=datetime(2020, 1, 1),
        min_value=datetime(2000, 1, 1),
        max_value=datetime.now()
    )

    end_date = st.date_input(
        "End Date",
        value=datetime.now(),
        min_value=start_date,
        max_value=datetime.now()
    )

    batch_months = st.slider("Batch Size (months)", 3, 12, 6)
    max_workers = st.slider("Parallel Downloads", 5, 20, 10)

# Main content tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Asset Selection", "Clustering Analysis",
    "Portfolio Optimization", "Analytics Dashboard",
    "Monte Carlo & Risk"
])

with tab1:
    st.header("Asset Selection & Data Loading")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Load assets from YAML
        assets_config = data_manager.load_assets_from_yaml()

        if assets_config:
            symbols_by_category = data_manager.get_asset_symbols(assets_config)

            st.subheader("Select Assets by Category")

            selected_symbols = []

            for category_name, symbols_list in symbols_by_category.items():
                with st.expander(f"{category_name} ({len(symbols_list)} assets)", expanded=False):

                    # Select all/none buttons
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.button(f"Select All {category_name}", key=f"select_all_{category_name.replace(' ', '_')}"):
                            # Use set to avoid duplicates
                            unique_symbols = set()
                            for symbol_info in symbols_list:
                                symbol = symbol_info.get('symbol', '')
                                if symbol and symbol not in unique_symbols:
                                    unique_symbols.add(symbol)
                                    if symbol not in st.session_state.selected_symbols:
                                        st.session_state.selected_symbols.append(symbol)

                    with col_b:
                        if st.button(f"Clear {category_name}", key=f"clear_{category_name.replace(' ', '_')}"):
                            for symbol_info in symbols_list:
                                symbol = symbol_info.get('symbol', '')
                                if symbol in st.session_state.selected_symbols:
                                    st.session_state.selected_symbols.remove(symbol)

                    # Individual selections (handle duplicates)
                    displayed_symbols = set()
                    for i, symbol_info in enumerate(symbols_list[:50]):  # Limit display for performance
                        symbol = symbol_info.get('symbol', '')
                        name = symbol_info.get('name', symbol)
                        description = symbol_info.get('description', '')

                        # Skip if symbol already displayed to avoid duplicates
                        if symbol in displayed_symbols or not symbol:
                            continue
                        displayed_symbols.add(symbol)

                        # Create unique key using category and index
                        unique_key = f"check_{category_name}_{i}_{symbol}"

                        if st.checkbox(
                            f"{symbol} - {name}",
                            value=symbol in st.session_state.selected_symbols,
                            key=unique_key,
                            help=description
                        ):
                            if symbol not in st.session_state.selected_symbols:
                                st.session_state.selected_symbols.append(symbol)
                        else:
                            if symbol in st.session_state.selected_symbols:
                                st.session_state.selected_symbols.remove(symbol)

            # Manual symbol input
            st.subheader("Add Custom Symbols")
            custom_symbols = st.text_area(
                "Enter symbols (one per line or comma-separated)",
                placeholder="AAPL\nGOOGL\nMSFT"
            )

            if st.button("Add Custom Symbols"):
                if custom_symbols:
                    # Parse symbols
                    symbols = custom_symbols.replace(',', '\n').split('\n')
                    symbols = [s.strip().upper() for s in symbols if s.strip()]

                    for symbol in symbols:
                        if symbol not in st.session_state.selected_symbols:
                            st.session_state.selected_symbols.append(symbol)

                    st.success(f"Added {len(symbols)} custom symbols")

        else:
            st.error("Could not load assets.yaml file. Please ensure it exists in the current directory.")

    with col2:
        st.subheader("Selected Assets")
        st.info(f"Total Selected: {len(st.session_state.selected_symbols)}")

        if st.session_state.selected_symbols:
            # Display selected symbols
            selected_df = pd.DataFrame({
                'Symbol': st.session_state.selected_symbols
            })
            st.dataframe(selected_df, height=300)

            # Clear all button
            if st.button("Clear All Selections", type="secondary"):
                st.session_state.selected_symbols = []
                st.rerun()

        # Data loading section
        st.subheader("Load Data")

        if st.button("🚀 Load Historical Data", type="primary", disabled=len(st.session_state.selected_symbols) == 0):
            if st.session_state.selected_symbols:
                with st.spinner("Loading historical data... This may take several minutes."):

                    # Load data
                    raw_data = data_manager.fetch_data_in_batches(
                        st.session_state.selected_symbols,
                        start_date=start_date.strftime("%Y-%m-%d"),
                        end_date=end_date.strftime("%Y-%m-%d"),
                        batch_months=batch_months,
                        max_workers=max_workers
                    )

                    # Clean data
                    cleaned_data = data_manager.clean_data(raw_data)

                    if cleaned_data:
                        st.session_state.portfolio_data = cleaned_data
                        st.session_state.data_loaded = True

                        st.success(f"✅ Loaded data for {len(cleaned_data)} symbols")

                        # Show data summary
                        summary_data = []
                        for symbol, data in cleaned_data.items():
                            summary_data.append({
                                'Symbol': symbol,
                                'Data Points': len(data),
                                'Start Date': data.index.min().strftime("%Y-%m-%d"),
                                'End Date': data.index.max().strftime("%Y-%m-%d"),
                                'Latest Price': f"${data['close'].iloc[-1]:.2f}"
                            })

                        summary_df = pd.DataFrame(summary_data)
                        st.dataframe(summary_df)
                    else:
                        st.error("No valid data could be loaded. Please check your symbol selection.")

with tab2:
    st.header("Clustering Analysis")

    if not st.session_state.data_loaded:
        st.warning("Please load data first in the Asset Selection tab.")
    else:
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Clustering Parameters")

            # Calculate features
            if st.button("Calculate Features", type="primary"):
                with st.spinner("Calculating features for clustering..."):
                    features_df = data_manager.calculate_features(st.session_state.portfolio_data)
                    st.session_state.features_df = features_df
                    st.success(f"Calculated {len(features_df.columns)} features for {len(features_df)} assets")

            if 'features_df' in st.session_state and not st.session_state.features_df.empty:
                features_df = st.session_state.features_df

                st.subheader("Feature Selection")
                available_features = features_df.columns.tolist()
                selected_features = st.multiselect(
                    "Select features for clustering",
                    available_features,
                    default=available_features[:6]
                )

                st.subheader("Clustering Method")
                clustering_method = st.selectbox(
                    "Choose clustering algorithm",
                    ["K-Means", "Hierarchical", "DBSCAN", "PCA + K-Means"]
                )

                # Method-specific parameters
                if clustering_method in ["K-Means", "PCA + K-Means"]:
                    n_clusters = st.slider("Number of Clusters", 2, 10, 5)

                    if st.button("Find Optimal Clusters"):
                        with st.spinner("Finding optimal number of clusters..."):
                            clustering_engine = ClusteringEngine()
                            results = clustering_engine.find_optimal_clusters(
                                features_df[selected_features], max_clusters=10
                            )

                            if results:
                                # Plot elbow curve
                                silhouette_scores = [results[k]['silhouette_score'] for k in sorted(results.keys())]
                                inertias = [results[k]['inertia'] for k in sorted(results.keys()) if results[k]['inertia']]

                                import plotly.graph_objects as go
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(
                                    x=list(sorted(results.keys())),
                                    y=silhouette_scores,
                                    mode='lines+markers',
                                    name='Silhouette Score'
                                ))
                                fig.update_layout(
                                    title="Optimal Number of Clusters",
                                    xaxis_title="Number of Clusters",
                                    yaxis_title="Silhouette Score"
                                )
                                st.plotly_chart(fig, use_container_width=True)

                elif clustering_method == "Hierarchical":
                    linkage_method = st.selectbox("Linkage Method", ["ward", "complete", "average", "single"])
                    n_clusters = st.slider("Number of Clusters", 2, 10, 5)

                elif clustering_method == "DBSCAN":
                    eps = st.slider("Epsilon", 0.1, 2.0, 0.5, 0.1)
                    min_samples = st.slider("Min Samples", 3, 20, 5)

                # Dimensionality reduction options
                if clustering_method == "PCA + K-Means":
                    n_components = st.slider("PCA Components", 2, min(10, len(selected_features)), 3)

                # Run clustering
                if st.button("Run Clustering Analysis", type="primary"):
                    with st.spinner("Running clustering analysis..."):
                        clustering_engine = ClusteringEngine()

                        if clustering_method == "K-Means":
                            results = clustering_engine.fit_kmeans(
                                features_df[selected_features], n_clusters
                            )
                        elif clustering_method == "Hierarchical":
                            results = clustering_engine.fit_hierarchical(
                                features_df[selected_features], n_clusters, linkage_method
                            )
                        elif clustering_method == "DBSCAN":
                            results = clustering_engine.fit_dbscan(
                                features_df[selected_features], eps, min_samples
                            )
                        elif clustering_method == "PCA + K-Means":
                            pca_df, pca_model = clustering_engine.apply_pca(
                                features_df[selected_features], n_components
                            )
                            results = clustering_engine.fit_kmeans(pca_df, n_clusters)
                            results['pca_features'] = pca_df

                        st.session_state.cluster_results = results
                        st.success("Clustering analysis completed!")

        with col2:
            if 'cluster_results' in st.session_state and st.session_state.cluster_results:
                results = st.session_state.cluster_results

                st.subheader("Clustering Results")

                # Display metrics
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Silhouette Score", f"{results.get('silhouette_score', 0):.3f}")
                with col_b:
                    if 'n_clusters' in results:
                        st.metric("Number of Clusters", results['n_clusters'])
                    elif 'labels' in results:
                        n_clusters = len(np.unique(results['labels']))
                        st.metric("Number of Clusters", n_clusters)

                # Visualization
                if 'features_df' in st.session_state:
                    features_df = st.session_state.features_df

                    # Select features for visualization
                    viz_features = st.multiselect(
                        "Select features for visualization",
                        features_df.columns.tolist(),
                        default=features_df.columns.tolist()[:3]
                    )

                    if len(viz_features) >= 2:
                        if len(viz_features) == 2:
                            fig = visualizer.plot_cluster_scatter_2d(
                                features_df[viz_features],
                                results['labels'],
                                viz_features[0],
                                viz_features[1],
                                f"{clustering_method} Clustering Results"
                            )
                        else:
                            fig = visualizer.plot_cluster_scatter_3d(
                                features_df[viz_features],
                                results['labels'],
                                viz_features[0],
                                viz_features[1],
                                viz_features[2],
                                f"{clustering_method} Clustering Results (3D)"
                            )

                        st.plotly_chart(fig, use_container_width=True)

                # Cluster composition
                if 'labels' in results and 'symbols' in results:
                    st.subheader("Cluster Composition")

                    cluster_df = pd.DataFrame({
                        'Symbol': results['symbols'],
                        'Cluster': results['labels']
                    })

                    cluster_summary = cluster_df['Cluster'].value_counts().sort_index()
                    st.bar_chart(cluster_summary)

                    # Show detailed composition
                    for cluster_id in sorted(cluster_df['Cluster'].unique()):
                        if cluster_id == -1:
                            st.write("**Noise Points:**")
                        else:
                            st.write(f"**Cluster {cluster_id}:**")

                        cluster_symbols = cluster_df[cluster_df['Cluster'] == cluster_id]['Symbol'].tolist()
                        st.write(", ".join(cluster_symbols))

with tab3:
    st.header("Portfolio Optimization")

    if not st.session_state.data_loaded:
        st.warning("Please load data first in the Asset Selection tab.")
    elif 'cluster_results' not in st.session_state or not st.session_state.cluster_results:
        st.warning("Please run clustering analysis first.")
    else:
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Optimization Parameters")

            optimization_method = st.selectbox(
                "Optimization Method",
                [
                    "Sortino Optimization",
                    "Hierarchical Risk Parity",
                    "CVaR Optimization",
                    "Bayesian Optimization"
                ]
            )

            # Common parameters
            lookback_period = st.slider("Lookback Period (days)", 63, 756, 252)

            # Method-specific parameters
            if optimization_method == "Sortino Optimization":
                target_return = st.slider("Target Annual Return", 0.05, 0.25, 0.10, 0.01)
                use_target = st.checkbox("Use target return constraint")
            elif optimization_method == "CVaR Optimization":
                alpha = st.slider("Confidence Level (α)", 0.01, 0.10, 0.05, 0.01)
                target_return = st.slider("Target Annual Return", 0.05, 0.25, 0.10, 0.01)
                use_target = st.checkbox("Use target return constraint")
            elif optimization_method == "Bayesian Optimization":
                n_calls = st.slider("Optimization Iterations", 20, 100, 50)

            # Risk constraints
            st.subheader("Risk Constraints")
            max_weight = st.slider("Maximum Asset Weight", 0.10, 0.50, 0.30, 0.05)
            concentration_penalty = st.slider("Concentration Penalty", 0.0, 0.5, 0.1, 0.05)

            if st.button("🎯 Optimize Portfolio", type="primary"):
                with st.spinner("Optimizing portfolio... This may take a few minutes."):
                    try:
                        portfolio_optimizer = PortfolioOptimizer()

                        # Calculate returns and covariance
                        mu, S = portfolio_optimizer.calculate_returns_and_cov(
                            st.session_state.portfolio_data, lookback_period
                        )

                        if mu is not None and S is not None:
                            cluster_labels = st.session_state.cluster_results.get('labels', None)

                            if optimization_method == "Sortino Optimization":
                                target = target_return if use_target else None
                                result = portfolio_optimizer.optimize_sortino_portfolio(
                                    mu, S, st.session_state.portfolio_data, target
                                )

                            elif optimization_method == "Hierarchical Risk Parity":
                                if cluster_labels is not None:
                                    result = portfolio_optimizer.hierarchical_risk_parity(
                                        mu, S, cluster_labels
                                    )
                                else:
                                    st.error("Hierarchical Risk Parity requires clustering results.")
                                    result = None

                            elif optimization_method == "CVaR Optimization":
                                # Prepare returns DataFrame
                                returns_dict = {}
                                for symbol in mu.index:
                                    if symbol in st.session_state.portfolio_data:
                                        returns = st.session_state.portfolio_data[symbol]['close'].pct_change().dropna()
                                        returns_dict[symbol] = returns.tail(lookback_period)

                                returns_df = pd.DataFrame(returns_dict).dropna()
                                target = target_return if use_target else None
                                result = portfolio_optimizer.cvar_optimization(mu, returns_df, alpha, target)

                            elif optimization_method == "Bayesian Optimization":
                                bayesian_optimizer = BayesianOptimizer(st.session_state.portfolio_data)
                                result = bayesian_optimizer.optimize_weights(mu.index.tolist(), n_calls)

                                # Convert to standard format
                                if 'weights' in result:
                                    weights = result['weights']
                                    result = {
                                        'weights': weights,
                                        'expected_return': (weights * mu).sum(),
                                        'expected_volatility': np.sqrt(weights.T @ S @ weights),
                                        'status': result.get('status', 'optimal')
                                    }

                            if result and 'weights' in result:
                                st.session_state.portfolio_weights = result['weights']
                                st.session_state.optimization_result = result

                                # Show detailed status message
                                status = result.get('status', 'unknown')
                                if status == 'optimal':
                                    st.success(f"✅ Portfolio optimization completed successfully using advanced {optimization_method}!")
                                elif status == 'pypfopt_fallback':
                                    st.warning(f"⚠️ Portfolio optimization completed using PyPortfolioOpt fallback (CVXPY not available). Consider installing CVXPY for advanced Sortino optimization.")
                                elif status == 'equal_weight_fallback':
                                    st.error(f"❌ Portfolio optimization failed - using equal weights. Check debug output for details.")
                                else:
                                    st.info(f"Portfolio optimization completed with status: {status}")

                                # Show weight distribution info
                                weights = result['weights']
                                max_weight = weights.max()
                                min_weight = weights[weights > 0].min() if (weights > 0).any() else 0
                                n_assets = (weights > 0.001).sum()  # Assets with >0.1% allocation

                                st.info(f"📊 Portfolio composed of {n_assets} assets with weights ranging from {min_weight:.1%} to {max_weight:.1%}")
                            else:
                                st.error("Portfolio optimization failed - no valid result returned.")
                        else:
                            st.error("Could not calculate returns and covariance matrix.")

                    except Exception as e:
                        st.error(f"Optimization error: {str(e)}")

        with col2:
            if 'portfolio_weights' in st.session_state and not st.session_state.portfolio_weights.empty:
                weights = st.session_state.portfolio_weights

                st.subheader("Optimized Portfolio")

                # Display key metrics
                if 'optimization_result' in st.session_state:
                    result = st.session_state.optimization_result

                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Expected Return", f"{result.get('expected_return', 0):.2%}")
                    with col_b:
                        st.metric("Expected Volatility", f"{result.get('expected_volatility', 0):.2%}")
                    with col_c:
                        if 'sortino_ratio' in result:
                            st.metric("Sortino Ratio", f"{result.get('sortino_ratio', 0):.3f}")
                        else:
                            sharpe = result.get('expected_return', 0) / result.get('expected_volatility', 1)
                            st.metric("Sharpe Ratio", f"{sharpe:.3f}")

                # Portfolio composition
                cluster_labels = st.session_state.cluster_results.get('labels', None)
                fig_composition = visualizer.plot_portfolio_composition(
                    weights, cluster_labels, "Optimized Portfolio Composition"
                )
                st.plotly_chart(fig_composition, use_container_width=True)

                # Weights table
                st.subheader("Portfolio Weights")
                weights_df = pd.DataFrame({
                    'Symbol': weights.index,
                    'Weight': weights.values,
                    'Weight (%)': (weights.values * 100).round(2)
                }).sort_values('Weight', ascending=False)

                st.dataframe(weights_df, use_container_width=True)

                # Download weights
                csv = weights_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Portfolio Weights",
                    csv,
                    "portfolio_weights.csv",
                    "text/csv"
                )

with tab4:
    st.header("Analytics Dashboard")

    if not st.session_state.data_loaded:
        st.warning("Please load data first in the Asset Selection tab.")
    elif 'portfolio_weights' not in st.session_state or st.session_state.portfolio_weights.empty:
        st.warning("Please optimize a portfolio first.")
    else:
        weights = st.session_state.portfolio_weights

        # Calculate comprehensive metrics
        with st.spinner("Calculating portfolio analytics..."):
            portfolio_metrics = calculate_portfolio_metrics(
                weights, st.session_state.portfolio_data
            )

        if portfolio_metrics:
            # Key metrics dashboard
            st.subheader("📊 Portfolio Performance Metrics")

            # Create metrics grid
            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                st.metric(
                    "Annual Return",
                    f"{portfolio_metrics.get('annual_return', 0):.2%}",
                    help="Annualized portfolio return"
                )
                st.metric(
                    "Annual Volatility",
                    f"{portfolio_metrics.get('annual_volatility', 0):.2%}",
                    help="Annualized portfolio volatility"
                )

            with col2:
                st.metric(
                    "Sharpe Ratio",
                    f"{portfolio_metrics.get('sharpe_ratio', 0):.3f}",
                    help="Risk-adjusted return (Sharpe ratio)"
                )
                st.metric(
                    "Sortino Ratio",
                    f"{portfolio_metrics.get('sortino_ratio', 0):.3f}",
                    help="Downside risk-adjusted return"
                )

            with col3:
                st.metric(
                    "Max Drawdown",
                    f"{portfolio_metrics.get('max_drawdown', 0):.2%}",
                    help="Maximum peak-to-trough decline"
                )
                st.metric(
                    "VaR (5%)",
                    f"{portfolio_metrics.get('var_5', 0):.2%}",
                    help="Value at Risk (5% confidence)"
                )

            with col4:
                st.metric(
                    "CVaR (5%)",
                    f"{portfolio_metrics.get('cvar_5', 0):.2%}",
                    help="Conditional Value at Risk"
                )
                st.metric(
                    "Effective Assets",
                    f"{portfolio_metrics.get('effective_num_assets', 0):.1f}",
                    help="Effective number of assets (1/HHI)"
                )

            with col5:
                st.metric(
                    "Diversification Ratio",
                    f"{portfolio_metrics.get('diversification_ratio', 0):.3f}",
                    help="Portfolio diversification benefit"
                )
                st.metric(
                    "Concentration (HHI)",
                    f"{portfolio_metrics.get('herfindahl_index', 0):.3f}",
                    help="Herfindahl-Hirschman Index"
                )

            # Visualizations
            col_left, col_right = st.columns(2)

            with col_left:
                # Performance radar chart
                fig_radar = visualizer.plot_performance_metrics(
                    portfolio_metrics, "Portfolio Performance Radar"
                )
                st.plotly_chart(fig_radar, use_container_width=True)

                # Risk-return analysis
                if 'features_df' in st.session_state:
                    cluster_labels = st.session_state.cluster_results.get('labels', None)
                    fig_risk_return = visualizer.plot_risk_return_scatter(
                        st.session_state.features_df, cluster_labels,
                        "Risk vs Return by Clusters"
                    )
                    st.plotly_chart(fig_risk_return, use_container_width=True)

            with col_right:
                # Time series performance
                selected_symbols = weights[weights > 0.01].index.tolist()[:10]
                fig_timeseries = visualizer.plot_time_series_comparison(
                    st.session_state.portfolio_data, selected_symbols, weights,
                    "Portfolio vs Individual Assets Performance"
                )
                st.plotly_chart(fig_timeseries, use_container_width=True)

                # Correlation heatmap
                correlation_matrix = data_manager.get_correlation_matrix(
                    st.session_state.portfolio_data
                )
                if not correlation_matrix.empty:
                    cluster_labels = st.session_state.cluster_results.get('labels', None)
                    fig_corr = visualizer.plot_correlation_heatmap(
                        correlation_matrix, cluster_labels, "Asset Correlation Matrix"
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)

            # Efficient frontier
            st.subheader("📈 Efficient Frontier Analysis")

            if 'optimization_result' in st.session_state:
                portfolio_optimizer = PortfolioOptimizer()
                mu, S = portfolio_optimizer.calculate_returns_and_cov(
                    st.session_state.portfolio_data
                )

                if mu is not None and S is not None:
                    cluster_labels = st.session_state.cluster_results.get('labels', None)
                    fig_frontier = visualizer.plot_efficient_frontier(
                        mu, S, st.session_state.optimization_result, cluster_labels,
                        "Efficient Frontier with Optimal Portfolio"
                    )
                    st.plotly_chart(fig_frontier, use_container_width=True)

with tab5:
    st.header("Monte Carlo Simulation & Risk Analysis")

    if not st.session_state.data_loaded:
        st.warning("Please load data first in the Asset Selection tab.")
    elif 'portfolio_weights' not in st.session_state or st.session_state.portfolio_weights.empty:
        st.warning("Please optimize a portfolio first.")
    else:
        weights = st.session_state.portfolio_weights

        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Simulation Parameters")

            n_simulations = st.slider("Number of Simulations", 100, 5000, 1000, 100)
            simulation_days = st.slider("Simulation Period (days)", 30, 756, 252)

            confidence_levels = st.multiselect(
                "Confidence Levels (%)",
                [90, 95, 99],
                default=[95]
            )

            if st.button("🎲 Run Monte Carlo Simulation", type="primary"):
                with st.spinner("Running Monte Carlo simulation..."):
                    # Calculate portfolio returns
                    portfolio_metrics = calculate_portfolio_metrics(
                        weights, st.session_state.portfolio_data
                    )

                    if portfolio_metrics and 'portfolio_returns' in portfolio_metrics:
                        portfolio_returns = portfolio_metrics['portfolio_returns']

                        # Store simulation results
                        st.session_state.simulation_results = {
                            'portfolio_returns': portfolio_returns,
                            'n_simulations': n_simulations,
                            'simulation_days': simulation_days,
                            'confidence_levels': confidence_levels
                        }

                        st.success("Monte Carlo simulation completed!")

        with col2:
            if 'simulation_results' in st.session_state:
                sim_results = st.session_state.simulation_results
                portfolio_returns = sim_results['portfolio_returns']

                # Monte Carlo visualization
                fig_mc = visualizer.plot_monte_carlo_simulation(
                    portfolio_returns,
                    sim_results['n_simulations'],
                    sim_results['simulation_days'],
                    "Monte Carlo Portfolio Simulation"
                )
                st.plotly_chart(fig_mc, use_container_width=True)

                # Risk metrics
                st.subheader("Risk Analysis Results")

                # Calculate statistics
                mean_return = portfolio_returns.mean()
                std_return = portfolio_returns.std()

                # Simulation statistics
                np.random.seed(42)
                final_values = []

                for _ in range(sim_results['n_simulations']):
                    random_returns = np.random.normal(
                        mean_return, std_return, sim_results['simulation_days']
                    )
                    final_value = (1 + random_returns).prod()
                    final_values.append(final_value)

                final_values = np.array(final_values)

                # Display risk metrics
                col_a, col_b, col_c = st.columns(3)

                with col_a:
                    st.metric("Mean Final Value", f"{np.mean(final_values):.3f}")
                    st.metric("Std Final Value", f"{np.std(final_values):.3f}")

                with col_b:
                    st.metric("Best Case", f"{np.max(final_values):.3f}")
                    st.metric("Worst Case", f"{np.min(final_values):.3f}")

                with col_c:
                    prob_loss = np.sum(final_values < 1) / len(final_values)
                    st.metric("Probability of Loss", f"{prob_loss:.1%}")

                    median_return = (np.median(final_values) - 1) * 100
                    st.metric("Median Return", f"{median_return:.1%}")

                # Confidence intervals
                st.subheader("Confidence Intervals")

                for conf_level in sim_results['confidence_levels']:
                    lower_percentile = (100 - conf_level) / 2
                    upper_percentile = 100 - lower_percentile

                    lower_bound = np.percentile(final_values, lower_percentile)
                    upper_bound = np.percentile(final_values, upper_percentile)

                    st.write(f"**{conf_level}% Confidence Interval:** "
                            f"{lower_bound:.3f} to {upper_bound:.3f}")

# Footer
st.markdown("---")
st.markdown(
    "🎯 **Advanced Cluster Portfolio Optimizer** | "
    "Built with Streamlit, scikit-learn, PyPortfolioOpt, and Plotly | "
    "Data from Yahoo Finance"
)
