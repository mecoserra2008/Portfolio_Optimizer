import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram
# import networkx as nx  # Removed unused import
from typing import Dict, List, Tuple, Optional

class PortfolioVisualizer:
    def __init__(self):
        self.color_palette = px.colors.qualitative.Set3
        self.theme = {
            'bgcolor': '#f8f9fa',
            'paper_bgcolor': 'white',
            'font_family': 'Arial, sans-serif',
            'font_size': 12,
            'title_size': 16
        }

    def plot_cluster_scatter_2d(self, features_df: pd.DataFrame, cluster_labels: np.ndarray,
                               x_col: str, y_col: str, title: str = "2D Cluster Visualization") -> go.Figure:
        """Create 2D scatter plot of clusters"""
        if features_df.empty:
            return go.Figure()

        df_plot = features_df.copy()
        df_plot['cluster'] = cluster_labels
        df_plot['symbol'] = df_plot.index

        # Create color map
        unique_clusters = np.unique(cluster_labels)
        color_map = {cluster: self.color_palette[i % len(self.color_palette)]
                    for i, cluster in enumerate(unique_clusters)}

        fig = go.Figure()

        for cluster in unique_clusters:
            cluster_data = df_plot[df_plot['cluster'] == cluster]
            cluster_name = f'Noise' if cluster == -1 else f'Cluster {cluster}'

            fig.add_trace(go.Scatter(
                x=cluster_data[x_col],
                y=cluster_data[y_col],
                mode='markers',
                name=cluster_name,
                text=cluster_data['symbol'],
                hovertemplate=f'<b>%{{text}}</b><br>{x_col}: %{{x:.3f}}<br>{y_col}: %{{y:.3f}}<extra></extra>',
                marker=dict(
                    size=8,
                    color=color_map[cluster],
                    opacity=0.7,
                    line=dict(width=1, color='white')
                )
            ))

        fig.update_layout(
            title=title,
            xaxis_title=x_col.replace('_', ' ').title(),
            yaxis_title=y_col.replace('_', ' ').title(),
            template='plotly_white',
            hovermode='closest',
            showlegend=True
        )

        return fig

    def plot_cluster_scatter_3d(self, features_df: pd.DataFrame, cluster_labels: np.ndarray,
                               x_col: str, y_col: str, z_col: str,
                               title: str = "3D Cluster Visualization") -> go.Figure:
        """Create 3D scatter plot of clusters"""
        if features_df.empty:
            return go.Figure()

        df_plot = features_df.copy()
        df_plot['cluster'] = cluster_labels
        df_plot['symbol'] = df_plot.index

        # Create color map
        unique_clusters = np.unique(cluster_labels)
        color_map = {cluster: self.color_palette[i % len(self.color_palette)]
                    for i, cluster in enumerate(unique_clusters)}

        fig = go.Figure()

        for cluster in unique_clusters:
            cluster_data = df_plot[df_plot['cluster'] == cluster]
            cluster_name = f'Noise' if cluster == -1 else f'Cluster {cluster}'

            fig.add_trace(go.Scatter3d(
                x=cluster_data[x_col],
                y=cluster_data[y_col],
                z=cluster_data[z_col],
                mode='markers',
                name=cluster_name,
                text=cluster_data['symbol'],
                hovertemplate=f'<b>%{{text}}</b><br>{x_col}: %{{x:.3f}}<br>{y_col}: %{{y:.3f}}<br>{z_col}: %{{z:.3f}}<extra></extra>',
                marker=dict(
                    size=6,
                    color=color_map[cluster],
                    opacity=0.7,
                    line=dict(width=1, color='white')
                )
            ))

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=x_col.replace('_', ' ').title(),
                yaxis_title=y_col.replace('_', ' ').title(),
                zaxis_title=z_col.replace('_', ' ').title()
            ),
            template='plotly_white',
            showlegend=True
        )

        return fig

    def plot_dendrogram(self, linkage_matrix: np.ndarray, symbols: List[str],
                       title: str = "Hierarchical Clustering Dendrogram") -> go.Figure:
        """Create dendrogram for hierarchical clustering"""
        if linkage_matrix is None or len(symbols) == 0:
            return go.Figure()

        # Create dendrogram data
        dend_data = dendrogram(linkage_matrix, labels=symbols, no_plot=True)

        fig = go.Figure()

        # Plot dendrogram lines
        for i in range(len(dend_data['icoord'])):
            x = dend_data['icoord'][i]
            y = dend_data['dcoord'][i]

            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode='lines',
                line=dict(color='blue', width=1),
                hoverinfo='skip',
                showlegend=False
            ))

        # Add labels
        fig.update_layout(
            title=title,
            xaxis=dict(
                title='Symbols',
                tickmode='array',
                tickvals=dend_data['leaves'],
                ticktext=[symbols[i] for i in dend_data['leaves']],
                tickangle=45
            ),
            yaxis_title='Distance',
            template='plotly_white',
            height=600
        )

        return fig

    def plot_portfolio_composition(self, weights: pd.Series, cluster_labels: np.ndarray = None,
                                 title: str = "Portfolio Composition") -> go.Figure:
        """Create portfolio composition chart"""
        if weights.empty:
            return go.Figure()

        # Filter out zero weights
        weights_filtered = weights[weights > 0.001]

        if cluster_labels is not None and len(cluster_labels) == len(weights):
            # Group by clusters
            cluster_weights = {}
            symbols_by_cluster = {}

            for i, (symbol, weight) in enumerate(weights_filtered.items()):
                cluster_id = cluster_labels[weights.index.get_loc(symbol)]
                cluster_name = f'Cluster {cluster_id}' if cluster_id != -1 else 'Noise'

                if cluster_name not in cluster_weights:
                    cluster_weights[cluster_name] = 0
                    symbols_by_cluster[cluster_name] = []

                cluster_weights[cluster_name] += weight
                symbols_by_cluster[cluster_name].append(f"{symbol}: {weight:.1%}")

            # Create sunburst chart
            labels = []
            parents = []
            values = []
            colors = []

            # Add root
            labels.append("Portfolio")
            parents.append("")
            values.append(1.0)
            colors.append('#f0f0f0')

            # Add clusters
            for cluster_name, cluster_weight in cluster_weights.items():
                labels.append(cluster_name)
                parents.append("Portfolio")
                values.append(cluster_weight)
                colors.append(self.color_palette[len(labels) % len(self.color_palette)])

                # Add individual stocks
                for symbol_info in symbols_by_cluster[cluster_name]:
                    symbol = symbol_info.split(':')[0]
                    weight = weights_filtered[symbol]
                    labels.append(symbol)
                    parents.append(cluster_name)
                    values.append(weight)
                    colors.append('#lightgray')

            fig = go.Figure(go.Sunburst(
                labels=labels,
                parents=parents,
                values=values,
                branchvalues="total",
                hovertemplate='<b>%{label}</b><br>Weight: %{value:.1%}<extra></extra>',
                marker_colors=colors
            ))

        else:
            # Simple treemap
            fig = go.Figure(go.Treemap(
                labels=weights_filtered.index,
                values=weights_filtered.values,
                parents=[""] * len(weights_filtered),
                texttemplate="<b>%{label}</b><br>%{value:.1%}",
                hovertemplate='<b>%{label}</b><br>Weight: %{value:.1%}<extra></extra>'
            ))

        fig.update_layout(
            title=title,
            template='plotly_white'
        )

        return fig

    def plot_efficient_frontier(self, returns: pd.Series, cov_matrix: pd.DataFrame,
                               optimal_portfolio: Dict = None,
                               cluster_labels: np.ndarray = None,
                               title: str = "Efficient Frontier") -> go.Figure:
        """Plot efficient frontier with cluster-colored points"""
        if returns.empty or cov_matrix.empty:
            return go.Figure()

        # Generate random portfolios for comparison
        n_assets = len(returns)
        n_portfolios = 1000

        np.random.seed(42)
        results = np.zeros((3, n_portfolios))

        for i in range(n_portfolios):
            # Random weights
            weights = np.random.random(n_assets)
            weights = weights / np.sum(weights)

            # Portfolio return and risk
            portfolio_return = np.sum(returns * weights)
            portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

            # Sharpe ratio
            sharpe = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0

            results[0, i] = portfolio_risk
            results[1, i] = portfolio_return
            results[2, i] = sharpe

        fig = go.Figure()

        # Plot random portfolios
        if cluster_labels is not None:
            # Color by cluster assignment (use first asset's cluster for portfolio color)
            unique_clusters = np.unique(cluster_labels)
            for cluster in unique_clusters:
                cluster_mask = np.array([cluster_labels[0] == cluster] * n_portfolios)
                if np.any(cluster_mask):
                    fig.add_trace(go.Scatter(
                        x=results[0, cluster_mask],
                        y=results[1, cluster_mask],
                        mode='markers',
                        name=f'Random Portfolios (Cluster {cluster})',
                        marker=dict(
                            size=4,
                            color=self.color_palette[cluster % len(self.color_palette)],
                            opacity=0.3
                        ),
                        hovertemplate='Risk: %{x:.3f}<br>Return: %{y:.3f}<extra></extra>'
                    ))
        else:
            fig.add_trace(go.Scatter(
                x=results[0],
                y=results[1],
                mode='markers',
                name='Random Portfolios',
                marker=dict(
                    size=4,
                    color=results[2],
                    colorscale='Viridis',
                    colorbar=dict(title="Sharpe Ratio"),
                    opacity=0.6
                ),
                hovertemplate='Risk: %{x:.3f}<br>Return: %{y:.3f}<extra></extra>'
            ))

        # Plot optimal portfolio
        if optimal_portfolio and 'expected_volatility' in optimal_portfolio:
            fig.add_trace(go.Scatter(
                x=[optimal_portfolio['expected_volatility']],
                y=[optimal_portfolio['expected_return']],
                mode='markers',
                name='Optimal Portfolio',
                marker=dict(
                    size=15,
                    color='red',
                    symbol='star'
                ),
                hovertemplate='<b>Optimal Portfolio</b><br>Risk: %{x:.3f}<br>Return: %{y:.3f}<extra></extra>'
            ))

        fig.update_layout(
            title=title,
            xaxis_title='Risk (Volatility)',
            yaxis_title='Expected Return',
            template='plotly_white',
            hovermode='closest'
        )

        return fig

    def plot_correlation_heatmap(self, correlation_matrix: pd.DataFrame,
                                cluster_labels: np.ndarray = None,
                                title: str = "Correlation Matrix") -> go.Figure:
        """Create correlation heatmap with cluster annotations"""
        if correlation_matrix.empty:
            return go.Figure()

        # Reorder by clusters if available
        if cluster_labels is not None and len(cluster_labels) == len(correlation_matrix):
            # Create DataFrame with clusters
            cluster_df = pd.DataFrame({
                'symbol': correlation_matrix.index,
                'cluster': cluster_labels
            })
            cluster_df = cluster_df.sort_values(['cluster', 'symbol'])
            reordered_symbols = cluster_df['symbol'].tolist()

            correlation_matrix = correlation_matrix.loc[reordered_symbols, reordered_symbols]

        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=np.round(correlation_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 8},
            hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>'
        ))

        fig.update_layout(
            title=title,
            template='plotly_white',
            height=max(400, len(correlation_matrix) * 20),
            width=max(400, len(correlation_matrix) * 20)
        )

        return fig

    def plot_performance_metrics(self, portfolio_metrics: Dict,
                                title: str = "Portfolio Performance Metrics") -> go.Figure:
        """Create radar chart of portfolio performance metrics"""
        if not portfolio_metrics:
            return go.Figure()

        # Define metrics to display
        metrics = {
            'Annual Return': portfolio_metrics.get('annual_return', 0),
            'Volatility': -portfolio_metrics.get('annual_volatility', 0),  # Negative for better display
            'Sharpe Ratio': portfolio_metrics.get('sharpe_ratio', 0),
            'Sortino Ratio': portfolio_metrics.get('sortino_ratio', 0),
            'Max Drawdown': portfolio_metrics.get('max_drawdown', 0),
            'Diversification': portfolio_metrics.get('diversification_ratio', 0)
        }

        # Normalize values for radar chart
        values = list(metrics.values())
        categories = list(metrics.keys())

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Portfolio Metrics',
            line=dict(color='blue'),
            fillcolor='rgba(0, 0, 255, 0.1)'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[-1, 1]
                )
            ),
            showlegend=True,
            title=title,
            template='plotly_white'
        )

        return fig

    def plot_time_series_comparison(self, data_dict: Dict, selected_symbols: List[str],
                                   weights: pd.Series = None,
                                   title: str = "Time Series Comparison") -> go.Figure:
        """Plot time series comparison of selected assets and portfolio"""
        if not data_dict or not selected_symbols:
            return go.Figure()

        fig = go.Figure()

        # Plot individual assets
        for symbol in selected_symbols[:10]:  # Limit to 10 for readability
            if symbol in data_dict:
                data = data_dict[symbol]
                if not data.empty:
                    normalized_prices = data['close'] / data['close'].iloc[0]
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=normalized_prices,
                        mode='lines',
                        name=symbol,
                        line=dict(width=1),
                        opacity=0.7,
                        hovertemplate=f'{symbol}<br>Date: %{{x}}<br>Normalized Price: %{{y:.2f}}<extra></extra>'
                    ))

        # Plot portfolio if weights are provided
        if weights is not None and not weights.empty:
            portfolio_values = []
            common_dates = None

            for symbol, weight in weights.items():
                if symbol in data_dict and weight > 0:
                    data = data_dict[symbol]['close']
                    if common_dates is None:
                        common_dates = data.index
                    else:
                        common_dates = common_dates.intersection(data.index)

            if common_dates is not None and len(common_dates) > 0:
                portfolio_series = pd.Series(0, index=common_dates)

                for symbol, weight in weights.items():
                    if symbol in data_dict and weight > 0:
                        asset_prices = data_dict[symbol]['close'].reindex(common_dates)
                        normalized_asset = asset_prices / asset_prices.iloc[0]
                        portfolio_series += normalized_asset * weight

                fig.add_trace(go.Scatter(
                    x=portfolio_series.index,
                    y=portfolio_series,
                    mode='lines',
                    name='Portfolio',
                    line=dict(width=3, color='red'),
                    hovertemplate='Portfolio<br>Date: %{x}<br>Normalized Value: %{y:.2f}<extra></extra>'
                ))

        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Normalized Price',
            template='plotly_white',
            hovermode='x unified'
        )

        return fig

    def plot_risk_return_scatter(self, features_df: pd.DataFrame, cluster_labels: np.ndarray = None,
                                title: str = "Risk vs Return Analysis") -> go.Figure:
        """Create risk-return scatter plot colored by clusters"""
        if features_df.empty:
            return go.Figure()

        # Use 1Y metrics if available
        x_col = 'volatility_1Y' if 'volatility_1Y' in features_df.columns else features_df.columns[1]
        y_col = 'return_1Y' if 'return_1Y' in features_df.columns else features_df.columns[0]

        return self.plot_cluster_scatter_2d(features_df, cluster_labels, x_col, y_col, title)

    def create_dashboard_layout(self, portfolio_metrics: Dict) -> Dict:
        """Create layout data for Streamlit dashboard"""
        if not portfolio_metrics:
            return {}

        # Key metrics for display
        metrics_display = {
            "Annual Return": f"{portfolio_metrics.get('annual_return', 0):.2%}",
            "Annual Volatility": f"{portfolio_metrics.get('annual_volatility', 0):.2%}",
            "Sharpe Ratio": f"{portfolio_metrics.get('sharpe_ratio', 0):.3f}",
            "Sortino Ratio": f"{portfolio_metrics.get('sortino_ratio', 0):.3f}",
            "Max Drawdown": f"{portfolio_metrics.get('max_drawdown', 0):.2%}",
            "VaR (5%)": f"{portfolio_metrics.get('var_5', 0):.2%}",
            "CVaR (5%)": f"{portfolio_metrics.get('cvar_5', 0):.2%}",
            "Effective # Assets": f"{portfolio_metrics.get('effective_num_assets', 0):.1f}",
            "Diversification Ratio": f"{portfolio_metrics.get('diversification_ratio', 0):.3f}",
            "Concentration (HHI)": f"{portfolio_metrics.get('herfindahl_index', 0):.3f}"
        }

        return metrics_display

    def plot_monte_carlo_simulation(self, portfolio_returns: pd.Series,
                                   n_simulations: int = 1000,
                                   n_days: int = 252,
                                   title: str = "Monte Carlo Simulation") -> go.Figure:
        """Plot Monte Carlo simulation of portfolio performance"""
        if portfolio_returns.empty:
            return go.Figure()

        # Calculate statistics
        mean_return = portfolio_returns.mean()
        std_return = portfolio_returns.std()

        # Run simulations
        np.random.seed(42)
        simulations = []

        for _ in range(n_simulations):
            random_returns = np.random.normal(mean_return, std_return, n_days)
            cumulative = (1 + random_returns).cumprod()
            simulations.append(cumulative)

        simulations = np.array(simulations)

        fig = go.Figure()

        # Plot percentiles
        percentiles = [5, 25, 50, 75, 95]
        colors = ['red', 'orange', 'blue', 'orange', 'red']
        names = ['5th percentile', '25th percentile', 'Median', '75th percentile', '95th percentile']

        for i, (p, color, name) in enumerate(zip(percentiles, colors, names)):
            percentile_path = np.percentile(simulations, p, axis=0)
            fig.add_trace(go.Scatter(
                x=list(range(n_days)),
                y=percentile_path,
                mode='lines',
                name=name,
                line=dict(color=color, width=2 if p == 50 else 1)
            ))

        # Add some individual paths for visualization
        for i in range(min(20, n_simulations)):
            fig.add_trace(go.Scatter(
                x=list(range(n_days)),
                y=simulations[i],
                mode='lines',
                name=f'Simulation {i+1}' if i < 3 else '',
                line=dict(color='lightblue', width=0.5),
                opacity=0.3,
                showlegend=i < 3,
                hoverinfo='skip'
            ))

        fig.update_layout(
            title=title,
            xaxis_title='Days',
            yaxis_title='Portfolio Value (Normalized)',
            template='plotly_white',
            hovermode='x unified'
        )

        return fig