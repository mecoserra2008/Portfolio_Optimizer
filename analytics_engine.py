import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.optimize import minimize
try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False

try:
    from pypfopt import EfficientFrontier, risk_models, expected_returns
    from pypfopt import objective_functions, DiscreteAllocation
    from pypfopt.hierarchical_portfolio import HRPOpt
    HAS_PYPFOPT = True
except ImportError:
    HAS_PYPFOPT = False
try:
    from skopt import gp_minimize
    from skopt.space import Real
    HAS_SKOPT = True
except ImportError:
    HAS_SKOPT = False
# import empyrical  # Commented out due to compatibility issues
import warnings
warnings.filterwarnings('ignore')

class ClusteringEngine:
    def __init__(self):
        self.scaler = StandardScaler()
        self.clusters_ = None
        self.cluster_labels_ = None
        self.silhouette_score_ = None

    def fit_kmeans(self, features_df: pd.DataFrame, n_clusters: int = 5,
                   random_state: int = 42) -> dict:
        """Perform K-means clustering"""
        if features_df.empty:
            return {}

        # Scale features
        features_scaled = self.scaler.fit_transform(features_df)

        # Fit K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(features_scaled)

        # Calculate silhouette score
        silhouette_avg = silhouette_score(features_scaled, cluster_labels)

        self.cluster_labels_ = cluster_labels
        self.silhouette_score_ = silhouette_avg

        return {
            'labels': cluster_labels,
            'centers': kmeans.cluster_centers_,
            'silhouette_score': silhouette_avg,
            'inertia': kmeans.inertia_,
            'symbols': features_df.index.tolist()
        }

    def fit_hierarchical(self, features_df: pd.DataFrame, n_clusters: int = 5,
                        linkage_method: str = 'ward') -> dict:
        """Perform hierarchical clustering"""
        if features_df.empty:
            return {}

        # Scale features
        features_scaled = self.scaler.fit_transform(features_df)

        # Fit hierarchical clustering
        hierarchical = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage_method
        )
        cluster_labels = hierarchical.fit_predict(features_scaled)

        # Calculate silhouette score
        silhouette_avg = silhouette_score(features_scaled, cluster_labels)

        # Create linkage matrix for dendrogram
        linkage_matrix = linkage(features_scaled, method=linkage_method)

        self.cluster_labels_ = cluster_labels
        self.silhouette_score_ = silhouette_avg

        return {
            'labels': cluster_labels,
            'silhouette_score': silhouette_avg,
            'linkage_matrix': linkage_matrix,
            'symbols': features_df.index.tolist()
        }

    def fit_dbscan(self, features_df: pd.DataFrame, eps: float = 0.5,
                   min_samples: int = 5) -> dict:
        """Perform DBSCAN clustering"""
        if features_df.empty:
            return {}

        # Scale features
        features_scaled = self.scaler.fit_transform(features_df)

        # Fit DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(features_scaled)

        # Calculate silhouette score (only if we have more than 1 cluster)
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        silhouette_avg = 0

        if n_clusters > 1:
            # Filter out noise points for silhouette calculation
            mask = cluster_labels != -1
            if mask.sum() > 1:
                silhouette_avg = silhouette_score(
                    features_scaled[mask], cluster_labels[mask]
                )

        self.cluster_labels_ = cluster_labels
        self.silhouette_score_ = silhouette_avg

        return {
            'labels': cluster_labels,
            'n_clusters': n_clusters,
            'n_noise': list(cluster_labels).count(-1),
            'silhouette_score': silhouette_avg,
            'symbols': features_df.index.tolist()
        }

    def apply_pca(self, features_df: pd.DataFrame, n_components: int = 3) -> tuple:
        """Apply PCA for dimensionality reduction"""
        if features_df.empty:
            return pd.DataFrame(), None

        features_scaled = self.scaler.fit_transform(features_df)

        pca = PCA(n_components=n_components)
        features_pca = pca.fit_transform(features_scaled)

        # Create DataFrame with PCA results
        pca_columns = [f'PC{i+1}' for i in range(n_components)]
        pca_df = pd.DataFrame(
            features_pca,
            index=features_df.index,
            columns=pca_columns
        )

        return pca_df, pca

    def apply_tsne(self, features_df: pd.DataFrame, n_components: int = 2,
                   perplexity: float = 30.0, random_state: int = 42) -> pd.DataFrame:
        """Apply t-SNE for visualization"""
        if features_df.empty:
            return pd.DataFrame()

        features_scaled = self.scaler.fit_transform(features_df)

        tsne = TSNE(
            n_components=n_components,
            perplexity=min(perplexity, len(features_df) - 1),
            random_state=random_state
        )
        features_tsne = tsne.fit_transform(features_scaled)

        # Create DataFrame with t-SNE results
        tsne_columns = [f'TSNE{i+1}' for i in range(n_components)]
        tsne_df = pd.DataFrame(
            features_tsne,
            index=features_df.index,
            columns=tsne_columns
        )

        return tsne_df

    def find_optimal_clusters(self, features_df: pd.DataFrame, max_clusters: int = 10,
                             method: str = 'kmeans') -> dict:
        """Find optimal number of clusters using elbow method and silhouette analysis"""
        if features_df.empty:
            return {}

        features_scaled = self.scaler.fit_transform(features_df)
        results = {}

        for k in range(2, min(max_clusters + 1, len(features_df))):
            if method == 'kmeans':
                model = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = model.fit_predict(features_scaled)
                inertia = model.inertia_
            elif method == 'hierarchical':
                model = AgglomerativeClustering(n_clusters=k)
                labels = model.fit_predict(features_scaled)
                inertia = None

            silhouette_avg = silhouette_score(features_scaled, labels)

            results[k] = {
                'silhouette_score': silhouette_avg,
                'inertia': inertia,
                'labels': labels
            }

        return results

class PortfolioOptimizer:
    def __init__(self):
        self.weights_ = None
        self.expected_return_ = None
        self.expected_volatility_ = None

    def calculate_returns_and_cov(self, data_dict: dict, period: int = 252) -> tuple:
        """Calculate expected returns and covariance matrix"""
        returns_dict = {}

        for symbol, data in data_dict.items():
            if not data.empty and len(data) >= period:
                # Ensure timezone-naive index
                if data.index.tz is not None:
                    data_copy = data.copy()
                    data_copy.index = data_copy.index.tz_localize(None)
                else:
                    data_copy = data

                returns = data_copy['close'].pct_change().dropna().tail(period)
                if len(returns) >= period * 0.8:
                    returns_dict[symbol] = returns

        if not returns_dict:
            return None, None

        # Create returns DataFrame
        returns_df = pd.DataFrame(returns_dict)
        returns_df = returns_df.dropna()

        if len(returns_df) < 50:  # Need minimum data
            return None, None

        # Calculate expected returns and covariance
        if HAS_PYPFOPT:
            mu = expected_returns.mean_historical_return(returns_df, frequency=252)
            S = risk_models.sample_cov(returns_df, frequency=252)
        else:
            # Manual calculation as fallback
            mu = returns_df.mean() * 252  # Annualized returns
            S = returns_df.cov() * 252    # Annualized covariance

        return mu, S

    def optimize_sortino_portfolio(self, mu: pd.Series, S: pd.DataFrame,
                                  data_dict: dict, target_return: float = None) -> dict:
        """Optimize portfolio using Sortino ratio instead of Sharpe"""
        symbols = list(mu.index)
        n = len(symbols)

        print(f"[DEBUG] Starting Sortino optimization for {n} symbols")
        print(f"[DEBUG] HAS_CVXPY: {HAS_CVXPY}, HAS_PYPFOPT: {HAS_PYPFOPT}")
        print(f"[DEBUG] Target return: {target_return}")

        if not HAS_CVXPY:
            # Fallback to PyPortfolioOpt for basic optimization if available
            if HAS_PYPFOPT:
                try:
                    print("[DEBUG] Using PyPortfolioOpt fallback")
                    ef = EfficientFrontier(mu, S)
                    ef.add_constraint(lambda w: w >= 0)  # Long only
                    ef.add_constraint(lambda w: w <= 0.3)  # Max weight constraint

                    if target_return is not None:
                        print(f"[DEBUG] Optimizing for target return: {target_return}")
                        weights = ef.efficient_return(target_return)
                    else:
                        print("[DEBUG] Optimizing for max Sharpe")
                        weights = ef.max_sharpe()

                    cleaned_weights = ef.clean_weights()
                    weights_series = pd.Series(cleaned_weights)

                    print(f"[DEBUG] PyPortfolioOpt optimization successful")
                    print(f"[DEBUG] Weights range: {weights_series.min():.4f} to {weights_series.max():.4f}")

                    portfolio_return = (weights_series * mu).sum()
                    portfolio_vol = np.sqrt(weights_series.T @ S @ weights_series)

                    return {
                        'weights': weights_series,
                        'expected_return': portfolio_return,
                        'expected_volatility': portfolio_vol,
                        'sortino_ratio': 0,  # Not calculated without cvxpy
                        'status': 'pypfopt_fallback'
                    }
                except Exception as e:
                    print(f"[DEBUG] PyPortfolioOpt failed: {e}")
                    pass
            else:
                print("[DEBUG] PyPortfolioOpt not available")

        else:
            # Full cvxpy implementation
            try:
                print("[DEBUG] Using CVXPY for Sortino optimization")
                returns_dict = {}
                for symbol in symbols:
                    if symbol in data_dict:
                        returns = data_dict[symbol]['close'].pct_change().dropna()
                        returns_dict[symbol] = returns

                returns_df = pd.DataFrame(returns_dict).dropna()
                print(f"[DEBUG] Returns data shape: {returns_df.shape}")

                if returns_df.empty:
                    print("[DEBUG] Empty returns data, falling back")
                    raise ValueError("Empty returns data")

                # Calculate downside deviation (more robust approach)
                downside_returns = returns_df[returns_df < 0].fillna(0)

                # Ensure we have enough downside observations
                downside_count = (downside_returns != 0).sum()
                print(f"[DEBUG] Downside observations per asset: {downside_count.to_dict()}")

                # If too few downside observations, use modified approach
                if downside_count.min() < 10:
                    print("[DEBUG] Too few downside observations, using negative deviation from mean")
                    # Use deviations below mean instead of below zero
                    mean_returns = returns_df.mean()
                    downside_returns = (returns_df - mean_returns).where(returns_df < mean_returns, 0)

                downside_cov = downside_returns.cov() * 252

                # Regularize downside covariance matrix to ensure positive definiteness
                eigenvals = np.linalg.eigvals(downside_cov.values)
                min_eigenval = np.min(eigenvals)
                print(f"[DEBUG] Minimum eigenvalue of downside covariance: {min_eigenval:.6f}")

                if min_eigenval <= 1e-6:
                    # Add regularization to make matrix positive definite
                    regularization = max(1e-6 - min_eigenval, 1e-4)
                    downside_cov += np.eye(downside_cov.shape[0]) * regularization
                    print(f"[DEBUG] Added regularization: {regularization:.6f}")

                    # Verify positive definiteness after regularization
                    eigenvals_reg = np.linalg.eigvals(downside_cov.values)
                    min_eigenval_reg = np.min(eigenvals_reg)
                    print(f"[DEBUG] Minimum eigenvalue after regularization: {min_eigenval_reg:.6f}")

                    if min_eigenval_reg <= 0:
                        print("[DEBUG] Regularization failed, falling back to PyPortfolioOpt")
                        raise ValueError("Regularization failed")

                # Optimization variables
                w = cp.Variable(n)
                portfolio_return = mu.values @ w
                portfolio_downside_risk = cp.quad_form(w, downside_cov.values)

                # Constraints (relaxed for better feasibility)
                constraints = [
                    cp.sum(w) == 1,  # Weights sum to 1
                    w >= 0.01,       # Minimum weight to avoid singularity
                    w <= 0.8         # Relaxed maximum weight
                ]

                # Only add return constraint if it's reasonable
                if target_return is not None and target_return > mu.max():
                    print(f"[DEBUG] Target return {target_return:.4f} exceeds maximum possible {mu.max():.4f}, ignoring constraint")
                elif target_return is not None:
                    constraints.append(portfolio_return >= target_return)

                # Debug: Check expected returns
                print(f"[DEBUG] Expected returns: min={mu.min():.4f}, max={mu.max():.4f}, mean={mu.mean():.4f}")

                # Since Sortino ratio optimization is not DCP-compliant, we'll minimize downside risk for a given return
                # or maximize return subject to downside risk constraint
                if target_return is not None:
                    # Minimize downside risk subject to target return
                    objective = cp.Minimize(portfolio_downside_risk)
                else:
                    # For negative expected returns, just minimize risk
                    if mu.max() < 0:
                        print("[DEBUG] All negative expected returns, minimizing risk only")
                        objective = cp.Minimize(portfolio_downside_risk)
                    else:
                        # Maximize return with penalty for downside risk (convex approximation)
                        lambda_penalty = 0.1  # Reduced penalty to make problem more feasible
                        objective = cp.Minimize(-portfolio_return + lambda_penalty * portfolio_downside_risk)

                # Solve with multiple solver attempts
                problem = cp.Problem(objective, constraints)
                print("[DEBUG] Solving CVXPY problem...")

                # Try different solvers in order of preference
                solvers_to_try = [
                    (cp.ECOS, {'max_iters': 1000}),
                    (cp.OSQP, {'max_iter': 5000, 'eps_abs': 1e-6, 'eps_rel': 1e-6}),
                    (cp.SCS, {'max_iters': 2500})
                ]

                solved = False
                for solver, solver_args in solvers_to_try:
                    try:
                        print(f"[DEBUG] Trying solver: {solver}")
                        problem.solve(solver=solver, **solver_args)
                        print(f"[DEBUG] Problem status: {problem.status}")

                        if problem.status == cp.OPTIMAL:
                            solved = True
                            break
                        elif problem.status in [cp.OPTIMAL_INACCURATE]:
                            print("[DEBUG] Inaccurate solution, but acceptable")
                            solved = True
                            break
                    except Exception as e:
                        print(f"[DEBUG] Solver {solver} failed: {e}")
                        continue

                if not solved:
                    print("[DEBUG] All CVXPY solvers failed, falling back to PyPortfolioOpt")
                    raise ValueError("All CVXPY solvers failed")

                if w.value is not None:
                    weights = pd.Series(w.value, index=symbols)
                    weights = weights.clip(lower=0)  # Ensure non-negative
                    weights = weights / weights.sum()  # Normalize

                    print(f"[DEBUG] CVXPY optimization successful")
                    print(f"[DEBUG] Weights range: {weights.min():.4f} to {weights.max():.4f}")

                    # Calculate portfolio metrics
                    portfolio_return = (weights * mu).sum()
                    portfolio_vol = np.sqrt(weights.T @ S @ weights)

                    # Calculate Sortino ratio
                    portfolio_returns = (returns_df * weights).sum(axis=1)
                    downside_vol = portfolio_returns[portfolio_returns < 0].std() * np.sqrt(252)
                    sortino_ratio = portfolio_return / downside_vol if downside_vol > 0 else 0

                    return {
                        'weights': weights,
                        'expected_return': portfolio_return,
                        'expected_volatility': portfolio_vol,
                        'sortino_ratio': sortino_ratio,
                        'status': 'optimal'
                    }
                else:
                    print("[DEBUG] CVXPY solver returned None for weights")

            except Exception as e:
                print(f"[DEBUG] CVXPY optimization failed: {e}")

        # Final fallback to equal weight
        print("[DEBUG] All optimization methods failed, using equal weights")
        equal_weights = pd.Series(1/n, index=symbols)
        return {
            'weights': equal_weights,
            'expected_return': (equal_weights * mu).sum(),
            'expected_volatility': np.sqrt(equal_weights.T @ S @ equal_weights),
            'sortino_ratio': 0,
            'status': 'equal_weight_fallback'
        }

    def hierarchical_risk_parity(self, mu: pd.Series, S: pd.DataFrame,
                                cluster_labels: np.ndarray) -> dict:
        """Implement Hierarchical Risk Parity using cluster structure"""
        symbols = list(mu.index)

        # Create cluster-based hierarchy
        unique_clusters = np.unique(cluster_labels)
        cluster_weights = {}

        # Equal risk allocation across clusters
        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id
            cluster_symbols = [symbols[i] for i in range(len(symbols)) if cluster_mask[i]]

            if len(cluster_symbols) == 0:
                continue

            # Get cluster covariance matrix
            cluster_cov = S.loc[cluster_symbols, cluster_symbols]

            # Calculate inverse volatility weights within cluster
            diag_vol = np.sqrt(np.diag(cluster_cov))
            inv_vol_weights = 1 / diag_vol
            inv_vol_weights = inv_vol_weights / inv_vol_weights.sum()

            cluster_weights[cluster_id] = pd.Series(
                inv_vol_weights, index=cluster_symbols
            )

        # Combine cluster weights (equal allocation across clusters)
        if not cluster_weights:
            # Fallback to equal weights
            equal_weights = pd.Series(1/len(symbols), index=symbols)
            return {
                'weights': equal_weights,
                'expected_return': (equal_weights * mu).sum(),
                'expected_volatility': np.sqrt(equal_weights.T @ S @ equal_weights),
                'status': 'equal_weight_fallback'
            }

        final_weights = pd.Series(0.0, index=symbols)
        cluster_allocation = 1.0 / len(unique_clusters)

        for cluster_id, weights in cluster_weights.items():
            final_weights.loc[weights.index] = weights * cluster_allocation

        # Normalize weights
        final_weights = final_weights / final_weights.sum()

        # Calculate portfolio metrics
        portfolio_return = (final_weights * mu).sum()
        portfolio_vol = np.sqrt(final_weights.T @ S @ final_weights)

        return {
            'weights': final_weights,
            'expected_return': portfolio_return,
            'expected_volatility': portfolio_vol,
            'status': 'optimal'
        }

    def cvar_optimization(self, mu: pd.Series, returns_df: pd.DataFrame,
                         alpha: float = 0.05, target_return: float = None) -> dict:
        """Optimize portfolio using Conditional Value at Risk (CVaR)"""
        n = len(mu)
        T = len(returns_df)

        if not HAS_CVXPY:
            # Fallback to basic optimization if PyPortfolioOpt available
            if HAS_PYPFOPT:
                try:
                    ef = EfficientFrontier(mu, returns_df.cov() * 252)
                    ef.add_constraint(lambda w: w >= 0)  # Long only
                    ef.add_constraint(lambda w: w <= 0.25)  # Max 25% in any asset

                    if target_return is not None:
                        weights = ef.efficient_return(target_return)
                    else:
                        weights = ef.min_volatility()

                    cleaned_weights = ef.clean_weights()
                    weights_series = pd.Series(cleaned_weights)

                    portfolio_return = (weights_series * mu).sum()
                    portfolio_vol = np.sqrt(weights_series.T @ (returns_df.cov() * 252) @ weights_series)

                    return {
                        'weights': weights_series,
                        'expected_return': portfolio_return,
                        'expected_volatility': portfolio_vol,
                        'status': 'pypfopt_fallback'
                    }
                except:
                    pass

        else:
            # Full CVaR optimization with cvxpy
            # Decision variables
            w = cp.Variable(n)  # Portfolio weights
            z = cp.Variable(T)  # Auxiliary variables for CVaR
            eta = cp.Variable()  # VaR level

            # Portfolio returns for each time period
            portfolio_returns = returns_df.values @ w

            # Constraints
            constraints = [
                cp.sum(w) == 1,  # Weights sum to 1
                w >= 0,          # Long only
                w <= 0.25,       # Max 25% in any asset
                z >= 0,          # Auxiliary variables non-negative
                z >= -portfolio_returns - eta  # CVaR constraint
            ]

            if target_return is not None:
                constraints.append(mu.values @ w >= target_return)

            # Objective: Minimize CVaR
            cvar = eta + (1/alpha) * cp.sum(z) / T
            objective = cp.Minimize(cvar)

            # Solve
            problem = cp.Problem(objective, constraints)
            try:
                problem.solve(solver=cp.ECOS, max_iters=1000)

                if w.value is not None:
                    weights = pd.Series(w.value, index=mu.index)
                    weights = weights.clip(lower=0)
                    weights = weights / weights.sum()

                    # Calculate metrics
                    portfolio_return = (weights * mu).sum()
                    portfolio_returns_series = (returns_df * weights).sum(axis=1)
                    portfolio_vol = portfolio_returns_series.std() * np.sqrt(252)

                    return {
                        'weights': weights,
                        'expected_return': portfolio_return,
                        'expected_volatility': portfolio_vol,
                        'cvar': cvar.value,
                        'status': 'optimal'
                    }
            except:
                pass

        # Final fallback
        equal_weights = pd.Series(1/n, index=mu.index)
        return {
            'weights': equal_weights,
            'expected_return': (equal_weights * mu).sum(),
            'expected_volatility': 0.15,  # Rough estimate
            'status': 'equal_weight_fallback'
        }

class BayesianOptimizer:
    def __init__(self, data_dict: dict):
        self.data_dict = data_dict
        self.bounds = None

    def objective_function(self, weights: np.ndarray, symbols: list,
                          lookback_period: int = 63) -> float:
        """
        Objective function for Bayesian optimization
        Returns negative Sortino ratio (to minimize)
        """
        if len(weights) != len(symbols):
            return 1000  # Penalty for incorrect dimensions

        weights = np.array(weights)
        weights = np.clip(weights, 0, 1)
        weights = weights / weights.sum()  # Normalize

        try:
            # Calculate portfolio returns
            returns_list = []
            for i, symbol in enumerate(symbols):
                if symbol in self.data_dict and weights[i] > 0:
                    returns = self.data_dict[symbol]['close'].pct_change().dropna()
                    returns_list.append(returns.tail(lookback_period) * weights[i])

            if not returns_list:
                return 1000

            portfolio_returns = pd.concat(returns_list, axis=1).sum(axis=1)

            if len(portfolio_returns) < 20:
                return 1000

            # Calculate Sortino ratio
            mean_return = portfolio_returns.mean() * 252
            downside_returns = portfolio_returns[portfolio_returns < 0]

            if len(downside_returns) == 0:
                return -mean_return  # Return negative of return if no downside

            downside_std = downside_returns.std() * np.sqrt(252)
            sortino_ratio = mean_return / downside_std if downside_std > 0 else 0

            # Add penalties for concentration
            concentration_penalty = np.sum(weights**2)  # Herfindahl index

            # Add penalty for low returns
            if mean_return < 0.05:  # Less than 5% annual return
                return 1000

            return -sortino_ratio + concentration_penalty * 0.1

        except:
            return 1000

    def optimize_weights(self, symbols: list, n_calls: int = 50) -> dict:
        """Optimize portfolio weights using Bayesian optimization"""
        if not symbols:
            return {}

        n_assets = len(symbols)

        if not HAS_SKOPT:
            # Fallback to equal weights if scikit-optimize not available
            equal_weights = pd.Series(1/n_assets, index=symbols)
            return {
                'weights': equal_weights,
                'status': 'equal_weight_fallback'
            }

        # Define bounds for each weight (0 to 0.3 for each asset)
        bounds = [Real(0.0, 0.3, name=f'w_{i}') for i in range(n_assets)]

        # Initial guess (equal weights)
        initial_weights = [1.0 / n_assets] * n_assets

        try:
            result = gp_minimize(
                func=lambda w: self.objective_function(w, symbols),
                dimensions=bounds,
                n_calls=n_calls,
                x0=initial_weights,
                random_state=42,
                acq_func='EI'  # Expected Improvement
            )

            optimal_weights = np.array(result.x)
            optimal_weights = np.clip(optimal_weights, 0, 1)
            optimal_weights = optimal_weights / optimal_weights.sum()

            weights_series = pd.Series(optimal_weights, index=symbols)

            return {
                'weights': weights_series,
                'fun_value': result.fun,
                'n_calls': len(result.func_vals),
                'status': 'optimal'
            }

        except Exception as e:
            print(f"Bayesian optimization failed: {e}")
            # Fallback to equal weights
            equal_weights = pd.Series(1/n_assets, index=symbols)
            return {
                'weights': equal_weights,
                'status': 'equal_weight_fallback'
            }

def calculate_portfolio_metrics(weights: pd.Series, data_dict: dict,
                               period: int = 252) -> dict:
    """Calculate comprehensive portfolio metrics"""
    if weights.empty:
        return {}

    # Calculate portfolio returns
    returns_list = []
    for symbol, weight in weights.items():
        if symbol in data_dict and weight > 0:
            data = data_dict[symbol]
            # Ensure timezone-naive index
            if data.index.tz is not None:
                data = data.copy()
                data.index = data.index.tz_localize(None)

            returns = data['close'].pct_change().dropna()
            returns_list.append(returns.tail(period) * weight)

    if not returns_list:
        return {}

    portfolio_returns = pd.concat(returns_list, axis=1).sum(axis=1)

    if len(portfolio_returns) < 50:
        return {}

    # Basic metrics (manual calculation since empyrical has compatibility issues)
    annual_return = portfolio_returns.mean() * 252
    annual_volatility = portfolio_returns.std() * np.sqrt(252)

    # Sharpe ratio (assuming 0% risk-free rate)
    sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0

    # Sortino ratio
    downside_returns = portfolio_returns[portfolio_returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else annual_volatility
    sortino_ratio = annual_return / downside_std if downside_std > 0 else 0

    # Max drawdown
    cumulative = (1 + portfolio_returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    # Risk metrics
    var_5 = np.percentile(portfolio_returns, 5)
    cvar_5 = portfolio_returns[portfolio_returns <= var_5].mean()

    # Concentration metrics
    herfindahl_index = np.sum(weights**2)
    effective_num_assets = 1 / herfindahl_index

    # Diversification ratio
    individual_vols = []
    for symbol, weight in weights.items():
        if symbol in data_dict and weight > 0:
            returns = data_dict[symbol]['close'].pct_change().dropna().tail(period)
            individual_vols.append(returns.std() * np.sqrt(252) * weight)

    if individual_vols:
        weighted_avg_vol = sum(individual_vols)
        diversification_ratio = weighted_avg_vol / annual_volatility if annual_volatility > 0 else 1
    else:
        diversification_ratio = 1

    return {
        'annual_return': annual_return,
        'annual_volatility': annual_volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'var_5': var_5,
        'cvar_5': cvar_5,
        'herfindahl_index': herfindahl_index,
        'effective_num_assets': effective_num_assets,
        'diversification_ratio': diversification_ratio,
        'portfolio_returns': portfolio_returns
    }