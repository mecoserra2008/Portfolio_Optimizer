# Advanced Cluster Portfolio Optimizer

A sophisticated financial portfolio optimization tool that leverages machine learning clustering algorithms to identify asset relationships and construct optimized portfolios. Built with Python, Streamlit, and advanced quantitative finance libraries.

## Purpose

This application combines cluster analysis with modern portfolio theory to create data-driven investment portfolios. It performs unsupervised learning on financial assets to identify natural groupings based on risk-return characteristics, then applies various optimization techniques to construct efficient portfolios within and across these clusters.

## Architecture

The application follows a modular architecture with clear separation of concerns:

### Core Components

- **`streamlit_app.py`**: Main Streamlit web interface with multi-tab layout
- **`analytics_engine.py`**: Core clustering and optimization algorithms
- **`data_manager.py`**: Financial data fetching, caching, and preprocessing
- **`visualizations.py`**: Interactive plotting and chart generation
- **`assets.yaml`**: Comprehensive asset universe configuration

### Key Features

1. **Multi-Algorithm Clustering**: K-Means, DBSCAN, Hierarchical clustering
2. **Portfolio Optimization**: Mean-variance, hierarchical risk parity, Bayesian optimization
3. **Advanced Analytics**: Risk metrics, performance attribution, correlation analysis
4. **Interactive Visualizations**: 2D/3D cluster plots, efficient frontiers, correlation heatmaps
5. **Real-time Data**: Yahoo Finance integration with intelligent caching

## Installation

### Prerequisites

- Python 3.8+ (tested up to 3.13)
- pip package manager
- Internet connection for data fetching

### Setup

```bash
# Clone or download the repository
cd Cluster_analysis

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Optional Dependencies

The application gracefully handles missing optional packages:

- **CVXPy**: Required for advanced convex optimization algorithms
- **PyPortfolioOpt**: Enables additional optimization methods
- **scikit-optimize**: Required for Bayesian hyperparameter optimization

## Usage

### Starting the Application

```bash
streamlit run streamlit_app.py
```

The interface will be available at `http://localhost:8501`

### Workflow

1. **Data Loading**:
   - Select assets from predefined categories or custom symbols
   - Configure date ranges and data parameters
   - Leverage built-in caching for performance

2. **Feature Engineering**:
   - Automatic calculation of financial metrics (returns, volatility, Sharpe ratio, etc.)
   - Feature scaling and preprocessing
   - Missing data handling

3. **Cluster Analysis**:
   - Choose clustering algorithm and parameters
   - Visualize clusters in 2D/3D space
   - Analyze cluster characteristics and silhouette scores

4. **Portfolio Construction**:
   - Select optimization method (mean-variance, HRP, etc.)
   - Configure constraints and objectives
   - Generate efficient frontier plots

5. **Analysis & Export**:
   - Review risk-return metrics
   - Analyze correlation structures
   - Export results and visualizations

### Configuration

#### Asset Universe

Edit `assets.yaml` to customize available assets:

```yaml
asset_categories:
  custom_category:
    name: "Custom Assets"
    symbols:
      - symbol: "AAPL"
        name: "Apple Inc."
        category: "Tech"
        description: "Technology stock"
```

#### Caching

Data is automatically cached in `data_cache/` directory. Cache files include:
- Raw price data (`.pkl` format)
- Processed features
- Asset metadata

## Implementation Details

### Clustering Algorithms

- **K-Means**: Partitional clustering with configurable cluster count
- **DBSCAN**: Density-based clustering for irregular cluster shapes
- **Hierarchical**: Agglomerative clustering with linkage options

### Optimization Methods

- **Mean-Variance Optimization**: Classical Markowitz approach
- **Hierarchical Risk Parity (HRP)**: Machine learning-based risk budgeting
- **Bayesian Optimization**: Hyperparameter tuning for cluster parameters

### Risk Models

- Sample covariance matrix with optional shrinkage
- Exponentially weighted moving averages
- Factor models (when sufficient data available)

### Performance Metrics

- Sharpe ratio, Sortino ratio, maximum drawdown
- Value at Risk (VaR) calculations
- Risk-adjusted returns across multiple time horizons

## Limitations

### Data Quality
- Relies on Yahoo Finance API availability and accuracy
- Historical data limitations for newer assets
- Market data delays and potential gaps

### Algorithmic Constraints
- Clustering assumes stationarity in asset relationships
- Optimization based on historical patterns may not predict future performance
- Limited handling of extreme market regimes

### Technical Limitations
- Memory usage scales with asset universe size and data history
- Some optimization methods require proprietary solvers (CVXPy)
- Real-time data processing not implemented

### Financial Disclaimers
- No consideration of transaction costs or market impact
- Assumes perfect liquidity and divisibility
- Does not account for regulatory constraints or tax implications
- Results should not be considered investment advice

## Performance Considerations

### Optimization
- Parallel data fetching for multiple assets
- Intelligent caching reduces API calls
- Efficient DataFrame operations using vectorized calculations

### Scalability
- Tested with up to 500 assets simultaneously
- Memory usage approximately 100MB per 100 assets with 5 years of data
- Clustering algorithms may become slow with >1000 assets

### Testing
- Unit tests in `test_*.py` files
- Integration tests for optimization pipeline
- Performance benchmarks for clustering algorithms


## Dependencies

### Core Libraries
- **streamlit**: Web application framework
- **pandas/numpy**: Data manipulation and numerical computing
- **scikit-learn**: Machine learning algorithms
- **plotly**: Interactive visualizations
- **yfinance**: Financial data API

### Optimization Stack
- **scipy**: Scientific computing and optimization
- **cvxpy**: Convex optimization modeling
- **pypfopt**: Portfolio optimization utilities
- **scikit-optimize**: Bayesian optimization

### File I/O
- **pyyaml**: Configuration file parsing
- **joblib**: Efficient serialization for caching

## License

This project is intended for educational and research purposes. Financial calculations should be independently verified before any investment decisions.
