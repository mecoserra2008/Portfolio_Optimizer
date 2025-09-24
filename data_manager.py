import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yaml
import joblib
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import streamlit as st
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class DataManager:
    def __init__(self, cache_dir: str = "data_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    @staticmethod
    def ensure_timezone_naive(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure DataFrame has timezone-naive datetime index"""
        if df.index.tz is not None:
            df = df.copy()
            df.index = df.index.tz_localize(None)
        return df

    def load_assets_from_yaml(self, yaml_path: str = "assets.yaml") -> Dict:
        """Load assets configuration from YAML file"""
        try:
            with open(yaml_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            st.error(f"Error loading assets.yaml: {e}")
            return {}

    def get_asset_symbols(self, assets_config: Dict) -> Dict[str, List[Dict]]:
        """Extract symbols organized by category from assets config"""
        symbols_by_category = {}

        if 'asset_categories' in assets_config:
            for category, data in assets_config['asset_categories'].items():
                if 'symbols' in data:
                    # Deduplicate symbols within each category
                    unique_symbols = []
                    seen_symbols = set()

                    for symbol_info in data['symbols']:
                        symbol = symbol_info.get('symbol', '')
                        if symbol and symbol not in seen_symbols:
                            seen_symbols.add(symbol)
                            unique_symbols.append(symbol_info)

                    symbols_by_category[data.get('name', category)] = unique_symbols

        return symbols_by_category

    def _get_cache_filename(self, symbol: str) -> Path:
        """Get cache filename for a symbol"""
        return self.cache_dir / f"{symbol.replace('^', '_').replace('/', '_')}.pkl"

    def _load_cached_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load cached data for a symbol"""
        cache_file = self._get_cache_filename(symbol)
        if cache_file.exists():
            try:
                return joblib.load(cache_file)
            except:
                return None
        return None

    def _save_cached_data(self, symbol: str, data: pd.DataFrame):
        """Save data to cache"""
        cache_file = self._get_cache_filename(symbol)
        try:
            joblib.dump(data, cache_file)
        except Exception as e:
            print(f"Warning: Could not cache data for {symbol}: {e}")

    def _fetch_symbol_batch(self, symbol: str, start_date: str, end_date: str,
                           max_retries: int = 3) -> Optional[pd.DataFrame]:
        """Fetch data for a single symbol with retry logic"""
        for attempt in range(max_retries):
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date, interval='1d')

                if data.empty:
                    return None

                # Clean column names and add symbol
                data.columns = [col.lower().replace(' ', '_') for col in data.columns]
                data['symbol'] = symbol

                # Ensure timezone-naive timestamps to avoid comparison issues
                data = self.ensure_timezone_naive(data)

                return data

            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Failed to fetch {symbol} after {max_retries} attempts: {e}")
                    return None
                time.sleep(1)

        return None

    def fetch_data_in_batches(self, symbols: List[str], start_date: str = "2000-01-01",
                             end_date: Optional[str] = None, batch_months: int = 6,
                             max_workers: int = 10) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data in time batches to manage memory efficiently

        Args:
            symbols: List of stock symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format (None for today)
            batch_months: Number of months per batch
            max_workers: Maximum number of concurrent downloads
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        # Convert to datetime
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        all_data = {}

        # Check for cached data first
        cached_symbols = []
        new_symbols = []

        for symbol in symbols:
            cached_data = self._load_cached_data(symbol)
            if cached_data is not None and not cached_data.empty:
                # Ensure timezone-naive for comparison
                cached_data = self.ensure_timezone_naive(cached_data)

                # Check if cached data is recent enough
                last_date = cached_data.index.max()
                end_timestamp = pd.Timestamp(end_dt).tz_localize(None) if pd.Timestamp(end_dt).tz is not None else pd.Timestamp(end_dt)

                if last_date >= end_timestamp - pd.Timedelta(days=7):
                    all_data[symbol] = cached_data
                    cached_symbols.append(symbol)
                else:
                    new_symbols.append(symbol)
            else:
                new_symbols.append(symbol)

        if cached_symbols:
            st.info(f"Loaded {len(cached_symbols)} symbols from cache")

        if not new_symbols:
            return all_data

        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        total_batches = len(new_symbols)

        # Process symbols in parallel batches
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all symbol downloads
            future_to_symbol = {}

            for i, symbol in enumerate(new_symbols):
                future = executor.submit(
                    self._fetch_complete_history,
                    symbol, start_date, end_date, batch_months
                )
                future_to_symbol[future] = symbol

            # Process completed downloads
            for i, future in enumerate(as_completed(future_to_symbol)):
                symbol = future_to_symbol[future]

                try:
                    data = future.result()
                    if data is not None and not data.empty:
                        all_data[symbol] = data
                        self._save_cached_data(symbol, data)

                except Exception as e:
                    print(f"Error processing {symbol}: {e}")

                # Update progress
                progress = (i + 1) / total_batches
                progress_bar.progress(progress)
                status_text.text(f"Downloaded {i + 1}/{total_batches} symbols")

        progress_bar.empty()
        status_text.empty()

        return all_data

    def _fetch_complete_history(self, symbol: str, start_date: str, end_date: str,
                               batch_months: int) -> Optional[pd.DataFrame]:
        """Fetch complete history for a symbol in batches"""
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        all_batches = []
        current_start = start_dt

        while current_start < end_dt:
            # Calculate batch end date
            batch_end = min(
                current_start + timedelta(days=batch_months * 30),
                end_dt
            )

            # Fetch batch
            batch_data = self._fetch_symbol_batch(
                symbol,
                current_start.strftime("%Y-%m-%d"),
                batch_end.strftime("%Y-%m-%d")
            )

            if batch_data is not None and not batch_data.empty:
                all_batches.append(batch_data)

            current_start = batch_end
            time.sleep(0.1)  # Rate limiting

        if all_batches:
            combined_data = pd.concat(all_batches)
            combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
            combined_data.sort_index(inplace=True)
            return combined_data

        return None

    def calculate_features(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate features for clustering analysis

        Features include:
        - Returns (1M, 3M, 6M, 1Y)
        - Volatility (realized, rolling)
        - Risk metrics (VaR, max drawdown)
        - Momentum indicators
        - Correlation features
        """
        features_list = []

        for symbol, data in data_dict.items():
            if data.empty:
                continue

            try:
                # Ensure timezone-naive index
                data = self.ensure_timezone_naive(data)

                # Calculate returns
                returns = data['close'].pct_change().dropna()

                if len(returns) < 252:  # Need at least 1 year of data
                    continue

                features = {'symbol': symbol}

                # Return metrics (annualized)
                periods = [21, 63, 126, 252]  # 1M, 3M, 6M, 1Y
                period_names = ['1M', '3M', '6M', '1Y']

                for period, name in zip(periods, period_names):
                    if len(returns) >= period:
                        period_return = returns.tail(period).mean() * 252
                        period_vol = returns.tail(period).std() * np.sqrt(252)
                        features[f'return_{name}'] = period_return
                        features[f'volatility_{name}'] = period_vol
                        features[f'sharpe_{name}'] = period_return / period_vol if period_vol > 0 else 0

                # Risk metrics
                if len(returns) >= 252:
                    # VaR (5%)
                    var_5 = np.percentile(returns, 5)
                    features['var_5'] = var_5

                    # CVaR (Expected Shortfall)
                    cvar_5 = returns[returns <= var_5].mean()
                    features['cvar_5'] = cvar_5

                    # Maximum drawdown
                    cumulative = (1 + returns).cumprod()
                    running_max = cumulative.expanding().max()
                    drawdown = (cumulative - running_max) / running_max
                    features['max_drawdown'] = drawdown.min()

                    # Sortino ratio (downside deviation)
                    downside_returns = returns[returns < 0]
                    if len(downside_returns) > 0:
                        downside_vol = downside_returns.std() * np.sqrt(252)
                        features['sortino_1Y'] = (returns.mean() * 252) / downside_vol if downside_vol > 0 else 0
                    else:
                        features['sortino_1Y'] = features.get('sharpe_1Y', 0)

                # Momentum features
                if len(returns) >= 252:
                    # Price momentum
                    price_momentum_3M = (data['close'].iloc[-1] / data['close'].iloc[-63] - 1) if len(data) >= 63 else 0
                    price_momentum_6M = (data['close'].iloc[-1] / data['close'].iloc[-126] - 1) if len(data) >= 126 else 0
                    features['momentum_3M'] = price_momentum_3M
                    features['momentum_6M'] = price_momentum_6M

                    # Volume trend (if available)
                    if 'volume' in data.columns:
                        vol_trend = data['volume'].tail(21).mean() / data['volume'].tail(63).mean() if len(data) >= 63 else 1
                        features['volume_trend'] = vol_trend

                # Technical indicators
                if len(data) >= 200:
                    # RSI-like momentum
                    gains = returns[returns > 0]
                    losses = returns[returns < 0].abs()

                    if len(gains) > 0 and len(losses) > 0:
                        avg_gain = gains.tail(14).mean()
                        avg_loss = losses.tail(14).mean()
                        rs = avg_gain / avg_loss if avg_loss > 0 else 0
                        rsi = 100 - (100 / (1 + rs))
                        features['rsi'] = rsi

                features_list.append(features)

            except Exception as e:
                print(f"Error calculating features for {symbol}: {e}")
                continue

        if not features_list:
            return pd.DataFrame()

        features_df = pd.DataFrame(features_list)
        features_df.set_index('symbol', inplace=True)

        # Fill any remaining NaNs
        features_df = features_df.fillna(features_df.median())

        return features_df

    def get_correlation_matrix(self, data_dict: Dict[str, pd.DataFrame],
                              period: int = 252) -> pd.DataFrame:
        """Calculate correlation matrix for the given symbols"""
        returns_dict = {}

        for symbol, data in data_dict.items():
            if not data.empty and len(data) >= period:
                # Ensure timezone-naive index
                data = self.ensure_timezone_naive(data)

                returns = data['close'].pct_change().dropna().tail(period)
                if len(returns) >= period * 0.8:  # At least 80% of required data
                    returns_dict[symbol] = returns

        if not returns_dict:
            return pd.DataFrame()

        # Align dates
        returns_df = pd.DataFrame(returns_dict)
        returns_df = returns_df.dropna()

        if returns_df.empty:
            return pd.DataFrame()

        return returns_df.corr()

    def clean_data(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Clean and validate the data"""
        cleaned_data = {}

        for symbol, data in data_dict.items():
            if data.empty:
                continue

            # Ensure timezone-naive index
            data = self.ensure_timezone_naive(data)

            # Remove data with too many missing values
            if data.isnull().sum().sum() > len(data) * 0.5:
                continue

            # Forward fill small gaps
            data = data.fillna(method='ffill', limit=5)

            # Remove extreme outliers in returns
            returns = data['close'].pct_change()
            q99 = returns.quantile(0.99)
            q1 = returns.quantile(0.01)

            # Cap extreme returns
            if abs(q99) > 0.5 or abs(q1) > 0.5:  # More than 50% daily return
                data = data[~((returns > 0.5) | (returns < -0.5))]

            if len(data) >= 252:  # Minimum 1 year of data
                cleaned_data[symbol] = data

        return cleaned_data