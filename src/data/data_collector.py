"""
Stock data collection from multiple sources.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from alpha_vantage.timeseries import TimeSeries
from typing import Optional, Dict, Any, List
import time
from datetime import datetime, timedelta
import requests
from pathlib import Path

from ..utils.logger import get_logger
from ..utils.config import config

logger = get_logger(__name__)

class StockDataCollector:
    """Collect stock data from multiple sources with caching and error handling."""

    def __init__(self):
        """Initialize the data collector."""
        self.data_sources_config = config.get_data_sources()
        self.cache_dir = config.data_storage_path / "raw"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Alpha Vantage client if enabled
        if self.data_sources_config.get('alpha_vantage', {}).get('enabled', False):
            try:
                self.alpha_vantage = TimeSeries(
                    key=config.alpha_vantage_api_key,
                    output_format='pandas'
                )
            except Exception as e:
                logger.warning(f"Failed to initialize Alpha Vantage client: {e}")
                self.alpha_vantage = None
        else:
            self.alpha_vantage = None

    def get_stock_data(
        self,
        symbol: str,
        period: str = "2y",
        interval: str = "1d",
        source: str = "yahoo",
        use_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Get stock data from specified source.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            source: Data source ('yahoo' or 'alpha_vantage')
            use_cache: Whether to use cached data

        Returns:
            DataFrame with stock data or None if failed
        """
        logger.info(f"Fetching data for {symbol} from {source}")

        # Check cache first
        if use_cache:
            cached_data = self._load_from_cache(symbol, period, interval, source)
            if cached_data is not None:
                logger.info(f"Using cached data for {symbol}")
                return cached_data

        # Fetch data based on source
        if source == "yahoo":
            data = self._fetch_yahoo_data(symbol, period, interval)
        elif source == "alpha_vantage":
            data = self._fetch_alpha_vantage_data(symbol)
        elif source == "google":
            # Experimental: use Yahoo backend due to Google Finance API limitations
            data = self._fetch_google_data(symbol, period, interval)
        else:
            logger.error(f"Unknown data source: {source}")
            return None

        # Fallback to Yahoo if primary source failed (helps when Alpha Vantage isn't configured)
        if data is None and source != "yahoo":
            logger.info(f"Primary source '{source}' failed for {symbol}. Falling back to Yahoo Finance...")
            data = self._fetch_yahoo_data(symbol, period, interval)

        # Cache the data if successful
        if data is not None and use_cache:
            self._save_to_cache(data, symbol, period, interval, source)

        return data

    def _fetch_yahoo_data(
        self,
        symbol: str,
        period: str = "2y",
        interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """
        Fetch data from Yahoo Finance.

        Args:
            symbol: Stock symbol
            period: Data period
            interval: Data interval

        Returns:
            DataFrame with stock data or None if failed
        """
        try:
            # Attempt direct fetch
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)

            # Fallback: try uppercased symbol (common for Yahoo Finance)
            if data is None or data.empty:
                sym_upper = symbol.upper()
                if sym_upper != symbol:
                    logger.info(f"Retrying with uppercased symbol: {sym_upper}")
                    t_upper = yf.Ticker(sym_upper)
                    data = t_upper.history(period=period, interval=interval)
                    if data is not None and not data.empty:
                        symbol = sym_upper

            # If empty and looks like Indian symbol without suffix, try NSE/BSE suffixes
            if (data is None or data.empty):
                base = sym_upper if 'sym_upper' in locals() else symbol.upper()
                if not base.endswith(('.NS', '.BO')) and base.isalpha():
                    for suffix in ['.NS', '.BO']:
                        try_symbol = f"{base}{suffix}"
                        logger.info(f"Retrying with market suffix: {try_symbol}")
                        t2 = yf.Ticker(try_symbol)
                        data = t2.history(period=period, interval=interval)
                        if data is not None and not data.empty:
                            symbol = try_symbol
                            break

            if data is None or data.empty:
                logger.warning(f"No data found for symbol {symbol}")
                return None

            # Reset index to make date a column
            data.reset_index(inplace=True)

            # Standardize column names
            data.columns = [col.lower().replace(' ', '_') for col in data.columns]

            # Rename date to timestamp
            data.rename(columns={'date': 'timestamp'}, inplace=True)

            # Convert timezone-aware datetime to timezone-naive
            if 'timestamp' in data.columns and hasattr(data['timestamp'].dtype, 'tz') and data['timestamp'].dtype.tz is not None:
                data['timestamp'] = data['timestamp'].dt.tz_localize(None)

            # Add symbol column
            data['symbol'] = symbol

            logger.info(f"Successfully fetched {len(data)} records for {symbol} from Yahoo Finance")
            return data

        except Exception as e:
            logger.error(f"Error fetching data from Yahoo Finance for {symbol}: {e}")
            return None

    def _fetch_alpha_vantage_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Fetch data from Alpha Vantage.

        Args:
            symbol: Stock symbol

        Returns:
            DataFrame with stock data or None if failed
        """
        if self.alpha_vantage is None:
            logger.warning("Alpha Vantage client not initialized")
            return None

        try:
            data, meta_data = self.alpha_vantage.get_daily(symbol=symbol, outputsize='full')

            if data.empty:
                logger.warning(f"No data found for symbol {symbol} from Alpha Vantage")
                return None

            # Standardize column names
            data.columns = [col.split('. ')[1].lower().replace(' ', '_') for col in data.columns]

            # Add symbol column
            data['symbol'] = symbol

            # Reset index to make date a column
            data.reset_index(inplace=True)
            data.rename(columns={'date': 'timestamp'}, inplace=True)

            # Sort by date (oldest first)
            data = data.sort_values('timestamp').reset_index(drop=True)

            logger.info(f"Successfully fetched {len(data)} records for {symbol} from Alpha Vantage")
            return data

        except Exception as e:
            logger.error(f"Error fetching data from Alpha Vantage for {symbol}: {e}")
            return None

    def _fetch_google_data(
        self,
        symbol: str,
        period: str = "2y",
        interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """
        Fetch data from Google Finance (experimental placeholder).
        Note: There is no official Google Finance API. This method currently proxies to Yahoo Finance
        to ensure consistent data availability.
        """
        logger.info(f"[Google] Fetching data for {symbol} via Yahoo proxy")
        return self._fetch_yahoo_data(symbol, period, interval)

    def _load_from_cache(
        self,
        symbol: str,
        period: str,
        interval: str,
        source: str
    ) -> Optional[pd.DataFrame]:
        """Load data from cache if available and recent."""
        cache_file = self.cache_dir / f"{symbol}_{source}_{period}_{interval}.csv"

        if not cache_file.exists():
            return None

        try:
            # Check if cache is recent (less than 1 day old for daily data)
            cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if cache_age > timedelta(hours=6):  # Refresh cache every 6 hours
                logger.info(f"Cache for {symbol} is outdated, will fetch fresh data")
                return None

            data = pd.read_csv(cache_file, parse_dates=['timestamp'])
            logger.info(f"Loaded {len(data)} records from cache for {symbol}")
            return data

        except Exception as e:
            logger.warning(f"Error loading cache for {symbol}: {e}")
            return None

    @staticmethod
    def detect_currency(symbol: str) -> str:
        """Detect the currency based on symbol/market suffix.
        Returns currency code like 'USD', 'INR', etc.
        """
        s = symbol.upper()
        if s.endswith('.NS') or s.endswith('.BO'):
            return 'INR'
        return 'USD'

    def _save_to_cache(
        self,
        data: pd.DataFrame,
        symbol: str,
        period: str,
        interval: str,
        source: str
    ) -> None:
        """Save data to cache."""
        try:
            cache_file = self.cache_dir / f"{symbol}_{source}_{period}_{interval}.csv"
            data.to_csv(cache_file, index=False)
            logger.info(f"Saved data to cache: {cache_file}")
        except Exception as e:
            logger.warning(f"Error saving cache for {symbol}: {e}")

    def get_multiple_stocks(
        self,
        symbols: List[str],
        period: str = "2y",
        interval: str = "1d",
        source: str = "yahoo"
    ) -> Dict[str, pd.DataFrame]:
        """
        Get data for multiple stocks.

        Args:
            symbols: List of stock symbols
            period: Data period
            interval: Data interval
            source: Data source

        Returns:
            Dictionary mapping symbols to DataFrames
        """
        results = {}

        for symbol in symbols:
            data = self.get_stock_data(symbol, period, interval, source)
            if data is not None:
                results[symbol] = data

            # Add delay to avoid rate limiting
            if source == "alpha_vantage":
                time.sleep(12)  # Alpha Vantage free tier: 5 calls per minute

        return results

    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate stock data quality.

        Args:
            data: Stock data DataFrame

        Returns:
            True if data is valid, False otherwise
        """
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

        # Check required columns
        if not all(col in data.columns for col in required_columns):
            logger.error(f"Missing required columns. Expected: {required_columns}")
            return False

        # Check for empty data
        if data.empty:
            logger.error("Data is empty")
            return False

        # Check for null values in critical columns
        critical_columns = ['open', 'high', 'low', 'close']
        if data[critical_columns].isnull().any().any():
            logger.warning("Found null values in critical price columns")

        # Check for negative prices
        if (data[critical_columns] < 0).any().any():
            logger.error("Found negative prices in data")
            return False

        # Check high >= low
        if (data['high'] < data['low']).any():
            logger.error("Found records where high < low")
            return False

        logger.info("Data validation passed")
        return True
