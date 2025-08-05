"""
Unit tests for data collection module.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.data_collector import StockDataCollector

class TestStockDataCollector:
    """Test cases for StockDataCollector."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.collector = StockDataCollector()
        
        # Sample data for testing
        self.sample_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=100, freq='D'),
            'open': np.random.uniform(100, 200, 100),
            'high': np.random.uniform(150, 250, 100),
            'low': np.random.uniform(50, 150, 100),
            'close': np.random.uniform(100, 200, 100),
            'volume': np.random.randint(1000000, 10000000, 100),
            'symbol': ['AAPL'] * 100
        })
        
        # Ensure high >= low and other constraints
        self.sample_data['high'] = np.maximum(
            self.sample_data['high'],
            np.maximum(self.sample_data['open'], self.sample_data['close'])
        )
        self.sample_data['low'] = np.minimum(
            self.sample_data['low'],
            np.minimum(self.sample_data['open'], self.sample_data['close'])
        )
    
    def test_init(self):
        """Test collector initialization."""
        assert self.collector is not None
        assert hasattr(self.collector, 'data_sources_config')
        assert hasattr(self.collector, 'cache_dir')
    
    @patch('yfinance.Ticker')
    def test_fetch_yahoo_data_success(self, mock_ticker):
        """Test successful Yahoo Finance data fetching."""
        # Mock yfinance response
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = self.sample_data.set_index('timestamp')[
            ['open', 'high', 'low', 'close', 'volume']
        ]
        mock_ticker.return_value = mock_ticker_instance
        
        # Test data fetching
        result = self.collector._fetch_yahoo_data('AAPL', '1y', '1d')
        
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert 'symbol' in result.columns
        assert len(result) > 0
        mock_ticker.assert_called_once_with('AAPL')
    
    @patch('yfinance.Ticker')
    def test_fetch_yahoo_data_empty(self, mock_ticker):
        """Test Yahoo Finance data fetching with empty response."""
        # Mock empty response
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = pd.DataFrame()
        mock_ticker.return_value = mock_ticker_instance
        
        # Test data fetching
        result = self.collector._fetch_yahoo_data('INVALID', '1y', '1d')
        
        assert result is None
    
    @patch('yfinance.Ticker')
    def test_fetch_yahoo_data_exception(self, mock_ticker):
        """Test Yahoo Finance data fetching with exception."""
        # Mock exception
        mock_ticker.side_effect = Exception("Network error")
        
        # Test data fetching
        result = self.collector._fetch_yahoo_data('AAPL', '1y', '1d')
        
        assert result is None
    
    def test_validate_data_valid(self):
        """Test data validation with valid data."""
        result = self.collector.validate_data(self.sample_data)
        assert result is True
    
    def test_validate_data_missing_columns(self):
        """Test data validation with missing columns."""
        invalid_data = self.sample_data.drop(columns=['close'])
        result = self.collector.validate_data(invalid_data)
        assert result is False
    
    def test_validate_data_empty(self):
        """Test data validation with empty data."""
        empty_data = pd.DataFrame()
        result = self.collector.validate_data(empty_data)
        assert result is False
    
    def test_validate_data_negative_prices(self):
        """Test data validation with negative prices."""
        invalid_data = self.sample_data.copy()
        invalid_data.loc[0, 'close'] = -10
        result = self.collector.validate_data(invalid_data)
        assert result is False
    
    def test_validate_data_high_less_than_low(self):
        """Test data validation with high < low."""
        invalid_data = self.sample_data.copy()
        invalid_data.loc[0, 'high'] = 50
        invalid_data.loc[0, 'low'] = 100
        result = self.collector.validate_data(invalid_data)
        assert result is False
    
    @patch.object(StockDataCollector, '_fetch_yahoo_data')
    def test_get_stock_data_success(self, mock_fetch):
        """Test successful stock data retrieval."""
        mock_fetch.return_value = self.sample_data
        
        result = self.collector.get_stock_data('AAPL', '1y', '1d', 'yahoo', use_cache=False)
        
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        mock_fetch.assert_called_once()
    
    @patch.object(StockDataCollector, '_fetch_yahoo_data')
    def test_get_stock_data_failure(self, mock_fetch):
        """Test stock data retrieval failure."""
        mock_fetch.return_value = None
        
        result = self.collector.get_stock_data('INVALID', '1y', '1d', 'yahoo', use_cache=False)
        
        assert result is None
    
    def test_get_multiple_stocks(self):
        """Test multiple stock data retrieval."""
        with patch.object(self.collector, 'get_stock_data') as mock_get:
            mock_get.return_value = self.sample_data
            
            symbols = ['AAPL', 'GOOGL']
            results = self.collector.get_multiple_stocks(symbols, source='yahoo')
            
            assert len(results) == 2
            assert 'AAPL' in results
            assert 'GOOGL' in results
            assert mock_get.call_count == 2

if __name__ == "__main__":
    pytest.main([__file__])
