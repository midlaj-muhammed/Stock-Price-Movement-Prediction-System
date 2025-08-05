"""
Technical indicators calculation for stock data.
"""

import pandas as pd
import numpy as np
import ta
from typing import Dict, List, Optional
from ..utils.logger import get_logger

logger = get_logger(__name__)

class TechnicalIndicators:
    """Calculate various technical indicators for stock data."""
    
    def __init__(self):
        """Initialize technical indicators calculator."""
        pass
    
    def calculate_moving_averages(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate various moving averages.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with moving average indicators
        """
        df = data.copy()
        
        # Simple Moving Averages
        df['sma_5'] = ta.trend.sma_indicator(df['close'], window=5)
        df['sma_10'] = ta.trend.sma_indicator(df['close'], window=10)
        df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
        
        # Exponential Moving Averages
        df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
        df['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)
        df['ema_50'] = ta.trend.ema_indicator(df['close'], window=50)
        
        # Volume Moving Averages
        df['volume_sma_10'] = ta.trend.sma_indicator(df['volume'], window=10)
        df['volume_sma_20'] = ta.trend.sma_indicator(df['volume'], window=20)
        
        # Moving Average Ratios
        df['price_sma_ratio_5'] = df['close'] / df['sma_5']
        df['price_sma_ratio_20'] = df['close'] / df['sma_20']
        df['sma_ratio_5_20'] = df['sma_5'] / df['sma_20']
        
        return df
    
    def calculate_momentum_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate momentum-based indicators.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with momentum indicators
        """
        df = data.copy()
        
        # RSI (Relative Strength Index)
        df['rsi_14'] = ta.momentum.rsi(df['close'], window=14)
        df['rsi_7'] = ta.momentum.rsi(df['close'], window=7)
        
        # Stochastic Oscillator
        df['stoch_k'] = ta.momentum.stoch(df['high'], df['low'], df['close'])
        df['stoch_d'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'])
        
        # Williams %R
        df['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df['close'])
        
        # Rate of Change
        df['roc_10'] = ta.momentum.roc(df['close'], window=10)
        df['roc_20'] = ta.momentum.roc(df['close'], window=20)
        
        # Commodity Channel Index
        df['cci_20'] = ta.trend.cci(df['high'], df['low'], df['close'], window=20)
        
        return df
    
    def calculate_volatility_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volatility-based indicators.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with volatility indicators
        """
        df = data.copy()
        
        # Bollinger Bands
        bb_indicator = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb_indicator.bollinger_hband()
        df['bb_middle'] = bb_indicator.bollinger_mavg()
        df['bb_lower'] = bb_indicator.bollinger_lband()
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Average True Range
        df['atr_14'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
        df['atr_20'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=20)
        
        # Keltner Channels
        kc_indicator = ta.volatility.KeltnerChannel(df['high'], df['low'], df['close'])
        df['kc_upper'] = kc_indicator.keltner_channel_hband()
        df['kc_middle'] = kc_indicator.keltner_channel_mband()
        df['kc_lower'] = kc_indicator.keltner_channel_lband()
        
        # Donchian Channels
        dc_indicator = ta.volatility.DonchianChannel(df['high'], df['low'], df['close'])
        df['dc_upper'] = dc_indicator.donchian_channel_hband()
        df['dc_middle'] = dc_indicator.donchian_channel_mband()
        df['dc_lower'] = dc_indicator.donchian_channel_lband()
        
        return df
    
    def calculate_trend_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trend-based indicators.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with trend indicators
        """
        df = data.copy()
        
        # MACD
        macd_indicator = ta.trend.MACD(df['close'])
        df['macd'] = macd_indicator.macd()
        df['macd_signal'] = macd_indicator.macd_signal()
        df['macd_histogram'] = macd_indicator.macd_diff()
        
        # ADX (Average Directional Index)
        df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
        df['adx_pos'] = ta.trend.adx_pos(df['high'], df['low'], df['close'], window=14)
        df['adx_neg'] = ta.trend.adx_neg(df['high'], df['low'], df['close'], window=14)
        
        # Parabolic SAR
        df['psar'] = ta.trend.psar_up(df['high'], df['low'], df['close'])
        
        # Aroon
        aroon_indicator = ta.trend.AroonIndicator(df['high'], df['low'], window=25)
        df['aroon_up'] = aroon_indicator.aroon_up()
        df['aroon_down'] = aroon_indicator.aroon_down()
        df['aroon_oscillator'] = df['aroon_up'] - df['aroon_down']
        
        # Ichimoku
        ichimoku_indicator = ta.trend.IchimokuIndicator(df['high'], df['low'])
        df['ichimoku_a'] = ichimoku_indicator.ichimoku_a()
        df['ichimoku_b'] = ichimoku_indicator.ichimoku_b()
        df['ichimoku_base'] = ichimoku_indicator.ichimoku_base_line()
        df['ichimoku_conversion'] = ichimoku_indicator.ichimoku_conversion_line()
        
        return df
    
    def calculate_volume_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volume-based indicators.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with volume indicators
        """
        df = data.copy()
        
        # On-Balance Volume
        df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
        
        # Volume Price Trend
        df['vpt'] = ta.volume.volume_price_trend(df['close'], df['volume'])
        
        # Accumulation/Distribution Line
        df['ad_line'] = ta.volume.acc_dist_index(df['high'], df['low'], df['close'], df['volume'])
        
        # Chaikin Money Flow
        df['cmf'] = ta.volume.chaikin_money_flow(df['high'], df['low'], df['close'], df['volume'])
        
        # Money Flow Index
        df['mfi'] = ta.volume.money_flow_index(df['high'], df['low'], df['close'], df['volume'])
        
        # Volume Weighted Average Price
        df['vwap'] = ta.volume.volume_weighted_average_price(df['high'], df['low'], df['close'], df['volume'])
        
        # Volume ratios
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        return df
    
    def calculate_custom_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate custom indicators.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with custom indicators
        """
        df = data.copy()
        
        # Price position within the day's range
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        df['price_position'] = df['price_position'].fillna(0.5)  # Fill NaN with neutral position
        
        # Gap indicators
        df['gap'] = df['open'] - df['close'].shift(1)
        df['gap_percent'] = df['gap'] / df['close'].shift(1)
        
        # Candlestick patterns (simplified)
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
        df['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
        df['total_range'] = df['high'] - df['low']
        
        # Doji pattern (small body relative to range)
        df['is_doji'] = (df['body_size'] / df['total_range'] < 0.1).astype(int)
        
        # Hammer pattern (small body, long lower shadow)
        df['is_hammer'] = ((df['lower_shadow'] > 2 * df['body_size']) & 
                          (df['upper_shadow'] < df['body_size'])).astype(int)
        
        # Shooting star pattern (small body, long upper shadow)
        df['is_shooting_star'] = ((df['upper_shadow'] > 2 * df['body_size']) & 
                                 (df['lower_shadow'] < df['body_size'])).astype(int)
        
        # Price momentum
        df['momentum_1'] = df['close'] / df['close'].shift(1) - 1
        df['momentum_3'] = df['close'] / df['close'].shift(3) - 1
        df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        
        # Volatility measures
        df['true_range'] = np.maximum(df['high'] - df['low'],
                                     np.maximum(abs(df['high'] - df['close'].shift(1)),
                                               abs(df['low'] - df['close'].shift(1))))
        
        # Support and resistance levels (simplified)
        df['resistance_20'] = df['high'].rolling(window=20).max()
        df['support_20'] = df['low'].rolling(window=20).min()
        df['resistance_distance'] = (df['resistance_20'] - df['close']) / df['close']
        df['support_distance'] = (df['close'] - df['support_20']) / df['close']
        
        return df
    
    def calculate_all_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all technical indicators
        """
        logger.info("Calculating all technical indicators")
        
        df = data.copy()
        
        # Calculate all indicator groups
        df = self.calculate_moving_averages(df)
        df = self.calculate_momentum_indicators(df)
        df = self.calculate_volatility_indicators(df)
        df = self.calculate_trend_indicators(df)
        df = self.calculate_volume_indicators(df)
        df = self.calculate_custom_indicators(df)
        
        # Remove rows with NaN values (from indicator calculations)
        initial_rows = len(df)
        df = df.dropna().reset_index(drop=True)
        final_rows = len(df)
        
        logger.info(f"Calculated technical indicators. Rows: {initial_rows} -> {final_rows}")
        logger.info(f"Total features: {len(df.columns)}")
        
        return df
    
    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """
        Get feature groups for importance analysis.
        
        Returns:
            Dictionary mapping group names to feature lists
        """
        return {
            'price_features': ['open', 'high', 'low', 'close', 'price_position', 'body_size'],
            'volume_features': ['volume', 'obv', 'vpt', 'ad_line', 'cmf', 'mfi', 'vwap', 'volume_ratio'],
            'moving_averages': ['sma_5', 'sma_10', 'sma_20', 'sma_50', 'ema_12', 'ema_26', 'ema_50'],
            'momentum': ['rsi_14', 'rsi_7', 'stoch_k', 'stoch_d', 'williams_r', 'roc_10', 'roc_20'],
            'volatility': ['bb_upper', 'bb_lower', 'bb_width', 'atr_14', 'atr_20', 'true_range'],
            'trend': ['macd', 'macd_signal', 'adx', 'aroon_oscillator', 'psar'],
            'patterns': ['is_doji', 'is_hammer', 'is_shooting_star'],
            'momentum_custom': ['momentum_1', 'momentum_3', 'momentum_5'],
            'support_resistance': ['resistance_distance', 'support_distance']
        }
