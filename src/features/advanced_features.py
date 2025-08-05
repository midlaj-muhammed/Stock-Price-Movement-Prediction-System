"""
Advanced feature engineering for better model accuracy.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
import warnings
warnings.filterwarnings('ignore')

# Try to import talib, fallback to manual implementations
try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False
    print("TA-Lib not available, using manual implementations")

class AdvancedFeatureEngineer:
    """Advanced feature engineering with sophisticated techniques."""

    def __init__(self):
        self.scalers = {}
        self.feature_selectors = {}
        self.pca_transformers = {}

    def _sma(self, data: np.ndarray, period: int) -> np.ndarray:
        """Simple Moving Average - manual implementation."""
        return pd.Series(data).rolling(window=period).mean().values

    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Exponential Moving Average - manual implementation."""
        return pd.Series(data).ewm(span=period).mean().values

    def _rsi(self, data: np.ndarray, period: int = 14) -> np.ndarray:
        """RSI - manual implementation."""
        delta = pd.Series(data).diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return (100 - (100 / (1 + rs))).values

    def _macd(self, data: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9):
        """MACD - manual implementation."""
        ema_fast = self._ema(data, fast)
        ema_slow = self._ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = self._ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def _bollinger_bands(self, data: np.ndarray, period: int = 20, std_dev: float = 2):
        """Bollinger Bands - manual implementation."""
        sma = self._sma(data, period)
        std = pd.Series(data).rolling(window=period).std().values
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower
        
    def create_advanced_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create advanced technical indicators for better predictions."""
        
        df = data.copy()
        
        # Ensure we have the required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Convert to numpy arrays
        open_prices = df['open'].values
        high_prices = df['high'].values
        low_prices = df['low'].values
        close_prices = df['close'].values
        volume = df['volume'].values
        
        # === TREND INDICATORS ===
        
        # Multiple timeframe moving averages
        for period in [5, 10, 20, 50, 100, 200]:
            if HAS_TALIB:
                df[f'sma_{period}'] = talib.SMA(close_prices, timeperiod=period)
                df[f'ema_{period}'] = talib.EMA(close_prices, timeperiod=period)
            else:
                df[f'sma_{period}'] = self._sma(close_prices, period)
                df[f'ema_{period}'] = self._ema(close_prices, period)

            # Price relative to moving averages
            df[f'close_sma_{period}_ratio'] = close_prices / df[f'sma_{period}']
            df[f'close_ema_{period}_ratio'] = close_prices / df[f'ema_{period}']
        
        # MACD with multiple settings
        for fast, slow, signal in [(12, 26, 9), (5, 35, 5), (19, 39, 9)]:
            if HAS_TALIB:
                macd, macd_signal, macd_hist = talib.MACD(
                    close_prices, fastperiod=fast, slowperiod=slow, signalperiod=signal
                )
            else:
                macd, macd_signal, macd_hist = self._macd(close_prices, fast, slow, signal)

            df[f'macd_{fast}_{slow}'] = macd
            df[f'macd_signal_{fast}_{slow}'] = macd_signal
            df[f'macd_hist_{fast}_{slow}'] = macd_hist
        
        # Parabolic SAR (simplified version)
        if HAS_TALIB:
            df['sar'] = talib.SAR(high_prices, low_prices)
        else:
            # Simplified SAR approximation
            df['sar'] = df['close'].rolling(20).mean()
        df['sar_trend'] = (close_prices > df['sar']).astype(int)

        # === MOMENTUM INDICATORS ===

        # RSI with multiple periods
        for period in [7, 14, 21, 30]:
            if HAS_TALIB:
                df[f'rsi_{period}'] = talib.RSI(close_prices, timeperiod=period)
            else:
                df[f'rsi_{period}'] = self._rsi(close_prices, period)
            
            # RSI divergence signals
            df[f'rsi_{period}_oversold'] = (df[f'rsi_{period}'] < 30).astype(int)
            df[f'rsi_{period}_overbought'] = (df[f'rsi_{period}'] > 70).astype(int)
        
        # Stochastic oscillators
        for k_period, d_period in [(14, 3), (21, 5), (5, 3)]:
            slowk, slowd = talib.STOCH(
                high_prices, low_prices, close_prices,
                fastk_period=k_period, slowk_period=d_period, slowd_period=d_period
            )
            df[f'stoch_k_{k_period}'] = slowk
            df[f'stoch_d_{k_period}'] = slowd
            df[f'stoch_cross_{k_period}'] = (slowk > slowd).astype(int)
        
        # Williams %R
        for period in [14, 21]:
            df[f'willr_{period}'] = talib.WILLR(high_prices, low_prices, close_prices, timeperiod=period)
        
        # Rate of Change
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = talib.ROC(close_prices, timeperiod=period)
        
        # === VOLATILITY INDICATORS ===
        
        # Bollinger Bands with multiple settings
        for period, std_dev in [(20, 2), (20, 1.5), (10, 2)]:
            upper, middle, lower = talib.BBANDS(
                close_prices, timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev
            )
            df[f'bb_upper_{period}_{std_dev}'] = upper
            df[f'bb_middle_{period}_{std_dev}'] = middle
            df[f'bb_lower_{period}_{std_dev}'] = lower
            
            # Bollinger Band position
            df[f'bb_position_{period}_{std_dev}'] = (close_prices - lower) / (upper - lower)
            df[f'bb_squeeze_{period}_{std_dev}'] = (upper - lower) / middle
        
        # Average True Range
        for period in [14, 21]:
            df[f'atr_{period}'] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=period)
            df[f'atr_ratio_{period}'] = df[f'atr_{period}'] / close_prices
        
        # === VOLUME INDICATORS ===
        
        # Volume moving averages
        for period in [10, 20, 50]:
            df[f'volume_sma_{period}'] = talib.SMA(volume, timeperiod=period)
            df[f'volume_ratio_{period}'] = volume / df[f'volume_sma_{period}']
        
        # On Balance Volume
        df['obv'] = talib.OBV(close_prices, volume)
        df['obv_sma_10'] = talib.SMA(df['obv'].values, timeperiod=10)
        
        # Volume Price Trend
        df['vpt'] = talib.AD(high_prices, low_prices, close_prices, volume)
        
        # === PRICE ACTION FEATURES ===
        
        # Price changes and returns
        for period in [1, 2, 3, 5, 10]:
            df[f'return_{period}'] = df['close'].pct_change(period)
            df[f'high_low_ratio_{period}'] = df['high'].rolling(period).max() / df['low'].rolling(period).min()
        
        # Candlestick patterns (simplified)
        df['doji'] = (abs(df['close'] - df['open']) / (df['high'] - df['low']) < 0.1).astype(int)
        df['hammer'] = ((df['close'] > df['open']) & 
                       ((df['open'] - df['low']) > 2 * (df['close'] - df['open']))).astype(int)
        
        # Gap analysis
        df['gap_up'] = (df['open'] > df['close'].shift(1)).astype(int)
        df['gap_down'] = (df['open'] < df['close'].shift(1)).astype(int)
        
        # === MARKET STRUCTURE ===
        
        # Support and resistance levels
        df['resistance_20'] = df['high'].rolling(20).max()
        df['support_20'] = df['low'].rolling(20).min()
        df['price_position'] = (df['close'] - df['support_20']) / (df['resistance_20'] - df['support_20'])
        
        # Trend strength
        df['trend_strength'] = abs(df['close'] - df['sma_20']) / df['atr_14']
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between different indicators."""
        
        # RSI and MACD interaction
        if 'rsi_14' in df.columns and 'macd_12_26' in df.columns:
            df['rsi_macd_signal'] = ((df['rsi_14'] > 50) & (df['macd_12_26'] > 0)).astype(int)
        
        # Volume and price interaction
        if 'volume_ratio_20' in df.columns and 'return_1' in df.columns:
            df['volume_price_momentum'] = df['volume_ratio_20'] * df['return_1']
        
        # Bollinger Bands and RSI
        if 'bb_position_20_2' in df.columns and 'rsi_14' in df.columns:
            df['bb_rsi_signal'] = ((df['bb_position_20_2'] < 0.2) & (df['rsi_14'] < 30)).astype(int)
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame, target_col: str = 'close') -> pd.DataFrame:
        """Create lagged features for time series patterns."""
        
        # Create lagged versions of key indicators
        key_features = ['rsi_14', 'macd_12_26', 'bb_position_20_2', 'volume_ratio_20']
        
        for feature in key_features:
            if feature in df.columns:
                for lag in [1, 2, 3, 5]:
                    df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)
        
        # Rolling statistics
        for window in [5, 10, 20]:
            df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window).mean()
            df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window).std()
            df[f'{target_col}_rolling_skew_{window}'] = df[target_col].rolling(window).skew()
        
        return df
    
    def engineer_advanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Main method to engineer all advanced features."""
        
        print("Creating advanced technical indicators...")
        df = self.create_advanced_technical_indicators(data)
        
        print("Creating interaction features...")
        df = self.create_interaction_features(df)
        
        print("Creating lag features...")
        df = self.create_lag_features(df)
        
        # Remove infinite and NaN values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        print(f"Advanced feature engineering completed. Shape: {df.shape}")
        return df
    
    def select_best_features(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        n_features: int = 50,
        method: str = 'mutual_info'
    ) -> List[str]:
        """Select the best features using advanced selection methods."""
        
        # Remove NaN values for feature selection
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X_clean = X[mask]
        y_clean = y[mask]
        
        if method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
        else:
            selector = SelectKBest(score_func=f_classif, k=n_features)
        
        selector.fit(X_clean, y_clean)
        selected_features = X.columns[selector.get_support()].tolist()
        
        print(f"Selected {len(selected_features)} best features using {method}")
        return selected_features
    
    def scale_features_advanced(
        self, 
        X: pd.DataFrame, 
        method: str = 'robust',
        fit: bool = True
    ) -> pd.DataFrame:
        """Scale features using advanced scaling methods."""
        
        if method == 'robust':
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()
        
        if fit:
            X_scaled = scaler.fit_transform(X)
            self.scalers[method] = scaler
        else:
            if method not in self.scalers:
                raise ValueError(f"Scaler for method '{method}' not fitted yet")
            X_scaled = self.scalers[method].transform(X)
        
        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
