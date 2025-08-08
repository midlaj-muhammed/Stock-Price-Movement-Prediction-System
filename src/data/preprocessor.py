"""
Data preprocessing and preparation for machine learning models.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, Dict, Any, List
import joblib
from pathlib import Path

from ..utils.logger import get_logger
from ..utils.config import config

logger = get_logger(__name__)

class StockDataPreprocessor:
    """Preprocess stock data for machine learning models."""
    
    def __init__(self):
        """Initialize the preprocessor."""
        self.scalers = {}
        self.feature_columns = []
        self.target_columns = []
        self.lookback_window = config.get('features.lookback_window', 60)
        self.prediction_horizon = config.get('features.prediction_horizon', 1)
        
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features from raw stock data.
        
        Args:
            data: Raw stock data DataFrame
            
        Returns:
            DataFrame with prepared features
        """
        logger.info("Preparing features from raw data")
        
        # Make a copy to avoid modifying original data
        df = data.copy()
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Calculate returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Price-based features
        df['price_change'] = df['close'] - df['open']
        df['price_range'] = df['high'] - df['low']
        df['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
        df['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
        
        # Volume features
        df['volume_change'] = df['volume'].pct_change()
        df['price_volume'] = df['close'] * df['volume']
        
        # Volatility features
        df['volatility_5'] = df['returns'].rolling(window=5).std()
        df['volatility_10'] = df['returns'].rolling(window=10).std()
        df['volatility_20'] = df['returns'].rolling(window=20).std()
        
        # Gap features
        df['gap'] = df['open'] - df['close'].shift(1)
        df['gap_percent'] = df['gap'] / df['close'].shift(1)
        
        # Intraday features
        df['intraday_return'] = (df['close'] - df['open']) / df['open']
        df['overnight_return'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        
        # Remove rows with NaN values
        df = df.dropna().reset_index(drop=True)
        
        logger.info(f"Prepared {len(df)} records with {len(df.columns)} features")
        return df
    
    def create_targets(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create target variables for prediction.
        
        Args:
            data: DataFrame with features
            
        Returns:
            DataFrame with target variables added
        """
        logger.info("Creating target variables")
        
        df = data.copy()
        
        # Binary classification target (next day price movement)
        df['target_direction'] = (df['close'].shift(-self.prediction_horizon) > df['close']).astype(int)
        
        # Regression target (next day closing price)
        df['target_price'] = df['close'].shift(-self.prediction_horizon)
        
        # Regression target (next day return)
        df['target_return'] = (df['close'].shift(-self.prediction_horizon) - df['close']) / df['close']
        
        # Remove rows with NaN targets
        df = df.dropna().reset_index(drop=True)
        
        logger.info(f"Created targets for {len(df)} records")
        return df
    
    def scale_features(
        self,
        data: pd.DataFrame,
        feature_columns: List[str],
        scaler_type: str = "minmax",
        fit_scaler: bool = True
    ) -> pd.DataFrame:
        """
        Scale features using specified scaler.
        
        Args:
            data: DataFrame with features
            feature_columns: List of columns to scale
            scaler_type: Type of scaler ('minmax' or 'standard')
            fit_scaler: Whether to fit the scaler (True for training, False for inference)
            
        Returns:
            DataFrame with scaled features
        """
        logger.info(f"Scaling features using {scaler_type} scaler")
        
        df = data.copy()
        
        # Initialize scaler if not exists
        if scaler_type not in self.scalers:
            if scaler_type == "minmax":
                self.scalers[scaler_type] = MinMaxScaler()
            elif scaler_type == "standard":
                self.scalers[scaler_type] = StandardScaler()
            else:
                raise ValueError(f"Unknown scaler type: {scaler_type}")
        
        scaler = self.scalers[scaler_type]
        
        # Fit and transform or just transform
        if fit_scaler:
            df[feature_columns] = scaler.fit_transform(df[feature_columns])
        else:
            df[feature_columns] = scaler.transform(df[feature_columns])
        
        return df
    
    def create_sequences(
        self,
        data: pd.DataFrame,
        feature_columns: List[str],
        target_columns: List[str],
        lookback_window: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series models with adaptive lookback window.

        Args:
            data: DataFrame with features and targets
            feature_columns: List of feature column names
            target_columns: List of target column names
            lookback_window: Number of time steps to look back

        Returns:
            Tuple of (X, y) arrays for model training
        """
        if lookback_window is None:
            lookback_window = self.lookback_window

        # Adaptive lookback window based on available data
        max_possible_lookback = len(data) - 1
        if lookback_window > max_possible_lookback:
            original_lookback = lookback_window
            lookback_window = max(1, max_possible_lookback)
            logger.warning(f"Reduced lookback window from {original_lookback} to {lookback_window} due to limited data")

        logger.info(f"Creating sequences with lookback window of {lookback_window}")

        # Extract features and targets
        features = data[feature_columns].values
        targets = data[target_columns].values

        X, y = [], []

        for i in range(lookback_window, len(data)):
            # Features: lookback_window time steps
            X.append(features[i-lookback_window:i])
            # Target: current time step
            y.append(targets[i])

        # If no sequences can be created, create at least one with available data
        if len(X) == 0 and len(data) > 0:
            logger.warning("No sequences created with standard method, using fallback")
            # Use all available data as a single sequence
            if len(data) >= 2:
                X.append(features[:-1])  # All but last row as features
                y.append(targets[-1])    # Last row as target
            else:
                # Single data point - duplicate it
                X.append(features)
                y.append(targets[0])

        X = np.array(X)
        y = np.array(y)

        logger.info(f"Created {len(X)} sequences with shape X: {X.shape}, y: {y.shape}")
        return X, y
    
    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        validation_size: float = 0.2,
        shuffle: bool = False,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            X: Feature arrays
            y: Target arrays
            test_size: Proportion of data for testing
            validation_size: Proportion of training data for validation
            shuffle: Whether to shuffle data before splitting
            random_state: Random state for reproducibility
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        logger.info("Splitting data into train, validation, and test sets")
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=shuffle, random_state=random_state
        )
        
        # Second split: separate train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=validation_size, shuffle=shuffle, random_state=random_state
        )
        
        logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_scalers(self, filepath: str) -> None:
        """Save fitted scalers to disk."""
        try:
            joblib.dump(self.scalers, filepath)
            logger.info(f"Saved scalers to {filepath}")
        except Exception as e:
            logger.error(f"Error saving scalers: {e}")
    
    def load_scalers(self, filepath: str) -> None:
        """Load fitted scalers from disk."""
        try:
            self.scalers = joblib.load(filepath)
            logger.info(f"Loaded scalers from {filepath}")
        except Exception as e:
            logger.error(f"Error loading scalers: {e}")
    
    def inverse_transform_targets(
        self,
        targets: np.ndarray,
        target_columns: List[str],
        scaler_type: str = "minmax"
    ) -> np.ndarray:
        """
        Inverse transform scaled target values.
        
        Args:
            targets: Scaled target values
            target_columns: List of target column names
            scaler_type: Type of scaler used
            
        Returns:
            Original scale target values
        """
        if scaler_type not in self.scalers:
            logger.warning(f"Scaler {scaler_type} not found, returning original values")
            return targets
        
        scaler = self.scalers[scaler_type]
        
        # Create dummy array with all features for inverse transform
        n_features = scaler.n_features_in_
        dummy_array = np.zeros((targets.shape[0], n_features))
        
        # Find indices of target columns in original feature set
        target_indices = [i for i, col in enumerate(self.feature_columns) if col in target_columns]
        
        # Place target values in correct positions
        for i, idx in enumerate(target_indices):
            if i < targets.shape[1]:
                dummy_array[:, idx] = targets[:, i]
        
        # Inverse transform and extract target columns
        inverse_transformed = scaler.inverse_transform(dummy_array)
        result = inverse_transformed[:, target_indices]
        
        return result
