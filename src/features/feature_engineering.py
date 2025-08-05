"""
Feature engineering pipeline for stock prediction.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

from .technical_indicators import TechnicalIndicators
from ..utils.logger import get_logger
from ..utils.config import config

logger = get_logger(__name__)

class FeatureEngineer:
    """Feature engineering pipeline for stock prediction models."""
    
    def __init__(self):
        """Initialize feature engineer."""
        self.technical_indicators = TechnicalIndicators()
        self.feature_selector = None
        self.pca_transformer = None
        self.selected_features = []
        self.feature_importance = {}
        
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Complete feature engineering pipeline.
        
        Args:
            data: Raw stock data DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Starting feature engineering pipeline")
        
        # Calculate technical indicators
        df = self.technical_indicators.calculate_all_indicators(data)
        
        # Add time-based features
        df = self._add_time_features(df)
        
        # Add lag features
        df = self._add_lag_features(df)
        
        # Add rolling statistics
        df = self._add_rolling_features(df)
        
        # Add interaction features
        df = self._add_interaction_features(df)
        
        # Clean up features
        df = self._clean_features(df)
        
        logger.info(f"Feature engineering completed. Final shape: {df.shape}")
        return df
    
    def _add_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        df = data.copy()
        
        # Convert timestamp to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Extract time components
        df['year'] = df['timestamp'].dt.year
        df['month'] = df['timestamp'].dt.month
        df['day'] = df['timestamp'].dt.day
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        df['week_of_year'] = df['timestamp'].dt.isocalendar().week
        df['quarter'] = df['timestamp'].dt.quarter
        
        # Cyclical encoding for time features
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        # Market session indicators
        df['is_monday'] = (df['day_of_week'] == 0).astype(int)
        df['is_friday'] = (df['day_of_week'] == 4).astype(int)
        df['is_month_end'] = (df['timestamp'].dt.is_month_end).astype(int)
        df['is_month_start'] = (df['timestamp'].dt.is_month_start).astype(int)
        df['is_quarter_end'] = (df['timestamp'].dt.is_quarter_end).astype(int)
        
        return df
    
    def _add_lag_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add lagged features."""
        df = data.copy()
        
        # Key features to lag
        lag_features = ['close', 'volume', 'returns', 'rsi_14', 'macd', 'bb_position']
        lag_periods = [1, 2, 3, 5, 10]
        
        for feature in lag_features:
            if feature in df.columns:
                for lag in lag_periods:
                    df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)
        
        return df
    
    def _add_rolling_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add rolling window statistics."""
        df = data.copy()
        
        # Features for rolling statistics
        rolling_features = ['close', 'volume', 'returns', 'high', 'low']
        windows = [5, 10, 20]
        
        for feature in rolling_features:
            if feature in df.columns:
                for window in windows:
                    # Rolling statistics
                    df[f'{feature}_mean_{window}'] = df[feature].rolling(window=window).mean()
                    df[f'{feature}_std_{window}'] = df[feature].rolling(window=window).std()
                    df[f'{feature}_min_{window}'] = df[feature].rolling(window=window).min()
                    df[f'{feature}_max_{window}'] = df[feature].rolling(window=window).max()
                    df[f'{feature}_median_{window}'] = df[feature].rolling(window=window).median()
                    
                    # Rolling ratios
                    df[f'{feature}_ratio_mean_{window}'] = df[feature] / df[f'{feature}_mean_{window}']
                    df[f'{feature}_zscore_{window}'] = (df[feature] - df[f'{feature}_mean_{window}']) / df[f'{feature}_std_{window}']
        
        return df
    
    def _add_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add interaction features."""
        df = data.copy()
        
        # Price-volume interactions
        if all(col in df.columns for col in ['close', 'volume']):
            df['price_volume_interaction'] = df['close'] * df['volume']
            df['price_volume_ratio'] = df['close'] / (df['volume'] + 1e-8)
        
        # Volatility-momentum interactions
        if all(col in df.columns for col in ['atr_14', 'rsi_14']):
            df['volatility_momentum'] = df['atr_14'] * df['rsi_14']
        
        # Trend-momentum interactions
        if all(col in df.columns for col in ['macd', 'adx']):
            df['trend_momentum'] = df['macd'] * df['adx']
        
        # Moving average interactions
        if all(col in df.columns for col in ['sma_5', 'sma_20']):
            df['ma_cross_signal'] = (df['sma_5'] > df['sma_20']).astype(int)
            df['ma_distance'] = (df['sma_5'] - df['sma_20']) / df['sma_20']
        
        # Bollinger Band interactions
        if all(col in df.columns for col in ['bb_position', 'rsi_14']):
            df['bb_rsi_interaction'] = df['bb_position'] * df['rsi_14']
        
        return df
    
    def _clean_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate features."""
        df = data.copy()
        
        # Remove infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Remove columns with too many NaN values (>50%)
        nan_threshold = 0.5
        nan_ratio = df.isnull().sum() / len(df)
        columns_to_drop = nan_ratio[nan_ratio > nan_threshold].index.tolist()
        
        if columns_to_drop:
            logger.info(f"Dropping columns with >50% NaN values: {columns_to_drop}")
            df = df.drop(columns=columns_to_drop)
        
        # Remove constant features
        constant_features = []
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].nunique() <= 1:
                constant_features.append(col)
        
        if constant_features:
            logger.info(f"Dropping constant features: {constant_features}")
            df = df.drop(columns=constant_features)
        
        # Fill remaining NaN values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(method='ffill').fillna(method='bfill')
        
        return df
    
    def select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method: str = "mutual_info",
        k: int = 50,
        task_type: str = "classification"
    ) -> List[str]:
        """
        Select top k features using specified method.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            method: Feature selection method ('mutual_info', 'f_test', 'correlation')
            k: Number of features to select
            task_type: 'classification' or 'regression'
            
        Returns:
            List of selected feature names
        """
        logger.info(f"Selecting top {k} features using {method} method")
        
        # Ensure we don't select more features than available
        k = min(k, X.shape[1])
        
        if method == "mutual_info":
            if task_type == "classification":
                selector = SelectKBest(score_func=mutual_info_classif, k=k)
            else:
                from sklearn.feature_selection import mutual_info_regression
                selector = SelectKBest(score_func=mutual_info_regression, k=k)
        elif method == "f_test":
            if task_type == "classification":
                selector = SelectKBest(score_func=f_classif, k=k)
            else:
                selector = SelectKBest(score_func=f_regression, k=k)
        elif method == "correlation":
            # Simple correlation-based selection
            correlations = abs(X.corrwith(y))
            selected_features = correlations.nlargest(k).index.tolist()
            self.selected_features = selected_features
            logger.info(f"Selected {len(selected_features)} features using correlation")
            return selected_features
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
        
        # Fit selector and get selected features
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        # Store feature importance scores
        self.feature_importance = dict(zip(X.columns, selector.scores_))
        self.selected_features = selected_features
        self.feature_selector = selector
        
        logger.info(f"Selected {len(selected_features)} features")
        return selected_features
    
    def apply_pca(
        self,
        X: pd.DataFrame,
        n_components: Optional[int] = None,
        variance_threshold: float = 0.95
    ) -> Tuple[np.ndarray, PCA]:
        """
        Apply PCA for dimensionality reduction.
        
        Args:
            X: Feature DataFrame
            n_components: Number of components (if None, use variance_threshold)
            variance_threshold: Cumulative variance threshold
            
        Returns:
            Tuple of (transformed_data, pca_transformer)
        """
        if n_components is None:
            # Determine number of components based on variance threshold
            pca_temp = PCA()
            pca_temp.fit(X)
            cumsum_variance = np.cumsum(pca_temp.explained_variance_ratio_)
            n_components = np.argmax(cumsum_variance >= variance_threshold) + 1
        
        logger.info(f"Applying PCA with {n_components} components")
        
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        
        self.pca_transformer = pca
        
        logger.info(f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")
        return X_pca, pca
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        return self.feature_importance
    
    def get_top_features(self, n: int = 20) -> List[str]:
        """Get top n important features."""
        if not self.feature_importance:
            logger.warning("No feature importance scores available")
            return []
        
        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [feature for feature, _ in sorted_features[:n]]
    
    def transform_new_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted transformers.
        
        Args:
            data: New data to transform
            
        Returns:
            Transformed data
        """
        # Apply feature engineering
        df = self.engineer_features(data)
        
        # Select features if selector is fitted
        if self.feature_selector is not None and self.selected_features:
            # Ensure all selected features are present
            missing_features = set(self.selected_features) - set(df.columns)
            if missing_features:
                logger.warning(f"Missing features in new data: {missing_features}")
                # Add missing features with zeros
                for feature in missing_features:
                    df[feature] = 0
            
            df = df[self.selected_features]
        
        return df
