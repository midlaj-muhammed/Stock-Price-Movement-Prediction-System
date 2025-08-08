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

        # Add advanced features for higher accuracy
        df = self._add_advanced_features(df)

        # Add market regime features
        df = self._add_market_regime_features(df)

        # Add volatility clustering features
        df = self._add_volatility_features(df)

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
        """Clean and validate features with robust handling of extreme values."""
        df = data.copy()
        logger.info("Starting feature cleaning and validation")

        # Step 1: Handle infinite values
        logger.info("Replacing infinite values with NaN")
        df = df.replace([np.inf, -np.inf], np.nan)

        # Step 2: Remove columns with too many NaN values (>50%)
        nan_threshold = 0.5
        nan_ratio = df.isnull().sum() / len(df)
        columns_to_drop = nan_ratio[nan_ratio > nan_threshold].index.tolist()

        if columns_to_drop:
            logger.info(f"Dropping columns with >50% NaN values: {columns_to_drop}")
            df = df.drop(columns=columns_to_drop)

        # Step 3: Remove constant features
        constant_features = []
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].nunique() <= 1:
                constant_features.append(col)

        if constant_features:
            logger.info(f"Dropping constant features: {constant_features}")
            df = df.drop(columns=constant_features)

        # Step 4: Ensure proper data types and handle extreme values
        numeric_columns = df.select_dtypes(include=[np.number]).columns

        # First, convert all numeric columns to float64 to avoid dtype issues
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].astype(np.float64)

        # Now handle extreme values and outliers
        for col in numeric_columns:
            if col in df.columns and df[col].notna().sum() > 0:
                # Calculate robust statistics
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1

                # Define outlier bounds (more conservative than 3*IQR)
                lower_bound = q1 - 3 * iqr
                upper_bound = q3 + 3 * iqr

                # Also use percentile-based bounds as backup
                percentile_lower = df[col].quantile(0.001)
                percentile_upper = df[col].quantile(0.999)

                # Use the more conservative bounds
                final_lower = max(lower_bound, percentile_lower) if not np.isnan(lower_bound) else percentile_lower
                final_upper = min(upper_bound, percentile_upper) if not np.isnan(upper_bound) else percentile_upper

                # Ensure bounds are finite
                if not np.isfinite(final_lower):
                    final_lower = df[col].min()
                if not np.isfinite(final_upper):
                    final_upper = df[col].max()

                # Clip extreme values
                outliers_count = ((df[col] < final_lower) | (df[col] > final_upper)).sum()
                if outliers_count > 0:
                    logger.info(f"Clipping {outliers_count} outliers in column {col}")
                    df[col] = df[col].clip(lower=final_lower, upper=final_upper)

        # Step 5: Fill remaining NaN values with multiple strategies
        for col in numeric_columns:
            if col in df.columns and df[col].isna().any():
                # Strategy 1: Forward fill
                df[col] = df[col].fillna(method='ffill')

                # Strategy 2: Backward fill
                df[col] = df[col].fillna(method='bfill')

                # Strategy 3: Use median for any remaining NaN
                if df[col].isna().any():
                    median_val = df[col].median()
                    if not np.isnan(median_val):
                        df[col] = df[col].fillna(median_val)
                    else:
                        # Last resort: use 0
                        df[col] = df[col].fillna(0)

        # Step 6: Final validation - ensure no infinite or NaN values remain
        inf_check = np.isinf(df.select_dtypes(include=[np.number])).any().any()
        nan_check = df.select_dtypes(include=[np.number]).isna().any().any()

        if inf_check:
            logger.warning("Found remaining infinite values, replacing with 0")
            df = df.replace([np.inf, -np.inf], 0)

        if nan_check:
            logger.warning("Found remaining NaN values, replacing with 0")
            df = df.fillna(0)

        # Step 7: Ensure all values are finite and within reasonable range
        for col in df.select_dtypes(include=[np.number]).columns:
            # Check for extremely large values that might cause overflow
            max_val = df[col].abs().max()
            if max_val > 1e10:  # Arbitrary large number threshold
                logger.warning(f"Column {col} has very large values (max: {max_val}), scaling down")
                df[col] = df[col] / (max_val / 1e6)  # Scale to reasonable range

        logger.info(f"Feature cleaning completed. Final shape: {df.shape}")

        # Final check
        final_inf_check = np.isinf(df.select_dtypes(include=[np.number])).any().any()
        final_nan_check = df.select_dtypes(include=[np.number]).isna().any().any()

        if final_inf_check or final_nan_check:
            logger.error("Still found infinite or NaN values after cleaning!")
            # Emergency cleanup
            df = df.replace([np.inf, -np.inf, np.nan], 0)

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
                logger.warning(f"Missing selected features: {missing_features}")

        return df

    def _add_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced features for higher accuracy."""
        logger.info("Adding advanced features")

        # Price momentum features
        for period in [3, 5, 10, 20]:
            df[f'momentum_{period}'] = df['close'].pct_change(period)
            df[f'momentum_ma_{period}'] = df[f'momentum_{period}'].rolling(5).mean()

        # Volume-price relationship (with safe calculations)
        price_change = df['close'].pct_change().fillna(0)
        volume_change = df['volume'].pct_change().fillna(0)
        df['volume_price_trend'] = (price_change * volume_change).rolling(5).mean()
        df['price_volume_ratio'] = df['close'] / (df['volume'] + 1e-6)  # Prevent division by zero

        # Support and resistance levels (with safe calculations)
        df['resistance_20'] = df['high'].rolling(20).max()
        df['support_20'] = df['low'].rolling(20).min()
        resistance_support_diff = df['resistance_20'] - df['support_20']
        # Prevent division by zero with a minimum difference
        resistance_support_diff = resistance_support_diff.where(resistance_support_diff > 1e-6, 1e-6)
        df['price_position'] = (df['close'] - df['support_20']) / resistance_support_diff

        # Fractal features
        df['local_max'] = (df['high'] > df['high'].shift(1)) & (df['high'] > df['high'].shift(-1))
        df['local_min'] = (df['low'] < df['low'].shift(1)) & (df['low'] < df['low'].shift(-1))
        df['fractal_score'] = df['local_max'].astype(int) - df['local_min'].astype(int)

        # Gap analysis
        df['gap_up'] = (df['open'] > df['close'].shift(1)) & (df['open'] - df['close'].shift(1) > df['close'].shift(1) * 0.02)
        df['gap_down'] = (df['open'] < df['close'].shift(1)) & (df['close'].shift(1) - df['open'] > df['close'].shift(1) * 0.02)

        return df

    def _add_market_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market regime detection features."""
        logger.info("Adding market regime features")

        # Trend strength
        df['trend_strength'] = abs(df['close'].rolling(20).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0]))

        # Market state (bull/bear/sideways) with safe calculations
        ma_short = df['close'].rolling(10).mean()
        ma_long = df['close'].rolling(50).mean()
        # Prevent division by zero or very small numbers
        ma_long_safe = ma_long.where(ma_long.abs() > 1e-6, 1e-6)
        df['market_state'] = np.where(ma_short > ma_long_safe * 1.02, 1,  # Bull
                                    np.where(ma_short < ma_long_safe * 0.98, -1, 0))  # Bear, Sideways

        # Volatility regime with safe calculations
        volatility = df['close'].pct_change().rolling(20).std()
        vol_ma = volatility.rolling(50).mean()
        # Prevent division by zero
        vol_ma_safe = vol_ma.where(vol_ma > 1e-6, 1e-6)
        df['vol_regime'] = np.where(volatility > vol_ma_safe * 1.5, 1,  # High vol
                                  np.where(volatility < vol_ma_safe * 0.5, -1, 0))  # Low vol, Normal

        return df

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility clustering and GARCH-like features."""
        logger.info("Adding volatility features")

        returns = df['close'].pct_change()

        # Realized volatility
        for period in [5, 10, 20]:
            df[f'realized_vol_{period}'] = returns.rolling(period).std()
            df[f'vol_of_vol_{period}'] = df[f'realized_vol_{period}'].rolling(period).std()

        # GARCH-like features with safe calculations
        df['returns_squared'] = returns ** 2
        df['garch_vol'] = df['returns_squared'].ewm(alpha=0.1).mean()

        # Volatility clustering with safe calculations
        vol_20_mean = df['realized_vol_20'].rolling(50).mean()
        df['vol_cluster'] = (df['realized_vol_20'] > vol_20_mean).astype(int)

        # Jump detection with safe calculations
        vol_threshold = df['realized_vol_20'] * 3
        vol_threshold = vol_threshold.where(vol_threshold > 1e-6, 1e-6)  # Prevent zero threshold
        df['jump_indicator'] = (abs(returns) > vol_threshold).astype(int)

        return df

        return df
