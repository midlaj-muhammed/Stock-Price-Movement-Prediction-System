"""
Training pipeline for stock prediction models.
"""

import os
# Force CPU usage before any TensorFlow imports
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import joblib
from pathlib import Path
import json
import time

from .data.data_collector import StockDataCollector
from .data.preprocessor import StockDataPreprocessor
from .features.feature_engineering import FeatureEngineer
from .models.lstm_model import LSTMStockModel
from .models.tcn_model import TCNStockModel
from .evaluation.metrics import ModelEvaluator
from .utils.logger import get_logger
from .utils.config import config

logger = get_logger(__name__)

class StockPredictionPipeline:
    """Complete training pipeline for stock prediction models."""
    
    def __init__(self, symbol: str, task_type: str = "classification"):
        """
        Initialize training pipeline.
        
        Args:
            symbol: Stock symbol to train on
            task_type: 'classification' or 'regression'
        """
        self.symbol = symbol
        self.task_type = task_type
        
        # Initialize components
        self.data_collector = StockDataCollector()
        self.preprocessor = StockDataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.evaluator = ModelEvaluator()
        
        # Models
        self.models = {
            'lstm': LSTMStockModel(task_type),
            'tcn': TCNStockModel(task_type)
        }
        
        # Data storage
        self.raw_data = None
        self.processed_data = None
        self.features = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        
        # Results storage
        self.training_results = {}
        self.evaluation_results = {}
        
    def collect_data(self, period: str = "2y", source: str = "yahoo") -> pd.DataFrame:
        """
        Collect stock data.
        
        Args:
            period: Data period
            source: Data source
            
        Returns:
            Raw stock data
        """
        logger.info(f"Collecting data for {self.symbol}")
        
        self.raw_data = self.data_collector.get_stock_data(
            symbol=self.symbol,
            period=period,
            source=source
        )
        
        if self.raw_data is None:
            raise ValueError(f"Failed to collect data for {self.symbol}")
        
        # Validate data
        if not self.data_collector.validate_data(self.raw_data):
            raise ValueError("Data validation failed")
        
        logger.info(f"Collected {len(self.raw_data)} records for {self.symbol}")
        return self.raw_data
    
    def engineer_features(self) -> pd.DataFrame:
        """
        Engineer features from raw data.
        
        Returns:
            Data with engineered features
        """
        if self.raw_data is None:
            raise ValueError("No raw data available. Call collect_data() first.")
        
        logger.info("Engineering features")
        
        # Engineer features
        self.features = self.feature_engineer.engineer_features(self.raw_data)
        
        # Prepare basic features
        self.processed_data = self.preprocessor.prepare_features(self.features)
        
        # Create targets
        self.processed_data = self.preprocessor.create_targets(self.processed_data)
        
        logger.info(f"Feature engineering completed. Shape: {self.processed_data.shape}")
        return self.processed_data
    
    def prepare_data_for_training(
        self,
        feature_selection_method: str = "mutual_info",
        n_features: int = 50,
        test_size: float = 0.2,
        validation_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for model training.
        
        Args:
            feature_selection_method: Method for feature selection
            n_features: Number of features to select
            test_size: Test set size
            validation_size: Validation set size
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        if self.processed_data is None:
            raise ValueError("No processed data available. Call engineer_features() first.")
        
        logger.info("Preparing data for training")
        
        # Define feature and target columns
        exclude_columns = ['timestamp', 'symbol', 'target_direction', 'target_price', 'target_return']
        feature_columns = [col for col in self.processed_data.columns if col not in exclude_columns]
        
        # Select target column based on task type
        if self.task_type == "classification":
            target_column = 'target_direction'
        else:
            target_column = 'target_price'  # or 'target_return'
        
        # Prepare features and targets
        X = self.processed_data[feature_columns]
        y = self.processed_data[target_column]
        
        # Feature selection
        # Ensure we don't select more features than available
        n_features = min(n_features, len(X.columns))

        selected_features = self.feature_engineer.select_features(
            X, y, method=feature_selection_method, k=n_features, task_type=self.task_type
        )

        X_selected = X[selected_features]

        # Store selected features for later use
        self.feature_engineer.selected_features = selected_features
        
        # Scale features
        X_scaled = self.preprocessor.scale_features(
            X_selected, selected_features, scaler_type="minmax", fit_scaler=True
        )

        # Validate scaled data
        logger.info("Validating scaled training data...")

        # Check for infinite values in scaled features
        inf_mask = np.isinf(X_scaled).any(axis=1)
        if inf_mask.any():
            logger.warning(f"Found {inf_mask.sum()} rows with infinite values in scaled features")
            X_scaled = X_scaled[~inf_mask]
            y = y[~inf_mask]

        # Check for NaN values in scaled features
        nan_mask = X_scaled.isna().any(axis=1)
        if nan_mask.any():
            logger.warning(f"Found {nan_mask.sum()} rows with NaN values in scaled features")
            X_scaled = X_scaled[~nan_mask]
            y = y[~nan_mask]

        # Check for NaN values in target
        target_nan_mask = y.isna()
        if target_nan_mask.any():
            logger.warning(f"Found {target_nan_mask.sum()} NaN values in target")
            X_scaled = X_scaled[~target_nan_mask]
            y = y[~target_nan_mask]

        # Final validation
        if len(X_scaled) == 0:
            raise ValueError("No valid training data remaining after validation")

        logger.info(f"Data validation completed. Features shape: {X_scaled.shape}, Target shape: {y.shape}")

        # Combine scaled features with targets for sequence creation
        combined_data = X_scaled.copy()
        combined_data[target_column] = y.values

        # Create sequences for time series models
        X_sequences, y_sequences = self.preprocessor.create_sequences(
            combined_data, selected_features, [target_column]
        )

        # Check if we have enough sequences
        if len(X_sequences) < 10:
            raise ValueError(f"Insufficient data for training. Only {len(X_sequences)} sequences created. "
                           f"Need at least 10. Try using more historical data or reducing lookback_window.")

        # Split data
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = \
            self.preprocessor.split_data(
                X_sequences, y_sequences.flatten(),
                test_size=test_size,
                validation_size=validation_size,
                shuffle=False  # Important for time series
            )
        
        logger.info(f"Data preparation completed. Train: {len(self.X_train)}, "
                   f"Val: {len(self.X_val)}, Test: {len(self.X_test)}")
        
        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test
    
    def train_model(
        self,
        model_name: str,
        **training_kwargs
    ) -> Dict[str, Any]:
        """
        Train a specific model.
        
        Args:
            model_name: Name of model to train ('lstm' or 'tcn')
            **training_kwargs: Additional training parameters
            
        Returns:
            Training history
        """
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        
        if self.X_train is None:
            raise ValueError("No training data available. Call prepare_data_for_training() first.")
        
        logger.info(f"Training {model_name} model")
        
        model = self.models[model_name]
        
        # Train model
        start_time = time.time()
        history = model.train(
            self.X_train, self.y_train,
            self.X_val, self.y_val,
            **training_kwargs
        )
        training_time = time.time() - start_time
        
        # Store results
        self.training_results[model_name] = {
            'history': history,
            'training_time': training_time,
            'model_config': model.model_config
        }
        
        logger.info(f"{model_name} training completed in {training_time:.2f} seconds")
        return history
    
    def train_all_models(self, **training_kwargs) -> Dict[str, Dict[str, Any]]:
        """
        Train all available models.
        
        Args:
            **training_kwargs: Training parameters
            
        Returns:
            Dictionary of training results for all models
        """
        logger.info("Training all models")
        
        for model_name in self.models.keys():
            try:
                self.train_model(model_name, **training_kwargs)
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                continue
        
        return self.training_results
    
    def evaluate_model(self, model_name: str) -> Dict[str, Any]:
        """
        Evaluate a trained model.
        
        Args:
            model_name: Name of model to evaluate
            
        Returns:
            Evaluation results
        """
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        
        model = self.models[model_name]
        
        if not model.is_trained:
            raise ValueError(f"Model {model_name} is not trained")
        
        logger.info(f"Evaluating {model_name} model")
        
        # Make predictions
        y_pred = model.predict(self.X_test)
        
        # Evaluate based on task type
        if self.task_type == "classification":
            results = self.evaluator.evaluate_classification(
                self.y_test, y_pred, y_pred_proba=y_pred
            )
        else:
            results = self.evaluator.evaluate_regression(
                self.y_test, y_pred
            )
        
        # Store results
        self.evaluation_results[model_name] = results
        
        # Generate report
        report = self.evaluator.generate_evaluation_report(
            results, model_name, self.task_type
        )
        
        logger.info(f"{model_name} evaluation completed")
        print(report)
        
        return results
    
    def evaluate_all_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate all trained models.
        
        Returns:
            Dictionary of evaluation results for all models
        """
        logger.info("Evaluating all models")
        
        for model_name in self.models.keys():
            if self.models[model_name].is_trained:
                try:
                    self.evaluate_model(model_name)
                except Exception as e:
                    logger.error(f"Error evaluating {model_name}: {e}")
                    continue
        
        return self.evaluation_results
    
    def save_models(self) -> Dict[str, str]:
        """
        Save all trained models.
        
        Returns:
            Dictionary mapping model names to saved file paths
        """
        saved_paths = {}
        
        for model_name, model in self.models.items():
            if model.is_trained:
                try:
                    filepath = model.save_model()
                    saved_paths[model_name] = filepath
                    logger.info(f"Saved {model_name} model to {filepath}")
                except Exception as e:
                    logger.error(f"Error saving {model_name}: {e}")
        
        return saved_paths
    
    def save_pipeline_results(self, filepath: Optional[str] = None) -> str:
        """
        Save complete pipeline results.
        
        Args:
            filepath: Custom filepath (optional)
            
        Returns:
            Path where results were saved
        """
        if filepath is None:
            results_dir = config.model_storage_path / "results"
            results_dir.mkdir(parents=True, exist_ok=True)
            filepath = results_dir / f"{self.symbol}_{self.task_type}_results.json"
        
        results = {
            'symbol': self.symbol,
            'task_type': self.task_type,
            'data_shape': self.processed_data.shape if self.processed_data is not None else None,
            'training_results': self.training_results,
            'evaluation_results': self.evaluation_results,
            'feature_importance': self.feature_engineer.get_feature_importance(),
            'top_features': self.feature_engineer.get_top_features(20)
        }
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            return obj
        
        try:
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=convert_numpy)
            
            logger.info(f"Pipeline results saved to {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error saving pipeline results: {e}")
            raise
    
    def run_complete_pipeline(
        self,
        period: str = "2y",
        source: str = "yahoo",
        feature_selection_method: str = "mutual_info",
        n_features: int = 50,
        **training_kwargs
    ) -> Dict[str, Any]:
        """
        Run the complete training and evaluation pipeline.
        
        Args:
            period: Data collection period
            source: Data source
            feature_selection_method: Feature selection method
            n_features: Number of features to select
            **training_kwargs: Training parameters
            
        Returns:
            Complete pipeline results
        """
        logger.info(f"Running complete pipeline for {self.symbol}")
        
        try:
            # Step 1: Collect data
            self.collect_data(period=period, source=source)
            
            # Step 2: Engineer features
            self.engineer_features()
            
            # Step 3: Prepare data
            self.prepare_data_for_training(
                feature_selection_method=feature_selection_method,
                n_features=n_features
            )
            
            # Step 4: Train models
            self.train_all_models(**training_kwargs)
            
            # Step 5: Evaluate models
            self.evaluate_all_models()
            
            # Step 6: Save results
            self.save_models()
            results_path = self.save_pipeline_results()
            
            logger.info(f"Complete pipeline finished for {self.symbol}")
            
            return {
                'training_results': self.training_results,
                'evaluation_results': self.evaluation_results,
                'results_path': results_path
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed for {self.symbol}: {e}")
            raise
