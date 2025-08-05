"""
Ensemble methods for combining multiple stock prediction models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib

from .base_model import BaseStockModel
from ..utils.logger import get_logger
from ..utils.config import config

logger = get_logger(__name__)

class ModelEnsemble(BaseStockModel):
    """Ensemble model combining multiple stock prediction models."""
    
    def __init__(self, task_type: str = "classification"):
        """
        Initialize ensemble model.
        
        Args:
            task_type: 'classification' or 'regression'
        """
        super().__init__("Ensemble", task_type)
        self.base_models = {}
        self.ensemble_method = "weighted_average"
        self.weights = {}
        self.meta_model = None
        
    def add_model(self, name: str, model: BaseStockModel, weight: float = 1.0) -> None:
        """
        Add a base model to the ensemble.
        
        Args:
            name: Name of the model
            model: Trained model instance
            weight: Weight for weighted averaging
        """
        if not model.is_trained:
            raise ValueError(f"Model {name} must be trained before adding to ensemble")
        
        if model.task_type != self.task_type:
            raise ValueError(f"Model {name} task type ({model.task_type}) doesn't match ensemble task type ({self.task_type})")
        
        self.base_models[name] = model
        self.weights[name] = weight
        
        logger.info(f"Added model {name} to ensemble with weight {weight}")
    
    def set_ensemble_method(self, method: str) -> None:
        """
        Set ensemble method.
        
        Args:
            method: Ensemble method ('simple_average', 'weighted_average', 'voting', 'stacking')
        """
        valid_methods = ['simple_average', 'weighted_average', 'voting', 'stacking']
        if method not in valid_methods:
            raise ValueError(f"Invalid ensemble method. Choose from: {valid_methods}")
        
        self.ensemble_method = method
        logger.info(f"Set ensemble method to {method}")
    
    def build_model(self, input_shape: Tuple, **kwargs) -> None:
        """Build ensemble model (not applicable for ensemble)."""
        logger.info("Ensemble model doesn't require building - uses pre-trained models")
        pass
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train ensemble model (meta-learning for stacking).
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            **kwargs: Additional parameters
            
        Returns:
            Training history
        """
        if not self.base_models:
            raise ValueError("No base models added to ensemble")
        
        logger.info(f"Training ensemble using {self.ensemble_method} method")
        
        if self.ensemble_method == "stacking":
            # Train meta-model for stacking
            self._train_meta_model(X_train, y_train, X_val, y_val)
        elif self.ensemble_method == "weighted_average":
            # Optimize weights based on validation performance
            if X_val is not None and y_val is not None:
                self._optimize_weights(X_val, y_val)
        
        self.is_trained = True
        
        training_history = {
            'ensemble_method': self.ensemble_method,
            'base_models': list(self.base_models.keys()),
            'weights': self.weights
        }
        
        self.training_history = training_history
        logger.info("Ensemble training completed")
        
        return training_history
    
    def _train_meta_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> None:
        """Train meta-model for stacking ensemble."""
        logger.info("Training meta-model for stacking")
        
        # Get base model predictions
        base_predictions_train = self._get_base_predictions(X_train)
        
        # Initialize meta-model
        if self.task_type == "classification":
            self.meta_model = LogisticRegression(random_state=42)
        else:
            self.meta_model = LinearRegression()
        
        # Train meta-model
        self.meta_model.fit(base_predictions_train, y_train)
        
        # Validate if validation data is provided
        if X_val is not None and y_val is not None:
            base_predictions_val = self._get_base_predictions(X_val)
            val_predictions = self.meta_model.predict(base_predictions_val)
            
            if self.task_type == "classification":
                val_score = accuracy_score(y_val, (val_predictions > 0.5).astype(int))
                logger.info(f"Meta-model validation accuracy: {val_score:.4f}")
            else:
                val_score = mean_squared_error(y_val, val_predictions)
                logger.info(f"Meta-model validation MSE: {val_score:.4f}")
    
    def _optimize_weights(self, X_val: np.ndarray, y_val: np.ndarray) -> None:
        """Optimize ensemble weights based on validation performance."""
        logger.info("Optimizing ensemble weights")
        
        from scipy.optimize import minimize
        
        def objective(weights):
            # Normalize weights
            weights = weights / np.sum(weights)
            
            # Get weighted predictions
            predictions = self._weighted_average_predict(X_val, weights)
            
            # Calculate loss
            if self.task_type == "classification":
                # Use log loss for classification
                predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
                loss = -np.mean(y_val * np.log(predictions) + (1 - y_val) * np.log(1 - predictions))
            else:
                # Use MSE for regression
                loss = mean_squared_error(y_val, predictions)
            
            return loss
        
        # Initial weights
        n_models = len(self.base_models)
        initial_weights = np.ones(n_models) / n_models
        
        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=[(0, 1) for _ in range(n_models)],
            constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        )
        
        if result.success:
            optimized_weights = result.x
            model_names = list(self.base_models.keys())
            
            for i, name in enumerate(model_names):
                self.weights[name] = optimized_weights[i]
            
            logger.info(f"Optimized weights: {dict(zip(model_names, optimized_weights))}")
        else:
            logger.warning("Weight optimization failed, using equal weights")
    
    def _get_base_predictions(self, X: np.ndarray) -> np.ndarray:
        """Get predictions from all base models."""
        predictions = []
        
        for name, model in self.base_models.items():
            pred = model.predict(X)
            predictions.append(pred)
        
        return np.column_stack(predictions)
    
    def _weighted_average_predict(self, X: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
        """Make weighted average predictions."""
        if weights is None:
            weights = np.array([self.weights[name] for name in self.base_models.keys()])
        
        predictions = self._get_base_predictions(X)
        weighted_pred = np.average(predictions, axis=1, weights=weights)
        
        return weighted_pred
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make ensemble predictions.
        
        Args:
            X: Input features
            
        Returns:
            Ensemble predictions
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions")
        
        if not self.base_models:
            raise ValueError("No base models in ensemble")
        
        self.validate_input(X)
        
        if self.ensemble_method == "simple_average":
            predictions = self._get_base_predictions(X)
            return np.mean(predictions, axis=1)
        
        elif self.ensemble_method == "weighted_average":
            return self._weighted_average_predict(X)
        
        elif self.ensemble_method == "voting":
            if self.task_type == "classification":
                # Majority voting for classification
                predictions = self._get_base_predictions(X)
                binary_predictions = (predictions > 0.5).astype(int)
                return np.mean(binary_predictions, axis=1)
            else:
                # Average for regression
                predictions = self._get_base_predictions(X)
                return np.mean(predictions, axis=1)
        
        elif self.ensemble_method == "stacking":
            if self.meta_model is None:
                raise ValueError("Meta-model not trained for stacking")
            
            base_predictions = self._get_base_predictions(X)
            return self.meta_model.predict(base_predictions)
        
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
    
    def get_model_contributions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get individual model contributions to ensemble predictions.
        
        Args:
            X: Input features
            
        Returns:
            Dictionary mapping model names to their predictions
        """
        contributions = {}
        
        for name, model in self.base_models.items():
            contributions[name] = model.predict(X)
        
        return contributions
    
    def get_model_summary(self) -> str:
        """Get ensemble model summary."""
        summary = f"Ensemble Model ({self.task_type})\n"
        summary += f"Method: {self.ensemble_method}\n"
        summary += f"Base Models: {len(self.base_models)}\n\n"
        
        for name, weight in self.weights.items():
            summary += f"  {name}: weight={weight:.4f}\n"
        
        if self.meta_model is not None:
            summary += f"\nMeta-model: {type(self.meta_model).__name__}\n"
        
        return summary
    
    def save_model(self, filepath: Optional[str] = None) -> str:
        """
        Save ensemble model.
        
        Args:
            filepath: Custom filepath (optional)
            
        Returns:
            Path where model was saved
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before saving")
        
        if filepath is None:
            filepath = self.model_dir / f"{self.model_name}_ensemble.pkl"
        
        try:
            # Save ensemble configuration
            ensemble_data = {
                'ensemble_method': self.ensemble_method,
                'weights': self.weights,
                'meta_model': self.meta_model,
                'task_type': self.task_type,
                'base_model_names': list(self.base_models.keys())
            }
            
            joblib.dump(ensemble_data, filepath)
            
            # Save metadata
            metadata = {
                'model_name': self.model_name,
                'task_type': self.task_type,
                'is_trained': self.is_trained,
                'ensemble_method': self.ensemble_method,
                'base_models': list(self.base_models.keys()),
                'weights': self.weights
            }
            
            import json
            metadata_path = str(filepath).replace('.pkl', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"Ensemble model saved to {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error saving ensemble model: {e}")
            raise
    
    def load_model(self, filepath: str, base_models: Dict[str, BaseStockModel]) -> None:
        """
        Load ensemble model.
        
        Args:
            filepath: Path to saved ensemble model
            base_models: Dictionary of base models
        """
        try:
            # Load ensemble configuration
            ensemble_data = joblib.load(filepath)
            
            self.ensemble_method = ensemble_data['ensemble_method']
            self.weights = ensemble_data['weights']
            self.meta_model = ensemble_data['meta_model']
            self.task_type = ensemble_data['task_type']
            
            # Set base models
            for name in ensemble_data['base_model_names']:
                if name in base_models:
                    self.base_models[name] = base_models[name]
                else:
                    logger.warning(f"Base model {name} not provided")
            
            self.is_trained = True
            
            logger.info(f"Ensemble model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading ensemble model: {e}")
            raise
