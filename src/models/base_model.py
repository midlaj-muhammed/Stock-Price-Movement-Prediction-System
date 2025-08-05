"""
Base model class for stock prediction models.
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import joblib
from pathlib import Path
import json

from ..utils.logger import get_logger
from ..utils.config import config

logger = get_logger(__name__)

class BaseStockModel(ABC):
    """Abstract base class for stock prediction models."""
    
    def __init__(self, model_name: str, task_type: str = "classification"):
        """
        Initialize base model.
        
        Args:
            model_name: Name of the model
            task_type: 'classification' or 'regression'
        """
        self.model_name = model_name
        self.task_type = task_type
        self.model = None
        self.is_trained = False
        self.training_history = {}
        self.model_config = {}
        self.feature_columns = []
        
        # Model storage paths
        self.model_dir = config.model_storage_path / "saved" / model_name
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
    @abstractmethod
    def build_model(self, input_shape: Tuple, **kwargs) -> None:
        """Build the model architecture."""
        pass
    
    @abstractmethod
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities (for classification models).
        
        Args:
            X: Input features
            
        Returns:
            Prediction probabilities
        """
        if self.task_type != "classification":
            raise ValueError("predict_proba is only available for classification models")
        
        return self.predict(X)
    
    def save_model(self, filepath: Optional[str] = None) -> str:
        """
        Save the trained model.
        
        Args:
            filepath: Custom filepath (optional)
            
        Returns:
            Path where model was saved
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        if filepath is None:
            filepath = self.model_dir / f"{self.model_name}_model.pkl"
        
        try:
            # Save model
            joblib.dump(self.model, filepath)
            
            # Save metadata
            metadata = {
                'model_name': self.model_name,
                'task_type': self.task_type,
                'is_trained': self.is_trained,
                'model_config': self.model_config,
                'feature_columns': self.feature_columns,
                'training_history': self.training_history
            }
            
            metadata_path = str(filepath).replace('.pkl', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"Model saved to {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model.
        
        Args:
            filepath: Path to the saved model
        """
        try:
            # Load model
            self.model = joblib.load(filepath)
            
            # Load metadata
            metadata_path = filepath.replace('.pkl', '_metadata.json')
            if Path(metadata_path).exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                self.model_name = metadata.get('model_name', self.model_name)
                self.task_type = metadata.get('task_type', self.task_type)
                self.is_trained = metadata.get('is_trained', True)
                self.model_config = metadata.get('model_config', {})
                self.feature_columns = metadata.get('feature_columns', [])
                self.training_history = metadata.get('training_history', {})
            
            logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'model_name': self.model_name,
            'task_type': self.task_type,
            'is_trained': self.is_trained,
            'model_config': self.model_config,
            'feature_columns': self.feature_columns,
            'training_history': self.training_history
        }
    
    def set_feature_columns(self, feature_columns: List[str]) -> None:
        """Set feature column names."""
        self.feature_columns = feature_columns
    
    def validate_input(self, X: np.ndarray) -> None:
        """
        Validate input data.
        
        Args:
            X: Input data to validate
        """
        if not isinstance(X, np.ndarray):
            raise TypeError("Input must be a numpy array")
        
        if X.size == 0:
            raise ValueError("Input array is empty")
        
        if np.isnan(X).any():
            raise ValueError("Input contains NaN values")
        
        if np.isinf(X).any():
            raise ValueError("Input contains infinite values")
    
    def get_training_history(self) -> Dict[str, Any]:
        """Get training history."""
        return self.training_history
    
    def reset_model(self) -> None:
        """Reset the model to untrained state."""
        self.model = None
        self.is_trained = False
        self.training_history = {}
        logger.info(f"Model {self.model_name} has been reset")
    
    @abstractmethod
    def get_model_summary(self) -> str:
        """Get model architecture summary."""
        pass
