"""
LSTM model implementation for stock price prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
import warnings
warnings.filterwarnings('ignore')

# Configure TensorFlow to avoid GPU issues
try:
    # Try to configure GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    # If GPU configuration fails, force CPU usage
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from .base_model import BaseStockModel
from ..utils.logger import get_logger
from ..utils.config import config

logger = get_logger(__name__)

class LSTMStockModel(BaseStockModel):
    """LSTM model for stock price prediction."""
    
    def __init__(self, task_type: str = "classification"):
        """
        Initialize LSTM model.
        
        Args:
            task_type: 'classification' or 'regression'
        """
        super().__init__("LSTM", task_type)
        self.model_config = config.get_model_config("lstm")
        
        # Set random seeds for reproducibility
        tf.random.set_seed(42)
        np.random.seed(42)
    
    def build_model(self, input_shape: Tuple, **kwargs) -> None:
        """
        Build LSTM model architecture.
        
        Args:
            input_shape: Shape of input data (timesteps, features)
            **kwargs: Additional model parameters
        """
        logger.info(f"Building LSTM model for {self.task_type} with input shape: {input_shape}")
        
        # Get model configuration
        units = kwargs.get('units', self.model_config.get('units', [128, 64, 32]))
        dropout = kwargs.get('dropout', self.model_config.get('dropout', 0.2))
        recurrent_dropout = kwargs.get('recurrent_dropout', self.model_config.get('recurrent_dropout', 0.2))
        learning_rate = kwargs.get('learning_rate', self.model_config.get('learning_rate', 0.001))
        l2_reg = kwargs.get('l2_reg', 0.001)
        
        # Build sequential model
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(
            units[0],
            return_sequences=True if len(units) > 1 else False,
            input_shape=input_shape,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            kernel_regularizer=l2(l2_reg),
            name='lstm_1'
        ))
        model.add(BatchNormalization())
        
        # Additional LSTM layers
        for i, unit in enumerate(units[1:], 2):
            return_sequences = i < len(units)  # Return sequences for all but last layer
            model.add(LSTM(
                unit,
                return_sequences=return_sequences,
                dropout=dropout,
                recurrent_dropout=recurrent_dropout,
                kernel_regularizer=l2(l2_reg),
                name=f'lstm_{i}'
            ))
            model.add(BatchNormalization())
        
        # Dense layers
        model.add(Dense(64, activation='relu', kernel_regularizer=l2(l2_reg)))
        model.add(Dropout(dropout))
        model.add(Dense(32, activation='relu', kernel_regularizer=l2(l2_reg)))
        model.add(Dropout(dropout))
        
        # Output layer
        if self.task_type == "classification":
            model.add(Dense(1, activation='sigmoid', name='output'))
            loss = 'binary_crossentropy'
            metrics = ['accuracy']
        else:  # regression
            model.add(Dense(1, activation='linear', name='output'))
            loss = 'mse'
            metrics = ['mae']
        
        # Compile model
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        self.model = model
        self.model_config.update(kwargs)
        
        logger.info("LSTM model built successfully")
        logger.info(f"Model summary:\n{model.summary()}")
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the LSTM model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            **kwargs: Additional training parameters
            
        Returns:
            Training history
        """
        logger.info("Starting LSTM model training")
        
        # Validate inputs
        self.validate_input(X_train)
        self.validate_input(y_train)
        
        if X_val is not None:
            self.validate_input(X_val)
            self.validate_input(y_val)
        
        # Get training configuration
        batch_size = kwargs.get('batch_size', self.model_config.get('batch_size', 32))
        epochs = kwargs.get('epochs', self.model_config.get('epochs', 100))
        patience = kwargs.get('patience', self.model_config.get('patience', 10))
        verbose = kwargs.get('verbose', 1)
        
        # Build model if not already built
        if self.model is None:
            input_shape = (X_train.shape[1], X_train.shape[2])
            self.build_model(input_shape, **kwargs)
        
        # Prepare callbacks
        callbacks = []
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss' if X_val is not None else 'loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Reduce learning rate on plateau
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss' if X_val is not None else 'loss',
            factor=0.5,
            patience=patience//2,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # Model checkpoint
        checkpoint_path = self.model_dir / f"{self.model_name}_checkpoint.h5"
        checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss' if X_val is not None else 'loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
        callbacks.append(checkpoint)
        
        # Prepare validation data
        validation_data = (X_val, y_val) if X_val is not None else None
        
        # Train model
        try:
            history = self.model.fit(
                X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=validation_data,
                callbacks=callbacks,
                verbose=verbose,
                shuffle=True
            )
            
            self.is_trained = True
            self.training_history = history.history
            
            logger.info("LSTM model training completed successfully")
            return history.history
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise
    
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Make predictions with the trained model.

        Args:
            X: Input features
            **kwargs: Additional arguments (verbose will be filtered out)

        Returns:
            Predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        self.validate_input(X)

        try:
            # Filter out 'verbose' argument as it's not supported in predict
            filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'verbose'}
            predictions = self.model.predict(X, verbose=0, **filtered_kwargs)

            # For classification, return probabilities
            if self.task_type == "classification":
                return predictions.flatten()
            else:
                return predictions.flatten()
                
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise
    
    def predict_classes(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict classes for classification tasks.
        
        Args:
            X: Input features
            threshold: Classification threshold
            
        Returns:
            Predicted classes
        """
        if self.task_type != "classification":
            raise ValueError("predict_classes is only available for classification models")
        
        probabilities = self.predict(X)
        return (probabilities > threshold).astype(int)
    
    def get_model_summary(self) -> str:
        """Get model architecture summary."""
        if self.model is None:
            return "Model not built yet"
        
        summary_lines = []
        self.model.summary(print_fn=lambda x: summary_lines.append(x))
        return '\n'.join(summary_lines)
    
    def save_model(self, filepath: Optional[str] = None) -> str:
        """
        Save the trained LSTM model.
        
        Args:
            filepath: Custom filepath (optional)
            
        Returns:
            Path where model was saved
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        if filepath is None:
            filepath = self.model_dir / f"{self.model_name}_model.h5"
        
        try:
            # Save Keras model
            self.model.save(filepath)
            
            # Save metadata
            metadata = {
                'model_name': self.model_name,
                'task_type': self.task_type,
                'is_trained': self.is_trained,
                'model_config': self.model_config,
                'feature_columns': self.feature_columns,
                'training_history': self.training_history
            }
            
            import json
            metadata_path = str(filepath).replace('.h5', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"LSTM model saved to {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error saving LSTM model: {e}")
            raise
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained LSTM model.
        
        Args:
            filepath: Path to the saved model
        """
        try:
            # Load Keras model
            self.model = tf.keras.models.load_model(filepath)
            
            # Load metadata
            import json
            metadata_path = filepath.replace('.h5', '_metadata.json')
            if Path(metadata_path).exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                self.model_name = metadata.get('model_name', self.model_name)
                self.task_type = metadata.get('task_type', self.task_type)
                self.is_trained = metadata.get('is_trained', True)
                self.model_config = metadata.get('model_config', {})
                self.feature_columns = metadata.get('feature_columns', [])
                self.training_history = metadata.get('training_history', {})
            
            logger.info(f"LSTM model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading LSTM model: {e}")
            raise
    
    def plot_training_history(self) -> None:
        """Plot training history."""
        if not self.training_history:
            logger.warning("No training history available")
            return
        
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot loss
        axes[0].plot(self.training_history['loss'], label='Training Loss')
        if 'val_loss' in self.training_history:
            axes[0].plot(self.training_history['val_loss'], label='Validation Loss')
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        
        # Plot metrics
        metric_key = 'accuracy' if self.task_type == 'classification' else 'mae'
        if metric_key in self.training_history:
            axes[1].plot(self.training_history[metric_key], label=f'Training {metric_key.upper()}')
            if f'val_{metric_key}' in self.training_history:
                axes[1].plot(self.training_history[f'val_{metric_key}'], label=f'Validation {metric_key.upper()}')
            axes[1].set_title(f'Model {metric_key.upper()}')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel(metric_key.upper())
            axes[1].legend()
        
        plt.tight_layout()
        plt.show()
