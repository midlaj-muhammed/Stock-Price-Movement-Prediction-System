"""
Enhanced LSTM model with advanced fine-tuning techniques for better accuracy.
"""

import os
# Force CPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, BatchNormalization, 
    Bidirectional, Attention, MultiHeadAttention,
    LayerNormalization, Add, Input, Concatenate
)
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,
    LearningRateScheduler
)
from tensorflow.keras.regularizers import l1_l2
import warnings
warnings.filterwarnings('ignore')

class EnhancedLSTMModel:
    """Enhanced LSTM model with advanced techniques for better accuracy."""
    
    def __init__(self, task_type: str = "classification"):
        self.task_type = task_type
        self.model = None
        self.history = None
        
    def build_advanced_model(self, input_shape: Tuple[int, int], **kwargs) -> Model:
        """Build an advanced LSTM model with attention and residual connections."""
        
        # Hyperparameters with better defaults
        lstm_units = kwargs.get('lstm_units', [128, 96, 64])
        dense_units = kwargs.get('dense_units', [128, 64, 32])
        dropout_rate = kwargs.get('dropout_rate', 0.3)
        recurrent_dropout = kwargs.get('recurrent_dropout', 0.2)
        l1_reg = kwargs.get('l1_reg', 0.001)
        l2_reg = kwargs.get('l2_reg', 0.001)
        use_attention = kwargs.get('use_attention', True)
        use_bidirectional = kwargs.get('use_bidirectional', True)
        
        # Input layer
        inputs = Input(shape=input_shape, name='input_layer')
        x = inputs
        
        # Add noise for regularization
        x = tf.keras.layers.GaussianNoise(0.01)(x)
        
        # LSTM layers with residual connections
        lstm_outputs = []
        for i, units in enumerate(lstm_units):
            if use_bidirectional:
                lstm_layer = Bidirectional(
                    LSTM(
                        units,
                        return_sequences=True if i < len(lstm_units) - 1 else False,
                        dropout=dropout_rate,
                        recurrent_dropout=recurrent_dropout,
                        kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
                        name=f'bidirectional_lstm_{i}'
                    ),
                    name=f'bidirectional_{i}'
                )
            else:
                lstm_layer = LSTM(
                    units,
                    return_sequences=True if i < len(lstm_units) - 1 else False,
                    dropout=dropout_rate,
                    recurrent_dropout=recurrent_dropout,
                    kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
                    name=f'lstm_{i}'
                )
            
            x = lstm_layer(x)
            x = LayerNormalization(name=f'layer_norm_{i}')(x)
            lstm_outputs.append(x)
            
            # Skip connection for deeper networks
            if i > 0 and i < len(lstm_units) - 1:
                # Reshape for residual connection if needed
                if lstm_outputs[i-1].shape[-1] != x.shape[-1]:
                    residual = Dense(x.shape[-1], name=f'residual_projection_{i}')(lstm_outputs[i-1])
                else:
                    residual = lstm_outputs[i-1]
                x = Add(name=f'residual_add_{i}')([x, residual])
        
        # Attention mechanism
        if use_attention and len(lstm_outputs) > 1:
            # Use the second-to-last LSTM output for attention
            attention_input = lstm_outputs[-2]  # This should have return_sequences=True
            if len(attention_input.shape) == 3:  # (batch, time, features)
                attention = MultiHeadAttention(
                    num_heads=4,
                    key_dim=attention_input.shape[-1] // 4,
                    name='multi_head_attention'
                )(attention_input, attention_input)
                attention = LayerNormalization(name='attention_norm')(attention)
                
                # Global average pooling to reduce to 2D
                attention = tf.keras.layers.GlobalAveragePooling1D(name='attention_pooling')(attention)
                
                # Concatenate with LSTM output
                x = Concatenate(name='attention_concat')([x, attention])
        
        # Dense layers with advanced regularization
        for i, units in enumerate(dense_units):
            x = Dense(
                units,
                activation='relu',
                kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
                name=f'dense_{i}'
            )(x)
            x = BatchNormalization(name=f'batch_norm_{i}')(x)
            x = Dropout(dropout_rate, name=f'dropout_{i}')(x)
        
        # Output layer
        if self.task_type == "classification":
            outputs = Dense(1, activation='sigmoid', name='output')(x)
            loss = 'binary_crossentropy'
            metrics = ['accuracy', 'precision', 'recall']
        else:
            outputs = Dense(1, activation='linear', name='output')(x)
            loss = 'huber'  # More robust than MSE
            metrics = ['mae', 'mse']
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs, name='enhanced_lstm')
        
        # Advanced optimizer
        optimizer = AdamW(
            learning_rate=0.001,
            weight_decay=0.01,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        return model
    
    def create_advanced_callbacks(self, model_path: str) -> List:
        """Create advanced callbacks for better training."""
        
        callbacks = []
        
        # Early stopping with patience
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1,
            mode='min'
        )
        callbacks.append(early_stopping)
        
        # Learning rate scheduling
        def cosine_annealing(epoch, lr):
            """Cosine annealing learning rate schedule."""
            import math
            T_max = 50  # Maximum number of epochs
            eta_min = 1e-6  # Minimum learning rate
            return eta_min + (0.001 - eta_min) * (1 + math.cos(math.pi * epoch / T_max)) / 2
        
        lr_scheduler = LearningRateScheduler(cosine_annealing, verbose=0)
        callbacks.append(lr_scheduler)
        
        # Reduce LR on plateau as backup
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-6,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # Model checkpoint
        checkpoint = ModelCheckpoint(
            model_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
            mode='min'
        )
        callbacks.append(checkpoint)
        
        return callbacks
    
    def train_with_advanced_techniques(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        **kwargs
    ) -> Dict[str, Any]:
        """Train model with advanced techniques."""
        
        # Build advanced model
        self.model = self.build_advanced_model(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            **kwargs
        )
        
        print("Enhanced LSTM Model Architecture:")
        self.model.summary()
        
        # Training parameters
        epochs = kwargs.get('epochs', 100)
        batch_size = kwargs.get('batch_size', 32)
        model_path = kwargs.get('model_path', 'models/saved/enhanced_lstm.h5')
        
        # Create callbacks
        callbacks = self.create_advanced_callbacks(model_path)
        
        # Data augmentation for time series
        def augment_data(X, y, noise_factor=0.01):
            """Add noise augmentation to training data."""
            X_aug = X + np.random.normal(0, noise_factor, X.shape)
            return X_aug, y
        
        # Train with augmentation
        print("Training Enhanced LSTM with advanced techniques...")
        
        # Original training
        history1 = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs // 2,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
            shuffle=True
        )
        
        # Store the history from the first training phase
        self.history = history1.history
        
        return self.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the trained model."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        predictions = self.model.predict(X, verbose=0)
        
        if self.task_type == "classification":
            return (predictions > 0.5).astype(int).flatten()
        else:
            return predictions.flatten()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities for classification."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        if self.task_type != "classification":
            raise ValueError("predict_proba only available for classification")
        
        return self.model.predict(X, verbose=0).flatten()
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save")
        
        self.model.save(filepath)
        print(f"Enhanced LSTM model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        self.model = tf.keras.models.load_model(filepath)
        print(f"Enhanced LSTM model loaded from {filepath}")
