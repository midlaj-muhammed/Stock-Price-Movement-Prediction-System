"""
Unit tests for model implementations.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.lstm_model import LSTMStockModel
from models.tcn_model import TCNStockModel
from models.ensemble import ModelEnsemble

class TestLSTMStockModel:
    """Test cases for LSTM model."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.model = LSTMStockModel("classification")
        
        # Sample data
        self.X_train = np.random.random((100, 60, 10))  # 100 samples, 60 timesteps, 10 features
        self.y_train = np.random.randint(0, 2, 100)     # Binary classification
        self.X_val = np.random.random((20, 60, 10))
        self.y_val = np.random.randint(0, 2, 20)
    
    def test_init_classification(self):
        """Test LSTM model initialization for classification."""
        model = LSTMStockModel("classification")
        assert model.task_type == "classification"
        assert model.model_name == "LSTM"
        assert not model.is_trained
    
    def test_init_regression(self):
        """Test LSTM model initialization for regression."""
        model = LSTMStockModel("regression")
        assert model.task_type == "regression"
        assert model.model_name == "LSTM"
        assert not model.is_trained
    
    def test_build_model_classification(self):
        """Test building LSTM model for classification."""
        input_shape = (60, 10)
        self.model.build_model(input_shape)
        
        assert self.model.model is not None
        assert len(self.model.model.layers) > 0
    
    def test_build_model_regression(self):
        """Test building LSTM model for regression."""
        model = LSTMStockModel("regression")
        input_shape = (60, 10)
        model.build_model(input_shape)
        
        assert model.model is not None
        assert len(model.model.layers) > 0
    
    @patch('tensorflow.keras.models.Sequential.fit')
    def test_train_success(self, mock_fit):
        """Test successful model training."""
        # Mock training history
        mock_history = Mock()
        mock_history.history = {'loss': [0.5, 0.4, 0.3], 'accuracy': [0.7, 0.8, 0.9]}
        mock_fit.return_value = mock_history
        
        # Build model first
        self.model.build_model((60, 10))
        
        # Train model
        history = self.model.train(self.X_train, self.y_train, self.X_val, self.y_val, epochs=3)
        
        assert self.model.is_trained
        assert 'loss' in history
        assert 'accuracy' in history
        mock_fit.assert_called_once()
    
    def test_train_without_model(self):
        """Test training without building model first."""
        # Should build model automatically
        with patch('tensorflow.keras.models.Sequential.fit') as mock_fit:
            mock_history = Mock()
            mock_history.history = {'loss': [0.5], 'accuracy': [0.7]}
            mock_fit.return_value = mock_history
            
            history = self.model.train(self.X_train, self.y_train, epochs=1)
            assert self.model.model is not None
    
    def test_predict_without_training(self):
        """Test prediction without training."""
        with pytest.raises(ValueError, match="Model must be trained"):
            self.model.predict(self.X_val)
    
    @patch('tensorflow.keras.models.Sequential.predict')
    def test_predict_classification(self, mock_predict):
        """Test prediction for classification."""
        # Mock prediction
        mock_predict.return_value = np.array([[0.8], [0.3], [0.9]])
        
        # Set model as trained
        self.model.is_trained = True
        self.model.model = Mock()
        
        predictions = self.model.predict(self.X_val[:3])
        
        assert len(predictions) == 3
        assert all(0 <= p <= 1 for p in predictions)
        mock_predict.assert_called_once()
    
    def test_predict_classes(self):
        """Test class prediction."""
        with patch.object(self.model, 'predict') as mock_predict:
            mock_predict.return_value = np.array([0.8, 0.3, 0.9])
            
            classes = self.model.predict_classes(self.X_val[:3])
            
            assert len(classes) == 3
            assert all(c in [0, 1] for c in classes)
            np.testing.assert_array_equal(classes, [1, 0, 1])
    
    def test_validate_input_valid(self):
        """Test input validation with valid data."""
        # Should not raise exception
        self.model.validate_input(self.X_train)
    
    def test_validate_input_nan(self):
        """Test input validation with NaN values."""
        invalid_data = self.X_train.copy()
        invalid_data[0, 0, 0] = np.nan
        
        with pytest.raises(ValueError, match="Input contains NaN values"):
            self.model.validate_input(invalid_data)
    
    def test_validate_input_inf(self):
        """Test input validation with infinite values."""
        invalid_data = self.X_train.copy()
        invalid_data[0, 0, 0] = np.inf
        
        with pytest.raises(ValueError, match="Input contains infinite values"):
            self.model.validate_input(invalid_data)
    
    def test_get_model_summary_no_model(self):
        """Test getting model summary without building model."""
        summary = self.model.get_model_summary()
        assert "Model not built yet" in summary
    
    def test_get_model_summary_with_model(self):
        """Test getting model summary with built model."""
        self.model.build_model((60, 10))
        summary = self.model.get_model_summary()
        assert len(summary) > 0
        assert "Model not built yet" not in summary

class TestTCNStockModel:
    """Test cases for TCN model."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.model = TCNStockModel("classification")
        self.X_train = np.random.random((100, 60, 10))
        self.y_train = np.random.randint(0, 2, 100)
    
    def test_init(self):
        """Test TCN model initialization."""
        assert self.model.task_type == "classification"
        assert self.model.model_name == "TCN"
        assert not self.model.is_trained
    
    def test_build_model(self):
        """Test building TCN model."""
        input_shape = (60, 10)
        self.model.build_model(input_shape)
        
        assert self.model.model is not None
        assert len(self.model.model.layers) > 0

class TestModelEnsemble:
    """Test cases for model ensemble."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.ensemble = ModelEnsemble("classification")
        
        # Mock base models
        self.mock_model1 = Mock()
        self.mock_model1.is_trained = True
        self.mock_model1.task_type = "classification"
        self.mock_model1.predict.return_value = np.array([0.8, 0.3, 0.9])
        
        self.mock_model2 = Mock()
        self.mock_model2.is_trained = True
        self.mock_model2.task_type = "classification"
        self.mock_model2.predict.return_value = np.array([0.7, 0.4, 0.8])
        
        self.X_test = np.random.random((3, 60, 10))
        self.y_test = np.array([1, 0, 1])
    
    def test_init(self):
        """Test ensemble initialization."""
        assert self.ensemble.task_type == "classification"
        assert self.ensemble.model_name == "Ensemble"
        assert len(self.ensemble.base_models) == 0
    
    def test_add_model_success(self):
        """Test adding model to ensemble."""
        self.ensemble.add_model("model1", self.mock_model1, weight=1.0)
        
        assert "model1" in self.ensemble.base_models
        assert self.ensemble.weights["model1"] == 1.0
    
    def test_add_model_not_trained(self):
        """Test adding untrained model to ensemble."""
        untrained_model = Mock()
        untrained_model.is_trained = False
        
        with pytest.raises(ValueError, match="must be trained"):
            self.ensemble.add_model("untrained", untrained_model)
    
    def test_add_model_wrong_task_type(self):
        """Test adding model with wrong task type."""
        wrong_model = Mock()
        wrong_model.is_trained = True
        wrong_model.task_type = "regression"
        
        with pytest.raises(ValueError, match="task type"):
            self.ensemble.add_model("wrong", wrong_model)
    
    def test_set_ensemble_method(self):
        """Test setting ensemble method."""
        self.ensemble.set_ensemble_method("weighted_average")
        assert self.ensemble.ensemble_method == "weighted_average"
    
    def test_set_invalid_ensemble_method(self):
        """Test setting invalid ensemble method."""
        with pytest.raises(ValueError, match="Invalid ensemble method"):
            self.ensemble.set_ensemble_method("invalid_method")
    
    def test_predict_simple_average(self):
        """Test prediction with simple average."""
        # Add models
        self.ensemble.add_model("model1", self.mock_model1)
        self.ensemble.add_model("model2", self.mock_model2)
        
        # Set method and train
        self.ensemble.set_ensemble_method("simple_average")
        self.ensemble.train(self.X_test, self.y_test)
        
        # Make predictions
        predictions = self.ensemble.predict(self.X_test)
        
        # Should be average of [0.8, 0.3, 0.9] and [0.7, 0.4, 0.8]
        expected = np.array([0.75, 0.35, 0.85])
        np.testing.assert_array_almost_equal(predictions, expected)
    
    def test_predict_weighted_average(self):
        """Test prediction with weighted average."""
        # Add models with different weights
        self.ensemble.add_model("model1", self.mock_model1, weight=0.7)
        self.ensemble.add_model("model2", self.mock_model2, weight=0.3)
        
        # Set method and train
        self.ensemble.set_ensemble_method("weighted_average")
        self.ensemble.train(self.X_test, self.y_test)
        
        # Make predictions
        predictions = self.ensemble.predict(self.X_test)
        
        # Should be weighted average
        pred1 = np.array([0.8, 0.3, 0.9])
        pred2 = np.array([0.7, 0.4, 0.8])
        expected = 0.7 * pred1 + 0.3 * pred2
        
        np.testing.assert_array_almost_equal(predictions, expected)
    
    def test_predict_without_training(self):
        """Test prediction without training ensemble."""
        self.ensemble.add_model("model1", self.mock_model1)
        
        with pytest.raises(ValueError, match="must be trained"):
            self.ensemble.predict(self.X_test)
    
    def test_predict_without_models(self):
        """Test prediction without base models."""
        self.ensemble.is_trained = True
        
        with pytest.raises(ValueError, match="No base models"):
            self.ensemble.predict(self.X_test)
    
    def test_get_model_contributions(self):
        """Test getting individual model contributions."""
        self.ensemble.add_model("model1", self.mock_model1)
        self.ensemble.add_model("model2", self.mock_model2)
        
        contributions = self.ensemble.get_model_contributions(self.X_test)
        
        assert "model1" in contributions
        assert "model2" in contributions
        np.testing.assert_array_equal(contributions["model1"], [0.8, 0.3, 0.9])
        np.testing.assert_array_equal(contributions["model2"], [0.7, 0.4, 0.8])

if __name__ == "__main__":
    pytest.main([__file__])
