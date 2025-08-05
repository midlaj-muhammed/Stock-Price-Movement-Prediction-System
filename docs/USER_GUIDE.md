# User Guide

This guide provides comprehensive instructions for using the Stock Price Movement Prediction System.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Web Interface](#web-interface)
3. [Command Line Interface](#command-line-interface)
4. [Configuration](#configuration)
5. [Model Training](#model-training)
6. [Making Predictions](#making-predictions)
7. [Model Evaluation](#model-evaluation)
8. [Advanced Usage](#advanced-usage)

## Getting Started

### Quick Start

1. **Install the system** following the [Installation Guide](INSTALLATION.md)
2. **Set up API keys** in your `.env` file
3. **Start the web interface**:
   ```bash
   streamlit run src/web/app.py
   ```
4. **Open your browser** to `http://localhost:8501`

### Basic Workflow

1. **Data Collection**: Fetch historical stock data
2. **Feature Engineering**: Create technical indicators and features
3. **Model Training**: Train LSTM and TCN models
4. **Evaluation**: Assess model performance
5. **Prediction**: Make future price predictions

## Web Interface

The web interface provides an intuitive way to interact with the system.

### Main Dashboard

- **Stock Symbol Input**: Enter any valid stock symbol (e.g., AAPL, GOOGL)
- **Prediction Type**: Choose between classification (up/down) or regression (price value)
- **Data Parameters**: Configure data period and source
- **Model Selection**: Select which models to use

### Training Models

1. **Configure Parameters**:
   - Stock symbol
   - Prediction type
   - Data period (1y, 2y, 5y, max)
   - Number of features (20-100)
   - Training epochs (10-200)

2. **Click "Train Models"**:
   - System will collect data
   - Engineer features
   - Train selected models
   - Display results

3. **View Results**:
   - Training time
   - Model performance metrics
   - Feature importance

### Making Predictions

1. **Ensure models are trained** for your stock symbol
2. **Click "Make Prediction"**:
   - System fetches latest data
   - Processes through trained models
   - Displays predictions with confidence scores

3. **Interpret Results**:
   - **Classification**: Direction (UP/DOWN) with confidence percentage
   - **Regression**: Predicted price with change amount and percentage

### Performance Analysis

1. **Click "Analyze Performance"**:
   - View model comparison table
   - See detailed metrics for each model
   - Examine confusion matrices (classification)
   - Review error analysis (regression)

## Command Line Interface

The CLI provides programmatic access to all system features.

### Basic Commands

```bash
# Train models
python src/main.py --symbol AAPL --mode train --task-type classification

# Make predictions
python src/main.py --symbol AAPL --mode predict

# Evaluate models
python src/main.py --symbol AAPL --mode evaluate

# Compare models
python src/main.py --symbol AAPL --mode compare
```

### Advanced Options

```bash
# Full training with custom parameters
python src/main.py \
    --symbol AAPL \
    --mode train \
    --task-type classification \
    --period 2y \
    --source yahoo \
    --models lstm tcn \
    --n-features 50 \
    --epochs 100 \
    --batch-size 32
```

### Command Reference

| Option | Description | Default | Choices |
|--------|-------------|---------|---------|
| `--symbol` | Stock symbol | Required | Any valid symbol |
| `--mode` | Operation mode | Required | train, predict, evaluate, compare |
| `--task-type` | Prediction task | classification | classification, regression |
| `--period` | Data period | 2y | 1y, 2y, 5y, max |
| `--source` | Data source | yahoo | yahoo, alpha_vantage |
| `--models` | Models to use | lstm tcn | lstm, tcn, ensemble |
| `--n-features` | Number of features | 50 | 20-100 |
| `--epochs` | Training epochs | 50 | 10-200 |
| `--batch-size` | Batch size | 32 | 16, 32, 64, 128 |

## Configuration

### Configuration File

The system uses `config.yaml` for configuration:

```yaml
# Data Sources
data_sources:
  yahoo_finance:
    enabled: true
    period: "2y"
    interval: "1d"

# Model Configuration
models:
  lstm:
    units: [128, 64, 32]
    dropout: 0.2
    epochs: 100
    batch_size: 32
  
  tcn:
    nb_filters: 64
    kernel_size: 3
    epochs: 100
    batch_size: 32
```

### Environment Variables

Set in `.env` file:

```bash
# API Keys
ALPHA_VANTAGE_API_KEY=your_api_key_here

# Storage Paths
MODEL_STORAGE_PATH=models/
DATA_STORAGE_PATH=data/

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/stock_prediction.log
```

## Model Training

### Training Process

1. **Data Collection**:
   - Fetches historical stock data
   - Validates data quality
   - Caches for future use

2. **Feature Engineering**:
   - Calculates technical indicators
   - Creates time-based features
   - Applies feature selection

3. **Data Preparation**:
   - Scales features
   - Creates sequences for time series models
   - Splits into train/validation/test sets

4. **Model Training**:
   - Trains LSTM and TCN models
   - Uses early stopping and learning rate reduction
   - Saves best models

### Training Tips

- **Start Small**: Use fewer epochs for initial testing
- **Monitor Performance**: Watch for overfitting
- **Adjust Parameters**: Tune based on validation results
- **Use GPU**: Enable GPU for faster training

### Model Architecture

#### LSTM Model
- Multiple LSTM layers with dropout
- Batch normalization
- Dense layers for final prediction
- Configurable architecture

#### TCN Model
- Temporal convolutional layers
- Dilated convolutions for long sequences
- Skip connections
- Global pooling

## Making Predictions

### Prediction Types

#### Classification (Up/Down)
- Predicts if stock price will go up or down
- Returns probability score (0-1)
- Threshold of 0.5 for binary decision

#### Regression (Price Value)
- Predicts actual next-day closing price
- Returns continuous value
- Can calculate price change and percentage

### Prediction Process

1. **Data Collection**: Fetch recent stock data
2. **Feature Engineering**: Apply same transformations as training
3. **Model Inference**: Run through trained models
4. **Post-processing**: Convert outputs to interpretable results

### Interpreting Results

#### Classification Results
```
Direction: 📈 UP
Confidence: 78.5%
Probability: 0.785
```

#### Regression Results
```
Predicted Price: $150.25
Current Price: $148.50
Change: +$1.75 (+1.18%)
```

## Model Evaluation

### Evaluation Metrics

#### Classification Metrics
- **Accuracy**: Overall correctness
- **Precision**: True positive rate
- **Recall**: Sensitivity
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve

#### Regression Metrics
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **R² Score**: Coefficient of determination
- **MAPE**: Mean Absolute Percentage Error
- **Directional Accuracy**: Correct direction prediction

#### Trading-Specific Metrics
- **True Positive Rate**: Correctly predicted up movements
- **True Negative Rate**: Correctly predicted down movements
- **Matthews Correlation Coefficient**: Balanced metric

### Performance Analysis

1. **Model Comparison**: Compare different models side-by-side
2. **Error Analysis**: Understand where models fail
3. **Feature Importance**: See which features matter most
4. **Statistical Significance**: Test if differences are meaningful

## Advanced Usage

### Custom Models

Add your own models by extending the base model class:

```python
from src.models.base_model import BaseStockModel

class CustomModel(BaseStockModel):
    def __init__(self, task_type="classification"):
        super().__init__("Custom", task_type)
    
    def build_model(self, input_shape, **kwargs):
        # Implement your model architecture
        pass
    
    def train(self, X_train, y_train, **kwargs):
        # Implement training logic
        pass
    
    def predict(self, X):
        # Implement prediction logic
        pass
```

### Ensemble Methods

Combine multiple models for better performance:

```python
from src.models.ensemble import ModelEnsemble

# Create ensemble
ensemble = ModelEnsemble("classification")

# Add trained models
ensemble.add_model("lstm", lstm_model, weight=0.6)
ensemble.add_model("tcn", tcn_model, weight=0.4)

# Set ensemble method
ensemble.set_ensemble_method("weighted_average")

# Train ensemble (for stacking)
ensemble.train(X_train, y_train, X_val, y_val)

# Make predictions
predictions = ensemble.predict(X_test)
```

### Batch Processing

Process multiple stocks:

```python
from src.training_pipeline import StockPredictionPipeline

symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]
results = {}

for symbol in symbols:
    pipeline = StockPredictionPipeline(symbol, "classification")
    result = pipeline.run_complete_pipeline()
    results[symbol] = result
```

### Custom Features

Add custom technical indicators:

```python
from src.features.technical_indicators import TechnicalIndicators

class CustomIndicators(TechnicalIndicators):
    def calculate_custom_rsi(self, data, window=14):
        # Implement custom RSI calculation
        pass
    
    def calculate_all_indicators(self, data):
        # Call parent method
        data = super().calculate_all_indicators(data)
        
        # Add custom indicators
        data = self.calculate_custom_rsi(data)
        
        return data
```

## Best Practices

### Data Quality
- Always validate data before training
- Handle missing values appropriately
- Check for data leakage

### Model Training
- Use proper train/validation/test splits
- Monitor for overfitting
- Save model checkpoints
- Use early stopping

### Feature Engineering
- Start with basic features
- Add domain-specific indicators
- Use feature selection
- Avoid look-ahead bias

### Evaluation
- Use multiple metrics
- Test on out-of-sample data
- Consider transaction costs
- Validate statistical significance

### Production Deployment
- Monitor model performance
- Retrain regularly
- Handle API failures gracefully
- Log all predictions

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce batch size or sequence length
2. **Training Slow**: Use GPU or reduce model complexity
3. **Poor Performance**: Try different features or hyperparameters
4. **API Errors**: Check API keys and rate limits

### Performance Tips

1. **Use GPU**: Enable CUDA for faster training
2. **Optimize Batch Size**: Find optimal batch size for your hardware
3. **Feature Selection**: Use fewer but more relevant features
4. **Early Stopping**: Prevent overfitting and save time

## Support

For additional help:

1. Check the [FAQ](FAQ.md)
2. Review [API Documentation](API.md)
3. Browse [Examples](../examples/)
4. Submit [GitHub Issues](https://github.com/your-username/stock-price-prediction/issues)
