# Installation Guide

This guide will help you set up the Stock Price Movement Prediction System on your local machine.

## Prerequisites

- Python 3.8 or higher
- Git
- 4GB+ RAM (recommended for model training)
- Internet connection (for data fetching)

## Quick Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd stock-price-prediction
```

### 2. Create Virtual Environment

```bash
# Using venv
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your API keys
nano .env  # or use your preferred editor
```

Required environment variables:
- `ALPHA_VANTAGE_API_KEY`: Get from [Alpha Vantage](https://www.alphavantage.co/support/#api-key)

### 5. Create Required Directories

```bash
mkdir -p data/{raw,processed} models/{saved,checkpoints} logs
```

### 6. Verify Installation

```bash
# Test data collection
python -c "from src.data.data_collector import StockDataCollector; print('✅ Data collector works')"

# Test model imports
python -c "from src.models.lstm_model import LSTMStockModel; print('✅ LSTM model works')"

# Test web app
streamlit run src/web/app.py --server.port 8501
```

## Alternative Installation Methods

### Using pip (if package is published)

```bash
pip install stock-price-prediction
```

### Using Docker

```bash
# Build Docker image
docker build -t stock-prediction .

# Run container
docker run -p 8501:8501 stock-prediction
```

### Using conda

```bash
# Create conda environment
conda create -n stock-prediction python=3.9
conda activate stock-prediction

# Install dependencies
pip install -r requirements.txt
```

## Development Installation

For development and contributing:

```bash
# Clone repository
git clone <repository-url>
cd stock-price-prediction

# Create development environment
python -m venv venv-dev
source venv-dev/bin/activate  # or venv-dev\Scripts\activate on Windows

# Install in development mode
pip install -e .

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Troubleshooting

### Common Issues

#### 1. TensorFlow Installation Issues

```bash
# For CPU-only installation
pip install tensorflow-cpu

# For GPU support (requires CUDA)
pip install tensorflow-gpu
```

#### 2. TA-Lib Installation Issues

**Windows:**
```bash
# Download wheel from https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
pip install TA_Lib-0.4.25-cp39-cp39-win_amd64.whl
```

**macOS:**
```bash
brew install ta-lib
pip install TA-Lib
```

**Linux:**
```bash
sudo apt-get install libta-lib-dev
pip install TA-Lib
```

#### 3. Memory Issues

If you encounter memory issues during training:

```bash
# Reduce batch size in config.yaml
batch_size: 16  # instead of 32

# Or use CPU-only TensorFlow
export CUDA_VISIBLE_DEVICES=""
```

#### 4. API Key Issues

Make sure your Alpha Vantage API key is valid:

```bash
# Test API key
python -c "
import os
from alpha_vantage.timeseries import TimeSeries
ts = TimeSeries(key='YOUR_API_KEY')
data, meta = ts.get_daily('AAPL', outputsize='compact')
print('✅ API key works')
"
```

### Performance Optimization

#### 1. GPU Setup (Optional)

For faster training with GPU support:

```bash
# Install CUDA toolkit (version compatible with TensorFlow)
# Download from: https://developer.nvidia.com/cuda-toolkit

# Install cuDNN
# Download from: https://developer.nvidia.com/cudnn

# Verify GPU availability
python -c "
import tensorflow as tf
print('GPU Available:', tf.config.list_physical_devices('GPU'))
"
```

#### 2. Memory Optimization

```bash
# Set TensorFlow memory growth
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Or limit GPU memory
python -c "
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
"
```

## Verification

After installation, verify everything works:

### 1. Run Tests

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_data_collector.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### 2. Quick Training Test

```bash
# Train a small model
python src/main.py --symbol AAPL --mode train --epochs 5 --batch-size 16
```

### 3. Web Interface Test

```bash
# Start web interface
streamlit run src/web/app.py

# Open browser to http://localhost:8501
```

### 4. Command Line Test

```bash
# Test data collection
python src/main.py --symbol AAPL --mode predict

# Test evaluation
python src/main.py --symbol AAPL --mode evaluate
```

## Next Steps

After successful installation:

1. **Configure API Keys**: Set up your Alpha Vantage API key in `.env`
2. **Read Documentation**: Check out the [User Guide](USER_GUIDE.md)
3. **Try Examples**: Run the example notebooks in `examples/`
4. **Customize Configuration**: Modify `config.yaml` for your needs

## Getting Help

If you encounter issues:

1. Check the [Troubleshooting](#troubleshooting) section
2. Search existing [GitHub Issues](https://github.com/your-username/stock-price-prediction/issues)
3. Create a new issue with:
   - Your operating system
   - Python version
   - Error message
   - Steps to reproduce

## System Requirements

### Minimum Requirements
- Python 3.8+
- 4GB RAM
- 2GB free disk space
- Internet connection

### Recommended Requirements
- Python 3.9+
- 8GB+ RAM
- 10GB+ free disk space
- GPU with 4GB+ VRAM (for faster training)
- Stable internet connection
