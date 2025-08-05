# 📈 Stock Price Movement Prediction System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive machine learning system for predicting stock price movements using advanced deep learning models including LSTM and TCN architectures. The system features an intuitive web interface, supports 100+ stock symbols across multiple sectors, and provides real-time predictions with technical analysis.

## 🚀 Features

### 📊 **Comprehensive Stock Coverage**
- **100+ Stock Symbols** organized by industry sectors
- **7 Major Categories**: Technology, Finance, Healthcare, Consumer, Energy, Industrial, Real Estate
- **Smart Stock Selection**: Popular stocks, high volatility, and stable dividend stocks
- **Interactive Search**: Find stocks by company name or symbol

### 🧠 **Advanced Machine Learning**
- **LSTM Networks**: Long Short-Term Memory models for time series prediction
- **TCN Models**: Temporal Convolutional Networks for sequence modeling
- **50+ Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages, and more
- **Feature Engineering**: Automated feature selection and preprocessing
- **Multiple Prediction Types**: Classification (Up/Down) and Regression (Price Values)

### 🌐 **User-Friendly Interface**
- **Web Application**: Beautiful Streamlit-based interface
- **Real-time Training**: Watch model training progress live
- **Interactive Charts**: Visualize predictions and performance metrics
- **Stock Selector**: Enhanced UI with category browsing and search
- **Performance Analytics**: Detailed model evaluation and metrics

### ⚡ **Production Ready**
- **CPU Optimized**: Runs efficiently without GPU requirements
- **Robust Error Handling**: Comprehensive validation and user feedback
- **Data Caching**: Intelligent caching for faster performance
- **Containerized**: Docker support for easy deployment
- **Scalable Architecture**: Modular design for easy extension

## 🛠 Technology Stack

- **Python 3.8+**: Core programming language
- **TensorFlow 2.x**: Deep learning framework
- **Streamlit**: Web application framework
- **yfinance**: Real-time stock data collection
- **pandas & numpy**: Data manipulation and analysis
- **scikit-learn**: Machine learning utilities
- **plotly**: Interactive visualizations
- **ta-lib**: Technical analysis library

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Internet connection for data fetching

### Quick Setup

1. **Clone the repository**
```bash
git clone https://github.com/midlaj-muhammed/Stock-Price-Movement-Prediction-System.git
cd Stock-Price-Movement-Prediction-System
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env file with your configuration
```

### Alternative Installation Methods

#### Using Make
```bash
make setup
make install
```

#### Using Docker
```bash
docker-compose up --build
```

## 🎯 Quick Start

### 1. Web Interface (Recommended)
Launch the interactive web application:

```bash
# CPU-only mode (recommended)
CUDA_VISIBLE_DEVICES=-1 streamlit run src/web/app.py

# Or using make
make web
```

Open your browser to `http://localhost:8501` and:
1. Select a stock from 100+ available symbols
2. Choose prediction type (Classification/Regression)
3. Configure training parameters
4. Train your model and make predictions

### 2. Command Line Interface

#### Train a Model
```bash
python src/main.py --symbol AAPL --mode train --task classification
```

#### Make Predictions
```bash
python src/main.py --symbol AAPL --mode predict
```

#### Quick Fine-tuning
```bash
python fine_tune_simple.py
```

### 3. Interactive Stock Selection
```bash
python stock_selector.py
```

## 📋 Usage Examples

### Web Interface Workflow

1. **Stock Selection**: Choose from popular stocks or browse by category
   - 🌟 Popular: AAPL, MSFT, GOOGL, TSLA, NVDA
   - 📂 By Category: Technology, Finance, Healthcare, etc.
   - 🔍 Search: Find stocks by company name

2. **Model Configuration**:
   - Prediction Type: Classification (Up/Down) or Regression (Price)
   - Data Period: 1y, 2y, 5y, or max
   - Model Type: LSTM or TCN
   - Training Parameters: Epochs, batch size, features

3. **Training & Prediction**:
   - Real-time training progress
   - Model performance metrics
   - Interactive prediction charts
   - Downloadable results

### Command Line Examples

#### Basic Training
```bash
# Train LSTM model for Apple stock
python src/main.py --symbol AAPL --mode train --task classification --epochs 50

# Train with custom parameters
python src/main.py --symbol MSFT --mode train --task regression --period 2y --features 30
```

#### Batch Processing
```bash
# Train models for multiple stocks
for symbol in AAPL MSFT GOOGL TSLA; do
    python src/main.py --symbol $symbol --mode train --task classification
done
```

#### Advanced Fine-tuning
```bash
# Interactive fine-tuning with stock selection
python fine_tune_simple.py

# CPU-only training
CUDA_VISIBLE_DEVICES=-1 python train_cpu.py
```

### Python API Usage

```python
from src.training_pipeline import StockPredictionPipeline
from src.data.data_collector import StockDataCollector

# Initialize pipeline
pipeline = StockPredictionPipeline("AAPL", "classification")

# Collect and process data
pipeline.collect_data(period="2y", source="yahoo")
pipeline.engineer_features()
pipeline.prepare_data_for_training(n_features=30)

# Train model
pipeline.train_model("lstm", epochs=50, batch_size=32)

# Evaluate performance
results = pipeline.evaluate_model("lstm")
print(f"Accuracy: {results['accuracy']:.4f}")

# Make predictions
predictions = pipeline.predict_latest("lstm")
```

## 📊 Available Stock Symbols

The system includes 100+ carefully selected stocks across major sectors:

### 🌟 Popular Stocks (Top 20)
AAPL, MSFT, GOOGL, AMZN, TSLA, META, NVDA, NFLX, JPM, JNJ, UNH, PG, HD, BAC, DIS, ADBE, CRM, V, MA, WMT

### 📂 By Category
- **Technology (25)**: AAPL, MSFT, GOOGL, AMZN, TSLA, META, NVDA, etc.
- **Finance (15)**: JPM, BAC, V, MA, BRK.B, GS, MS, etc.
- **Healthcare (15)**: JNJ, UNH, PFE, ABBV, TMO, ABT, etc.
- **Consumer (15)**: WMT, PG, KO, HD, MCD, DIS, etc.
- **Energy (10)**: XOM, CVX, COP, EOG, etc.
- **Industrial (10)**: BA, CAT, GE, MMM, etc.
- **Real Estate (10)**: AMT, PLD, CCI, EQIX, etc.

### ⚡ Special Categories
- **High Volatility**: TSLA, NVDA, AMD, NFLX, ZOOM, etc.
- **Stable/Dividend**: AAPL, MSFT, JNJ, PG, KO, etc.

## 🏗 Project Structure

```
Stock-Price-Movement-Prediction-System/
├── src/
│   ├── data/
│   │   ├── data_collector.py      # Stock data collection
│   │   ├── preprocessor.py        # Data preprocessing
│   │   └── stock_symbols.py       # 100+ stock symbols database
│   ├── features/
│   │   ├── feature_engineering.py # Feature engineering pipeline
│   │   └── technical_indicators.py # 50+ technical indicators
│   ├── models/
│   │   ├── lstm_model.py          # LSTM implementation
│   │   ├── tcn_model.py           # TCN implementation
│   │   └── base_model.py          # Base model class
│   ├── evaluation/
│   │   └── metrics.py             # Evaluation metrics
│   ├── web/
│   │   └── app.py                 # Streamlit web interface
│   ├── training_pipeline.py       # Complete training pipeline
│   └── main.py                    # CLI entry point
├── data/
│   ├── raw/                       # Raw stock data cache
│   └── processed/                 # Processed datasets
├── models/
│   └── saved/                     # Trained model files
├── tests/                         # Unit tests
├── docs/                          # Documentation
├── docker/                        # Docker configuration
├── Makefile                       # Development commands
├── Dockerfile                     # Container definition
├── docker-compose.yml             # Multi-container setup
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment variables template
└── README.md                      # This file
```

## 🔧 Development

### Setting Up Development Environment

```bash
# Clone and setup
git clone https://github.com/midlaj-muhammed/Stock-Price-Movement-Prediction-System.git
cd Stock-Price-Movement-Prediction-System

# Setup development environment
make setup-dev

# Run tests
make test

# Code formatting
make format

# Type checking
make lint
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/test_models.py
python -m pytest tests/test_data.py

# Run with coverage
python -m pytest --cov=src tests/
```

## 📈 Model Performance

### Classification Metrics
- **Accuracy**: 60-75% (varies by stock and market conditions)
- **Precision**: 0.65-0.80
- **Recall**: 0.60-0.75
- **F1-Score**: 0.62-0.77
- **ROC-AUC**: 0.65-0.85

### Regression Metrics
- **RMSE**: 2-8% of stock price
- **MAE**: 1-5% of stock price
- **MAPE**: 3-12%
- **R² Score**: 0.4-0.8

*Note: Performance varies significantly based on market volatility, stock characteristics, and training data quality.*

## 🐳 Docker Deployment

### Quick Start with Docker

```bash
# Build and run with docker-compose
docker-compose up --build

# Access web interface at http://localhost:8501
```

### Manual Docker Build

```bash
# Build image
docker build -t stock-prediction .

# Run container
docker run -p 8501:8501 stock-prediction
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Quick Contribution Steps

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This system is for educational and research purposes only. Stock market predictions are inherently uncertain and should not be used as the sole basis for investment decisions. Always consult with qualified financial professionals before making investment choices.

## 🙏 Acknowledgments

- Yahoo Finance for providing free stock data API
- TensorFlow team for the excellent deep learning framework
- Streamlit for the intuitive web application framework
- The open-source community for various technical analysis libraries

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/midlaj-muhammed/Stock-Price-Movement-Prediction-System/issues)
- **Discussions**: [GitHub Discussions](https://github.com/midlaj-muhammed/Stock-Price-Movement-Prediction-System/discussions)
- **Documentation**: [Wiki](https://github.com/midlaj-muhammed/Stock-Price-Movement-Prediction-System/wiki)

---

**⭐ Star this repository if you find it helpful!**
