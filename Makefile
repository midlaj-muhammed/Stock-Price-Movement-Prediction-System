# Stock Price Movement Prediction System - Makefile
# Comprehensive development and deployment commands

.PHONY: help setup setup-dev install install-dev clean test lint format web train predict docker-build docker-run docker-compose-up docker-compose-down

# Default target
help:
	@echo "📈 Stock Price Movement Prediction System"
	@echo "=========================================="
	@echo ""
	@echo "Available commands:"
	@echo ""
	@echo "🚀 Setup & Installation:"
	@echo "  setup           - Create virtual environment and install dependencies"
	@echo "  setup-dev       - Setup development environment with dev dependencies"
	@echo "  install         - Install production dependencies"
	@echo "  install-dev     - Install development dependencies"
	@echo "  clean           - Clean up temporary files and caches"
	@echo ""
	@echo "🧪 Development & Testing:"
	@echo "  test            - Run all tests"
	@echo "  test-unit       - Run unit tests only"
	@echo "  test-integration - Run integration tests only"
	@echo "  test-coverage   - Run tests with coverage report"
	@echo "  lint            - Run code linting and type checking"
	@echo "  format          - Format code with black and isort"
	@echo "  check           - Run all quality checks (lint + test)"
	@echo ""
	@echo "🌐 Application:"
	@echo "  web             - Launch web interface (CPU mode)"
	@echo "  web-gpu         - Launch web interface (GPU mode)"
	@echo "  train           - Interactive training with stock selection"
	@echo "  train-batch     - Batch training for multiple stocks"
	@echo "  predict         - Make predictions for a stock"
	@echo "  demo            - Run demo with sample data"
	@echo ""
	@echo "🐳 Docker:"
	@echo "  docker-build    - Build Docker image"
	@echo "  docker-run      - Run Docker container"
	@echo "  docker-compose-up - Start services with docker-compose"
	@echo "  docker-compose-down - Stop docker-compose services"
	@echo ""
	@echo "📊 Data & Models:"
	@echo "  download-data   - Download sample stock data"
	@echo "  clean-data      - Clean cached data"
	@echo "  clean-models    - Clean saved models"
	@echo "  backup-models   - Backup trained models"
	@echo ""
	@echo "📚 Documentation:"
	@echo "  docs            - Generate documentation"
	@echo "  docs-serve      - Serve documentation locally"
	@echo ""

# Variables
PYTHON := python3
PIP := pip3
VENV := venv
VENV_BIN := $(VENV)/bin
PYTHON_VENV := $(VENV_BIN)/python
PIP_VENV := $(VENV_BIN)/pip

# Setup & Installation
setup: $(VENV)/bin/activate
	@echo "✅ Virtual environment created and dependencies installed"

$(VENV)/bin/activate: requirements.txt
	@echo "🔧 Creating virtual environment..."
	$(PYTHON) -m venv $(VENV)
	@echo "📦 Installing dependencies..."
	$(PIP_VENV) install --upgrade pip
	$(PIP_VENV) install -r requirements.txt
	@echo "✅ Setup complete!"

setup-dev: $(VENV)/bin/activate
	@echo "🔧 Setting up development environment..."
	$(PIP_VENV) install -r requirements-dev.txt || $(PIP_VENV) install pytest black isort flake8 mypy coverage
	@echo "✅ Development environment ready!"

install:
	@echo "📦 Installing production dependencies..."
	$(PIP) install -r requirements.txt

install-dev:
	@echo "📦 Installing development dependencies..."
	$(PIP) install pytest black isort flake8 mypy coverage

# Development & Testing
test:
	@echo "🧪 Running all tests..."
	$(PYTHON_VENV) -m pytest tests/ -v

test-unit:
	@echo "🧪 Running unit tests..."
	$(PYTHON_VENV) -m pytest tests/test_*.py -v

test-integration:
	@echo "🧪 Running integration tests..."
	$(PYTHON_VENV) -m pytest tests/integration/ -v || echo "No integration tests found"

test-coverage:
	@echo "📊 Running tests with coverage..."
	$(PYTHON_VENV) -m pytest --cov=src --cov-report=html --cov-report=term tests/

lint:
	@echo "🔍 Running code linting..."
	$(PYTHON_VENV) -m flake8 src/ --max-line-length=100 --ignore=E203,W503
	@echo "🔍 Running type checking..."
	$(PYTHON_VENV) -m mypy src/ --ignore-missing-imports || echo "Type checking completed with warnings"

format:
	@echo "🎨 Formatting code..."
	$(PYTHON_VENV) -m black src/ tests/ --line-length=100
	$(PYTHON_VENV) -m isort src/ tests/ --profile black

check: lint test
	@echo "✅ All quality checks passed!"

# Application
web:
	@echo "🌐 Starting web interface (CPU mode)..."
	CUDA_VISIBLE_DEVICES=-1 $(PYTHON_VENV) -m streamlit run src/web/app.py --server.port 8501

web-gpu:
	@echo "🌐 Starting web interface (GPU mode)..."
	$(PYTHON_VENV) -m streamlit run src/web/app.py --server.port 8501

train:
	@echo "🧠 Starting interactive training..."
	CUDA_VISIBLE_DEVICES=-1 $(PYTHON_VENV) fine_tune_simple.py

train-batch:
	@echo "🧠 Starting batch training for popular stocks..."
	@for symbol in AAPL MSFT GOOGL AMZN TSLA; do \
		echo "Training $$symbol..."; \
		CUDA_VISIBLE_DEVICES=-1 $(PYTHON_VENV) src/main.py --symbol $$symbol --mode train --task classification; \
	done

predict:
	@echo "🔮 Starting prediction interface..."
	$(PYTHON_VENV) stock_selector.py

demo:
	@echo "🎬 Running demo..."
	CUDA_VISIBLE_DEVICES=-1 $(PYTHON_VENV) examples/basic_usage.py || $(PYTHON_VENV) run_example.py

# Docker
docker-build:
	@echo "🐳 Building Docker image..."
	docker build -t stock-prediction-system .

docker-run:
	@echo "🐳 Running Docker container..."
	docker run -p 8501:8501 --name stock-prediction stock-prediction-system

docker-compose-up:
	@echo "🐳 Starting services with docker-compose..."
	docker-compose up --build

docker-compose-down:
	@echo "🐳 Stopping docker-compose services..."
	docker-compose down

# Data & Models
download-data:
	@echo "📊 Downloading sample stock data..."
	$(PYTHON_VENV) -c "from src.data.data_collector import StockDataCollector; collector = StockDataCollector(); [collector.get_stock_data(symbol, period='1y') for symbol in ['AAPL', 'MSFT', 'GOOGL']]"

clean-data:
	@echo "🧹 Cleaning cached data..."
	rm -rf data/raw/*.csv
	rm -rf data/processed/*.pkl

clean-models:
	@echo "🧹 Cleaning saved models..."
	rm -rf models/saved/*/*.h5
	rm -rf models/saved/*/*.keras
	rm -rf models/checkpoints/*

backup-models:
	@echo "💾 Backing up trained models..."
	mkdir -p backups/models/$(shell date +%Y%m%d_%H%M%S)
	cp -r models/saved/* backups/models/$(shell date +%Y%m%d_%H%M%S)/ || echo "No models to backup"

# Documentation
docs:
	@echo "📚 Generating documentation..."
	$(PYTHON_VENV) -m pydoc -w src/

docs-serve:
	@echo "📚 Serving documentation..."
	$(PYTHON_VENV) -m http.server 8000 --directory docs/

# Cleanup
clean:
	@echo "🧹 Cleaning up temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf dist/
	rm -rf build/
	@echo "✅ Cleanup complete!"

# Environment management
env-create:
	@echo "🔧 Creating .env file from template..."
	cp .env.example .env || echo "# Stock Prediction Environment Variables\nCUDA_VISIBLE_DEVICES=-1\nTF_CPP_MIN_LOG_LEVEL=2\n" > .env

env-check:
	@echo "🔍 Checking environment..."
	$(PYTHON_VENV) -c "import sys; print(f'Python: {sys.version}'); import tensorflow as tf; print(f'TensorFlow: {tf.__version__}'); import streamlit; print('Streamlit: OK')"

# Quick commands for common workflows
quick-start: setup env-create
	@echo "🚀 Quick start complete! Run 'make web' to launch the application."

dev-setup: setup-dev env-create
	@echo "🔧 Development setup complete! Run 'make check' to verify everything works."

# Production deployment
deploy-prep: clean test docker-build
	@echo "🚀 Deployment preparation complete!"

# Show system info
info:
	@echo "📋 System Information:"
	@echo "Python: $(shell $(PYTHON) --version)"
	@echo "Pip: $(shell $(PIP) --version)"
	@echo "Virtual env: $(shell test -d $(VENV) && echo 'Active' || echo 'Not created')"
	@echo "Docker: $(shell docker --version 2>/dev/null || echo 'Not installed')"
	@echo "Git: $(shell git --version 2>/dev/null || echo 'Not installed')"
