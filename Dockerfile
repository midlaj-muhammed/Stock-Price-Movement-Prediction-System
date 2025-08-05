# Stock Price Movement Prediction System - Multi-stage Docker Build
# Optimized for CPU-only TensorFlow deployment

# Build stage
FROM python:3.9-slim as builder

# Set build arguments
ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.9-slim as production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_VISIBLE_DEVICES=-1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create app user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Create application directory
WORKDIR /app

# Create necessary directories
RUN mkdir -p data/raw data/processed models/saved logs && \
    chown -R appuser:appuser /app

# Copy application code
COPY --chown=appuser:appuser . .

# Create .env file with default settings
RUN echo "CUDA_VISIBLE_DEVICES=-1" > .env && \
    echo "TF_CPP_MIN_LOG_LEVEL=2" >> .env && \
    echo "STREAMLIT_SERVER_PORT=8501" >> .env && \
    chown appuser:appuser .env

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command - run web interface
CMD ["streamlit", "run", "src/web/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Alternative commands (can be overridden)
# For training: docker run <image> python fine_tune_simple.py
# For CLI: docker run <image> python src/main.py --symbol AAPL --mode train
# For stock selector: docker run <image> python stock_selector.py

# Labels for metadata
LABEL maintainer="Midlaj Muhammed" \
      version="1.0" \
      description="Stock Price Movement Prediction System" \
      repository="https://github.com/midlaj-muhammed/Stock-Price-Movement-Prediction-System"
