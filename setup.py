"""
Setup script for Stock Price Movement Prediction System.
"""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="stock-price-prediction",
    version="1.0.0",
    author="Stock Prediction Team",
    author_email="team@stockprediction.com",
    description="Comprehensive stock price movement prediction system using LSTM and TCN models",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/stock-price-prediction",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "stock-predict=src.main:main",
            "stock-web=src.web.app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt"],
    },
    zip_safe=False,
    keywords="stock prediction, machine learning, LSTM, TCN, financial analysis, time series",
    project_urls={
        "Bug Reports": "https://github.com/your-username/stock-price-prediction/issues",
        "Source": "https://github.com/your-username/stock-price-prediction",
        "Documentation": "https://stock-price-prediction.readthedocs.io/",
    },
)
