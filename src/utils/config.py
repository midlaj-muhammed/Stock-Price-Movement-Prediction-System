"""
Configuration management for the stock prediction system.
"""

import os
import yaml
from typing import Dict, Any
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration manager for the stock prediction system."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to the configuration YAML file
        """
        self.config_path = Path(config_path)
        self._config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing configuration file: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation, e.g., 'models.lstm.units')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
    
    def get_data_sources(self) -> Dict[str, Any]:
        """Get data sources configuration."""
        return self.get('data_sources', {})
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get model-specific configuration."""
        return self.get(f'models.{model_name}', {})
    
    def get_features_config(self) -> Dict[str, Any]:
        """Get feature engineering configuration."""
        return self.get('features', {})
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self.get('training', {})
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """Get evaluation configuration."""
        return self.get('evaluation', {})
    
    def get_web_config(self) -> Dict[str, Any]:
        """Get web interface configuration."""
        return self.get('web_interface', {})
    
    @property
    def alpha_vantage_api_key(self) -> str:
        """Get Alpha Vantage API key from environment variables."""
        api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        if not api_key:
            raise ValueError("ALPHA_VANTAGE_API_KEY not found in environment variables")
        return api_key
    
    @property
    def model_storage_path(self) -> Path:
        """Get model storage path."""
        path = os.getenv('MODEL_STORAGE_PATH', 'models/')
        return Path(path)
    
    @property
    def data_storage_path(self) -> Path:
        """Get data storage path."""
        path = os.getenv('DATA_STORAGE_PATH', 'data/')
        return Path(path)
    
    @property
    def log_level(self) -> str:
        """Get logging level."""
        return os.getenv('LOG_LEVEL', 'INFO')
    
    @property
    def log_file(self) -> str:
        """Get log file path."""
        return os.getenv('LOG_FILE', 'logs/stock_prediction.log')

# Global configuration instance
config = Config()
