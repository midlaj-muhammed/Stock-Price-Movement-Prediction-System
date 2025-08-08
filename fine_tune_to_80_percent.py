#!/usr/bin/env python3
"""
Fine-tune stock prediction models to achieve 80% accuracy.
This script implements advanced techniques to improve model performance.
"""

import os
# Force CPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from src.training_pipeline import StockPredictionPipeline
from src.data.data_collector import StockDataCollector
from src.utils.logger import get_logger

logger = get_logger(__name__)

class AdvancedModelTuner:
    """Advanced model tuning to achieve 80% accuracy."""
    
    def __init__(self, symbol: str = "AAPL"):
        self.symbol = symbol
        self.target_accuracy = 0.80
        self.best_accuracy = 0.0
        self.best_config = None
        
    def get_hyperparameter_configs(self):
        """Define hyperparameter configurations to try."""
        configs = [
            # Configuration 1: Deeper network with more regularization
            {
                'name': 'deep_regularized',
                'units': [256, 128, 64, 32],
                'dropout': 0.3,
                'recurrent_dropout': 0.3,
                'learning_rate': 0.0005,
                'l2_reg': 0.01,
                'epochs': 100,
                'batch_size': 32,
                'n_features': 75
            },
            # Configuration 2: Wide network with moderate depth
            {
                'name': 'wide_moderate',
                'units': [512, 256, 128],
                'dropout': 0.25,
                'recurrent_dropout': 0.25,
                'learning_rate': 0.001,
                'l2_reg': 0.005,
                'epochs': 80,
                'batch_size': 64,
                'n_features': 100
            },
            # Configuration 3: Balanced approach with attention-like mechanism
            {
                'name': 'balanced_attention',
                'units': [384, 192, 96],
                'dropout': 0.2,
                'recurrent_dropout': 0.2,
                'learning_rate': 0.0008,
                'l2_reg': 0.008,
                'epochs': 120,
                'batch_size': 48,
                'n_features': 85
            },
            # Configuration 4: Conservative but stable
            {
                'name': 'conservative_stable',
                'units': [128, 64],
                'dropout': 0.4,
                'recurrent_dropout': 0.4,
                'learning_rate': 0.0003,
                'l2_reg': 0.02,
                'epochs': 150,
                'batch_size': 16,
                'n_features': 60
            },
            # Configuration 5: High capacity with strong regularization
            {
                'name': 'high_capacity',
                'units': [768, 384, 192, 96],
                'dropout': 0.35,
                'recurrent_dropout': 0.35,
                'learning_rate': 0.0002,
                'l2_reg': 0.015,
                'epochs': 200,
                'batch_size': 24,
                'n_features': 120
            }
        ]
        return configs
    
    def train_with_config(self, config):
        """Train model with specific configuration."""
        print(f"\n{'='*60}")
        print(f"🚀 Training with configuration: {config['name']}")
        print(f"{'='*60}")
        
        try:
            # Initialize pipeline
            pipeline = StockPredictionPipeline(self.symbol, "classification")
            
            # Collect more data for better training
            print("📊 Collecting extended dataset...")
            data = pipeline.collect_data(period="5y", source="yahoo")
            if data is None or len(data) < 500:
                print(f"❌ Insufficient data: {len(data) if data is not None else 0} records")
                return None
            
            print(f"✅ Collected {len(data)} records")
            
            # Engineer features with specified count
            print(f"🔧 Engineering {config['n_features']} features...")
            features = pipeline.engineer_features()
            
            # Prepare data with enhanced preprocessing
            print("🔄 Preparing training data...")
            pipeline.prepare_data_for_training(
                n_features=config['n_features'],
                test_size=0.2,
                validation_size=0.15
            )
            
            # Train LSTM model with custom configuration
            print(f"🧠 Training LSTM model...")
            training_kwargs = {
                'epochs': config['epochs'],
                'batch_size': config['batch_size'],
                'verbose': 1,
                'units': config['units'],
                'dropout': config['dropout'],
                'recurrent_dropout': config['recurrent_dropout'],
                'learning_rate': config['learning_rate'],
                'l2_reg': config['l2_reg']
            }
            
            pipeline.train_model('lstm', **training_kwargs)
            
            # Evaluate model
            print("📈 Evaluating model performance...")
            results = pipeline.evaluate_all_models()
            
            if 'lstm' in results:
                accuracy = results['lstm']['accuracy']
                print(f"🎯 Achieved accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
                
                if accuracy > self.best_accuracy:
                    self.best_accuracy = accuracy
                    self.best_config = config.copy()
                    self.best_config['accuracy'] = accuracy
                    
                    # Save best model
                    model_path = f"models/best_model_{self.symbol}_{accuracy:.4f}.pkl"
                    pipeline.save_models()
                    print(f"💾 New best model saved! Accuracy: {accuracy:.4f}")
                
                return {
                    'config': config,
                    'accuracy': accuracy,
                    'results': results['lstm']
                }
            else:
                print("❌ Model training failed")
                return None
                
        except Exception as e:
            print(f"❌ Error in configuration {config['name']}: {e}")
            return None
    
    def fine_tune_to_target(self):
        """Fine-tune models until target accuracy is reached."""
        print(f"🎯 Target accuracy: {self.target_accuracy*100:.1f}%")
        print(f"📈 Training symbol: {self.symbol}")
        
        configs = self.get_hyperparameter_configs()
        results = []
        
        for i, config in enumerate(configs, 1):
            print(f"\n🔄 Configuration {i}/{len(configs)}")
            result = self.train_with_config(config)
            
            if result:
                results.append(result)
                accuracy = result['accuracy']
                
                if accuracy >= self.target_accuracy:
                    print(f"\n🎉 TARGET ACHIEVED! Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
                    print(f"🏆 Best configuration: {config['name']}")
                    break
            
            print(f"📊 Current best: {self.best_accuracy:.4f} ({self.best_accuracy*100:.2f}%)")
        
        # Summary
        print(f"\n{'='*60}")
        print("📋 FINE-TUNING SUMMARY")
        print(f"{'='*60}")
        
        if self.best_accuracy >= self.target_accuracy:
            print(f"✅ SUCCESS! Achieved {self.best_accuracy:.4f} ({self.best_accuracy*100:.2f}%)")
            print(f"🏆 Best configuration: {self.best_config['name']}")
        else:
            print(f"⚠️  Best achieved: {self.best_accuracy:.4f} ({self.best_accuracy*100:.2f}%)")
            print(f"🎯 Target was: {self.target_accuracy:.4f} ({self.target_accuracy*100:.2f}%)")
        
        # Show all results
        if results:
            print(f"\n📊 All Results:")
            for result in sorted(results, key=lambda x: x['accuracy'], reverse=True):
                config_name = result['config']['name']
                accuracy = result['accuracy']
                print(f"  {config_name:20} → {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        return self.best_accuracy >= self.target_accuracy

def main():
    """Main function to run fine-tuning."""
    print("🚀 Advanced Model Fine-Tuning for 80% Accuracy")
    print("=" * 60)
    
    # You can change the symbol here
    symbol = input("Enter stock symbol (default: AAPL): ").strip().upper() or "AAPL"
    
    tuner = AdvancedModelTuner(symbol)
    success = tuner.fine_tune_to_target()
    
    if success:
        print(f"\n🎉 Successfully fine-tuned model for {symbol} to 80%+ accuracy!")
        print("💡 You can now use this model for predictions in the web app.")
    else:
        print(f"\n⚠️  Could not reach 80% accuracy for {symbol}.")
        print("💡 Try with a different stock symbol or adjust hyperparameters.")
        print("💡 Some stocks are inherently harder to predict than others.")

if __name__ == "__main__":
    main()
