"""
Basic usage examples for the Stock Price Movement Prediction System.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from training_pipeline import StockPredictionPipeline
from data.data_collector import StockDataCollector
from models.lstm_model import LSTMStockModel
from models.tcn_model import TCNStockModel
from models.ensemble import ModelEnsemble

def example_1_data_collection():
    """Example 1: Basic data collection."""
    print("Example 1: Data Collection")
    print("=" * 40)
    
    # Initialize data collector
    collector = StockDataCollector()
    
    # Collect data for Apple
    print("Collecting data for AAPL...")
    data = collector.get_stock_data("AAPL", period="1y", source="yahoo")
    
    if data is not None:
        print(f"✅ Successfully collected {len(data)} records")
        print(f"Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
        print(f"Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
        
        # Show recent data
        print("\nRecent data:")
        print(data[['timestamp', 'open', 'high', 'low', 'close', 'volume']].tail())
    else:
        print("❌ Failed to collect data")
    
    print("\n")

def example_2_simple_training():
    """Example 2: Simple model training."""
    print("Example 2: Simple Model Training")
    print("=" * 40)
    
    # Initialize pipeline
    pipeline = StockPredictionPipeline("AAPL", "classification")
    
    try:
        print("Starting training pipeline...")
        
        # Collect data
        print("📊 Collecting data...")
        pipeline.collect_data(period="1y", source="yahoo")
        
        # Engineer features
        print("🔧 Engineering features...")
        pipeline.engineer_features()
        
        # Prepare data
        print("📋 Preparing data...")
        pipeline.prepare_data_for_training(n_features=30)
        
        # Train LSTM model only (faster for example)
        print("🚀 Training LSTM model...")
        pipeline.train_model("lstm", epochs=10, batch_size=32, verbose=0)
        
        # Evaluate model
        print("📈 Evaluating model...")
        results = pipeline.evaluate_model("lstm")
        
        print(f"✅ Training completed!")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"F1 Score: {results['f1_score']:.4f}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
    
    print("\n")

def example_3_model_comparison():
    """Example 3: Compare multiple models."""
    print("Example 3: Model Comparison")
    print("=" * 40)
    
    # Initialize pipeline
    pipeline = StockPredictionPipeline("MSFT", "classification")
    
    try:
        print("Training multiple models for comparison...")
        
        # Run complete pipeline with reduced epochs for speed
        results = pipeline.run_complete_pipeline(
            period="1y",
            n_features=25,
            epochs=5,  # Reduced for example
            batch_size=32
        )
        
        print("✅ Training completed!")
        
        # Compare results
        print("\nModel Comparison:")
        print("-" * 30)
        
        for model_name, eval_result in results['evaluation_results'].items():
            print(f"{model_name.upper()}:")
            print(f"  Accuracy: {eval_result['accuracy']:.4f}")
            print(f"  F1 Score: {eval_result['f1_score']:.4f}")
            print(f"  Precision: {eval_result['precision']:.4f}")
            print(f"  Recall: {eval_result['recall']:.4f}")
        
        # Find best model
        best_model = max(
            results['evaluation_results'].items(),
            key=lambda x: x[1]['f1_score']
        )
        
        print(f"\n🏆 Best Model: {best_model[0].upper()}")
        print(f"   F1 Score: {best_model[1]['f1_score']:.4f}")
        
    except Exception as e:
        print(f"❌ Comparison failed: {e}")
    
    print("\n")

def example_4_ensemble_model():
    """Example 4: Create ensemble model."""
    print("Example 4: Ensemble Model")
    print("=" * 40)
    
    try:
        # Create individual models
        lstm_model = LSTMStockModel("classification")
        tcn_model = TCNStockModel("classification")
        
        # For this example, we'll simulate trained models
        # In practice, you would train them first
        print("Note: This example shows ensemble setup.")
        print("In practice, you would train the base models first.")
        
        # Create ensemble
        ensemble = ModelEnsemble("classification")
        
        # Set ensemble method
        ensemble.set_ensemble_method("weighted_average")
        
        print("✅ Ensemble model created")
        print(f"Ensemble method: {ensemble.ensemble_method}")
        print(f"Task type: {ensemble.task_type}")
        
        # Show ensemble configuration
        print("\nEnsemble Configuration:")
        print(f"- Method: {ensemble.ensemble_method}")
        print(f"- Base models: {len(ensemble.base_models)}")
        
    except Exception as e:
        print(f"❌ Ensemble creation failed: {e}")
    
    print("\n")

def example_5_custom_configuration():
    """Example 5: Custom configuration."""
    print("Example 5: Custom Configuration")
    print("=" * 40)
    
    # Initialize pipeline with custom settings
    pipeline = StockPredictionPipeline("GOOGL", "regression")
    
    try:
        print("Training with custom configuration...")
        
        # Custom training parameters
        custom_params = {
            'period': '6mo',
            'source': 'yahoo',
            'n_features': 40,
            'epochs': 8,
            'batch_size': 16
        }
        
        print(f"Configuration: {custom_params}")
        
        # Run pipeline with custom parameters
        results = pipeline.run_complete_pipeline(**custom_params)
        
        print("✅ Custom training completed!")
        
        # Show regression results
        print("\nRegression Results:")
        print("-" * 30)
        
        for model_name, eval_result in results['evaluation_results'].items():
            print(f"{model_name.upper()}:")
            print(f"  RMSE: {eval_result['rmse']:.4f}")
            print(f"  MAE: {eval_result['mae']:.4f}")
            print(f"  R² Score: {eval_result['r2_score']:.4f}")
            print(f"  MAPE: {eval_result['mape']:.2f}%")
        
    except Exception as e:
        print(f"❌ Custom training failed: {e}")
    
    print("\n")

def example_6_batch_processing():
    """Example 6: Batch processing multiple stocks."""
    print("Example 6: Batch Processing")
    print("=" * 40)
    
    # List of stocks to process
    symbols = ["AAPL", "MSFT", "GOOGL"]
    results = {}
    
    print(f"Processing {len(symbols)} stocks...")
    
    for i, symbol in enumerate(symbols, 1):
        print(f"\n[{i}/{len(symbols)}] Processing {symbol}...")
        
        try:
            # Initialize pipeline
            pipeline = StockPredictionPipeline(symbol, "classification")
            
            # Quick training (reduced parameters for speed)
            result = pipeline.run_complete_pipeline(
                period="6mo",
                n_features=20,
                epochs=3,
                batch_size=32
            )
            
            results[symbol] = result
            print(f"✅ {symbol} completed")
            
        except Exception as e:
            print(f"❌ {symbol} failed: {e}")
            continue
    
    # Summary
    print(f"\n📊 Batch Processing Summary:")
    print("=" * 40)
    
    for symbol, result in results.items():
        print(f"\n{symbol}:")
        for model_name, eval_result in result['evaluation_results'].items():
            print(f"  {model_name}: Accuracy={eval_result['accuracy']:.3f}")
    
    print("\n")

def main():
    """Run all examples."""
    print("Stock Price Prediction System - Examples")
    print("=" * 50)
    print()
    
    # Run examples
    try:
        example_1_data_collection()
        example_2_simple_training()
        example_3_model_comparison()
        example_4_ensemble_model()
        example_5_custom_configuration()
        example_6_batch_processing()
        
        print("🎉 All examples completed!")
        
    except KeyboardInterrupt:
        print("\n⏹️ Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Examples failed: {e}")

if __name__ == "__main__":
    main()
