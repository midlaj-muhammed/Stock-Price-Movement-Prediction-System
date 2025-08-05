#!/usr/bin/env python3
"""
Quick example runner for the Stock Price Movement Prediction System.
"""

import os
import sys
from pathlib import Path

# Force TensorFlow to use CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def main():
    """Run a quick example to demonstrate the system."""
    
    print("🚀 Stock Price Movement Prediction System")
    print("=" * 50)
    print()
    
    # Check if we're in the right directory
    if not Path("src").exists():
        print("❌ Error: Please run this script from the project root directory")
        print("   Current directory:", os.getcwd())
        sys.exit(1)
    
    # Import after path setup
    try:
        from src.training_pipeline import StockPredictionPipeline
        from src.data.data_collector import StockDataCollector
        print("✅ Successfully imported modules")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Please ensure all dependencies are installed:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    
    # Quick data collection test
    print("\n📊 Testing data collection...")
    try:
        collector = StockDataCollector()
        data = collector.get_stock_data("AAPL", period="5d", source="yahoo")
        
        if data is not None:
            print(f"✅ Successfully collected {len(data)} records for AAPL")
            print(f"   Latest price: ${data['close'].iloc[-1]:.2f}")
        else:
            print("❌ Failed to collect data")
            return
            
    except Exception as e:
        print(f"❌ Data collection failed: {e}")
        return
    
    # Quick training example
    print("\n🚀 Running quick training example...")
    print("   (This will take a few minutes...)")
    
    try:
        # Initialize pipeline
        pipeline = StockPredictionPipeline("AAPL", "classification")
        
        # Collect data
        print("   📊 Collecting data...")
        pipeline.collect_data(period="1y", source="yahoo")
        
        # Engineer features
        print("   🔧 Engineering features...")
        pipeline.engineer_features()
        
        # Prepare data
        print("   📋 Preparing data...")
        pipeline.prepare_data_for_training(n_features=20)
        
        # Train LSTM model (quick training)
        print("   🧠 Training LSTM model...")
        pipeline.train_model("lstm", epochs=5, batch_size=32, verbose=0)
        
        # Evaluate
        print("   📈 Evaluating model...")
        results = pipeline.evaluate_model("lstm")
        
        # Show results
        print("\n🎉 Training completed successfully!")
        print("=" * 30)
        print(f"Accuracy:  {results['accuracy']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall:    {results['recall']:.4f}")
        print(f"F1 Score:  {results['f1_score']:.4f}")
        
        if results.get('roc_auc'):
            print(f"ROC AUC:   {results['roc_auc']:.4f}")
        
        print("\n💡 Next steps:")
        print("   1. Try the web interface: streamlit run src/web/app.py")
        print("   2. Run full training: python src/main.py --symbol AAPL --mode train")
        print("   3. Check out examples: python examples/basic_usage.py")
        print("   4. Read the documentation in docs/")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("\n🔧 Troubleshooting tips:")
        print("   1. Check your internet connection")
        print("   2. Ensure all dependencies are installed")
        print("   3. Try with a different stock symbol")
        print("   4. Check the logs in logs/ directory")
        return
    
    print("\n✅ Example completed successfully!")

if __name__ == "__main__":
    main()
