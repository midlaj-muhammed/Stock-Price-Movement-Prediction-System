#!/usr/bin/env python3
"""
Train models with CPU-only mode to avoid GPU issues.
"""

import os
import sys
from pathlib import Path

# Force CPU usage before any imports
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def main():
    """Train models with CPU-only configuration."""
    
    print("🚀 Stock Price Prediction - CPU Training")
    print("=" * 50)
    print("🔧 GPU disabled to avoid compilation issues")
    print("🧠 Training will use CPU (slower but stable)")
    print("=" * 50)
    
    try:
        # Import after setting environment variables
        from src.training_pipeline import StockPredictionPipeline
        
        # Get user input
        symbol = input("\n📊 Enter stock symbol (e.g., AAPL): ").upper().strip()
        if not symbol:
            symbol = "AAPL"
        
        task_type = input("🎯 Task type (classification/regression) [classification]: ").lower().strip()
        if task_type not in ['classification', 'regression']:
            task_type = 'classification'
        
        print(f"\n🚀 Training {task_type} model for {symbol}...")
        
        # Initialize pipeline
        pipeline = StockPredictionPipeline(symbol, task_type)
        
        # Run training with reduced parameters for stability
        results = pipeline.run_complete_pipeline(
            period="1y",  # 1 year of data
            n_features=30,  # Reduced features
            epochs=10,  # Fewer epochs
            batch_size=16  # Smaller batch size
        )
        
        print("\n🎉 Training completed successfully!")
        print("=" * 40)
        
        # Show results
        for model_name, eval_result in results['evaluation_results'].items():
            print(f"\n{model_name.upper()} Results:")
            if task_type == "classification":
                print(f"  Accuracy:  {eval_result['accuracy']:.4f}")
                print(f"  F1 Score:  {eval_result['f1_score']:.4f}")
                if eval_result.get('roc_auc'):
                    print(f"  ROC AUC:   {eval_result['roc_auc']:.4f}")
            else:
                print(f"  RMSE:      {eval_result['rmse']:.4f}")
                print(f"  R² Score:  {eval_result['r2_score']:.4f}")
                print(f"  MAPE:      {eval_result['mape']:.2f}%")
        
        print(f"\n💾 Results saved to: {results['results_path']}")
        print("\n💡 Next steps:")
        print("   1. Start web interface: python start_web_cpu.py")
        print("   2. Make predictions: python predict_cpu.py")
        print("   3. Try different stocks with this script")
        
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        print("\n🔧 Troubleshooting:")
        print("   1. Check internet connection")
        print("   2. Try a different stock symbol")
        print("   3. Ensure all dependencies are installed")

if __name__ == "__main__":
    main()
