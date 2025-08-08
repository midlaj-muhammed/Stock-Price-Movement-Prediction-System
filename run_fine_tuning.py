#!/usr/bin/env python3
"""
Quick script to run fine-tuning for 80% accuracy.
"""

import os
import sys

# Add the project root to Python path
sys.path.append('.')

# Force CPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():
    """Run the fine-tuning process."""
    print("🚀 Starting Fine-Tuning Process")
    print("=" * 50)
    
    try:
        from fine_tune_to_80_percent import AdvancedModelTuner
        
        # Default to AAPL, but you can change this
        symbol = "AAPL"
        print(f"📈 Fine-tuning model for: {symbol}")
        
        tuner = AdvancedModelTuner(symbol)
        success = tuner.fine_tune_to_target()
        
        if success:
            print(f"\n🎉 SUCCESS! Achieved 80%+ accuracy for {symbol}")
        else:
            print(f"\n⚠️  Could not reach 80% accuracy for {symbol}")
            print("💡 Try running with a different stock symbol")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure you're running from the project root directory")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Check the logs above for more details")

if __name__ == "__main__":
    main()
