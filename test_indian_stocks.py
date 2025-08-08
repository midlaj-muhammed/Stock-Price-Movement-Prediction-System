#!/usr/bin/env python3
"""
Test script to verify the training fixes work with Indian stocks.
"""

import os
import sys
import warnings

# Force CPU usage and suppress warnings
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append('.')

import numpy as np
import pandas as pd
from src.training_pipeline import StockPredictionPipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)

def test_single_stock(symbol):
    """Test a single stock symbol."""
    task_type = "classification"
    
    try:
        print(f"📊 Testing {symbol}")
        
        # Initialize pipeline
        pipeline = StockPredictionPipeline(symbol, task_type)
        
        # Collect data
        print("📈 Collecting data...")
        data = pipeline.collect_data(period="1y", source="yahoo")
        
        if data is None:
            print("❌ Failed to collect data")
            return False
        
        print(f"✅ Collected {len(data)} records")
        
        # Engineer features
        print("🔧 Engineering features...")
        features = pipeline.engineer_features()
        
        if features is None:
            print("❌ Failed to engineer features")
            return False
        
        print(f"✅ Engineered features. Shape: {features.shape}")
        
        # Check for infinite/NaN values in features
        inf_count = np.isinf(features.select_dtypes(include=[np.number])).sum().sum()
        nan_count = features.select_dtypes(include=[np.number]).isna().sum().sum()
        
        print(f"🔍 Feature validation:")
        print(f"   - Infinite values: {inf_count}")
        print(f"   - NaN values: {nan_count}")
        
        if inf_count > 0 or nan_count > 0:
            print("⚠️  Found infinite or NaN values in features")
            return False
        else:
            print("✅ No infinite or NaN values found")
        
        # Prepare data for training
        print("🔄 Preparing data for training...")
        pipeline.prepare_data_for_training(
            n_features=30,  # Use fewer features for testing
            test_size=0.2,
            validation_size=0.15
        )
        
        print(f"✅ Data preparation completed")
        print(f"   - Training shape: {pipeline.X_train.shape}")
        print(f"   - Validation shape: {pipeline.X_val.shape}")
        print(f"   - Test shape: {pipeline.X_test.shape}")
        
        # Check training data for infinite/NaN values
        train_inf = np.isinf(pipeline.X_train).sum()
        train_nan = np.isnan(pipeline.X_train).sum()
        
        print(f"🔍 Training data validation:")
        print(f"   - Infinite values: {train_inf}")
        print(f"   - NaN values: {train_nan}")
        
        if train_inf > 0 or train_nan > 0:
            print("❌ Found infinite or NaN values in training data")
            return False
        else:
            print("✅ Training data is clean")
        
        # Try a quick training run
        print("🧠 Testing model training...")
        pipeline.train_model(
            'lstm',
            epochs=2,  # Just 2 epochs for testing
            batch_size=32,
            verbose=0
        )
        print("✅ Model training completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ {symbol} test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_indian_stocks():
    """Test the training pipeline with Indian stocks."""
    
    print("🇮🇳 Testing Training Pipeline with Indian Stocks")
    print("=" * 60)
    
    # Test with Indian stocks
    indian_stocks = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"]
    
    results = {}
    
    for symbol in indian_stocks:
        print(f"\n{'='*60}")
        success = test_single_stock(symbol)
        results[symbol] = success
        
        if success:
            print(f"✅ {symbol} test PASSED")
        else:
            print(f"❌ {symbol} test FAILED")
    
    return results

def main():
    """Main test function."""
    results = test_indian_stocks()
    
    print("\n" + "=" * 60)
    print("🇮🇳 INDIAN STOCKS TEST RESULTS")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for symbol, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{symbol:<15} {status}")
        if success:
            passed += 1
    
    print(f"\nSummary: {passed}/{total} stocks passed")
    
    if passed == total:
        print("🎉 SUCCESS! All Indian stocks work correctly")
        print("✅ No more infinity or NaN value errors")
        print("💡 Indian stock market integration is working perfectly")
    else:
        print(f"⚠️  {total - passed} stocks failed")
        print("💡 Some Indian stocks may need additional handling")
    
    return passed == total

if __name__ == "__main__":
    main()
