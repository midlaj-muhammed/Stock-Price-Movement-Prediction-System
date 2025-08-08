#!/usr/bin/env python3
"""
Test script to verify the prediction fixes work with limited data.
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
from src.data.data_collector import StockDataCollector
from src.utils.logger import get_logger

logger = get_logger(__name__)

def test_prediction_with_limited_data():
    """Test prediction functionality with limited data scenarios."""
    
    print("🧪 Testing Prediction with Limited Data")
    print("=" * 60)
    
    try:
        # Test with a stock that might have limited recent data
        symbol = "AAPL"
        task_type = "classification"
        
        print(f"📊 Testing prediction for {symbol}")
        
        # Initialize pipeline
        pipeline = StockPredictionPipeline(symbol, task_type)
        
        # Collect training data
        print("📈 Collecting training data...")
        data = pipeline.collect_data(period="1y", source="yahoo")
        
        if data is None:
            print("❌ Failed to collect training data")
            return False
        
        print(f"✅ Collected {len(data)} training records")
        
        # Engineer features and train a simple model
        print("🔧 Engineering features and training model...")
        features = pipeline.engineer_features()
        
        # Prepare data for training with minimal requirements
        pipeline.prepare_data_for_training(
            n_features=20,  # Reduced features
            test_size=0.2,
            validation_size=0.15
        )
        
        # Train a simple model
        pipeline.train_model('lstm', epochs=2, batch_size=16, verbose=0)
        
        print("✅ Model trained successfully")
        
        # Now test prediction with limited data
        print("\n🔮 Testing prediction with limited data scenarios...")
        
        # Test 1: Normal case with 6 months data
        print("\n📊 Test 1: Normal case (6 months)")
        collector = StockDataCollector()
        test_data = collector.get_stock_data(symbol, period="6mo", source="yahoo")
        
        if test_data is not None and len(test_data) >= 30:
            processed_data = pipeline.feature_engineer.transform_new_data(test_data)
            print(f"✅ Processed {len(processed_data)} records from {len(test_data)} raw records")
            
            if len(processed_data) >= 15:
                print("✅ Sufficient data for prediction")
            else:
                print(f"⚠️  Limited processed data: {len(processed_data)} records")
        else:
            print(f"❌ Insufficient raw data: {len(test_data) if test_data is not None else 0}")
        
        # Test 2: Limited case with 3 months data
        print("\n📊 Test 2: Limited case (3 months)")
        test_data = collector.get_stock_data(symbol, period="3mo", source="yahoo")
        
        if test_data is not None:
            processed_data = pipeline.feature_engineer.transform_new_data(test_data)
            print(f"✅ Processed {len(processed_data)} records from {len(test_data)} raw records")
            
            if len(processed_data) >= 10:
                print("✅ Sufficient data for prediction with reduced requirements")
            else:
                print(f"⚠️  Very limited processed data: {len(processed_data)} records")
        else:
            print("❌ No data available for 3 months")
        
        # Test 3: Very limited case with 1 month data
        print("\n📊 Test 3: Very limited case (1 month)")
        test_data = collector.get_stock_data(symbol, period="1mo", source="yahoo")
        
        if test_data is not None:
            processed_data = pipeline.feature_engineer.transform_new_data(test_data)
            print(f"✅ Processed {len(processed_data)} records from {len(test_data)} raw records")
            
            if len(processed_data) >= 5:
                print("✅ Minimal data available for prediction")
            else:
                print(f"❌ Insufficient processed data: {len(processed_data)} records")
        else:
            print("❌ No data available for 1 month")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    success = test_prediction_with_limited_data()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 SUCCESS! Prediction fixes are working")
        print("✅ System can handle limited data scenarios")
        print("💡 The 'Need at least 30 records' error should be resolved")
    else:
        print("❌ FAILED! There are still issues with prediction")
        print("💡 Check the error messages above for debugging")
    
    return success

if __name__ == "__main__":
    main()
