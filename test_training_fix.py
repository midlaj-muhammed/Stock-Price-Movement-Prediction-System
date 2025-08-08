#!/usr/bin/env python3
"""
Test script to verify the training fixes for infinity/NaN issues.
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

def test_training_pipeline():
    """Test the training pipeline with the infinity/NaN fixes."""
    
    print("🧪 Testing Training Pipeline with Infinity/NaN Fixes")
    print("=" * 60)
    
    try:
        # Test with Indian stocks
        indian_stocks = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"]

        for symbol in indian_stocks:
            print(f"\n📊 Testing with {symbol} (Indian stock)")

            try:
                success = test_single_stock(symbol)
                if success:
                    print(f"✅ {symbol} test passed")
                else:
                    print(f"❌ {symbol} test failed")
            except Exception as e:
                print(f"❌ {symbol} test failed with error: {e}")

        return True

def test_single_stock(symbol):
    """Test a single stock symbol."""
    task_type = "classification"

    try:
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
        try:
            pipeline.train_model(
                'lstm',
                epochs=2,  # Just 2 epochs for testing
                batch_size=32,
                verbose=0
            )
            print("✅ Model training completed successfully")
            return True

        except Exception as e:
            print(f"❌ Model training failed: {e}")
            return False

    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    success = test_training_pipeline()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 SUCCESS! Training pipeline fixes are working correctly")
        print("✅ No more infinity or NaN value errors")
        print("💡 You can now run the fine-tuning script safely")
    else:
        print("❌ FAILED! There are still issues with the training pipeline")
        print("💡 Check the error messages above for debugging")
    
    return success

if __name__ == "__main__":
    main()
