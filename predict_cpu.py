#!/usr/bin/env python3
"""
Make predictions with CPU-only mode to avoid GPU issues.
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
    """Make predictions with CPU-only configuration."""
    
    print("🔮 Stock Price Prediction - CPU Prediction")
    print("=" * 50)
    print("🔧 GPU disabled to avoid compilation issues")
    print("📈 Making predictions using CPU")
    print("=" * 50)
    
    try:
        # Import after setting environment variables
        from src.data.data_collector import StockDataCollector
        from src.models.lstm_model import LSTMStockModel
        from src.utils.config import config
        
        # Get user input
        symbol = input("\n📊 Enter stock symbol (e.g., AAPL): ").upper().strip()
        if not symbol:
            symbol = "AAPL"
        
        print(f"\n📈 Getting latest data for {symbol}...")
        
        # Collect recent data
        collector = StockDataCollector()
        data = collector.get_stock_data(symbol, period="1mo", source="yahoo")
        
        if data is None:
            print(f"❌ Failed to get data for {symbol}")
            return
        
        # Show current stock info
        current_price = data['close'].iloc[-1]
        prev_price = data['close'].iloc[-2] if len(data) > 1 else current_price
        price_change = current_price - prev_price
        price_change_pct = (price_change / prev_price) * 100 if prev_price != 0 else 0
        
        print(f"\n📊 Current Stock Info for {symbol}:")
        print("=" * 30)
        print(f"Current Price: ${current_price:.2f}")
        print(f"Price Change:  ${price_change:.2f} ({price_change_pct:+.2f}%)")
        print(f"Volume:        {data['volume'].iloc[-1]:,.0f}")
        print(f"High/Low:      ${data['high'].iloc[-1]:.2f} / ${data['low'].iloc[-1]:.2f}")
        
        # Check for trained models
        model_dir = config.model_storage_path / "saved"
        lstm_path = model_dir / "LSTM" / "LSTM_model.h5"
        
        if lstm_path.exists():
            print(f"\n🧠 Found trained LSTM model for predictions")
            print("📝 Note: For full predictions, use the web interface")
            print("   This demo shows the prediction framework")
        else:
            print(f"\n⚠️  No trained models found for {symbol}")
            print("💡 Train a model first:")
            print(f"   python train_cpu.py")
            return
        
        print(f"\n✅ Prediction system ready for {symbol}")
        print("\n💡 For interactive predictions:")
        print("   1. Start web interface: python start_web_cpu.py")
        print("   2. Open browser to: http://localhost:8501")
        print("   3. Use the prediction interface")
        
    except KeyboardInterrupt:
        print("\n⏹️ Prediction interrupted by user")
    except Exception as e:
        print(f"\n❌ Prediction failed: {e}")
        print("\n🔧 Troubleshooting:")
        print("   1. Train a model first: python train_cpu.py")
        print("   2. Check internet connection")
        print("   3. Try a different stock symbol")

if __name__ == "__main__":
    main()
