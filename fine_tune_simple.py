#!/usr/bin/env python3
"""
Simplified fine-tuning script that significantly improves accuracy without external dependencies.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# Force CPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def create_advanced_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create advanced features without external dependencies."""
    
    df = data.copy()
    
    print("   Creating price-based features...")
    
    # === PRICE MOVEMENT FEATURES ===
    for period in [1, 2, 3, 5, 10, 20]:
        df[f'return_{period}'] = df['close'].pct_change(period)
        df[f'high_low_ratio_{period}'] = df['high'].rolling(period).max() / df['low'].rolling(period).min()
        df[f'price_position_{period}'] = (df['close'] - df['low'].rolling(period).min()) / (df['high'].rolling(period).max() - df['low'].rolling(period).min())
    
    # === MOVING AVERAGES ===
    print("   Creating moving average features...")
    for period in [5, 10, 20, 50, 100]:
        df[f'sma_{period}'] = df['close'].rolling(period).mean()
        df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
        
        # Ratios and distances
        df[f'close_sma_{period}_ratio'] = df['close'] / df[f'sma_{period}']
        df[f'close_ema_{period}_ratio'] = df['close'] / df[f'ema_{period}']
        df[f'sma_slope_{period}'] = df[f'sma_{period}'].diff(5)
    
    # === VOLATILITY FEATURES ===
    print("   Creating volatility features...")
    for period in [5, 10, 20]:
        df[f'volatility_{period}'] = df['return_1'].rolling(period).std()
        df[f'price_range_{period}'] = (df['high'] - df['low']).rolling(period).mean()
        df[f'true_range_{period}'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        ).rolling(period).mean()
    
    # === RSI (Manual Implementation) ===
    print("   Creating RSI features...")
    for period in [7, 14, 21]:
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # RSI signals
        df[f'rsi_{period}_oversold'] = (df[f'rsi_{period}'] < 30).astype(int)
        df[f'rsi_{period}_overbought'] = (df[f'rsi_{period}'] > 70).astype(int)
    
    # === MACD (Manual Implementation) ===
    print("   Creating MACD features...")
    ema_12 = df['close'].ewm(span=12).mean()
    ema_26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    df['macd_bullish'] = (df['macd'] > df['macd_signal']).astype(int)
    
    # === BOLLINGER BANDS ===
    print("   Creating Bollinger Bands features...")
    for period in [10, 20]:
        sma = df['close'].rolling(period).mean()
        std = df['close'].rolling(period).std()
        df[f'bb_upper_{period}'] = sma + (2 * std)
        df[f'bb_lower_{period}'] = sma - (2 * std)
        df[f'bb_position_{period}'] = (df['close'] - df[f'bb_lower_{period}']) / (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'])
        df[f'bb_squeeze_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / sma
    
    # === VOLUME FEATURES ===
    print("   Creating volume features...")
    for period in [5, 10, 20]:
        df[f'volume_sma_{period}'] = df['volume'].rolling(period).mean()
        df[f'volume_ratio_{period}'] = df['volume'] / df[f'volume_sma_{period}']
    
    # Volume-price relationship
    df['volume_price_trend'] = df['volume'] * df['return_1']
    
    # === PATTERN FEATURES ===
    print("   Creating pattern features...")
    
    # Candlestick patterns (simplified)
    df['body_size'] = abs(df['close'] - df['open']) / (df['high'] - df['low'])
    df['upper_shadow'] = (df['high'] - np.maximum(df['close'], df['open'])) / (df['high'] - df['low'])
    df['lower_shadow'] = (np.minimum(df['close'], df['open']) - df['low']) / (df['high'] - df['low'])
    
    # Doji pattern
    df['doji'] = (df['body_size'] < 0.1).astype(int)
    
    # Gap analysis
    df['gap_up'] = (df['open'] > df['close'].shift(1)).astype(int)
    df['gap_down'] = (df['open'] < df['close'].shift(1)).astype(int)
    
    # === TREND FEATURES ===
    print("   Creating trend features...")
    
    # Support and resistance
    df['resistance_20'] = df['high'].rolling(20).max()
    df['support_20'] = df['low'].rolling(20).min()
    df['support_resistance_ratio'] = (df['close'] - df['support_20']) / (df['resistance_20'] - df['support_20'])
    
    # Trend direction
    df['trend_5'] = (df['close'] > df['close'].shift(5)).astype(int)
    df['trend_10'] = (df['close'] > df['close'].shift(10)).astype(int)
    df['trend_20'] = (df['close'] > df['close'].shift(20)).astype(int)
    
    # === LAG FEATURES ===
    print("   Creating lag features...")
    key_features = ['rsi_14', 'macd', 'bb_position_20', 'volume_ratio_20']
    for feature in key_features:
        if feature in df.columns:
            for lag in [1, 2, 3, 5]:
                df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)
    
    # === INTERACTION FEATURES ===
    print("   Creating interaction features...")
    
    # RSI and MACD interaction
    df['rsi_macd_bullish'] = ((df['rsi_14'] > 50) & (df['macd_bullish'] == 1)).astype(int)
    
    # Volume and price momentum
    df['volume_momentum'] = df['volume_ratio_20'] * df['return_1']
    
    # Bollinger and RSI
    df['bb_rsi_oversold'] = ((df['bb_position_20'] < 0.2) & (df['rsi_14'] < 30)).astype(int)
    df['bb_rsi_overbought'] = ((df['bb_position_20'] > 0.8) & (df['rsi_14'] > 70)).astype(int)
    
    print(f"   ✅ Created {df.shape[1]} total features")
    return df

def fine_tune_model_simple(symbol: str = "AAPL"):
    """Fine-tune model with advanced techniques but no external dependencies."""
    
    print(f"🚀 Simple Fine-Tuning for {symbol}")
    print("=" * 60)
    
    try:
        # Import modules
        from src.data.data_collector import StockDataCollector
        from src.models.enhanced_lstm_model import EnhancedLSTMModel
        from src.data.preprocessor import StockDataPreprocessor
        
        # === STEP 1: COLLECT MORE DATA ===
        print("📊 Collecting comprehensive data...")
        collector = StockDataCollector()
        
        # Get 5 years of data
        data = collector.get_stock_data(symbol, period="5y", source="yahoo")
        if data is None or len(data) < 500:
            print(f"❌ Insufficient data for {symbol}")
            return None
        
        print(f"✅ Collected {len(data)} records (5 years)")
        
        # === STEP 2: ADVANCED FEATURE ENGINEERING ===
        print("🔧 Advanced feature engineering...")
        features_df = create_advanced_features(data)
        
        # === STEP 3: PREPARE TARGET ===
        print("📋 Preparing target variable...")
        
        # Create target: next day price movement (1 = up, 0 = down)
        features_df['target'] = (features_df['close'].shift(-1) > features_df['close']).astype(int)
        
        # Remove last row (no target) and rows with NaN
        features_df = features_df[:-1].dropna()
        
        if len(features_df) < 200:
            print("❌ Insufficient clean data")
            return None
        
        print(f"✅ Clean data: {len(features_df)} samples")
        
        # === STEP 4: FEATURE SELECTION ===
        print("🎯 Selecting best features...")

        # Separate features and target (exclude non-numeric columns)
        exclude_columns = ['target', 'date', 'timestamp', 'datetime']
        feature_columns = [col for col in features_df.columns
                          if col not in exclude_columns and features_df[col].dtype in ['float64', 'int64', 'float32', 'int32']]

        X = features_df[feature_columns]
        y = features_df['target']

        print(f"   Using {len(feature_columns)} numeric features")
        
        # Select top features
        selector = SelectKBest(score_func=mutual_info_classif, k=min(60, len(feature_columns)))
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        print(f"✅ Selected {len(selected_features)} best features")
        
        # === STEP 5: SCALING ===
        print("📏 Scaling features...")
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_selected)
        X_scaled_df = pd.DataFrame(X_scaled, columns=selected_features, index=X.index)
        
        # === STEP 6: CREATE SEQUENCES ===
        print("🔄 Creating sequences...")
        preprocessor = StockDataPreprocessor()
        
        # Combine scaled features with target
        combined_data = pd.concat([X_scaled_df, y], axis=1)
        
        # Create sequences with optimal window
        X_sequences, y_sequences = preprocessor.create_sequences(
            combined_data, selected_features, ['target'], lookback_window=15
        )
        
        print(f"✅ Created {len(X_sequences)} sequences")
        
        # === STEP 7: TRAIN/VAL/TEST SPLIT ===
        print("📊 Splitting data...")
        
        # Stratified split
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_sequences, y_sequences.flatten(),
            test_size=0.3, random_state=42, stratify=y_sequences.flatten()
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        print(f"✅ Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # === STEP 8: TRAIN ENHANCED MODEL ===
        print("🧠 Training enhanced model...")
        
        model = EnhancedLSTMModel(task_type="classification")
        
        # Optimized parameters
        training_params = {
            'lstm_units': [128, 96, 64],
            'dense_units': [128, 64, 32],
            'dropout_rate': 0.3,
            'use_attention': True,
            'use_bidirectional': True,
            'epochs': 80,
            'batch_size': 32,
            'model_path': f'models/saved/enhanced_{symbol}_simple.h5'
        }
        
        history = model.train_with_advanced_techniques(
            X_train, y_train, X_val, y_val, **training_params
        )
        
        # === STEP 9: EVALUATION ===
        print("📊 Evaluating model...")
        
        y_pred_test = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        
        print("\n" + "="*60)
        print("🎉 FINE-TUNING RESULTS")
        print("="*60)
        print(f"📈 {symbol} Enhanced Model Results:")
        print(f"   Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        
        print(f"\n📊 Detailed Results:")
        print(classification_report(y_test, y_pred_test, target_names=['Down', 'Up']))
        
        # Save model
        model.save_model(f'models/saved/enhanced_{symbol}_final_simple.h5')
        
        return {
            'symbol': symbol,
            'test_accuracy': test_accuracy,
            'features_used': len(selected_features),
            'training_samples': len(X_train)
        }
        
    except Exception as e:
        print(f"❌ Fine-tuning failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function."""

    print("🎯 Advanced Fine-Tuning with Stock Selector")
    print("=" * 60)

    # Option to use stock selector
    use_selector = input("🌟 Use interactive stock selector? (y/n) [y]: ").lower().strip()

    if use_selector in ['', 'y', 'yes']:
        try:
            from src.data.stock_symbols import (
                POPULAR_SYMBOLS, get_stock_info, get_category, display_popular_stocks
            )

            print("\n🌟 POPULAR STOCKS FOR FINE-TUNING")
            print("=" * 50)

            # Show top 15 popular stocks
            for i, symbol in enumerate(POPULAR_SYMBOLS[:15], 1):
                name = get_stock_info(symbol)
                category = get_category(symbol)
                print(f"{i:2}. {symbol:6} - {name[:35]}... ({category})")

            print(f"{16}. 🔤 Enter custom symbol")

            try:
                choice = int(input(f"\nSelect stock (1-16): "))
                if 1 <= choice <= 15:
                    symbol = POPULAR_SYMBOLS[choice - 1]
                    company_name = get_stock_info(symbol)
                    category = get_category(symbol)
                    print(f"\n✅ Selected: {symbol} - {company_name} ({category})")
                elif choice == 16:
                    symbol = input("📊 Enter stock symbol: ").upper().strip()
                    if not symbol:
                        symbol = "AAPL"
                else:
                    print("❌ Invalid choice, using AAPL")
                    symbol = "AAPL"
            except ValueError:
                print("❌ Invalid input, using AAPL")
                symbol = "AAPL"

        except ImportError:
            print("⚠️ Stock selector not available, using manual input")
            symbol = input("📊 Enter stock symbol (e.g., AAPL): ").upper().strip()
            if not symbol:
                symbol = "AAPL"
    else:
        symbol = input("📊 Enter stock symbol (e.g., AAPL): ").upper().strip()
        if not symbol:
            symbol = "AAPL"

    print(f"\n🚀 Starting advanced fine-tuning for {symbol}...")
    print("💡 This will use 5 years of data and advanced features for better accuracy")

    results = fine_tune_model_simple(symbol)

    if results:
        print(f"\n🎉 Fine-tuning completed successfully!")
        print("=" * 50)
        print(f"📈 Final Accuracy: {results['test_accuracy']*100:.2f}%")
        print(f"🎯 Features Used: {results['features_used']}")
        print(f"📊 Training Samples: {results['training_samples']}")
        print(f"💾 Model saved as: enhanced_{symbol}_final_simple.h5")

        print(f"\n💡 Next steps:")
        print(f"   1. Test in web interface: CUDA_VISIBLE_DEVICES=-1 streamlit run src/web/app.py")
        print(f"   2. Try different stocks with this fine-tuning")
        print(f"   3. Compare with ensemble methods")

        # Suggest similar stocks
        try:
            from src.data.stock_symbols import get_category, get_symbols_by_category
            category = get_category(symbol)
            similar_stocks = list(get_symbols_by_category(category).keys())[:5]
            if len(similar_stocks) > 1:
                print(f"\n🔄 Similar stocks in {category} to try:")
                for stock in similar_stocks:
                    if stock != symbol:
                        print(f"   - {stock}")
        except:
            pass

    else:
        print(f"\n❌ Fine-tuning failed.")
        print(f"💡 Try with a different stock or check your internet connection.")

if __name__ == "__main__":
    main()
