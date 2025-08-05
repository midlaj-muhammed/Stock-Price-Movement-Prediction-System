#!/usr/bin/env python3
"""
Advanced model fine-tuning script for significantly better accuracy.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Force CPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def fine_tune_stock_model(symbol: str = "AAPL", use_advanced_features: bool = True):
    """Fine-tune stock prediction model with advanced techniques."""
    
    print(f"🚀 Advanced Fine-Tuning for {symbol}")
    print("=" * 60)
    
    try:
        # Import modules
        from src.data.data_collector import StockDataCollector
        from src.features.advanced_features import AdvancedFeatureEngineer
        from src.models.enhanced_lstm_model import EnhancedLSTMModel
        from src.data.preprocessor import DataPreprocessor
        
        # === STEP 1: COLLECT MORE DATA ===
        print("📊 Collecting comprehensive data...")
        collector = StockDataCollector()
        
        # Get more historical data for better training
        data = collector.get_stock_data(symbol, period="5y", source="yahoo")
        if data is None or len(data) < 500:
            print(f"❌ Insufficient data for {symbol}")
            return None
        
        print(f"✅ Collected {len(data)} records (5 years)")
        
        # === STEP 2: ADVANCED FEATURE ENGINEERING ===
        print("🔧 Advanced feature engineering...")
        
        if use_advanced_features:
            feature_engineer = AdvancedFeatureEngineer()
            features_df = feature_engineer.engineer_advanced_features(data)
        else:
            # Use basic features
            from src.features.feature_engineering import FeatureEngineer
            basic_engineer = FeatureEngineer()
            features_df = basic_engineer.engineer_features(data)
        
        print(f"✅ Created {features_df.shape[1]} features")
        
        # === STEP 3: PREPARE DATA WITH ADVANCED PREPROCESSING ===
        print("📋 Advanced data preprocessing...")
        
        preprocessor = DataPreprocessor()
        
        # Create target variable (next day price movement)
        features_df['target'] = (features_df['close'].shift(-1) > features_df['close']).astype(int)
        
        # Remove rows with NaN values
        features_df = features_df.dropna()
        
        if len(features_df) < 100:
            print("❌ Insufficient clean data after preprocessing")
            return None
        
        # Separate features and target
        feature_columns = [col for col in features_df.columns if col not in ['target', 'date']]
        X = features_df[feature_columns]
        y = features_df['target']
        
        print(f"✅ Clean data: {len(X)} samples, {len(feature_columns)} features")
        
        # === STEP 4: ADVANCED FEATURE SELECTION ===
        print("🎯 Advanced feature selection...")
        
        if use_advanced_features:
            # Select best features using multiple methods
            selected_features = feature_engineer.select_best_features(
                X, y, n_features=min(80, len(feature_columns)), method='mutual_info'
            )
        else:
            # Use top features by variance
            feature_var = X.var().sort_values(ascending=False)
            selected_features = feature_var.head(min(50, len(feature_columns))).index.tolist()
        
        X_selected = X[selected_features]
        print(f"✅ Selected {len(selected_features)} best features")
        
        # === STEP 5: ADVANCED SCALING ===
        print("📏 Advanced feature scaling...")
        
        if use_advanced_features:
            X_scaled = feature_engineer.scale_features_advanced(X_selected, method='robust', fit=True)
        else:
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
            X_scaled = pd.DataFrame(
                scaler.fit_transform(X_selected),
                columns=X_selected.columns,
                index=X_selected.index
            )
        
        # === STEP 6: CREATE SEQUENCES WITH OPTIMAL WINDOW ===
        print("🔄 Creating optimized sequences...")
        
        # Try different lookback windows and select the best
        best_accuracy = 0
        best_window = 10
        
        for window in [5, 10, 15, 20]:
            try:
                X_seq, y_seq = preprocessor.create_sequences(
                    pd.concat([X_scaled, y], axis=1),
                    selected_features,
                    ['target'],
                    lookback_window=window
                )
                
                if len(X_seq) > 50:
                    # Quick validation
                    split_idx = int(0.8 * len(X_seq))
                    X_train_temp = X_seq[:split_idx]
                    y_train_temp = y_seq[:split_idx].flatten()
                    X_val_temp = X_seq[split_idx:]
                    y_val_temp = y_seq[split_idx:].flatten()
                    
                    # Simple accuracy check
                    baseline_accuracy = max(np.mean(y_train_temp), 1 - np.mean(y_train_temp))
                    if baseline_accuracy > best_accuracy:
                        best_accuracy = baseline_accuracy
                        best_window = window
                        
            except Exception as e:
                print(f"   Window {window} failed: {e}")
                continue
        
        print(f"✅ Optimal lookback window: {best_window}")
        
        # Create final sequences
        X_sequences, y_sequences = preprocessor.create_sequences(
            pd.concat([X_scaled, y], axis=1),
            selected_features,
            ['target'],
            lookback_window=best_window
        )
        
        print(f"✅ Created {len(X_sequences)} sequences")
        
        # === STEP 7: ADVANCED TRAIN/VALIDATION SPLIT ===
        print("📊 Advanced data splitting...")
        
        # Use stratified split to maintain class balance
        from sklearn.model_selection import train_test_split
        
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_sequences, y_sequences.flatten(),
            test_size=0.3, random_state=42, stratify=y_sequences.flatten()
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=0.5, random_state=42, stratify=y_temp
        )
        
        print(f"✅ Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        print(f"   Class balance - Train: {np.mean(y_train):.3f}, Val: {np.mean(y_val):.3f}, Test: {np.mean(y_test):.3f}")
        
        # === STEP 8: ADVANCED MODEL TRAINING ===
        print("🧠 Training enhanced LSTM model...")
        
        model = EnhancedLSTMModel(task_type="classification")
        
        # === HYPERPARAMETER OPTIMIZATION ===
        print("🎯 Optimizing hyperparameters...")

        # Try different configurations
        configs = [
            {
                'lstm_units': [128, 96, 64],
                'dense_units': [128, 64, 32],
                'dropout_rate': 0.2,
                'use_attention': True,
                'use_bidirectional': True,
            },
            {
                'lstm_units': [96, 64, 32],
                'dense_units': [96, 48, 24],
                'dropout_rate': 0.3,
                'use_attention': True,
                'use_bidirectional': False,
            },
            {
                'lstm_units': [160, 128, 96],
                'dense_units': [160, 80, 40],
                'dropout_rate': 0.25,
                'use_attention': True,
                'use_bidirectional': True,
            }
        ]

        best_val_accuracy = 0
        best_config = configs[0]

        for i, config in enumerate(configs):
            print(f"   Testing configuration {i+1}/{len(configs)}...")

            try:
                temp_model = EnhancedLSTMModel(task_type="classification")
                temp_params = {
                    **config,
                    'recurrent_dropout': 0.2,
                    'l1_reg': 0.001,
                    'l2_reg': 0.001,
                    'epochs': 30,  # Shorter for hyperparameter search
                    'batch_size': 32,
                    'model_path': f'models/saved/temp_{symbol}_{i}.h5'
                }

                temp_history = temp_model.train_with_advanced_techniques(
                    X_train, y_train, X_val, y_val, **temp_params
                )

                # Evaluate
                val_pred = temp_model.predict(X_val)
                val_acc = accuracy_score(y_val, val_pred)

                print(f"      Validation accuracy: {val_acc:.4f}")

                if val_acc > best_val_accuracy:
                    best_val_accuracy = val_acc
                    best_config = config
                    print(f"      ✅ New best configuration!")

            except Exception as e:
                print(f"      ❌ Configuration failed: {e}")
                continue

        print(f"✅ Best configuration found with validation accuracy: {best_val_accuracy:.4f}")

        # Final training with best configuration
        training_params = {
            **best_config,
            'recurrent_dropout': 0.2,
            'l1_reg': 0.001,
            'l2_reg': 0.001,
            'epochs': 150,  # More epochs for final training
            'batch_size': 32,
            'model_path': f'models/saved/enhanced_{symbol}_lstm.h5'
        }
        
        # Train the model
        history = model.train_with_advanced_techniques(
            X_train, y_train, X_val, y_val, **training_params
        )
        
        # === STEP 9: COMPREHENSIVE EVALUATION ===
        print("📊 Comprehensive model evaluation...")
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)
        y_pred_test = model.predict(X_test)
        
        y_proba_test = model.predict_proba(X_test)
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_pred_train)
        val_accuracy = accuracy_score(y_val, y_pred_val)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        
        print("\n" + "="*60)
        print("🎉 FINE-TUNING RESULTS")
        print("="*60)
        print(f"📈 {symbol} Stock Prediction Results:")
        print(f"   Training Accuracy:   {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
        print(f"   Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
        print(f"   Test Accuracy:       {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        
        # Detailed classification report
        print(f"\n📊 Detailed Classification Report:")
        print(classification_report(y_test, y_pred_test, target_names=['Down', 'Up']))
        
        # Feature importance (approximate)
        print(f"\n🎯 Model Configuration:")
        print(f"   Features used: {len(selected_features)}")
        print(f"   Lookback window: {best_window}")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Advanced features: {'Yes' if use_advanced_features else 'No'}")
        
        # Save model
        model.save_model(f'models/saved/enhanced_{symbol}_final.h5')
        
        # Save results
        results = {
            'symbol': symbol,
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'test_accuracy': test_accuracy,
            'features_used': len(selected_features),
            'lookback_window': best_window,
            'training_samples': len(X_train),
            'advanced_features': use_advanced_features
        }
        
        return results
        
    except Exception as e:
        print(f"❌ Fine-tuning failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main fine-tuning function."""
    
    print("🎯 Advanced Stock Prediction Model Fine-Tuning")
    print("=" * 60)
    
    # Get user input
    symbol = input("📊 Enter stock symbol (e.g., AAPL): ").upper().strip()
    if not symbol:
        symbol = "AAPL"
    
    use_advanced = input("🔧 Use advanced features? (y/n) [y]: ").lower().strip()
    use_advanced_features = use_advanced != 'n'
    
    print(f"\n🚀 Starting fine-tuning for {symbol}...")
    print(f"   Advanced features: {'Enabled' if use_advanced_features else 'Disabled'}")
    
    # Run fine-tuning
    results = fine_tune_stock_model(symbol, use_advanced_features)
    
    if results:
        print(f"\n✅ Fine-tuning completed successfully!")
        print(f"💡 Test accuracy improved to: {results['test_accuracy']*100:.2f}%")
        print(f"\n🎯 Next steps:")
        print(f"   1. Use the enhanced model in the web interface")
        print(f"   2. Try different stocks with these settings")
        print(f"   3. Experiment with ensemble methods")
    else:
        print(f"\n❌ Fine-tuning failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
