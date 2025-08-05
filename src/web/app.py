"""
Streamlit web application for stock price prediction.
"""

import os
# Force CPU usage before any TensorFlow imports
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import stock symbols
from src.data.stock_symbols import (
    TOP_STOCKS, ALL_SYMBOLS, POPULAR_SYMBOLS,
    VOLATILE_SYMBOLS, STABLE_SYMBOLS,
    get_stock_info, get_category, search_stocks,
    display_category_menu
)

from src.data.data_collector import StockDataCollector
from src.training_pipeline import StockPredictionPipeline
from src.models.lstm_model import LSTMStockModel
from src.models.tcn_model import TCNStockModel
from src.models.ensemble import ModelEnsemble
from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="Stock Price Prediction System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .prediction-box {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        text-align: center;
        margin: 1rem 0;
    }
    .success-box {
        background: linear-gradient(90deg, #56ab2f 0%, #a8e6cf 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .warning-box {
        background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def create_enhanced_stock_selector():
    """Create an enhanced stock selector with categories and popular stocks."""

    st.subheader("📊 Stock Selection")

    # Selection method
    selection_method = st.radio(
        "Choose selection method:",
        ["🔤 Enter Symbol", "🌟 Popular Stocks", "📂 By Category", "🔍 Search"],
        horizontal=True
    )

    symbol = "AAPL"  # Default

    if selection_method == "🔤 Enter Symbol":
        symbol = st.text_input(
            "Stock Symbol",
            value="AAPL",
            help="Enter a stock symbol (e.g., AAPL, GOOGL, TSLA)"
        ).upper()

    elif selection_method == "🌟 Popular Stocks":
        st.markdown("**Most Popular Stocks:**")

        # Create a more visual selection
        popular_display = []
        for sym in POPULAR_SYMBOLS[:20]:  # Top 20
            name = get_stock_info(sym)
            category = get_category(sym)
            popular_display.append(f"{sym} - {name[:25]}... ({category})")

        selected_option = st.selectbox(
            "Select a popular stock:",
            popular_display,
            help="Choose from the most traded stocks"
        )
        symbol = selected_option.split(" - ")[0]

    elif selection_method == "📂 By Category":
        st.markdown("**Select by Industry:**")

        # Category selection
        categories = list(TOP_STOCKS.keys())
        selected_category = st.selectbox(
            "Choose category:",
            categories,
            help="Browse stocks by industry sector"
        )

        # Stocks in selected category
        category_stocks = TOP_STOCKS[selected_category]
        stock_options = []
        for sym, name in category_stocks.items():
            stock_options.append(f"{sym} - {name[:35]}...")

        selected_stock = st.selectbox(
            f"Select from {selected_category}:",
            stock_options
        )
        symbol = selected_stock.split(" - ")[0]

    elif selection_method == "🔍 Search":
        search_query = st.text_input(
            "Search stocks:",
            placeholder="Enter company name or symbol...",
            help="Search by company name or stock symbol"
        )

        if search_query:
            search_results = search_stocks(search_query)
            if search_results:
                search_options = []
                for result in search_results[:10]:  # Limit to 10 results
                    search_options.append(
                        f"{result['symbol']} - {result['name'][:25]}... ({result['category']})"
                    )

                if search_options:
                    selected_search = st.selectbox(
                        "Search results:",
                        search_options
                    )
                    symbol = selected_search.split(" - ")[0]
                else:
                    st.warning("No stocks found matching your search.")
                    symbol = "AAPL"
            else:
                st.info("Enter a search term to find stocks.")
                symbol = "AAPL"
        else:
            symbol = "AAPL"

    # Display selected stock info
    if symbol and symbol in ALL_SYMBOLS:
        company_name = get_stock_info(symbol)
        category = get_category(symbol)

        st.success(f"✅ Selected: **{symbol}**")
        st.info(f"🏢 **{company_name}**")
        st.info(f"📂 Category: {category}")

        # Quick info badges
        col1, col2 = st.columns(2)

        if symbol in VOLATILE_SYMBOLS:
            col1.warning("⚡ High Volatility")
        if symbol in STABLE_SYMBOLS:
            col2.info("🛡️ Stable Stock")

    elif symbol:
        st.warning(f"⚠️ {symbol} not in our database. You can still use it, but company info won't be available.")

    return symbol

def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">📈 Stock Price Movement Prediction System</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")

        # Enhanced stock selection
        symbol = create_enhanced_stock_selector()
        
        # Task type selection
        task_type = st.selectbox(
            "Prediction Type",
            ["Classification (Up/Down)", "Regression (Price Value)"],
            help="Choose between binary classification or price regression"
        )
        task_type = "classification" if "Classification" in task_type else "regression"
        
        # Data collection parameters
        st.subheader("📊 Data Parameters")
        period = st.selectbox(
            "Data Period",
            ["1y", "2y", "5y", "max"],
            index=1,
            help="Historical data period to collect"
        )
        
        source = st.selectbox(
            "Data Source",
            ["yahoo", "alpha_vantage"],
            help="Choose data source"
        )
        
        # Model selection
        st.subheader("🤖 Model Selection")
        selected_models = st.multiselect(
            "Models to Use",
            ["LSTM", "TCN", "Ensemble"],
            default=["LSTM", "TCN"],
            help="Select models for prediction"
        )
        
        # Advanced options
        with st.expander("⚙️ Advanced Options"):
            n_features = st.slider(
                "Number of Features",
                min_value=20,
                max_value=100,
                value=50,
                help="Number of features to select for training"
            )
            
            epochs = st.slider(
                "Training Epochs",
                min_value=10,
                max_value=200,
                value=50,
                help="Number of training epochs"
            )
            
            batch_size = st.selectbox(
                "Batch Size",
                [16, 32, 64, 128],
                index=1,
                help="Training batch size"
            )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Action buttons
        col_train, col_predict, col_analyze = st.columns(3)
        
        with col_train:
            train_button = st.button("🚀 Train Models", type="primary", use_container_width=True)
        
        with col_predict:
            predict_button = st.button("🔮 Make Prediction", use_container_width=True)
        
        with col_analyze:
            analyze_button = st.button("📊 Analyze Performance", use_container_width=True)
    
    with col2:
        # Quick stats
        st.subheader("📋 Quick Info")
        st.info(f"**Symbol:** {symbol}")
        st.info(f"**Task:** {task_type.title()}")
        st.info(f"**Period:** {period}")
        st.info(f"**Models:** {', '.join(selected_models)}")
    
    # Main content based on button clicks
    if train_button:
        train_models(symbol, task_type, period, source, selected_models, n_features, epochs, batch_size)
    
    elif predict_button:
        make_predictions(symbol, task_type, selected_models)
    
    elif analyze_button:
        analyze_performance(symbol, task_type)
    
    else:
        # Default view - show stock data
        show_stock_overview(symbol, period, source)

def train_models(symbol, task_type, period, source, selected_models, n_features, epochs, batch_size):
    """Train selected models."""
    
    st.header("🚀 Model Training")
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Initialize pipeline
        status_text.text("Initializing training pipeline...")
        pipeline = StockPredictionPipeline(symbol, task_type)
        progress_bar.progress(10)
        
        # Collect data
        status_text.text("Collecting stock data...")
        data = pipeline.collect_data(period=period, source=source)
        progress_bar.progress(30)
        
        # Engineer features
        status_text.text("Engineering features...")
        features = pipeline.engineer_features()
        progress_bar.progress(50)
        
        # Prepare data
        status_text.text("Preparing data for training...")
        pipeline.prepare_data_for_training(n_features=n_features)
        progress_bar.progress(60)
        
        # Train models
        training_kwargs = {
            'epochs': epochs,
            'batch_size': batch_size,
            'verbose': 0
        }
        
        results = {}
        model_progress = 70
        progress_step = 25 / len(selected_models)
        
        for model_name in selected_models:
            if model_name.lower() in ['lstm', 'tcn']:
                status_text.text(f"Training {model_name} model...")
                pipeline.train_model(model_name.lower(), **training_kwargs)
                results[model_name] = pipeline.training_results[model_name.lower()]
                model_progress += progress_step
                progress_bar.progress(int(model_progress))
        
        # Evaluate models
        status_text.text("Evaluating models...")
        evaluation_results = pipeline.evaluate_all_models()
        progress_bar.progress(95)
        
        # Save models
        status_text.text("Saving models...")
        saved_paths = pipeline.save_models()
        progress_bar.progress(100)
        
        # Display results
        status_text.text("Training completed successfully!")
        
        st.markdown('<div class="success-box">✅ Training completed successfully!</div>', 
                   unsafe_allow_html=True)
        
        # Show training summary
        st.subheader("📊 Training Summary")
        
        for model_name, result in results.items():
            with st.expander(f"{model_name} Results"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Training Time", f"{result['training_time']:.2f}s")
                    st.metric("Epochs", epochs)
                    st.metric("Batch Size", batch_size)
                
                with col2:
                    if model_name.lower() in evaluation_results:
                        eval_result = evaluation_results[model_name.lower()]
                        if task_type == "classification":
                            st.metric("Accuracy", f"{eval_result['accuracy']:.4f}")
                            st.metric("F1 Score", f"{eval_result['f1_score']:.4f}")
                        else:
                            st.metric("RMSE", f"{eval_result['rmse']:.4f}")
                            st.metric("R² Score", f"{eval_result['r2_score']:.4f}")
        
        # Store results in session state
        st.session_state[f'{symbol}_{task_type}_results'] = {
            'pipeline': pipeline,
            'evaluation_results': evaluation_results,
            'saved_paths': saved_paths
        }
        
    except Exception as e:
        st.error(f"Training failed: {str(e)}")
        logger.error(f"Training failed for {symbol}: {e}")

def make_predictions(symbol, task_type, selected_models):
    """Make predictions with trained models."""
    
    st.header("🔮 Model Predictions")
    
    # Check if models are trained
    results_key = f'{symbol}_{task_type}_results'
    if results_key not in st.session_state:
        st.warning("⚠️ No trained models found. Please train models first.")
        return
    
    try:
        pipeline = st.session_state[results_key]['pipeline']
        
        # Get latest data for prediction
        collector = StockDataCollector()
        latest_data = collector.get_stock_data(symbol, period="1mo")
        
        if latest_data is None:
            st.error("Failed to fetch latest data for prediction")
            return
        
        # Show current stock info
        current_price = latest_data['close'].iloc[-1]
        prev_price = latest_data['close'].iloc[-2]
        price_change = current_price - prev_price
        price_change_pct = (price_change / prev_price) * 100
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Price", f"${current_price:.2f}")
        
        with col2:
            st.metric("Price Change", f"${price_change:.2f}", f"{price_change_pct:.2f}%")
        
        with col3:
            st.metric("Volume", f"{latest_data['volume'].iloc[-1]:,.0f}")
        
        with col4:
            st.metric("High/Low", f"${latest_data['high'].iloc[-1]:.2f} / ${latest_data['low'].iloc[-1]:.2f}")
        
        # Make predictions
        st.subheader("🎯 Predictions")
        
        # Process latest data through pipeline with robust error handling
        try:
            # Get more data for prediction (use 6 months for better feature engineering)
            latest_data = collector.get_stock_data(symbol, period="6mo", source="yahoo")

            if latest_data is None or len(latest_data) < 50:
                st.error(f"Insufficient recent data for prediction. Need at least 50 records, got {len(latest_data) if latest_data is not None else 0}")
                return

            # Process data through feature engineering
            processed_data = pipeline.feature_engineer.transform_new_data(latest_data)

            # Check if we have enough processed data
            min_required = max(pipeline.preprocessor.lookback_window + 5, 30)
            if len(processed_data) < min_required:
                st.error(f"Insufficient processed data for prediction. Need at least {min_required} records, got {len(processed_data)}")
                st.info("💡 Try training with more historical data or use a different stock symbol.")
                return

            # Get feature columns from the trained model
            if hasattr(pipeline.feature_engineer, 'selected_features') and pipeline.feature_engineer.selected_features:
                feature_columns = pipeline.feature_engineer.selected_features
            elif hasattr(pipeline.preprocessor, 'feature_columns') and pipeline.preprocessor.feature_columns:
                feature_columns = pipeline.preprocessor.feature_columns
            else:
                st.error("No feature columns available. Please train models first.")
                return

            # Ensure all selected features are present
            available_features = [col for col in feature_columns if col in processed_data.columns]

            if len(available_features) < len(feature_columns) * 0.8:  # Need at least 80% of features
                st.warning(f"Only {len(available_features)}/{len(feature_columns)} features available. Prediction may be less accurate.")

                # Use available features only
                feature_columns = available_features

                if len(feature_columns) < 10:  # Need minimum features
                    st.error("Too few features available for reliable prediction.")
                    return

            # Add missing features with appropriate values
            for feature in feature_columns:
                if feature not in processed_data.columns:
                    # Use median of existing similar features or zero
                    processed_data[feature] = 0

            # Select only the required features
            X_features = processed_data[feature_columns].copy()

            # Handle any remaining NaN values
            X_features = X_features.fillna(method='ffill').fillna(method='bfill').fillna(0)

            # Scale the features using the same scaler if available
            try:
                if hasattr(pipeline.preprocessor, 'scaler') and pipeline.preprocessor.scaler is not None:
                    X_scaled = pipeline.preprocessor.scaler.transform(X_features)
                    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_columns, index=X_features.index)
                else:
                    # Use robust scaling as fallback
                    from sklearn.preprocessing import RobustScaler
                    scaler = RobustScaler()
                    X_scaled = scaler.fit_transform(X_features)
                    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_columns, index=X_features.index)
            except Exception as e:
                st.warning(f"Scaling failed, using raw features: {e}")
                X_scaled_df = X_features

            # Create sequences with error handling
            lookback = min(pipeline.preprocessor.lookback_window, len(X_scaled_df) - 1)

            if lookback < 5:
                st.error(f"Insufficient data for sequence creation. Need at least 5 time steps, got {lookback}")
                return

            # Manual sequence creation for prediction
            sequences = []
            for i in range(lookback, len(X_scaled_df)):
                sequence = X_scaled_df.iloc[i-lookback:i].values
                sequences.append(sequence)

            if len(sequences) == 0:
                st.error("Could not create any sequences for prediction.")
                return

            X_pred = np.array(sequences)

            if len(X_pred) > 0:
                X_latest = X_pred[-1:] # Get the most recent sequence
                
                predictions = {}
                
                for model_name in selected_models:
                    if model_name.lower() in pipeline.models:
                        model = pipeline.models[model_name.lower()]
                        if model.is_trained:
                            pred = model.predict(X_latest)[0]
                            predictions[model_name] = pred
                
                # Display predictions
                if predictions:
                    for model_name, pred in predictions.items():
                        if task_type == "classification":
                            direction = "📈 UP" if pred > 0.5 else "📉 DOWN"
                            confidence = max(pred, 1-pred) * 100
                            
                            st.markdown(f"""
                            <div class="prediction-box">
                                <h3>{model_name} Prediction</h3>
                                <h2>{direction}</h2>
                                <p>Confidence: {confidence:.1f}%</p>
                                <p>Probability: {pred:.3f}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        else:
                            predicted_price = pred
                            price_diff = predicted_price - current_price
                            price_diff_pct = (price_diff / current_price) * 100
                            
                            st.markdown(f"""
                            <div class="prediction-box">
                                <h3>{model_name} Prediction</h3>
                                <h2>${predicted_price:.2f}</h2>
                                <p>Change: ${price_diff:.2f} ({price_diff_pct:.2f}%)</p>
                            </div>
                            """, unsafe_allow_html=True)
                
                else:
                    st.warning("No trained models available for prediction")
            
            else:
                st.error("Could not create prediction sequences")
                st.info("💡 This can happen when:")
                st.info("   • There's insufficient recent data")
                st.info("   • The stock has been delisted or suspended")
                st.info("   • Market data is temporarily unavailable")
                st.info("   • Try a different stock symbol or train with more data")

        except Exception as e:
            st.error(f"Prediction processing failed: {str(e)}")
            st.info("🔧 Troubleshooting steps:")
            st.info("   1. Try training the model first")
            st.info("   2. Use a different stock symbol")
            st.info("   3. Check if the stock symbol is valid")
            st.info("   4. Ensure you have internet connection")
            logger.error(f"Prediction processing failed: {e}")

            # Show debug info in expander
            with st.expander("🐛 Debug Information"):
                st.code(f"Error: {str(e)}")
                st.code(f"Symbol: {symbol}")
                if 'latest_data' in locals():
                    st.code(f"Data shape: {latest_data.shape if latest_data is not None else 'None'}")
                if 'processed_data' in locals():
                    st.code(f"Processed shape: {processed_data.shape if 'processed_data' in locals() else 'None'}")
                if 'feature_columns' in locals():
                    st.code(f"Features: {len(feature_columns) if 'feature_columns' in locals() else 'None'}")
    
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        logger.error(f"Prediction failed for {symbol}: {e}")

def analyze_performance(symbol, task_type):
    """Analyze model performance."""
    
    st.header("📊 Performance Analysis")
    
    # Check if models are trained
    results_key = f'{symbol}_{task_type}_results'
    if results_key not in st.session_state:
        st.warning("⚠️ No trained models found. Please train models first.")
        return
    
    try:
        evaluation_results = st.session_state[results_key]['evaluation_results']
        
        if not evaluation_results:
            st.warning("No evaluation results available")
            return
        
        # Performance comparison table
        st.subheader("📈 Model Comparison")
        
        comparison_data = []
        for model_name, results in evaluation_results.items():
            row = {'Model': model_name.upper()}
            
            if task_type == "classification":
                row.update({
                    'Accuracy': results['accuracy'],
                    'Precision': results['precision'],
                    'Recall': results['recall'],
                    'F1 Score': results['f1_score'],
                    'ROC AUC': results.get('roc_auc', 'N/A')
                })
            else:
                row.update({
                    'RMSE': results['rmse'],
                    'MAE': results['mae'],
                    'R² Score': results['r2_score'],
                    'MAPE': results['mape'],
                    'Directional Accuracy': results['directional_accuracy']
                })
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Detailed results for each model
        st.subheader("🔍 Detailed Results")
        
        for model_name, results in evaluation_results.items():
            with st.expander(f"{model_name.upper()} Detailed Results"):
                
                if task_type == "classification":
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Classification Metrics")
                        st.metric("Accuracy", f"{results['accuracy']:.4f}")
                        st.metric("Precision", f"{results['precision']:.4f}")
                        st.metric("Recall", f"{results['recall']:.4f}")
                        st.metric("F1 Score", f"{results['f1_score']:.4f}")
                        if results.get('roc_auc'):
                            st.metric("ROC AUC", f"{results['roc_auc']:.4f}")
                    
                    with col2:
                        st.subheader("Trading Metrics")
                        trading = results.get('trading_metrics', {})
                        st.metric("True Positive Rate", f"{trading.get('true_positive_rate', 0):.4f}")
                        st.metric("True Negative Rate", f"{trading.get('true_negative_rate', 0):.4f}")
                        st.metric("Matthews Correlation", f"{trading.get('matthews_correlation_coefficient', 0):.4f}")
                    
                    # Confusion Matrix
                    if 'confusion_matrix' in results:
                        st.subheader("Confusion Matrix")
                        cm = results['confusion_matrix']
                        fig = px.imshow(cm, 
                                      text_auto=True, 
                                      aspect="auto",
                                      title="Confusion Matrix")
                        st.plotly_chart(fig, use_container_width=True)
                
                else:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Regression Metrics")
                        st.metric("RMSE", f"{results['rmse']:.4f}")
                        st.metric("MAE", f"{results['mae']:.4f}")
                        st.metric("R² Score", f"{results['r2_score']:.4f}")
                        st.metric("MAPE", f"{results['mape']:.2f}%")
                        st.metric("Directional Accuracy", f"{results['directional_accuracy']:.4f}")
                    
                    with col2:
                        st.subheader("Error Analysis")
                        error_bounds = results.get('error_bounds', {})
                        for bound, percentage in error_bounds.items():
                            st.metric(f"Within {bound.replace('within_', '').replace('pct', '%')}", 
                                    f"{percentage:.1f}%")
    
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        logger.error(f"Analysis failed for {symbol}: {e}")

def show_stock_overview(symbol, period, source):
    """Show stock data overview."""
    
    st.header(f"📊 {symbol} Stock Overview")
    
    try:
        # Collect stock data
        collector = StockDataCollector()
        data = collector.get_stock_data(symbol, period=period, source=source)
        
        if data is None:
            st.error(f"Failed to fetch data for {symbol}")
            return
        
        # Basic info
        latest = data.iloc[-1]
        prev = data.iloc[-2]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Price", f"${latest['close']:.2f}")
        
        with col2:
            change = latest['close'] - prev['close']
            change_pct = (change / prev['close']) * 100
            st.metric("Change", f"${change:.2f}", f"{change_pct:.2f}%")
        
        with col3:
            st.metric("Volume", f"{latest['volume']:,.0f}")
        
        with col4:
            st.metric("High/Low", f"${latest['high']:.2f} / ${latest['low']:.2f}")
        
        # Price chart
        st.subheader("📈 Price Chart")
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('Price', 'Volume'),
            row_width=[0.7, 0.3]
        )
        
        # Price candlestick
        fig.add_trace(
            go.Candlestick(
                x=data['timestamp'],
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Volume bars
        fig.add_trace(
            go.Bar(
                x=data['timestamp'],
                y=data['volume'],
                name='Volume',
                marker_color='rgba(158,202,225,0.8)'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title=f"{symbol} Stock Price and Volume",
            xaxis_rangeslider_visible=False,
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Data summary
        st.subheader("📋 Data Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Price Statistics:**")
            price_stats = data[['open', 'high', 'low', 'close']].describe()
            st.dataframe(price_stats)
        
        with col2:
            st.write("**Volume Statistics:**")
            volume_stats = data[['volume']].describe()
            st.dataframe(volume_stats)
        
        # Recent data
        st.subheader("📅 Recent Data")
        st.dataframe(data.tail(10)[['timestamp', 'open', 'high', 'low', 'close', 'volume']], 
                    use_container_width=True)
    
    except Exception as e:
        st.error(f"Failed to load stock overview: {str(e)}")
        logger.error(f"Stock overview failed for {symbol}: {e}")

if __name__ == "__main__":
    main()
