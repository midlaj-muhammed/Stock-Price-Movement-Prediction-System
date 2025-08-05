"""
Main entry point for the stock price prediction system.
"""

import os
# Force CPU usage before any TensorFlow imports
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from training_pipeline import StockPredictionPipeline
from models.lstm_model import LSTMStockModel
from models.tcn_model import TCNStockModel
from models.ensemble import ModelEnsemble
from models.model_comparison import ModelComparator
from utils.logger import get_logger
from utils.config import config

logger = get_logger(__name__)

def main():
    """Main function with command line interface."""
    
    parser = argparse.ArgumentParser(description="Stock Price Movement Prediction System")
    
    # Required arguments
    parser.add_argument("--symbol", type=str, required=True,
                       help="Stock symbol to analyze (e.g., AAPL)")
    
    parser.add_argument("--mode", type=str, required=True,
                       choices=["train", "predict", "evaluate", "compare"],
                       help="Operation mode")
    
    # Optional arguments
    parser.add_argument("--task-type", type=str, default="classification",
                       choices=["classification", "regression"],
                       help="Task type: classification or regression")
    
    parser.add_argument("--period", type=str, default="2y",
                       choices=["1y", "2y", "5y", "max"],
                       help="Data period to collect")
    
    parser.add_argument("--source", type=str, default="yahoo",
                       choices=["yahoo", "alpha_vantage"],
                       help="Data source")
    
    parser.add_argument("--models", type=str, nargs="+", default=["lstm", "tcn"],
                       choices=["lstm", "tcn", "ensemble"],
                       help="Models to use")
    
    parser.add_argument("--n-features", type=int, default=50,
                       help="Number of features to select")
    
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Training batch size")
    
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Execute based on mode
    if args.mode == "train":
        train_models(args)
    elif args.mode == "predict":
        make_predictions(args)
    elif args.mode == "evaluate":
        evaluate_models(args)
    elif args.mode == "compare":
        compare_models(args)

def train_models(args):
    """Train models for the specified stock."""
    
    logger.info(f"Training models for {args.symbol}")
    
    try:
        # Initialize pipeline
        pipeline = StockPredictionPipeline(args.symbol, args.task_type)
        
        # Run complete pipeline
        results = pipeline.run_complete_pipeline(
            period=args.period,
            source=args.source,
            n_features=args.n_features,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        print(f"\n✅ Training completed successfully for {args.symbol}")
        print(f"Results saved to: {results['results_path']}")
        
        # Print summary
        print("\n📊 Training Summary:")
        for model_name, training_result in results['training_results'].items():
            print(f"  {model_name.upper()}: {training_result['training_time']:.2f}s")
        
        print("\n📈 Evaluation Summary:")
        for model_name, eval_result in results['evaluation_results'].items():
            if args.task_type == "classification":
                print(f"  {model_name.upper()}: Accuracy={eval_result['accuracy']:.4f}, "
                      f"F1={eval_result['f1_score']:.4f}")
            else:
                print(f"  {model_name.upper()}: RMSE={eval_result['rmse']:.4f}, "
                      f"R²={eval_result['r2_score']:.4f}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"❌ Training failed: {e}")
        sys.exit(1)

def make_predictions(args):
    """Make predictions with trained models."""
    
    logger.info(f"Making predictions for {args.symbol}")
    
    try:
        # Load trained models
        models = {}
        model_dir = config.model_storage_path / "saved"
        
        for model_name in args.models:
            if model_name == "lstm":
                model = LSTMStockModel(args.task_type)
                model_path = model_dir / "LSTM" / "LSTM_model.h5"
            elif model_name == "tcn":
                model = TCNStockModel(args.task_type)
                model_path = model_dir / "TCN" / "TCN_model.h5"
            else:
                continue
            
            if model_path.exists():
                model.load_model(str(model_path))
                models[model_name] = model
                logger.info(f"Loaded {model_name} model")
            else:
                logger.warning(f"Model file not found: {model_path}")
        
        if not models:
            print("❌ No trained models found. Please train models first.")
            sys.exit(1)
        
        # Collect recent data
        from data.data_collector import StockDataCollector
        collector = StockDataCollector()
        data = collector.get_stock_data(args.symbol, period="1mo", source=args.source)
        
        if data is None:
            print(f"❌ Failed to collect data for {args.symbol}")
            sys.exit(1)
        
        # Make predictions (simplified version)
        print(f"\n🔮 Predictions for {args.symbol}:")
        print(f"Current Price: ${data['close'].iloc[-1]:.2f}")
        
        for model_name, model in models.items():
            # This is a simplified prediction - in practice, you'd need to
            # process the data through the full feature engineering pipeline
            print(f"  {model_name.upper()}: Model loaded and ready for prediction")
        
        print("\n💡 Note: Full prediction requires running the complete pipeline.")
        print("Use the web interface for complete predictions.")
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        print(f"❌ Prediction failed: {e}")
        sys.exit(1)

def evaluate_models(args):
    """Evaluate trained models."""
    
    logger.info(f"Evaluating models for {args.symbol}")
    
    try:
        # Load evaluation results
        results_dir = config.model_storage_path / "results"
        results_file = results_dir / f"{args.symbol}_{args.task_type}_results.json"
        
        if not results_file.exists():
            print(f"❌ No evaluation results found for {args.symbol}")
            print("Please train models first.")
            sys.exit(1)
        
        import json
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        evaluation_results = results.get('evaluation_results', {})
        
        if not evaluation_results:
            print(f"❌ No evaluation results available for {args.symbol}")
            sys.exit(1)
        
        print(f"\n📊 Evaluation Results for {args.symbol}:")
        print("=" * 60)
        
        for model_name, eval_result in evaluation_results.items():
            print(f"\n{model_name.upper()} Model:")
            print("-" * 30)
            
            if args.task_type == "classification":
                print(f"Accuracy:     {eval_result['accuracy']:.4f}")
                print(f"Precision:    {eval_result['precision']:.4f}")
                print(f"Recall:       {eval_result['recall']:.4f}")
                print(f"F1 Score:     {eval_result['f1_score']:.4f}")
                if eval_result.get('roc_auc'):
                    print(f"ROC AUC:      {eval_result['roc_auc']:.4f}")
            else:
                print(f"RMSE:         {eval_result['rmse']:.4f}")
                print(f"MAE:          {eval_result['mae']:.4f}")
                print(f"R² Score:     {eval_result['r2_score']:.4f}")
                print(f"MAPE:         {eval_result['mape']:.2f}%")
                print(f"Dir. Accuracy: {eval_result['directional_accuracy']:.4f}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        print(f"❌ Evaluation failed: {e}")
        sys.exit(1)

def compare_models(args):
    """Compare multiple models."""
    
    logger.info(f"Comparing models for {args.symbol}")
    
    try:
        # Load evaluation results
        results_dir = config.model_storage_path / "results"
        results_file = results_dir / f"{args.symbol}_{args.task_type}_results.json"
        
        if not results_file.exists():
            print(f"❌ No results found for {args.symbol}")
            print("Please train models first.")
            sys.exit(1)
        
        import json
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        evaluation_results = results.get('evaluation_results', {})
        
        if len(evaluation_results) < 2:
            print(f"❌ Need at least 2 models for comparison")
            sys.exit(1)
        
        # Create comparison
        comparator = ModelComparator()
        
        # Convert results to comparison format
        comparison_data = {}
        for model_name, eval_result in evaluation_results.items():
            comparison_data[model_name] = eval_result
        
        # Generate comparison report
        print(f"\n📊 Model Comparison for {args.symbol}:")
        print("=" * 80)
        
        # Create summary table
        import pandas as pd
        
        if args.task_type == "classification":
            metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        else:
            metrics = ['rmse', 'mae', 'r2_score', 'mape', 'directional_accuracy']
        
        summary_data = []
        for model_name, results in evaluation_results.items():
            row = {'Model': model_name.upper()}
            for metric in metrics:
                if metric in results:
                    row[metric.upper()] = results[metric]
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False, float_format='%.4f'))
        
        # Best model
        if args.task_type == "classification":
            best_metric = 'f1_score'
        else:
            best_metric = 'r2_score'
        
        best_model = max(evaluation_results.items(), 
                        key=lambda x: x[1].get(best_metric, 0))
        
        print(f"\n🏆 Best Model: {best_model[0].upper()}")
        print(f"   {best_metric.upper()}: {best_model[1][best_metric]:.4f}")
        
    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        print(f"❌ Comparison failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
