"""
Model comparison and analysis tools.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from ..evaluation.metrics import ModelEvaluator
from ..utils.logger import get_logger

logger = get_logger(__name__)

class ModelComparator:
    """Compare and analyze multiple stock prediction models."""
    
    def __init__(self):
        """Initialize model comparator."""
        self.evaluator = ModelEvaluator()
        self.comparison_results = {}
        
    def compare_models(
        self,
        models: Dict[str, Any],
        X_test: np.ndarray,
        y_test: np.ndarray,
        task_type: str = "classification"
    ) -> Dict[str, Any]:
        """
        Compare multiple models on the same test set.
        
        Args:
            models: Dictionary of trained models
            X_test: Test features
            y_test: Test targets
            task_type: 'classification' or 'regression'
            
        Returns:
            Comparison results
        """
        logger.info(f"Comparing {len(models)} models")
        
        results = {}
        predictions = {}
        
        # Get predictions and evaluate each model
        for name, model in models.items():
            if not hasattr(model, 'is_trained') or not model.is_trained:
                logger.warning(f"Model {name} is not trained, skipping")
                continue
            
            try:
                # Get predictions
                y_pred = model.predict(X_test)
                predictions[name] = y_pred
                
                # Evaluate model
                if task_type == "classification":
                    eval_results = self.evaluator.evaluate_classification(
                        y_test, y_pred, y_pred_proba=y_pred
                    )
                else:
                    eval_results = self.evaluator.evaluate_regression(
                        y_test, y_pred
                    )
                
                results[name] = eval_results
                
            except Exception as e:
                logger.error(f"Error evaluating model {name}: {e}")
                continue
        
        # Create comparison summary
        comparison_summary = self._create_comparison_summary(results, task_type)
        
        # Statistical significance tests
        significance_tests = self._perform_significance_tests(
            predictions, y_test, task_type
        )
        
        self.comparison_results = {
            'individual_results': results,
            'summary': comparison_summary,
            'predictions': predictions,
            'significance_tests': significance_tests,
            'task_type': task_type
        }
        
        logger.info("Model comparison completed")
        return self.comparison_results
    
    def _create_comparison_summary(
        self,
        results: Dict[str, Dict[str, Any]],
        task_type: str
    ) -> pd.DataFrame:
        """Create summary comparison table."""
        
        if task_type == "classification":
            metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        else:
            metrics = ['rmse', 'mae', 'r2_score', 'mape', 'directional_accuracy']
        
        summary_data = []
        
        for model_name, model_results in results.items():
            row = {'Model': model_name}
            
            for metric in metrics:
                if metric in model_results:
                    row[metric.upper()] = model_results[metric]
                else:
                    row[metric.upper()] = None
            
            # Add trading-specific metrics for classification
            if task_type == "classification" and 'trading_metrics' in model_results:
                trading = model_results['trading_metrics']
                row['TPR'] = trading.get('true_positive_rate')
                row['TNR'] = trading.get('true_negative_rate')
                row['MCC'] = trading.get('matthews_correlation_coefficient')
            
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.set_index('Model')
        
        return summary_df
    
    def _perform_significance_tests(
        self,
        predictions: Dict[str, np.ndarray],
        y_true: np.ndarray,
        task_type: str
    ) -> Dict[str, Any]:
        """Perform statistical significance tests between models."""
        
        if len(predictions) < 2:
            return {}
        
        model_names = list(predictions.keys())
        significance_results = {}
        
        # Pairwise comparisons
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names[i+1:], i+1):
                
                pred1 = predictions[model1]
                pred2 = predictions[model2]
                
                if task_type == "classification":
                    # McNemar's test for classification
                    try:
                        mcnemar_result = self._mcnemar_test(y_true, pred1, pred2)
                        significance_results[f"{model1}_vs_{model2}"] = {
                            'test': 'McNemar',
                            'statistic': mcnemar_result['statistic'],
                            'p_value': mcnemar_result['p_value'],
                            'significant': mcnemar_result['p_value'] < 0.05
                        }
                    except Exception as e:
                        logger.warning(f"McNemar test failed for {model1} vs {model2}: {e}")
                
                else:
                    # Paired t-test for regression
                    try:
                        errors1 = np.abs(y_true - pred1)
                        errors2 = np.abs(y_true - pred2)
                        
                        t_stat, p_value = stats.ttest_rel(errors1, errors2)
                        
                        significance_results[f"{model1}_vs_{model2}"] = {
                            'test': 'Paired t-test',
                            'statistic': t_stat,
                            'p_value': p_value,
                            'significant': p_value < 0.05
                        }
                    except Exception as e:
                        logger.warning(f"Paired t-test failed for {model1} vs {model2}: {e}")
        
        return significance_results
    
    def _mcnemar_test(
        self,
        y_true: np.ndarray,
        pred1: np.ndarray,
        pred2: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """Perform McNemar's test for comparing two classifiers."""
        
        # Convert probabilities to binary predictions if needed
        if np.all((pred1 >= 0) & (pred1 <= 1)):
            pred1_binary = (pred1 > threshold).astype(int)
        else:
            pred1_binary = pred1.astype(int)
        
        if np.all((pred2 >= 0) & (pred2 <= 1)):
            pred2_binary = (pred2 > threshold).astype(int)
        else:
            pred2_binary = pred2.astype(int)
        
        # Create contingency table
        correct1 = (pred1_binary == y_true)
        correct2 = (pred2_binary == y_true)
        
        # McNemar table
        both_correct = np.sum(correct1 & correct2)
        model1_correct_model2_wrong = np.sum(correct1 & ~correct2)
        model1_wrong_model2_correct = np.sum(~correct1 & correct2)
        both_wrong = np.sum(~correct1 & ~correct2)
        
        # McNemar statistic
        b = model1_correct_model2_wrong
        c = model1_wrong_model2_correct
        
        if b + c == 0:
            return {'statistic': 0, 'p_value': 1.0}
        
        mcnemar_stat = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = 1 - stats.chi2.cdf(mcnemar_stat, 1)
        
        return {'statistic': mcnemar_stat, 'p_value': p_value}
    
    def plot_model_comparison(
        self,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 10)
    ) -> None:
        """Plot comprehensive model comparison."""
        
        if not self.comparison_results:
            logger.error("No comparison results available. Run compare_models() first.")
            return
        
        summary = self.comparison_results['summary']
        task_type = self.comparison_results['task_type']
        
        if task_type == "classification":
            self._plot_classification_comparison(summary, save_path, figsize)
        else:
            self._plot_regression_comparison(summary, save_path, figsize)
    
    def _plot_classification_comparison(
        self,
        summary: pd.DataFrame,
        save_path: Optional[str],
        figsize: Tuple[int, int]
    ) -> None:
        """Plot classification model comparison."""
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # Metrics to plot
        metrics = ['ACCURACY', 'PRECISION', 'RECALL', 'F1_SCORE', 'ROC_AUC']
        
        for i, metric in enumerate(metrics):
            if metric in summary.columns:
                row, col = i // 3, i % 3
                
                # Bar plot
                summary[metric].plot(kind='bar', ax=axes[row, col])
                axes[row, col].set_title(f'{metric}')
                axes[row, col].set_ylabel('Score')
                axes[row, col].tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for j, v in enumerate(summary[metric]):
                    if not pd.isna(v):
                        axes[row, col].text(j, v + 0.01, f'{v:.3f}', 
                                          ha='center', va='bottom')
        
        # Overall comparison radar chart
        if len(summary) > 1:
            self._plot_radar_chart(summary, axes[1, 2], metrics)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_regression_comparison(
        self,
        summary: pd.DataFrame,
        save_path: Optional[str],
        figsize: Tuple[int, int]
    ) -> None:
        """Plot regression model comparison."""
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # Metrics to plot
        metrics = ['RMSE', 'MAE', 'R2_SCORE', 'MAPE', 'DIRECTIONAL_ACCURACY']
        
        for i, metric in enumerate(metrics):
            if metric in summary.columns:
                row, col = i // 3, i % 3
                
                # Bar plot
                summary[metric].plot(kind='bar', ax=axes[row, col])
                axes[row, col].set_title(f'{metric}')
                axes[row, col].set_ylabel('Score')
                axes[row, col].tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for j, v in enumerate(summary[metric]):
                    if not pd.isna(v):
                        axes[row, col].text(j, v + 0.01, f'{v:.3f}', 
                                          ha='center', va='bottom')
        
        # Overall comparison radar chart
        if len(summary) > 1:
            # Normalize metrics for radar chart (invert RMSE and MAE)
            radar_data = summary[metrics].copy()
            if 'RMSE' in radar_data.columns:
                radar_data['RMSE'] = 1 / (1 + radar_data['RMSE'])
            if 'MAE' in radar_data.columns:
                radar_data['MAE'] = 1 / (1 + radar_data['MAE'])
            if 'MAPE' in radar_data.columns:
                radar_data['MAPE'] = 1 / (1 + radar_data['MAPE'] / 100)
            
            self._plot_radar_chart(radar_data, axes[1, 2], metrics)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_radar_chart(
        self,
        data: pd.DataFrame,
        ax: plt.Axes,
        metrics: List[str]
    ) -> None:
        """Plot radar chart for model comparison."""
        
        # Filter available metrics
        available_metrics = [m for m in metrics if m in data.columns and not data[m].isna().all()]
        
        if len(available_metrics) < 3:
            ax.text(0.5, 0.5, 'Insufficient data\nfor radar chart', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Number of variables
        N = len(available_metrics)
        
        # Compute angle for each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        # Initialize the plot
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # Draw axis labels
        ax.set_thetagrids(np.degrees(angles[:-1]), available_metrics)
        
        # Plot data for each model
        colors = plt.cm.Set1(np.linspace(0, 1, len(data)))
        
        for i, (model_name, row) in enumerate(data.iterrows()):
            values = [row[metric] for metric in available_metrics]
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, 
                   label=model_name, color=colors[i])
            ax.fill(angles, values, alpha=0.25, color=colors[i])
        
        ax.set_title('Model Comparison Radar Chart')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    def get_best_model(self, metric: str = None) -> Tuple[str, float]:
        """
        Get the best performing model based on a specific metric.
        
        Args:
            metric: Metric to use for comparison (if None, uses default)
            
        Returns:
            Tuple of (model_name, metric_value)
        """
        if not self.comparison_results:
            raise ValueError("No comparison results available")
        
        summary = self.comparison_results['summary']
        task_type = self.comparison_results['task_type']
        
        if metric is None:
            metric = 'F1_SCORE' if task_type == 'classification' else 'R2_SCORE'
        
        if metric not in summary.columns:
            raise ValueError(f"Metric {metric} not available in results")
        
        # For error metrics (RMSE, MAE, MAPE), lower is better
        error_metrics = ['RMSE', 'MAE', 'MAPE']
        
        if metric in error_metrics:
            best_idx = summary[metric].idxmin()
        else:
            best_idx = summary[metric].idxmax()
        
        best_value = summary.loc[best_idx, metric]
        
        return best_idx, best_value
    
    def generate_comparison_report(self) -> str:
        """Generate a comprehensive comparison report."""
        
        if not self.comparison_results:
            return "No comparison results available."
        
        summary = self.comparison_results['summary']
        task_type = self.comparison_results['task_type']
        significance = self.comparison_results['significance_tests']
        
        report = f"\n{'='*80}\n"
        report += f"MODEL COMPARISON REPORT - {task_type.upper()}\n"
        report += f"{'='*80}\n\n"
        
        # Summary table
        report += "PERFORMANCE SUMMARY:\n"
        report += "-" * 40 + "\n"
        report += summary.to_string(float_format='%.4f')
        report += "\n\n"
        
        # Best model for each metric
        report += "BEST MODELS BY METRIC:\n"
        report += "-" * 40 + "\n"
        
        for metric in summary.columns:
            if not summary[metric].isna().all():
                try:
                    best_model, best_value = self.get_best_model(metric)
                    report += f"{metric}: {best_model} ({best_value:.4f})\n"
                except:
                    continue
        
        # Statistical significance
        if significance:
            report += "\nSTATISTICAL SIGNIFICANCE TESTS:\n"
            report += "-" * 40 + "\n"
            
            for comparison, result in significance.items():
                significance_text = "Significant" if result['significant'] else "Not significant"
                report += f"{comparison}: {result['test']} p-value = {result['p_value']:.4f} ({significance_text})\n"
        
        report += f"\n{'='*80}\n"
        
        return report
