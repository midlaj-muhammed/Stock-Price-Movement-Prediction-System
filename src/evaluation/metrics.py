"""
Evaluation metrics for stock prediction models.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, mean_squared_error, 
    mean_absolute_error, r2_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from ..utils.logger import get_logger

logger = get_logger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluation for stock prediction."""
    
    def __init__(self):
        """Initialize model evaluator."""
        pass
    
    def evaluate_classification(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Evaluate classification model performance.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels (or probabilities if y_pred_proba is None)
            y_pred_proba: Predicted probabilities (optional)
            threshold: Classification threshold
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating classification model")
        
        # If probabilities are provided, convert to binary predictions
        if y_pred_proba is not None:
            y_pred_binary = (y_pred_proba > threshold).astype(int)
        else:
            # Assume y_pred contains probabilities if values are between 0 and 1
            if np.all((y_pred >= 0) & (y_pred <= 1)):
                y_pred_proba = y_pred
                y_pred_binary = (y_pred > threshold).astype(int)
            else:
                y_pred_binary = y_pred
                y_pred_proba = None
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred_binary)
        precision = precision_score(y_true, y_pred_binary, average='binary', zero_division=0)
        recall = recall_score(y_true, y_pred_binary, average='binary', zero_division=0)
        f1 = f1_score(y_true, y_pred_binary, average='binary', zero_division=0)
        
        # ROC AUC (if probabilities available)
        roc_auc = None
        if y_pred_proba is not None:
            try:
                roc_auc = roc_auc_score(y_true, y_pred_proba)
            except ValueError as e:
                logger.warning(f"Could not calculate ROC AUC: {e}")
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred_binary)
        
        # Classification report
        class_report = classification_report(y_true, y_pred_binary, output_dict=True)
        
        # Trading-specific metrics
        trading_metrics = self._calculate_trading_metrics(y_true, y_pred_binary, y_pred_proba)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'classification_report': class_report,
            'trading_metrics': trading_metrics
        }
        
        logger.info(f"Classification evaluation completed. Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        return results
    
    def evaluate_regression(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """
        Evaluate regression model performance.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating regression model")
        
        # Basic regression metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Mean Absolute Percentage Error (MAPE)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        
        # Directional accuracy (for price prediction)
        directional_accuracy = self._calculate_directional_accuracy(y_true, y_pred)
        
        # Additional metrics
        residuals = y_true - y_pred
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)
        
        # Percentage of predictions within certain error bounds
        error_bounds = self._calculate_error_bounds(y_true, y_pred)
        
        results = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2,
            'mape': mape,
            'directional_accuracy': directional_accuracy,
            'mean_residual': mean_residual,
            'std_residual': std_residual,
            'error_bounds': error_bounds,
            'residuals': residuals
        }
        
        logger.info(f"Regression evaluation completed. RMSE: {rmse:.4f}, R²: {r2:.4f}")
        return results
    
    def _calculate_trading_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calculate trading-specific metrics."""
        
        # True Positive Rate (Sensitivity)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # True Negative Rate (Specificity)
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Positive Predictive Value
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # Negative Predictive Value
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        # Matthews Correlation Coefficient
        mcc_num = (tp * tn) - (fp * fn)
        mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = mcc_num / mcc_den if mcc_den > 0 else 0
        
        return {
            'true_positive_rate': tpr,
            'true_negative_rate': tnr,
            'positive_predictive_value': ppv,
            'negative_predictive_value': npv,
            'matthews_correlation_coefficient': mcc
        }
    
    def _calculate_directional_accuracy(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """Calculate directional accuracy for price predictions."""
        if len(y_true) < 2:
            return 0.0
        
        # Calculate actual and predicted directions
        true_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        
        # Calculate accuracy
        directional_accuracy = np.mean(true_direction == pred_direction)
        return directional_accuracy
    
    def _calculate_error_bounds(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate percentage of predictions within error bounds."""
        relative_errors = np.abs((y_true - y_pred) / (y_true + 1e-8))
        
        bounds = [0.01, 0.05, 0.10, 0.20]  # 1%, 5%, 10%, 20%
        error_bounds = {}
        
        for bound in bounds:
            within_bound = np.mean(relative_errors <= bound) * 100
            error_bounds[f'within_{int(bound*100)}pct'] = within_bound
        
        return error_bounds
    
    def plot_classification_results(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None,
        save_path: Optional[str] = None
    ) -> None:
        """Plot classification results."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        
        # ROC Curve
        if y_pred_proba is not None:
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            axes[0, 1].plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(y_true, y_pred_proba):.3f})')
            axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random')
            axes[0, 1].set_xlabel('False Positive Rate')
            axes[0, 1].set_ylabel('True Positive Rate')
            axes[0, 1].set_title('ROC Curve')
            axes[0, 1].legend()
        
        # Prediction Distribution
        if y_pred_proba is not None:
            axes[1, 0].hist(y_pred_proba[y_true == 0], alpha=0.5, label='Class 0', bins=30)
            axes[1, 0].hist(y_pred_proba[y_true == 1], alpha=0.5, label='Class 1', bins=30)
            axes[1, 0].set_xlabel('Predicted Probability')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Prediction Distribution')
            axes[1, 0].legend()
        
        # Precision-Recall Curve
        if y_pred_proba is not None:
            from sklearn.metrics import precision_recall_curve, average_precision_score
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            ap_score = average_precision_score(y_true, y_pred_proba)
            axes[1, 1].plot(recall, precision, label=f'PR Curve (AP = {ap_score:.3f})')
            axes[1, 1].set_xlabel('Recall')
            axes[1, 1].set_ylabel('Precision')
            axes[1, 1].set_title('Precision-Recall Curve')
            axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_regression_results(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[str] = None
    ) -> None:
        """Plot regression results."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Actual vs Predicted
        axes[0, 0].scatter(y_true, y_pred, alpha=0.5)
        axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Values')
        axes[0, 0].set_ylabel('Predicted Values')
        axes[0, 0].set_title('Actual vs Predicted')
        
        # Residuals
        residuals = y_true - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.5)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Values')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residual Plot')
        
        # Residual Distribution
        axes[1, 0].hist(residuals, bins=30, alpha=0.7)
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Residual Distribution')
        
        # Time Series Plot (if applicable)
        axes[1, 1].plot(y_true, label='Actual', alpha=0.7)
        axes[1, 1].plot(y_pred, label='Predicted', alpha=0.7)
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Values')
        axes[1, 1].set_title('Time Series Comparison')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_evaluation_report(
        self,
        results: Dict[str, Any],
        model_name: str,
        task_type: str
    ) -> str:
        """Generate a comprehensive evaluation report."""
        
        report = f"\n{'='*60}\n"
        report += f"MODEL EVALUATION REPORT - {model_name.upper()}\n"
        report += f"Task Type: {task_type.upper()}\n"
        report += f"{'='*60}\n\n"
        
        if task_type == "classification":
            report += "CLASSIFICATION METRICS:\n"
            report += f"Accuracy:     {results['accuracy']:.4f}\n"
            report += f"Precision:    {results['precision']:.4f}\n"
            report += f"Recall:       {results['recall']:.4f}\n"
            report += f"F1-Score:     {results['f1_score']:.4f}\n"
            if results['roc_auc'] is not None:
                report += f"ROC AUC:      {results['roc_auc']:.4f}\n"
            
            report += "\nTRADING METRICS:\n"
            trading = results['trading_metrics']
            report += f"True Positive Rate:  {trading['true_positive_rate']:.4f}\n"
            report += f"True Negative Rate:  {trading['true_negative_rate']:.4f}\n"
            report += f"Positive Pred. Value: {trading['positive_predictive_value']:.4f}\n"
            report += f"Matthews Corr. Coef.: {trading['matthews_correlation_coefficient']:.4f}\n"
            
        else:  # regression
            report += "REGRESSION METRICS:\n"
            report += f"RMSE:         {results['rmse']:.4f}\n"
            report += f"MAE:          {results['mae']:.4f}\n"
            report += f"R² Score:     {results['r2_score']:.4f}\n"
            report += f"MAPE:         {results['mape']:.2f}%\n"
            report += f"Directional Accuracy: {results['directional_accuracy']:.4f}\n"
            
            report += "\nERROR BOUNDS:\n"
            for bound, percentage in results['error_bounds'].items():
                report += f"Within {bound.replace('within_', '').replace('pct', '%')}: {percentage:.1f}%\n"
        
        report += f"\n{'='*60}\n"
        
        return report
