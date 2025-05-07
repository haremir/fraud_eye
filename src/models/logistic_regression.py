"""
Logistic Regression model implementation for fraud detection.
"""

import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_curve, auc, roc_curve
)

from src.utils.logger import setup_logger
from src.base_model import BaseModel
import src.config as config
import src.constants as constants

logger = setup_logger(__name__)

class LogisticRegressionModel(BaseModel):
    """Logistic Regression implementation for fraud detection."""
    
    def __init__(self, params=None):
        """Initialize Logistic Regression model with parameters."""
        super().__init__(model_name="logistic_regression")
        self.params = params or config.LOGISTIC_REGRESSION_CONFIG
        self.model = None
        
    def train(self, X_train, y_train):
        """Train Logistic Regression model."""
        logger.info(f"Training Logistic Regression model with {X_train.shape[0]} samples")
        self.model = LogisticRegression(**self.params)
        self.model.fit(X_train, y_train)
        logger.info("Logistic Regression model training completed")
        return self
    
    def predict(self, X):
        """Make predictions with trained model."""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        logger.info(f"Making predictions on {X.shape[0]} samples")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Make probability predictions with trained model."""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        logger.info(f"Making probability predictions on {X.shape[0]} samples")
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test, threshold=0.5):
        """Evaluate model performance."""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        logger.info(f"Evaluating model on {X_test.shape[0]} samples")
        
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        results = {
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'threshold': threshold,
            'accuracy': class_report['accuracy'],
            'precision': class_report['1']['precision'],
            'recall': class_report['1']['recall'],
            'f1': class_report['1']['f1-score']
        }
        
        # Log results
        logger.info(f"Evaluation Results:")
        logger.info(f"ROC AUC: {roc_auc:.4f}")
        logger.info(f"PR AUC: {pr_auc:.4f}")
        logger.info(f"Precision: {class_report['1']['precision']:.4f}")
        logger.info(f"Recall: {class_report['1']['recall']:.4f}")
        logger.info(f"F1-Score: {class_report['1']['f1-score']:.4f}")
        
        return results
    
    def plot_evaluation_results(self, X_test, y_test, results=None):
        """Plot evaluation results including ROC curve and confusion matrix."""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        if results is None:
            results = self.evaluate(X_test, y_test)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'])
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        plt.title('Confusion Matrix - Logistic Regression')
        plt.tight_layout()
        
        reports_dir = constants.REPORTS_FIGURES_DIR
        os.makedirs(reports_dir, exist_ok=True)
        plt.savefig(os.path.join(reports_dir, 'logistic_regression_confusion_matrix.png'))
        plt.show()
        
        # Plot ROC curve
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, label=f'ROC AUC = {results["roc_auc"]:.4f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Logistic Regression')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(os.path.join(reports_dir, 'logistic_regression_roc_curve.png'))
        plt.tight_layout()
        plt.show()
        
        # Plot Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, label=f'PR AUC = {results["pr_auc"]:.4f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve - Logistic Regression')
        plt.legend(loc="lower left")
        plt.grid(True)
        plt.savefig(os.path.join(reports_dir, 'logistic_regression_pr_curve.png'))
        plt.tight_layout()
        plt.show()
        
        return results
    
    def get_feature_importance(self, feature_names=None, plot=True, top_n=20, figsize=(12, 8)):
        """Get and optionally plot feature importance (absolute coefficients)."""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        importance = np.abs(self.model.coef_[0])
        
        if feature_names is not None:
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            })
        else:
            importance_df = pd.DataFrame({
                'feature': [f'feature_{i}' for i in range(len(importance))],
                'importance': importance
            })
        
        importance_df = importance_df.sort_values('importance', ascending=False).head(top_n)
        
        if plot:
            plt.figure(figsize=figsize)
            sns.barplot(x='importance', y='feature', data=importance_df)
            plt.title(f'Top {top_n} Feature Importance (absolute coefficients)')
            plt.tight_layout()
            
            reports_dir = constants.REPORTS_FIGURES_DIR
            os.makedirs(reports_dir, exist_ok=True)
            plt.savefig(os.path.join(reports_dir, 'logistic_regression_feature_importance.png'))
            plt.show()
        
        return importance_df


def train_logistic_regression(X_train, y_train, params=None):
    """Train Logistic Regression model with given parameters."""
    model = LogisticRegressionModel(params=params)
    model.train(X_train, y_train)
    return model


def evaluate_logistic_regression(model, X_test, y_test, threshold=0.5):
    """Evaluate Logistic Regression model."""
    return model.evaluate(X_test, y_test, threshold=threshold)
