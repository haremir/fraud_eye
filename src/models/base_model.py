"""
Base model module that contains common functions for all models.
"""
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.logger import setup_logger

logger = setup_logger("base_model")

class BaseModel:
    """Base class for all fraud detection models"""
    
    def __init__(self, model_name):
        """Initialize base model
        
        Args:
            model_name (str): Name of the model
        """
        self.model_name = model_name
        self.model = None
        logger.info(f"Initialized {model_name} model")
    
    def save_model(self, filepath):
        """Save model to disk
        
        Args:
            filepath (str): Path to save the model
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        logger.info(f"Model saved to {filepath}")
    
    @staticmethod
    def load_model(filepath):
        """Load model from disk
        
        Args:
            filepath (str): Path to load the model from
            
        Returns:
            object: Loaded model
        """
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from {filepath}")
        return model
    
    def evaluate(self, X_test, y_test, threshold=0.5):
        """Evaluate model performance
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test labels
            threshold (float, optional): Classification threshold. Defaults to 0.5.
            
        Returns:
            dict: Dictionary with evaluation metrics
        """
        if hasattr(self.model, "predict_proba"):
            y_prob = self.model.predict_proba(X_test)[:, 1]
            y_pred = (y_prob >= threshold).astype(int)
        else:
            y_pred = self.model.predict(X_test)
            y_prob = y_pred  # Not all models have predict_proba
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        # Only calculate ROC AUC if we have probabilities
        if hasattr(self.model, "predict_proba"):
            metrics['roc_auc'] = roc_auc_score(y_test, y_prob)
        
        # Log metrics
        logger.info(f"Model evaluation results for {self.model_name}:")
        for metric, value in metrics.items():
            if metric != 'confusion_matrix':
                logger.info(f"{metric}: {value:.4f}")
        
        return metrics
    
    def plot_confusion_matrix(self, y_true, y_pred, figsize=(10, 8), save_path=None):
        """Plot confusion matrix
        
        Args:
            y_true (array-like): True labels
            y_pred (array-like): Predicted labels
            figsize (tuple, optional): Figure size. Defaults to (10, 8).
            save_path (str, optional): Path to save the figure. Defaults to None.
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {self.model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_roc_curve(self, y_true, y_score, figsize=(10, 8), save_path=None):
        """Plot ROC curve
        
        Args:
            y_true (array-like): True labels
            y_score (array-like): Predicted scores
            figsize (tuple, optional): Figure size. Defaults to (10, 8).
            save_path (str, optional): Path to save the figure. Defaults to None.
        """
        from sklearn.metrics import roc_curve, auc
        
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {self.model_name}')
        plt.legend(loc="lower right")
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"ROC curve saved to {save_path}")
        
        plt.show()