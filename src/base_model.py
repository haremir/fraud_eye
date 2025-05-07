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

logger = setup_logger(__name__)

class BaseModel:
    """Base class for all fraud detection models"""
    
    def __init__(self, model_name=None):
        """Initialize base model
        
        Args:
            model_name (str, optional): Name of the model. Defaults to None.
        """
        self.model_name = model_name
        self.model = None
        logger.info(f"Initialized {model_name if model_name else 'base'} model")
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train model with training data
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training labels
            X_val (pd.DataFrame, optional): Validation features. Defaults to None.
            y_val (pd.Series, optional): Validation labels. Defaults to None.
            
        Returns:
            self: The trained model object
        """
        raise NotImplementedError("Subclasses must implement train method")
    
    def predict(self, X):
        """Make predictions with trained model
        
        Args:
            X (pd.DataFrame): Features to predict
            
        Returns:
            np.ndarray: Predictions
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        return self.model.predict(X)
    
    def save(self, filepath):
        """Save model to disk
        
        Args:
            filepath (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")
            
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model from disk
        
        Args:
            filepath (str): Path to load the model from
            
        Returns:
            self: The model object with loaded model
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
            
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        logger.info(f"Model loaded from {filepath}")
        return self
    
    def evaluate(self, X_test, y_test, threshold=0.5):
        """Evaluate model performance
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test labels
            threshold (float, optional): Classification threshold. Defaults to 0.5.
            
        Returns:
            dict: Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
            
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
            'confusion_matrix': confusion_matrix(y_test, y_pred)
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
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Normal', 'Fraud'], 
                    yticklabels=['Normal', 'Fraud'])
        plt.title(f'Confusion Matrix - {self.model_name}')
        plt.ylabel('Gerçek Sınıf')
        plt.xlabel('Tahmin Edilen Sınıf')
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.tight_layout()
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
        plt.grid(True)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"ROC curve saved to {save_path}")
        
        plt.tight_layout()
        plt.show()
        
    def plot_precision_recall_curve(self, y_true, y_score, figsize=(10, 8), save_path=None):
        """Plot Precision-Recall curve
        
        Args:
            y_true (array-like): True labels
            y_score (array-like): Predicted scores
            figsize (tuple, optional): Figure size. Defaults to (10, 8).
            save_path (str, optional): Path to save the figure. Defaults to None.
        """
        from sklearn.metrics import precision_recall_curve, average_precision_score
        
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        avg_precision = average_precision_score(y_true, y_score)
        
        plt.figure(figsize=figsize)
        plt.step(recall, precision, color='b', where='post', lw=2,
                 label=f'PR curve (AP = {avg_precision:.4f})')
        plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title(f'Precision-Recall Curve - {self.model_name}')
        plt.legend(loc="lower left")
        plt.grid(True)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Precision-Recall curve saved to {save_path}")
        
        plt.tight_layout()
        plt.show()
        
    def get_feature_importance(self, feature_names=None, plot=True, top_n=20, figsize=(12, 8)):
        """Get feature importance (to be implemented by subclasses)
        
        Args:
            feature_names (list, optional): List of feature names. Defaults to None.
            plot (bool, optional): Whether to plot importance. Defaults to True.
            top_n (int, optional): Number of top features to show. Defaults to 20.
            figsize (tuple, optional): Figure size. Defaults to (12, 8).
            
        Returns:
            pd.DataFrame: DataFrame with feature importance
        """
        raise NotImplementedError("Feature importance must be implemented by subclasses if required")