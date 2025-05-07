import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_curve, auc
)
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.logger import setup_logger
from src.base_model import BaseModel
import src.config as config
import src.constants as constants

logger = setup_logger(__name__)

class XGBoostModel(BaseModel):
    """XGBoost implementation for fraud detection."""
    
    def __init__(self, params=None):
        """Initialize XGBoost model with parameters."""
        super().__init__(model_name="xgboost")
        self.params = params or config.XGBOOST_PARAMS
        self.model = None
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train XGBoost model."""
        logger.info(f"Training XGBoost model with {X_train.shape[0]} samples")
        
        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        
        # If validation set provided, use early stopping
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evallist = [(dtrain, 'train'), (dval, 'validation')]
            
            self.model = xgb.train(
                params=self.params,
                dtrain=dtrain,
                num_boost_round=self.params.get('num_boost_round', 1000),
                evals=evallist,
                early_stopping_rounds=self.params.get('early_stopping_rounds', 50),
                verbose_eval=self.params.get('verbose_eval', 100)
            )
            
            logger.info(f"Best iteration: {self.model.best_iteration}")
        else:
            # Train without early stopping
            self.model = xgb.train(
                params=self.params,
                dtrain=dtrain,
                num_boost_round=self.params.get('num_boost_round', 1000),
                verbose_eval=self.params.get('verbose_eval', 100)
            )
        
        # Get feature importance
        feature_importance = self.model.get_score(importance_type='gain')
        
        logger.info("XGBoost model training completed")
        return self
    
    def predict(self, X):
        """Make predictions with trained model."""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        logger.info(f"Making predictions on {X.shape[0]} samples")
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)
    
    def predict_proba(self, X):
        """Make probability predictions with trained model."""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        logger.info(f"Making probability predictions on {X.shape[0]} samples")
        dtest = xgb.DMatrix(X)
        probas = self.model.predict(dtest)
        # XGBoost returns only positive class probabilities
        # We need to convert to the format expected by scikit-learn (both classes)
        return np.vstack((1 - probas, probas)).T
    
    def evaluate(self, X_test, y_test, threshold=0.5):
        """Evaluate model performance."""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        logger.info(f"Evaluating model on {X_test.shape[0]} samples")
        
        # Get predictions
        dtest = xgb.DMatrix(X_test)
        y_pred_proba = self.model.predict(dtest)
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate metrics
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Calculate Precision-Recall AUC
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
        plt.xlabel('Tahmin Edilen Sınıf')
        plt.ylabel('Gerçek Sınıf')
        plt.title('Confusion Matrix - XGBoost')
        plt.tight_layout()
        
        reports_dir = constants.REPORTS_FIGURES_DIR
        os.makedirs(reports_dir, exist_ok=True)
        plt.savefig(os.path.join(reports_dir, 'xgboost_confusion_matrix.png'))
        plt.show()
        
        # Plot ROC curve
        dtest = xgb.DMatrix(X_test)
        y_pred_proba = self.model.predict(dtest)
        
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, label=f'ROC AUC = {results["roc_auc"]:.4f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - XGBoost')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(os.path.join(reports_dir, 'xgboost_roc_curve.png'))
        plt.tight_layout()
        plt.show()
        
        # Plot Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, label=f'PR AUC = {results["pr_auc"]:.4f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve - XGBoost')
        plt.legend(loc="lower left")
        plt.grid(True)
        plt.savefig(os.path.join(reports_dir, 'xgboost_pr_curve.png'))
        plt.tight_layout()
        plt.show()
        
        return results
    
    def get_feature_importance(self, feature_names=None, plot=True, top_n=20, figsize=(12, 8)):
        """Get and optionally plot feature importance."""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
            
        importance = self.model.get_score(importance_type='gain')
        
        if feature_names is not None:
            # Map numerical features to feature names if provided
            mapped_importance = {}
            for feat_idx, importance_val in importance.items():
                # XGBoost uses f0, f1, ... format for features
                if feat_idx.startswith('f'):
                    idx = int(feat_idx[1:])
                    if idx < len(feature_names):
                        mapped_importance[feature_names[idx]] = importance_val
                else:
                    mapped_importance[feat_idx] = importance_val
            importance = mapped_importance
        
        # Convert to DataFrame for easier manipulation
        importance_df = pd.DataFrame({
            'feature': list(importance.keys()),
            'importance': list(importance.values())
        })
        importance_df = importance_df.sort_values('importance', ascending=False).head(top_n)
        
        if plot:
            plt.figure(figsize=figsize)
            sns.barplot(x='importance', y='feature', data=importance_df)
            plt.title(f'Top {top_n} Feature Importance (gain)')
            plt.tight_layout()
            
            reports_dir = constants.REPORTS_FIGURES_DIR
            os.makedirs(reports_dir, exist_ok=True)
            plt.savefig(os.path.join(reports_dir, 'xgboost_feature_importance.png'))
            plt.show()
        
        return importance_df


def train_xgboost(X_train, y_train, X_val=None, y_val=None, params=None):
    """Train XGBoost model with given parameters."""
    model = XGBoostModel(params=params)
    model.train(X_train, y_train, X_val, y_val)
    return model


def evaluate_xgboost(model, X_test, y_test, threshold=0.5):
    """Evaluate XGBoost model."""
    return model.evaluate(X_test, y_test, threshold=threshold)