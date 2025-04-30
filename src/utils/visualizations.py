"""
Visualization utilities for fraud detection project.
Contains functions to generate plots for EDA and model evaluation.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
from src.utils.logger import setup_logger

logger = setup_logger("visualizations")

def plot_class_distribution(y, title="Class Distribution"):
    """
    Plot the distribution of classes in the target variable.
    
    Args:
        y (pd.Series): Target variable
        title (str): Plot title
    """
    plt.figure(figsize=(10, 6))
    class_counts = pd.Series(y).value_counts().sort_index()
    sns.barplot(x=class_counts.index, y=class_counts.values)
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Count")
    
    # Add percentage labels
    total = len(y)
    for i, count in enumerate(class_counts.values):
        percentage = 100 * count / total
        plt.text(i, count + 5, f"{percentage:.2f}%", ha='center')
    
    plt.tight_layout()
    logger.info(f"Generated class distribution plot")
    return plt.gcf()

def plot_correlation_matrix(X, title="Feature Correlation Matrix", figsize=(12, 10)):
    """
    Plot correlation matrix for input features.
    
    Args:
        X (pd.DataFrame): Feature dataframe
        title (str): Plot title
        figsize (tuple): Figure size
    """
    plt.figure(figsize=figsize)
    corr_matrix = X.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, annot=False, fmt='.2f')
    
    plt.title(title)
    plt.tight_layout()
    logger.info(f"Generated correlation matrix plot with {X.shape[1]} features")
    return plt.gcf()

def plot_feature_importance(model, feature_names, top_n=20, title="Feature Importance"):
    """
    Plot feature importance for tree-based models.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names (list): List of feature names
        top_n (int): Number of top features to display
        title (str): Plot title
    """
    try:
        importances = model.feature_importances_
    except AttributeError:
        logger.warning("Model doesn't have feature_importances_ attribute")
        return None
    
    indices = np.argsort(importances)[::-1]
    top_indices = indices[:top_n]
    
    plt.figure(figsize=(10, 8))
    plt.title(title)
    plt.bar(range(top_n), importances[top_indices])
    plt.xticks(range(top_n), [feature_names[i] for i in top_indices], rotation=90)
    plt.tight_layout()
    logger.info(f"Generated feature importance plot showing top {top_n} features")
    return plt.gcf()

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", normalize=True):
    """
    Plot confusion matrix for binary classification.
    
    Args:
        y_true (array): True labels
        y_pred (array): Predicted labels
        title (str): Plot title
        normalize (bool): Whether to normalize the confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
    else:
        fmt = 'd'
    
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', cbar=False,
                xticklabels=['Non-Fraud (0)', 'Fraud (1)'],
                yticklabels=['Non-Fraud (0)', 'Fraud (1)'])
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(title)
    plt.tight_layout()
    logger.info(f"Generated confusion matrix plot")
    return plt.gcf()

def plot_roc_curve(y_true, y_pred_proba, title="ROC Curve"):
    """
    Plot ROC curve for binary classification.
    
    Args:
        y_true (array): True labels
        y_pred_proba (array): Predicted probabilities for the positive class
        title (str): Plot title
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    logger.info(f"Generated ROC curve plot with AUC: {roc_auc:.3f}")
    return plt.gcf(), roc_auc

def plot_precision_recall_curve(y_true, y_pred_proba, title="Precision-Recall Curve"):
    """
    Plot precision-recall curve for binary classification.
    
    Args:
        y_true (array): True labels
        y_pred_proba (array): Predicted probabilities for the positive class
        title (str): Plot title
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2, 
             label=f'Precision-Recall curve (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower left")
    plt.tight_layout()
    logger.info(f"Generated precision-recall curve plot with AUC: {pr_auc:.3f}")
    return plt.gcf(), pr_auc

def plot_learning_curve(train_sizes, train_scores, test_scores, title="Learning Curve"):
    """
    Plot learning curve to evaluate model performance with varying training set sizes.
    
    Args:
        train_sizes (array): Training set sizes
        train_scores (array): Training scores
        test_scores (array): Test scores
        title (str): Plot title
    """
    plt.figure(figsize=(10, 6))
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
    plt.plot(train_sizes, test_mean, 'o-', color='red', label='Cross-validation score')
    
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='red')
    
    plt.title(title)
    plt.xlabel('Training Examples')
    plt.ylabel('Score')
    plt.grid(True)
    plt.legend(loc='best')
    plt.tight_layout()
    logger.info("Generated learning curve plot")
    return plt.gcf()

def plot_time_series(df, time_col='Time', value_col='Amount', class_col='Class', title="Transaction Time Series"):
    """
    Plot time series data with fraud transactions highlighted.
    
    Args:
        df (pd.DataFrame): DataFrame containing time series data
        time_col (str): Column name for time
        value_col (str): Column name for value to plot
        class_col (str): Column name for class (fraud/non-fraud)
        title (str): Plot title
    """
    plt.figure(figsize=(15, 8))
    
    # Plot all transactions
    plt.scatter(df[time_col], df[value_col], alpha=0.5, s=10, label='Non-Fraud', color='blue')
    
    # Highlight fraud transactions
    fraud_df = df[df[class_col] == 1]
    plt.scatter(fraud_df[time_col], fraud_df[value_col], alpha=0.7, s=30, label='Fraud', color='red')
    
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel(value_col)
    plt.legend()
    plt.tight_layout()
    
    logger.info(f"Generated time series plot with {len(fraud_df)} fraud transactions highlighted")
    return plt.gcf()

def save_figure(fig, filename, dpi=300):
    """
    Save figure to file.
    
    Args:
        fig (matplotlib.figure.Figure): Figure to save
        filename (str): Output filename
        dpi (int): DPI for saved figure
    """
    fig.savefig(filename, dpi=dpi, bbox_inches='tight')
    logger.info(f"Saved figure to {filename}")