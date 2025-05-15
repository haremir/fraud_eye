import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    average_precision_score, roc_auc_score
)
import io
import re
from typing import List, Dict, Union, Tuple, Optional


def classification_report_to_dataframe(report: str) -> pd.DataFrame:
    """
    Sklearn classification_report string çıktısını DataFrame'e dönüştürür.
    
    Args:
        report (str): sklearn.metrics.classification_report() fonksiyonunun çıktısı
        
    Returns:
        pd.DataFrame: Report içeriğini içeren DataFrame
    """
    report_data = []
    lines = report.split('\n')
    
    for line in lines[2:-3]:  # Header ve footer satırlarını atlayalım
        line = line.strip()
        if not line:
            continue
            
        # Satırı parçalara ayıralım
        row_data = re.split(r'\s+', line)
        
        if len(row_data) < 5:  # Eksik değerler varsa atlayalım
            continue
            
        class_name = row_data[0]
        precision = float(row_data[1])
        recall = float(row_data[2])
        f1_score = float(row_data[3])
        support = int(row_data[4])
        
        report_data.append({
            'class': class_name,
            'precision': precision,
            'recall': recall,
            'f1-score': f1_score,
            'support': support
        })
    
    # 'accuracy', 'macro avg' ve 'weighted avg' satırlarını ekleyelim
    for line in lines[-3:]:
        line = line.strip()
        if not line:
            continue
            
        row_data = re.split(r'\s+', line, maxsplit=1)
        
        if len(row_data) < 2:
            continue
            
        class_name = row_data[0]
        
        # Geri kalan sayıları ayıralım
        numbers = re.findall(r'\d+\.\d+|\d+', row_data[1])
        
        if len(numbers) >= 3:
            precision = float(numbers[0]) if numbers[0] != 'nan' else np.nan
            recall = float(numbers[1]) if numbers[1] != 'nan' else np.nan
            f1_score = float(numbers[2]) if numbers[2] != 'nan' else np.nan
            
            if len(numbers) >= 4:
                support = int(float(numbers[3]))
            else:
                support = np.nan
                
            report_data.append({
                'class': class_name,
                'precision': precision,
                'recall': recall,
                'f1-score': f1_score,
                'support': support
            })
    
    return pd.DataFrame(report_data)


def plot_confusion_matrix(y_true, y_pred, class_names=['Normal', 'Fraud'], title='Confusion Matrix', 
                        figsize=(8, 6), normalize=False):
    """
    Karmaşıklık matrisini görselleştirir.
    
    Args:
        y_true (array): Gerçek etiketler
        y_pred (array): Tahmin edilen etiketler
        class_names (list): Sınıf isimleri
        title (str): Grafik başlığı
        figsize (tuple): Grafik boyutu
        normalize (bool): Normalize edilsin mi?
        
    Returns:
        matplotlib.figure.Figure: Oluşturulan grafik nesnesi
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
               xticklabels=class_names,
               yticklabels=class_names)
    plt.title(title)
    plt.ylabel('Gerçek Sınıf')
    plt.xlabel('Tahmin Edilen Sınıf')
    plt.tight_layout()
    
    return plt.gcf()


def plot_roc_curve(y_true, y_proba, title='ROC Curve', figsize=(8, 6)):
    """
    ROC eğrisini çizer.
    
    Args:
        y_true (array): Gerçek etiketler
        y_proba (array): Pozitif sınıf olasılıkları
        title (str): Grafik başlığı
        figsize (tuple): Grafik boyutu
        
    Returns:
        tuple: (Figure, AUC değeri)
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    
    return plt.gcf(), roc_auc


def plot_precision_recall_curve(y_true, y_proba, title='Precision-Recall Curve', figsize=(8, 6)):
    """
    Kesinlik-Duyarlılık eğrisini çizer.
    
    Args:
        y_true (array): Gerçek etiketler
        y_proba (array): Pozitif sınıf olasılıkları
        title (str): Grafik başlığı
        figsize (tuple): Grafik boyutu
        
    Returns:
        tuple: (Figure, PR AUC değeri)
    """
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=figsize)
    plt.plot(recall, precision, label=f'PR AUC = {pr_auc:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.tight_layout()
    
    return plt.gcf(), pr_auc


def plot_classification_report(report, title='Sınıflandırma Raporu', figsize=(12, 7)):
    """
    Sınıflandırma raporunu görselleştirir.
    
    Args:
        report (str): Sklearn classification_report çıktısı
        title (str): Grafik başlığı
        figsize (tuple): Grafik boyutu
        
    Returns:
        tuple: (Heatmap Figure, Bar Plot Figure)
    """
    # Raporu DataFrame'e dönüştür
    report_df = classification_report_to_dataframe(report)
    
    # 1. Isı Haritası (Heatmap)
    plt.figure(figsize=figsize)
    
    # Isı haritası için değerleri hazırlayalım
    metrics_df = report_df.copy()
    metrics_df = metrics_df.set_index('class')
    metrics_df = metrics_df[['precision', 'recall', 'f1-score']]
    
    # NaN değerleri 0 ile dolduralım (görselleştirme için)
    metrics_df = metrics_df.fillna(0)
    
    # Isı haritası
    ax = sns.heatmap(metrics_df, annot=True, cmap="YlGnBu", fmt='.2f', 
                    linewidths=.5, vmin=0, vmax=1)
    plt.title(f'{title}', fontsize=14)
    plt.ylabel('Sınıf')
    plt.xlabel('Metrikler')
    
    # Support değerlerini ekleyelim
    for i, (_, row) in enumerate(report_df.iterrows()):
        plt.text(3.2, i + 0.5, f"Support: {row['support']}", 
                 va='center', ha='left', fontsize=9)
    
    plt.tight_layout()
    heatmap_fig = plt.gcf()
    
    # 2. Çubuk Grafik (Bar Plot)
    plt.figure(figsize=figsize)
    
    # Uzun format DataFrame'e dönüştürme
    metrics_long = pd.melt(report_df, 
                          id_vars=['class', 'support'],
                          value_vars=['precision', 'recall', 'f1-score'],
                          var_name='metric', 
                          value_name='value')
    
    # NaN değerlerini filtreleyebiliriz
    metrics_long = metrics_long.dropna(subset=['value'])
    
    # Çubuk grafik
    sns.barplot(x='class', y='value', hue='metric', data=metrics_long, palette='viridis')
    plt.title(f'{title} - Metrikler', fontsize=14)
    plt.ylabel('Değer', fontsize=12)
    plt.xlabel('Sınıf', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Metrik')
    
    # Support değerlerini ekleyelim
    for i, cls in enumerate(report_df['class'].unique()):
        support = report_df[report_df['class'] == cls]['support'].values[0]
        plt.text(i, -0.05, f"Support: {support}", ha='center', fontsize=9)
    
    plt.ylim(0, 1.1)
    plt.tight_layout()
    barplot_fig = plt.gcf()
    
    return heatmap_fig, barplot_fig


def plot_model_comparison_roc(models_dict, X_test, y_test, title='Model Karşılaştırması - ROC', figsize=(10, 8)):
    """
    Birden fazla modelin ROC eğrilerini karşılaştırır.
    
    Args:
        models_dict (dict): {model_adı: model_nesnesi} şeklinde sözlük
        X_test (array): Test verileri
        y_test (array): Test etiketleri
        title (str): Grafik başlığı
        figsize (tuple): Grafik boyutu
        
    Returns:
        matplotlib.figure.Figure: Oluşturulan grafik nesnesi
    """
    plt.figure(figsize=figsize)
    
    for model_name, model in models_dict.items():
        # Model tipine göre tahmin fonksiyonunu ayarlayalım
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = model.predict(X_test)
        
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.4f})')
    
    # Tesadüfi tahmin için çizgi
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    
    return plt.gcf()


def plot_model_comparison_pr(models_dict, X_test, y_test, title='Model Karşılaştırması - PR', figsize=(10, 8)):
    """
    Birden fazla modelin Precision-Recall eğrilerini karşılaştırır.
    
    Args:
        models_dict (dict): {model_adı: model_nesnesi} şeklinde sözlük
        X_test (array): Test verileri
        y_test (array): Test etiketleri
        title (str): Grafik başlığı
        figsize (tuple): Grafik boyutu
        
    Returns:
        matplotlib.figure.Figure: Oluşturulan grafik nesnesi
    """
    plt.figure(figsize=figsize)
    
    for model_name, model in models_dict.items():
        # Model tipine göre tahmin fonksiyonunu ayarlayalım
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = model.predict(X_test)
        
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        avg_precision = average_precision_score(y_test, y_proba)
        plt.step(recall, precision, where='post', lw=2, 
                 label=f'{model_name} (AP = {avg_precision:.4f})')
    
    # Rastgele sınıflandırıcı için çizgi (veri setinin ortalama pozitif oranı)
    plt.axhline(y=np.mean(y_test), linestyle='--', color='r')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(title)
    plt.legend(loc="best")
    plt.grid(True)
    
    return plt.gcf()


def plot_feature_importance(feature_importance, top_n=20, title='Özellik Önem Sıralaması', figsize=(12, 10)):
    """
    Özellik önem sıralamasını görselleştirir.
    
    Args:
        feature_importance (pd.DataFrame): 'feature' ve 'importance' sütunlarını içeren DataFrame
        top_n (int): Gösterilecek özellik sayısı
        title (str): Grafik başlığı
        figsize (tuple): Grafik boyutu
        
    Returns:
        matplotlib.figure.Figure: Oluşturulan grafik nesnesi
    """
    plt.figure(figsize=figsize)
    
    # En önemli N özelliği seç
    top_features = feature_importance.head(top_n)
    
    # Özellik önemini görselleştirelim
    ax = sns.barplot(x='importance', y='feature', data=top_features)
    
    # Değerleri çubukların yanına ekleyelim
    for i, v in enumerate(top_features['importance']):
        ax.text(v + 0.001, i, f"{v:.4f}", va='center')
    
    plt.title(f'{title} (Top {top_n})')
    plt.tight_layout()
    
    return plt.gcf()


def plot_threshold_optimization(y_test, y_proba, title='Eşik Değeri Optimizasyonu', figsize=(10, 6)):
    """
    Farklı eşik değerleri için metrik değişimini görselleştirir.
    
    Args:
        y_test (array): Gerçek etiketler
        y_proba (array): Pozitif sınıf olasılıkları
        title (str): Grafik başlığı
        figsize (tuple): Grafik boyutu
        
    Returns:
        tuple: (Figure, optimal_threshold, optimal_f1)
    """
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    thresholds = np.arange(0, 1, 0.01)
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    for threshold in thresholds:
        # Verilen eşik değerine göre tahminleri oluştur
        y_pred = (y_proba >= threshold).astype(int)
        
        # Metrikleri hesapla
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
    
    # En yüksek F1 skoruna sahip eşik değerini bulalım
    optimal_threshold_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_threshold_idx]
    
    # Sonuçları görselleştirelim
    plt.figure(figsize=figsize)
    plt.plot(thresholds, precision_scores, label='Precision')
    plt.plot(thresholds, recall_scores, label='Recall')
    plt.plot(thresholds, f1_scores, label='F1 Score')
    plt.axvline(x=0.5, color='r', linestyle='--', label='Default Threshold (0.5)')
    plt.axvline(x=optimal_threshold, color='g', linestyle='--', 
                label=f'Optimal Threshold ({optimal_threshold:.2f})')
    
    plt.title(title)
    plt.xlabel('Eşik Değeri')
    plt.ylabel('Skor')
    plt.legend()
    plt.grid(True)
    
    return plt.gcf(), optimal_threshold, f1_scores[optimal_threshold_idx]


def create_model_comparison_table(models_dict, X_test, y_test):
    """
    Modellerin performans metriklerini karşılaştıran bir tablo oluşturur.
    
    Args:
        models_dict (dict): {model_adı: model_nesnesi} şeklinde sözlük
        X_test (array): Test verileri
        y_test (array): Test etiketleri
        
    Returns:
        pd.DataFrame: Performans metriklerini içeren DataFrame
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    results = []
    
    for model_name, model in models_dict.items():
        # Model tipine göre tahmin fonksiyonunu ayarlayalım
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_proba >= 0.5).astype(int)
        else:
            y_pred = model.predict(X_test)
            y_proba = y_pred  # Olasılık değerleri yoksa, direkt tahminleri kullan
            
        # Metrikleri hesapla
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_proba)
        
        results.append({
            'Model': model_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'AUC': auc_score
        })
    
    return pd.DataFrame(results)

def save_figure(fig, filename, version='V1', reports_dir=None, show=True):
    """
    Görseli kaydeder ve istenirse ekranda gösterir.

    Args:
        fig (matplotlib.figure.Figure): Kaydedilecek figür
        filename (str): Dosya adı
        version (str): Versiyon etiketi (V1, V2, ...) - klasör adı olarak kullanılır
        reports_dir (str, optional): Rapor dizini. None ise '../reports' varsayılan değeri kullanılır.
        show (bool): Görsel ekranda gösterilsin mi?
    """
    # Varsayılan reports dizini
    if reports_dir is None:
        reports_dir = '../reports'

    # Versiyon klasörü yolu
    fig_path = os.path.join(reports_dir, 'figures', version, filename)

    # Klasörü oluştur (yoksa)
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)

    # Kaydet
    fig.savefig(fig_path, bbox_inches='tight', dpi=300)
    print(f"✔ Görsel kaydedildi: {fig_path}")

    # Göster (istenirse)
    if show:
        plt.show()
    else:
        plt.close(fig)