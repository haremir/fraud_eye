"""
Hiperparametreleri ve yapılandırma ayarlarını içeren modül.
"""

# Genel konfigürasyon
CONFIG = {
    "verbose": True,
    "debug": False,
    "log_level": "INFO",
}

# Veri ön işleme konfigürasyonu
PREPROCESSING_CONFIG = {
    "remove_outliers": True,
    "outlier_threshold": 3.0,  # Z-score eşiği
    "scale_method": "standard",  # 'standard', 'minmax', 'robust'
    "handle_imbalance": True,
    "imbalance_method": "smote",  # 'smote', 'adasyn', 'random_oversampling'
    "feature_selection": False,
    "n_features_to_select": 15,
}

# XGBoost model hiperparametreleri
XGBOOST_PARAMS = {
    "max_depth": 5,
    "learning_rate": 0.1,
    "n_estimators": 100,
    "objective": "binary:logistic",
    "booster": "gbtree",
    "n_jobs": -1,
    "gamma": 0.1,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "scale_pos_weight": 1.0,  # Sınıf dengesizliği için ayarlanacak
    "random_state": 42,
}

# Lojistik Regresyon hiperparametreleri
LOGISTIC_REGRESSION_PARAMS = {
    "penalty": "l2",
    "C": 1.0,
    "solver": "liblinear",
    "max_iter": 100,
    "class_weight": "balanced",
    "random_state": 42,
}

# Cross-validation konfigürasyonu
CV_CONFIG = {
    "n_splits": 5,
    "shuffle": True,
    "random_state": 42,
}

# Model değerlendirme konfigürasyonu
EVALUATION_CONFIG = {
    "metrics": ["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"],
    "threshold": 0.5,  # Sınıflandırma eşiği
}

# Grafik konfigürasyonu
VISUALIZATION_CONFIG = {
    "figsize": (12, 8),
    "dpi": 100,
    "style": "seaborn-whitegrid",
    "palette": "viridis",
    "save_format": "png",
}