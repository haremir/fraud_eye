"""
Sabit değerleri içeren modül.
"""
import os
from pathlib import Path

# Proje dizin yapısı
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
METRICS_DIR = os.path.join(REPORTS_DIR, "metrics")

# Veri setine ilişkin sabitler
DATA_FILENAME = "creditcard.csv"
RAW_DATA_PATH = os.path.join(RAW_DATA_DIR, DATA_FILENAME)
PROCESSED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "processed_" + DATA_FILENAME)

# Model dosya yolları
XGBOOST_MODEL_PATH = os.path.join(MODELS_DIR, "xgboost_v1.pkl")
PREPROCESSING_PIPELINE_PATH = os.path.join(MODELS_DIR, "preprocessing_pipeline.pkl")

# Veri özelliklerine ilişkin sabitler
TIME_COLUMN = "Time"
AMOUNT_COLUMN = "Amount"
CLASS_COLUMN = "Class"
FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)]
NUMERIC_COLUMNS = [TIME_COLUMN, AMOUNT_COLUMN] + FEATURE_COLUMNS

# Sınıf değerleri
NORMAL_CLASS = "0"
FRAUD_CLASS = "1"

# Performans metriklerine ilişkin sabitler
METRICS_FILE = os.path.join(METRICS_DIR, "model_metrics.json")

# Diğer sabitler
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.25  # Test setinin içinden validation seti oranı