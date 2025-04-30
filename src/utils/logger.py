"""
Projedeki loglama işlemlerini yöneten modül.
"""
import os
import logging
from datetime import datetime
from ..constants import PROJECT_ROOT
from ..config import CONFIG

# Log dizini
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Log dosya adı
LOG_FILENAME = f"fraud_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
LOG_PATH = os.path.join(LOG_DIR, LOG_FILENAME)

# Log seviyesi ayarı
LOG_LEVEL = getattr(logging, CONFIG.get("log_level", "INFO"))

# Logger formatı
formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s"
)

# Logger yapılandırması
def setup_logger(name):
    """
    İstenilen modül için özelleştirilmiş bir logger oluşturur.
    
    Args:
        name: Logger adı (genelde __name__ kullanılır)
        
    Returns:
        logging.Logger: Yapılandırılmış logger nesnesi
    """
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)
    
    # Konsol işleyicisi
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Dosya işleyicisi
    file_handler = logging.FileHandler(LOG_PATH)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Debug modu kontrolü
    if not CONFIG.get("debug", False):
        logger.debug = lambda *args, **kwargs: None
    
    return logger

def log_data_info(logger, df, message="Veri seti özeti:"):
    """
    Veri çerçevesi hakkında özet bilgileri loglar.
    
    Args:
        logger: Logger nesnesi
        df: pandas DataFrame
        message: Log mesajı başlığı
    """
    logger.info(f"{message}\n"
                f"Boyut: {df.shape}\n"
                f"Bellek kullanımı: {df.memory_usage().sum() / (1024 * 1024):.2f} MB\n"
                f"Sütunlar: {df.columns.tolist()}\n"
                f"Veri tipleri:\n{df.dtypes}")
    
    if hasattr(df, 'isnull'):
        null_counts = df.isnull().sum()
        if null_counts.sum() > 0:
            logger.warning(f"Eksik değerler:\n{null_counts[null_counts > 0]}")

def log_step(logger, step_name, start=True):
    """
    İşlem adımlarını daha görünür şekilde loglamak için kullanılır.
    
    Args:
        logger: Logger nesnesi
        step_name: İşlem adımının adı
        start: Adım başlangıcı ise True, bitişi ise False
    """
    action = "BAŞLANIYOR" if start else "TAMAMLANDI"
    logger.info(f"{'='*20} {step_name} {action} {'='*20}")

def log_model_summary(logger, model_name, params, metrics):
    """
    Model eğitim sonuçlarını loglar.
    
    Args:
        logger: Logger nesnesi
        model_name: Model adı
        params: Model parametreleri
        metrics: Model performans metrikleri
    """
    logger.info(f"Model: {model_name}")
    logger.info(f"Parametreler: {params}")
    logger.info(f"Metrikler:")
    
    for metric_name, value in metrics.items():
        if isinstance(value, (int, float)):
            logger.info(f"  - {metric_name}: {value:.4f}")
        else:
            logger.info(f"  - {metric_name}: {value}")