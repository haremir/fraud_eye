"""
Preprocessing pipeline for fraud detection models.
"""
import os
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler
from src.utils.logger import setup_logger
from src.constants import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TARGET, ID_FEATURES
from src.preprocessing.feature_engineering import (
    handle_missing_values, scale_features, create_time_features,
    create_amount_features, apply_pca, create_interaction_features
)

logger = setup_logger("preprocessing_pipeline")

class FraudPreprocessor:
    """Preprocessing pipeline for fraud detection"""
    
    def __init__(self, config=None):
        """Initialize preprocessing pipeline
        
        Args:
            config (dict, optional): Configuration dictionary. Defaults to None.
        """
        self.config = config or {}
        self.scalers = {}
        self.pca = None
        logger.info("Initialized FraudPreprocessor")
    
    def preprocess(self, df, training=True):
        """Apply preprocessing steps
        
        Args:
            df (pd.DataFrame): Input dataframe
            training (bool, optional): Whether in training mode. Defaults to True.
        
        Returns:
            pd.DataFrame: Preprocessed dataframe
        """
        logger.info(f"Preprocessing data {'(training mode)' if training else '(inference mode)'}")
        
        # Make a copy to avoid modifying the original dataframe
        df_copy = df.copy()
        
        # 1. Handle missing values
        missing_strategy = self.config.get("missing_strategy", "median")
        df_copy = handle_missing_values(df_copy, strategy=missing_strategy)
        
        # 2. Create time features if Time column exists
        if "Time" in df_copy.columns:
            df_copy = create_time_features(df_copy)
        
        # 3. Create amount features if Amount column exists
        if "Amount" in df_copy.columns:
            df_copy = create_amount_features(df_copy)
        
        # 4. Scale features
        scaler_type = self.config.get("scaler", "standard")
        if training:
            df_copy, scaler = scale_features(df_copy, scaler_type=scaler_type)
            self.scalers["main_scaler"] = scaler
        else:
            if "main_scaler" in self.scalers:
                scaler = self.scalers["main_scaler"]
                features = [f for f in NUMERICAL_FEATURES if f in df_copy.columns]
                df_copy[features] = scaler.transform(df_copy[features])
            else:
                logger.warning("No scaler found for inference, using new scaler")
                df_copy, _ = scale_features(df_copy, scaler_type=scaler_type)
        
        # 5. Create interaction features (optional)
        if self.config.get("use_interactions", False):
            df_copy = create_interaction_features(df_copy)
        
        # 6. Apply PCA (optional)
        if self.config.get("use_pca", False):
            n_components = self.config.get("pca_components", 0.95)
            if training:
                df_copy, pca = apply_pca(df_copy, n_components=n_components)
                self.pca = pca
            else:
                if self.pca is not None:
                    features = [col for col in df_copy.columns if col != TARGET and col in NUMERICAL_FEATURES]
                    pca_result = self.pca.transform(df_copy[features])
                    pca_cols = [f'pca_{i}' for i in range(pca_result.shape[1])]
                    pca_df = pd.DataFrame(pca_result, columns=pca_cols, index=df_copy.index)
                    
                    # Keep non-PCA columns
                    non_pca_cols = [col for col in df_copy.columns if col not in features]
                    df_copy = pd.concat([df_copy[non_pca_cols], pca_df], axis=1)
                else:
                    logger.warning("No PCA found for inference, skipping PCA transformation")
        
        logger.info(f"Preprocessing complete. Output shape: {df_copy.shape}")
        return df_copy
    
    def get_features_and_target(self, df, include_target=True):
        """Split dataframe into features and target
        
        Args:
            df (pd.DataFrame): Input dataframe
            include_target (bool, optional): Whether to include target. Defaults to True.
        
        Returns:
            tuple: Features dataframe and target series (if include_target is True)
        """
        logger.info("Splitting features and target")
        
        # Identify feature columns (excluding target and ID columns)
        feature_cols = [col for col in df.columns if col != TARGET and col not in ID_FEATURES]
        
        X = df[feature_cols]
        
        if include_target and TARGET in df.columns:
            y = df[TARGET]
            return X, y
        else:
            return X
    
    def save(self, filepath):
        """Save preprocessor to disk
        
        Args:
            filepath (str): Path to save the preprocessor
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Preprocessor saved to {filepath}")
    
    @staticmethod
    def load(filepath):
        """Load preprocessor from disk
        
        Args:
            filepath (str): Path to load the preprocessor from
            
        Returns:
            FraudPreprocessor: Loaded preprocessor
        """
        with open(filepath, 'rb') as f:
            preprocessor = pickle.load(f)
        logger.info(f"Preprocessor loaded from {filepath}")
        return preprocessor
    
    def get_feature_importance(self, features, model):
        """Get feature importance from model
        
        Args:
            features (list): List of feature names
            model: Trained model with feature_importances_ attribute
            
        Returns:
            pd.DataFrame: Dataframe with feature importance
        """
        if not hasattr(model, 'feature_importances_'):
            logger.warning("Model doesn't have feature_importances_ attribute")
            return None
        
        importance = model.feature_importances_
        if len(importance) != len(features):
            logger.warning(f"Feature count mismatch: {len(importance)} importances, {len(features)} features")
            return None
        
        # Create a dataframe of feature importances
        feature_importance = pd.DataFrame({'feature': features, 'importance': importance})
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        return feature_importance