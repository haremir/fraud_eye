"""
Feature engineering module for fraud detection.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, NearMiss, TomekLinks
from imblearn.combine import SMOTETomek, SMOTEENN
from src.utils.logger import setup_logger
from src.constants import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TARGET

logger = setup_logger("feature_engineering")

def handle_missing_values(df, strategy="median"):
    """Handle missing values in the dataset
    
    Args:
        df (pd.DataFrame): Input dataframe
        strategy (str, optional): Strategy for handling missing values. Defaults to "median".
    
    Returns:
        pd.DataFrame: Dataframe with handled missing values
    """
    logger.info(f"Handling missing values with strategy: {strategy}")
    
    # Make a copy to avoid modifying the original dataframe
    df_copy = df.copy()
    
    # Check for missing values
    missing_cols = df_copy.columns[df_copy.isna().any()].tolist()
    if missing_cols:
        logger.info(f"Found missing values in columns: {missing_cols}")
        
        for col in missing_cols:
            if col in NUMERICAL_FEATURES:
                if strategy == "median":
                    df_copy[col].fillna(df_copy[col].median(), inplace=True)
                elif strategy == "mean":
                    df_copy[col].fillna(df_copy[col].mean(), inplace=True)
                elif strategy == "zero":
                    df_copy[col].fillna(0, inplace=True)
            elif col in CATEGORICAL_FEATURES:
                df_copy[col].fillna(df_copy[col].mode().iloc[0], inplace=True)
    else:
        logger.info("No missing values found in the dataset")
    
    return df_copy

def handle_class_imbalance(X, y, method="smote", sampling_strategy=None, random_state=42):
    """Handle class imbalance in the dataset
    
    Args:
        X (pd.DataFrame): Features dataframe
        y (pd.Series): Target series
        method (str, optional): Method to handle imbalance. Defaults to "smote".
            Options: 
                - "smote": Synthetic Minority Over-sampling Technique
                - "adasyn": Adaptive Synthetic Sampling
                - "random_over": Random Over-sampling
                - "random_under": Random Under-sampling
                - "nearmiss": NearMiss Under-sampling
                - "tomek": Tomek Links Under-sampling
                - "smote_tomek": SMOTE + Tomek Links
                - "smote_enn": SMOTE + Edited Nearest Neighbors
                - "none": No resampling
        sampling_strategy (float, str, dict, optional): Sampling strategy. 
            When float: ratio of minority to majority class.
            When "auto": Automatically determined.
            When dict: Specific ratio for each class.
            Defaults to None (automatically determined based on method).
        random_state (int, optional): Random seed. Defaults to 42.
        
    Returns:
        tuple: Resampled X and y (pd.DataFrame, pd.Series)
    """
    logger.info(f"Handling class imbalance using method: {method}")
    
    # Check class distribution before
    class_dist_before = y.value_counts()
    logger.info(f"Class distribution before: {class_dist_before.to_dict()}")
    
    if method == "none":
        logger.info("No resampling requested, returning original data")
        return X, y
    
    # Set default sampling strategy if none provided
    if sampling_strategy is None:
        if method in ["random_under", "nearmiss", "tomek"]:
            # Default for undersampling: keep all minority samples
            sampling_strategy = "auto"
        else:
            # Default for oversampling: bring minority class to 50% of majority
            # For fraud detection, we don't want to create too many synthetic samples
            sampling_strategy = 0.5
    
    # Initialize resampler based on method
    if method == "smote":
        resampler = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    elif method == "adasyn":
        resampler = ADASYN(sampling_strategy=sampling_strategy, random_state=random_state)
    elif method == "random_over":
        resampler = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=random_state)
    elif method == "random_under":
        resampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=random_state)
    elif method == "nearmiss":
        resampler = NearMiss(sampling_strategy=sampling_strategy, version=2)
    elif method == "tomek":
        resampler = TomekLinks(sampling_strategy=sampling_strategy)
    elif method == "smote_tomek":
        resampler = SMOTETomek(sampling_strategy=sampling_strategy, random_state=random_state)
    elif method == "smote_enn":
        resampler = SMOTEENN(sampling_strategy=sampling_strategy, random_state=random_state)
    else:
        logger.warning(f"Unknown imbalance handling method: {method}. No resampling applied.")
        return X, y
    
    # Apply resampling
    logger.info(f"Applying {method} resampling...")
    X_resampled, y_resampled = resampler.fit_resample(X, y)
    
    # Convert back to pandas objects to maintain consistency
    if not isinstance(X_resampled, pd.DataFrame):
        X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    if not isinstance(y_resampled, pd.Series):
        y_resampled = pd.Series(y_resampled, name=y.name)
    
    # Check class distribution after
    class_dist_after = y_resampled.value_counts()
    logger.info(f"Class distribution after: {class_dist_after.to_dict()}")
    logger.info(f"Resampled data shape: {X_resampled.shape}, {y_resampled.shape}")
    
    return X_resampled, y_resampled

def scale_features(df, scaler_type="standard", features=None):
    """Scale numerical features
    
    Args:
        df (pd.DataFrame): Input dataframe
        scaler_type (str, optional): Type of scaler. Defaults to "standard".
        features (list, optional): List of features to scale. Defaults to None.
    
    Returns:
        tuple: Scaled dataframe and scaler object
    """
    logger.info(f"Scaling features using {scaler_type} scaler")
    
    # Make a copy to avoid modifying the original dataframe
    df_copy = df.copy()
    
    # If features not specified, use all numerical features
    if features is None:
        features = NUMERICAL_FEATURES
    
    # Filter out features that don't exist in the dataframe
    features = [f for f in features if f in df_copy.columns]
    
    if not features:
        logger.warning("No features to scale")
        return df_copy, None
    
    # Initialize scaler based on type
    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "robust":
        scaler = RobustScaler()
    else:
        logger.warning(f"Unknown scaler type: {scaler_type}. Using StandardScaler")
        scaler = StandardScaler()
    
    # Fit and transform
    df_copy[features] = scaler.fit_transform(df_copy[features])
    
    logger.info(f"Scaled {len(features)} features")
    return df_copy, scaler

def create_time_features(df, time_column="Time"):
    """Create time-based features from a time column
    
    Args:
        df (pd.DataFrame): Input dataframe
        time_column (str, optional): Time column name. Defaults to "Time".
    
    Returns:
        pd.DataFrame: Dataframe with additional time features
    """
    logger.info("Creating time-based features")
    
    # Make a copy to avoid modifying the original dataframe
    df_copy = df.copy()
    
    if time_column not in df_copy.columns:
        logger.warning(f"Time column {time_column} not found in the dataset")
        return df_copy
    
    # In this specific dataset, Time represents seconds elapsed between each transaction
    # Let's create some time-based features
    
    # Hour of the day (assuming Time is in seconds from a starting point)
    df_copy['hour_of_day'] = (df_copy[time_column] % (24 * 3600)) // 3600
    
    # Day of the week (0 = Monday, 6 = Sunday)
    # This is a simplification, actual day depends on starting point
    df_copy['day_of_week'] = ((df_copy[time_column] // (24 * 3600)) % 7).astype(int)
    
    # Is weekend
    df_copy['is_weekend'] = df_copy['day_of_week'].isin([5, 6]).astype(int)
    
    logger.info("Created time-based features: hour_of_day, day_of_week, is_weekend")
    return df_copy

def create_amount_features(df, amount_column="Amount"):
    """Create features based on transaction amount
    
    Args:
        df (pd.DataFrame): Input dataframe
        amount_column (str, optional): Amount column name. Defaults to "Amount".
    
    Returns:
        pd.DataFrame: Dataframe with additional amount features
    """
    logger.info("Creating amount-based features")
    
    # Make a copy to avoid modifying the original dataframe
    df_copy = df.copy()
    
    if amount_column not in df_copy.columns:
        logger.warning(f"Amount column {amount_column} not found in the dataset")
        return df_copy
    
    # Log transform of amount
    df_copy['log_amount'] = np.log1p(df_copy[amount_column])
    
    # Bin amount into categories
    amount_bins = [0, 10, 50, 100, 500, 1000, float('inf')]
    amount_labels = ['very_small', 'small', 'medium', 'large', 'very_large', 'huge']
    df_copy['amount_category'] = pd.cut(df_copy[amount_column], bins=amount_bins, labels=amount_labels)
    
    # One-hot encode amount category
    amount_dummies = pd.get_dummies(df_copy['amount_category'], prefix='amount_cat')
    df_copy = pd.concat([df_copy, amount_dummies], axis=1)
    
    # Drop the categorical column
    df_copy.drop('amount_category', axis=1, inplace=True)
    
    logger.info("Created amount-based features")
    return df_copy

def apply_pca(df, n_components=0.95, features=None):
    """Apply PCA to reduce dimensionality
    
    Args:
        df (pd.DataFrame): Input dataframe
        n_components (float or int, optional): Number of components or variance ratio. Defaults to 0.95.
        features (list, optional): List of features to use for PCA. Defaults to None.
    
    Returns:
        tuple: Dataframe with PCA components and PCA object
    """
    logger.info(f"Applying PCA with n_components={n_components}")
    
    # Make a copy to avoid modifying the original dataframe
    df_copy = df.copy()
    
    # If features not specified, use all numerical features
    if features is None:
        features = [col for col in df_copy.columns if col != TARGET and col in NUMERICAL_FEATURES]
    
    # Filter out features that don't exist in the dataframe
    features = [f for f in features if f in df_copy.columns]
    
    if not features:
        logger.warning("No features available for PCA")
        return df_copy, None
    
    # Initialize PCA
    pca = PCA(n_components=n_components)
    
    # Fit and transform
    pca_result = pca.fit_transform(df_copy[features])
    
    # Create new dataframe with PCA components
    if isinstance(n_components, float):
        n_components = pca_result.shape[1]
    
    pca_cols = [f'pca_{i}' for i in range(n_components)]
    pca_df = pd.DataFrame(pca_result, columns=pca_cols, index=df_copy.index)
    
    # Keep non-PCA columns
    non_pca_cols = [col for col in df_copy.columns if col not in features]
    result_df = pd.concat([df_copy[non_pca_cols], pca_df], axis=1)
    
    logger.info(f"PCA applied, reduced {len(features)} features to {n_components} components")
    return result_df, pca

def create_interaction_features(df, features=None, degree=2):
    """Create interaction features between numerical features
    
    Args:
        df (pd.DataFrame): Input dataframe
        features (list, optional): List of features to use. Defaults to None.
        degree (int, optional): Degree of interactions. Defaults to 2.
    
    Returns:
        pd.DataFrame: Dataframe with interaction features
    """
    logger.info(f"Creating interaction features with degree={degree}")
    
    # Make a copy to avoid modifying the original dataframe
    df_copy = df.copy()
    
    # If features not specified, use a subset of numerical features
    # Using all might create too many features
    if features is None:
        features = NUMERICAL_FEATURES[:5]  # Limit to first 5 numerical features
    
    # Filter out features that don't exist in the dataframe
    features = [f for f in features if f in df_copy.columns]
    
    if len(features) < 2:
        logger.warning("Not enough features for interactions")
        return df_copy
    
    # Create pairwise interactions
    if degree == 2:
        for i, feat1 in enumerate(features):
            for feat2 in features[i+1:]:
                interaction_name = f"{feat1}x{feat2}"
                df_copy[interaction_name] = df_copy[feat1] * df_copy[feat2]
    else:
        logger.warning(f"Degree {degree} not implemented yet")
    
    logger.info("Created interaction features")
    return df_copy

def create_cost_sensitive_weights(y, pos_weight=None):
    """Create sample weights for cost-sensitive learning
    
    Args:
        y (pd.Series): Target series
        pos_weight (float, optional): Weight for positive class. If None, automatically determined.
            Defaults to None.
    
    Returns:
        np.ndarray: Array of sample weights
    """
    logger.info("Creating cost-sensitive weights")
    
    # Count classes
    class_counts = y.value_counts()
    n_neg = class_counts.get(0, 0)
    n_pos = class_counts.get(1, 0)
    
    if n_pos == 0 or n_neg == 0:
        logger.warning("One class has zero samples, returning equal weights")
        return np.ones(len(y))
    
    # Calculate class weight if not provided
    if pos_weight is None:
        # Classic balanced class weight formula
        pos_weight = n_neg / n_pos
    
    logger.info(f"Positive class weight: {pos_weight}")
    
    # Create sample weights
    sample_weights = np.ones(len(y))
    sample_weights[y == 1] = pos_weight
    
    return sample_weights