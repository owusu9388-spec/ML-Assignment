"""
Preprocessing Module with Multiple Feature Configurations
Supports: F0_base, F1_engineered, F2_reduced_rfe, F3_reduced_pca
"""

import os
import warnings
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer

warnings.filterwarnings('ignore')

from config import *
from utils import Logger, save_pickle, check_missing_values, reduce_memory_usage



# CUSTOM TRANSFORMERS

class BinningTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for quantile binning
    Fits bin edges on training data to prevent leakage
    """
    
    def __init__(self, n_bins=5, columns=None):
        self.n_bins = n_bins
        self.columns = columns
        self.bin_edges_ = {}
    
    def fit(self, X, y=None):
        """Fit bin edges on training data"""
        df = pd.DataFrame(X, columns=self.columns) if self.columns else pd.DataFrame(X)
        
        for col in df.columns:
            try:
                # Compute quantile edges on training data only
                _, edges = pd.qcut(df[col], q=self.n_bins, retbins=True, duplicates='drop')
                self.bin_edges_[col] = edges
            except Exception:
                # If binning fails, store None (will skip this column)
                self.bin_edges_[col] = None
        
        return self
    
    def transform(self, X):
        """Apply binning using fitted edges"""
        df = pd.DataFrame(X, columns=self.columns) if self.columns else pd.DataFrame(X)
        
        binned_cols = []
        for col in df.columns:
            if self.bin_edges_.get(col) is not None:
                # Apply binning with fitted edges
                binned = pd.cut(df[col], bins=self.bin_edges_[col], labels=False, include_lowest=True)
                binned_cols.append(binned.fillna(-1))  # Fill NaN with -1 for out-of-range values
            else:
                # Keep original if binning not applicable
                binned_cols.append(df[col])
        
        return np.column_stack(binned_cols)


class LogTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for log transformation
    Handles zero and negative values
    """
    
    def __init__(self, columns=None, offset=1.0):
        self.columns = columns
        self.offset = offset
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Apply log(x + offset) transformation"""
        df = pd.DataFrame(X, columns=self.columns) if self.columns else pd.DataFrame(X)
        
        transformed = df.copy()
        for col in df.columns:
            # Shift values to ensure all positive
            min_val = df[col].min()
            if min_val <= 0:
                shift = abs(min_val) + self.offset
            else:
                shift = 0
            
            transformed[col] = np.log(df[col] + shift + self.offset)
        
        return transformed.values


class InteractionTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for feature interactions
    Creates pairwise products of specified features
    """
    
    def __init__(self, interaction_pairs=None, feature_names=None):
        self.interaction_pairs = interaction_pairs or []
        self.feature_names = feature_names
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Create interaction features"""
        df = pd.DataFrame(X, columns=self.feature_names) if self.feature_names else pd.DataFrame(X)
        
        if not self.interaction_pairs:
            return X
        
        interactions = []
        for col1, col2 in self.interaction_pairs:
            if col1 in df.columns and col2 in df.columns:
                interaction = df[col1] * df[col2]
                interactions.append(interaction.values.reshape(-1, 1))
        
        if interactions:
            return np.hstack([X, *interactions])
        return X



# FEATURE ENGINEERING FUNCTIONS

def identify_skewed_features(df: pd.DataFrame, threshold: float = 1.0, logger: Optional[Logger] = None) -> List[str]:
    """
    Identify numeric features with high skewness for log transformation
    
    Args:
        df: Input DataFrame
        threshold: Skewness threshold
        logger: Logger instance
    
    Returns:
        List of column names with high skewness
    """
    log = logger or Logger()
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    skewed_features = []
    
    for col in numeric_cols:
        # Skip columns with limited unique values
        if df[col].nunique() <= 10:
            continue
        
        # Compute skewness
        skewness = df[col].skew()
        
        if abs(skewness) > threshold:
            skewed_features.append(col)
            log.info(f"Column '{col}' has skewness {skewness:.2f} (> {threshold})")
    
    log.success(f"Identified {len(skewed_features)} skewed features for log transformation")
    return skewed_features


def identify_interaction_pairs(df: pd.DataFrame, threshold: float = 0.5, max_pairs: int = 10, logger: Optional[Logger] = None) -> List[Tuple[str, str]]:
    """
    Identify numeric feature pairs with high correlation for interactions
    
    Args:
        df: Input DataFrame
        threshold: Correlation threshold
        max_pairs: Maximum number of pairs to return
        logger: Logger instance
    
    Returns:
        List of feature pairs
    """
    log = logger or Logger()
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Compute correlation matrix
    corr_matrix = df[numeric_cols].corr().abs()
    
    # Get upper triangle (avoid duplicates)
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find pairs above threshold
    pairs = []
    for col in upper_tri.columns:
        high_corr = upper_tri[col][upper_tri[col] > threshold]
        for idx in high_corr.index:
            pairs.append((idx, col, high_corr[idx]))
    
    # Sort by correlation and take top pairs
    pairs = sorted(pairs, key=lambda x: x[2], reverse=True)[:max_pairs]
    pair_tuples = [(p[0], p[1]) for p in pairs]
    
    log.success(f"Identified {len(pair_tuples)} feature interaction pairs (correlation > {threshold})")
    for i, (col1, col2, corr) in enumerate(pairs, 1):
        log.info(f"  {i}. {col1} × {col2} (corr={corr:.3f})")
    
    return pair_tuples


def identify_binning_candidates(df: pd.DataFrame, min_unique: int = 50, logger: Optional[Logger] = None) -> List[str]:
    """
    Identify continuous numeric features suitable for binning
    
    Args:
        df: Input DataFrame
        min_unique: Minimum number of unique values
        logger: Logger instance
    
    Returns:
        List of column names
    """
    log = logger or Logger()
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    binning_cols = []
    
    for col in numeric_cols:
        if df[col].nunique() >= min_unique:
            binning_cols.append(col)
    
    log.success(f"Identified {len(binning_cols)} continuous features for binning")
    return binning_cols



# PREPROCESSOR BUILDERS

def build_base_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
    impute_strategy: str = 'knn',
    logger: Optional[Logger] = None
) -> ColumnTransformer:
    """
    Build base preprocessor (F0_base) without feature engineering
    
    Args:
        numeric_features: List of numeric feature names
        categorical_features: List of categorical feature names
        impute_strategy: Imputation strategy ('knn', 'mean', 'median')
        logger: Logger instance
    
    Returns:
        ColumnTransformer
    """
    log = logger or Logger()
    log.info("Building F0_base preprocessor (no feature engineering)")
    
    # Numeric pipeline
    if impute_strategy == 'knn':
        numeric_imputer = KNNImputer(n_neighbors=5)
    else:
        numeric_imputer = SimpleImputer(strategy=impute_strategy)
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', numeric_imputer),
        ('scaler', StandardScaler()),
    ])
    
    # Categorical pipeline
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')),
    ])
    
    # Combine
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
        ]
    )
    
    log.success("F0_base preprocessor created")
    return preprocessor


def build_engineered_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
    log_transform_cols: List[str],
    interaction_pairs: List[Tuple[str, str]],
    binning_cols: List[str],
    n_bins: int = 5,
    logger: Optional[Logger] = None
) -> ColumnTransformer:
    """
    Build engineered preprocessor (F1_engineered) with feature engineering
    
    Args:
        numeric_features: List of numeric feature names
        categorical_features: List of categorical feature names
        log_transform_cols: Columns for log transformation
        interaction_pairs: Feature pairs for interactions
        binning_cols: Columns for binning
        n_bins: Number of bins
        logger: Logger instance
    
    Returns:
        ColumnTransformer
    """
    log = logger or Logger()
    log.info("Building F1_engineered preprocessor (with feature engineering)")
    
    # Separate numeric features by transformation type
    log_cols = [c for c in numeric_features if c in log_transform_cols]
    bin_cols = [c for c in numeric_features if c in binning_cols and c not in log_transform_cols]
    regular_cols = [c for c in numeric_features if c not in log_cols and c not in bin_cols]
    
    transformers = []
    
    # Regular numeric features
    if regular_cols:
        numeric_transformer = Pipeline(steps=[
            ('imputer', KNNImputer(n_neighbors=5)),
            ('scaler', StandardScaler()),
        ])
        transformers.append(('num_regular', numeric_transformer, regular_cols))
    
    # Log-transformed features
    if log_cols:
        log_transformer = Pipeline(steps=[
            ('imputer', KNNImputer(n_neighbors=5)),
            ('log', LogTransformer(columns=log_cols)),
            ('scaler', StandardScaler()),
        ])
        transformers.append(('num_log', log_transformer, log_cols))
    
    # Binned features
    if bin_cols:
        bin_transformer = Pipeline(steps=[
            ('imputer', KNNImputer(n_neighbors=5)),
            ('binner', BinningTransformer(n_bins=n_bins, columns=bin_cols)),
        ])
        transformers.append(('num_binned', bin_transformer, bin_cols))
    
    # Categorical features
    if categorical_features:
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')),
        ])
        transformers.append(('cat', categorical_transformer, categorical_features))
    
    preprocessor = ColumnTransformer(transformers=transformers)
    
    log.success(f"F1_engineered preprocessor created:")
    log.info(f"  - Regular numeric: {len(regular_cols)} features")
    log.info(f"  - Log-transformed: {len(log_cols)} features")
    log.info(f"  - Binned: {len(bin_cols)} features")
    log.info(f"  - Categorical: {len(categorical_features)} features")
    
    return preprocessor


def build_rfe_preprocessor(
    preprocessor: ColumnTransformer,
    n_features: int = 20,
    logger: Optional[Logger] = None
) -> Pipeline:
    """
    Build RFE-based feature selection preprocessor (F2_reduced_rfe)
    
    Args:
        preprocessor: Base preprocessor
        n_features: Number of features to select
        logger: Logger instance
    
    Returns:
        Pipeline with RFE
    """
    log = logger or Logger()
    log.info(f"Building F2_reduced_rfe preprocessor (select {n_features} features)")
    
    # Use Logistic Regression as estimator for RFE
    rfe = RFE(
        estimator=LogisticRegression(solver='liblinear', max_iter=1000, random_state=RANDOM_STATE),
        n_features_to_select=n_features,
        step=1
    )
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('rfe', rfe),
    ])
    
    log.success("F2_reduced_rfe pipeline created")
    return pipeline


def build_pca_preprocessor(
    preprocessor: ColumnTransformer,
    n_components: float = 0.95,
    logger: Optional[Logger] = None
) -> Pipeline:
    """
    Build PCA-based feature extraction preprocessor (F3_reduced_pca)
    
    Args:
        preprocessor: Base preprocessor
        n_components: Number of components or variance to retain
        logger: Logger instance
    
    Returns:
        Pipeline with PCA
    """
    log = logger or Logger()
    log.info(f"Building F3_reduced_pca preprocessor (retain {n_components*100:.0f}% variance)")
    
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('pca', pca),
    ])
    
    log.success("F3_reduced_pca pipeline created")
    return pipeline



# MAIN PREPROCESSING FUNCTION

def preprocess_data(
    df: pd.DataFrame,
    target_col: str,
    feature_config: str = 'F0_base',
    test_size: float = 0.2,
    random_state: int = 42,
    logger: Optional[Logger] = None,
    export_dir: Optional[str] = None,
    dataset_name: Optional[str] = None
) -> Dict:
    """
    Main preprocessing function supporting multiple feature configurations
    
    Args:
        df: Input DataFrame
        target_col: Target column name
        feature_config: Feature configuration ('F0_base', 'F1_engineered', 'F2_reduced_rfe', 'F3_reduced_pca')
        test_size: Test set proportion
        random_state: Random seed
        logger: Logger instance
    
    Returns:
        Dictionary with preprocessed data and metadata
    """
    log = logger or Logger()
    log.section(f"PREPROCESSING: {feature_config}")
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Ensure binary target is numeric (0/1) to avoid sklearn scorer pos_label issues
    if y.dtype == 'object' or str(y.dtype).startswith('category'):
        y_str = y.astype(str).str.strip().str.lower()
        uniq = set(y_str.dropna().unique().tolist())
        if uniq.issubset({'no','yes'}):
            y = y_str.map({'no': 0, 'yes': 1}).astype(int)
            log.info("Encoded target labels: no->0, yes->1")
        elif uniq.issubset({'false','true'}):
            y = y_str.map({'false': 0, 'true': 1}).astype(int)
            log.info("Encoded target labels: false->0, true->1")
        else:
            # Fallback: label-encode (alphabetical). This keeps sklearn happy but check positive class meaning.
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = le.fit_transform(y_str).astype(int)
            try:
                log.warning(f"Target label-encoded with classes: {list(le.classes_)}")
            except Exception:
                pass
    
    # Identify numeric and categorical features
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    log.info(f"Features: {len(numeric_features)} numeric, {len(categorical_features)} categorical")
    
    # Train-test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    log.success(f"Train-test split: {len(X_train)} train, {len(X_test)} test")
    
    # Build preprocessor based on configuration
    if feature_config == 'F0_base':
        preprocessor = build_base_preprocessor(numeric_features, categorical_features, logger=log)
        
    elif feature_config == 'F1_engineered':
        # Identify features for engineering (on training data only)
        log_cols = identify_skewed_features(X_train[numeric_features], threshold=1.0, logger=log)
        interaction_pairs = identify_interaction_pairs(X_train[numeric_features], threshold=0.5, max_pairs=5, logger=log)
        binning_cols = identify_binning_candidates(X_train[numeric_features], min_unique=50, logger=log)
        
        preprocessor = build_engineered_preprocessor(
            numeric_features, categorical_features,
            log_cols, interaction_pairs, binning_cols,
            n_bins=5, logger=log
        )
        
    elif feature_config == 'F2_reduced_rfe':
        # Build base preprocessor first, then add RFE
        base_prep = build_base_preprocessor(numeric_features, categorical_features, logger=log)
        preprocessor = build_rfe_preprocessor(base_prep, n_features=20, logger=log)
        
    elif feature_config == 'F3_reduced_pca':
        # Build base preprocessor first, then add PCA
        base_prep = build_base_preprocessor(numeric_features, categorical_features, logger=log)
        preprocessor = build_pca_preprocessor(base_prep, n_components=0.95, logger=log)
        
    else:
        raise ValueError(f"Unknown feature_config: {feature_config}")
    
    # Fit and transform
    log.info("Fitting preprocessor on training data...")
    X_train_processed = preprocessor.fit_transform(X_train, y_train)
    
    log.info("Transforming test data...")
    X_test_processed = preprocessor.transform(X_test)
    
    log.success(f"Preprocessing complete: {X_train_processed.shape[1]} features")
    
    # Return results
    return {
        'X_train': X_train_processed,
        'X_test': X_test_processed,
        'y_train': y_train,
        'y_test': y_test,
        'preprocessor': preprocessor,
        'feature_config': feature_config,
        'original_features': {
            'numeric': numeric_features,
            'categorical': categorical_features,
        }
    }



# TESTING

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    
    # Create synthetic dataset
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                                n_redundant=5, n_classes=2, weights=[0.8, 0.2],
                                random_state=42)
    
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    df['target'] = y
    
    # Test each configuration
    logger = Logger("test_preprocessing")
    
    for config in ['F0_base', 'F1_engineered', 'F2_reduced_rfe', 'F3_reduced_pca']:
        print(f"\n{'='*80}")
        print(f"Testing: {config}")
        print(f"{'='*80}")
        
        result = preprocess_data(df, 'target', feature_config=config, logger=logger)
        
        print(f"Train shape: {result['X_train'].shape}")
        print(f"Test shape: {result['X_test'].shape}")
    
    logger.success("All preprocessing configurations tested successfully!")