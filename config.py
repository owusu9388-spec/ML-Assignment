"""
Centralized Configuration for Capstone Project
All experiment parameters in one place for easy modification
"""

import os

# DATASET CONFIGURATION
DATASET = "online"  # Options: "online", "telco"

# Dataset-specific settings
DATASET_CONFIG = {
    "online": {
        "file": "online_shoppers_intention.csv",
        "target": "Revenue",
        "positive_class": True,  # True = revenue generated
    },
    "telco": {
        "file": "WA_Fn-UseC_-Telco-Customer-Churn.csv",
        "target": "Churn",
        "positive_class": "yes",  # "yes" = churned
    }
}

# EXPERIMENT CONFIGURATION

# Random seed for reproducibility
RANDOM_STATE = 42

# Sampling strategies to compare
SAMPLING_STRATEGIES = [
    "A0_baseline",      # No resampling
    "A1_smote",         # SMOTE only
    "A2_tomek_smote",   # TomekLinks + SMOTE
  #   "A3_undersample",   # Random undersampling
  #   "A4_smoteenn",      # SMOTE + ENN (hybrid)
]

# Feature configurations to compare
FEATURE_CONFIGS = [
    "F0_base",          # Raw features only (no engineering)
    "F1_engineered",    # Engineered features (log, interactions, bins)
    "F2_reduced_rfe",   # RFE feature selection
    "F3_reduced_pca",   # PCA dimensionality reduction
]

# Models to train and compare
MODELS_TO_TRAIN = [
    "LogisticRegression",
    "RandomForest",
    "GradientBoosting",
  #   "SVM",
  #   "KNN",
  #   "DecisionTree",
  #   "NaiveBayes",
]

# FEATURE ENGINEERING CONFIGURATION


# Feature engineering settings
FEATURE_ENGINEERING = {
    "log_transform_cols": [],  # Will be populated based on skewness
    "skewness_threshold": 1.0,  # Threshold for applying log transform
    "interaction_pairs": [],    # Will be populated based on correlation
    "binning_cols": [],         # Will be populated for continuous features
    "n_bins": 5,                # Number of bins for quantile binning
}

# Feature selection settings
FEATURE_SELECTION = {
    "rfe": {
        "n_features_to_select": 20,  # Or 0.5 for 50% of features
        "step": 1,
        "estimator": "LogisticRegression",  # Base estimator for RFE
    },
    "pca": {
        "n_components": 0.95,  # Retain 95% of variance
        "whiten": False,
    },
}


# PREPROCESSING CONFIGURATION

# Train-test split
TEST_SIZE = 0.2
STRATIFY = True

# Imputation
IMPUTATION = {
    "numeric": {
        "strategy": "knn",  # Options: "knn", "mean", "median"
        "n_neighbors": 5,   # For KNN imputer
    },
    "categorical": {
        "strategy": "most_frequent",
    }
}

# Scaling
SCALING_METHOD = "standard"  # Options: "standard", "minmax", "robust"

# Encoding
ENCODING = {
    "method": "onehot",  # Options: "onehot", "label"
    "drop_first": True,
    "handle_unknown": "ignore",
}

# Outlier handling
OUTLIER_DETECTION = {
    "method": "iqr",
    "threshold": 1.5,  # IQR multiplier
    "action": "keep",  # Options: "keep", "remove", "cap"
}


# SAMPLING CONFIGURATION

SAMPLING_CONFIG = {
    "smote": {
        "k_neighbors": 5,
        "random_state": RANDOM_STATE,
    },
    "tomek": {
        "sampling_strategy": "auto",
    },
    "undersample": {
        "sampling_strategy": "auto",  # Balance classes
        "random_state": RANDOM_STATE,
    },
    "smoteenn": {
        "random_state": RANDOM_STATE,
    }
}


# MODEL HYPERPARAMETERS
# Hyperparameter grids for tuning
PARAM_GRIDS = {
    "LogisticRegression": {
        "classifier__C": [0.01, 0.1, 1.0, 10.0],
        "classifier__penalty": ["l1", "l2"],
        "classifier__solver": ["liblinear"],
        "classifier__max_iter": [1000],
    },
    "RandomForest": {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__max_depth": [None, 10, 20, 30],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__max_features": ["sqrt", "log2"],
    },
    "GradientBoosting": {
        "classifier__n_estimators": [100, 200],
        "classifier__learning_rate": [0.01, 0.1, 0.2],
        "classifier__max_depth": [3, 5, 7],
        "classifier__min_samples_split": [2, 5],
        "classifier__subsample": [0.8, 1.0],
    },
    "SVM": {
        "classifier__C": [0.1, 1.0, 10.0],
        "classifier__kernel": ["rbf", "linear"],
        "classifier__gamma": ["scale", "auto"],
    },
    "KNN": {
        "classifier__n_neighbors": [3, 5, 7, 9, 11],
        "classifier__weights": ["uniform", "distance"],
        "classifier__metric": ["euclidean", "manhattan"],
    },
}

# Cross-validation settings
CV_CONFIG = {
    "method": "stratified",  # "stratified" or "standard"
    "n_splits": 5,
    "shuffle": True,
    "random_state": RANDOM_STATE,
}

# Grid search settings
GRID_SEARCH_CONFIG = {
    "scoring": "f1",  # Primary metric
    "n_jobs": -1,     # Use all CPU cores
    "verbose": 1,
    "cv": CV_CONFIG["n_splits"],
    "refit": True,
}


# EVALUATION CONFIGURATION
# Threshold tuning
THRESHOLD_TUNING = {
    "enabled": True,
    "method": "f1_max",  # Options: "f1_max", "recall_target"
    "recall_target": 0.80,  # If method is "recall_target"
}

# Metrics to compute
METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
]


# OUTPUT CONFIGURATION
# Directory structure
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_BASE = os.path.join(BASE_DIR, "results")

# Specific output directories
DIRS = {
    "results": OUTPUT_BASE,
    "eda": os.path.join(OUTPUT_BASE, "{dataset}", "eda"),
    "models": os.path.join(OUTPUT_BASE, "{dataset}", "models"),
    "sampling": os.path.join(OUTPUT_BASE, "{dataset}", "{sampling}", "{features}"),
    "visualizations": os.path.join(OUTPUT_BASE, "{dataset}", "visualizations"),
}

# File naming conventions
FILE_NAMES = {
    "master_results": "master_results.csv",
    "model_comparison": "model_comparison.csv",
    "best_model": "best_model_{sampling}_{features}.pkl",
    "preprocessor": "preprocessor_{features}.pkl",
    "feature_names": "feature_names_{features}.pkl",
    "experiment_log": "experiment_log.txt",
}

# Visualization settings
VIZ_CONFIG = {
    "dpi": 300,
    "style": "whitegrid",
    "context": "paper",
    "palette": "husl",
    "figure_sizes": {
        "small": (6, 4),
        "medium": (10, 6),
        "large": (14, 8),
        "wide": (16, 6),
    }
}


# LOGGING CONFIGURATION

LOGGING = {
    "level": "INFO",
    "format": "[%(levelname)s] %(asctime)s - %(message)s",
    "datefmt": "%Y-%m-%d %H:%M:%S",
    "file": os.path.join(OUTPUT_BASE, "experiment.log"),
}


# HELPER FUNCTIONS

def get_dataset_config(dataset_name=None):
    """Get configuration for specific dataset"""
    dataset_name = dataset_name or DATASET
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return DATASET_CONFIG[dataset_name]


def get_output_dir(dir_type, dataset=None, sampling=None, features=None):
    """Get output directory path with proper formatting"""
    dataset = dataset or DATASET
    dir_template = DIRS.get(dir_type, DIRS["results"])
    dir_path = dir_template.format(
        dataset=dataset,
        sampling=sampling or "",
        features=features or "",
    )
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


def get_file_path(file_type, **kwargs):
    """Get file path with proper formatting"""
    filename = FILE_NAMES.get(file_type, f"{file_type}.pkl")
    return filename.format(**kwargs)


if __name__ == "__main__":
    # Test configuration
    print("=" * 80)
    print("CONFIGURATION TEST")
    print("=" * 80)
    print(f"\nDataset: {DATASET}")
    print(f"Dataset config: {get_dataset_config()}")
    print(f"\nSampling strategies: {SAMPLING_STRATEGIES}")
    print(f"Feature configs: {FEATURE_CONFIGS}")
    print(f"Models: {MODELS_TO_TRAIN}")
    print(f"\nOutput base: {OUTPUT_BASE}")
    print(f"EDA directory: {get_output_dir('eda')}")
    print("\nConfiguration loaded successfully!")


# Backward-compatible aliases (runner convenience)
RESULTS_DIR = OUTPUT_BASE
# Set False to skip GridSearchCV / hyperparameter tuning
TUNE_HYPERPARAMS = True


# EDA + EXPORTS (Set A spirit)
# Toggle exploratory analysis outputs (plots/tables) and optional exports.
RUN_EDA = True
SAVE_EDA_TABLES = True
SAVE_EDA_PLOTS = True

# Outlier reporting (IQR). Non-destructive: does NOT remove rows unless you do so manually.
EXPORT_OUTLIERS = True
OUTLIER_CONFIG = {
    "iqr_k_mild": 1.5,
    "iqr_k_extreme": 3.0,
    "min_unique_continuous": 10,
}

# Optional: export processed train/test arrays + labels for inspection (Set A habit)
EXPORT_PROCESSED_DATASETS = True
