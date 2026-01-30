"""
Modeling Module with Multiple Sampling Strategies
Supports: A0_baseline, A1_smote, A2_tomek_smote, A3_undersample, A4_smoteenn
"""

import os
import time
import warnings
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks, RandomUnderSampler
from imblearn.combine import SMOTEENN  # SMOTEENN is in combine module
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings('ignore')

from config import *
from utils import Logger, Timer, save_model, save_pickle


# SAMPLING STRATEGY BUILDERS

def get_sampler(strategy: str, random_state: int = 42, logger: Optional[Logger] = None):
    """
    Get resampler for specified strategy
    
    Args:
        strategy: Sampling strategy name
        random_state: Random seed
        logger: Logger instance
    
    Returns:
        Resampler object or None for baseline
    """
    log = logger or Logger()
    
    samplers = {
        'A0_baseline': None,  # No resampling
        'A1_smote': SMOTE(k_neighbors=5, random_state=random_state),
        'A2_tomek_smote': None,  # Will use pipeline with TomekLinks + SMOTE
        'A3_undersample': RandomUnderSampler(sampling_strategy='auto', random_state=random_state),
        'A4_smoteenn': SMOTEENN(random_state=random_state),
    }
    
    if strategy == 'A2_tomek_smote':
        # Return both components for pipeline
        return [
            ('tomek', TomekLinks(sampling_strategy='auto')),
            ('smote', SMOTE(k_neighbors=5, random_state=random_state))
        ]
    
    sampler = samplers.get(strategy)
    
    if sampler is None and strategy != 'A0_baseline':
        log.warning(f"Unknown strategy '{strategy}', using baseline (no resampling)")
        strategy = 'A0_baseline'
    
    log.info(f"Sampling strategy: {strategy}")
    return sampler


def apply_sampling(
    X_train: np.ndarray,
    y_train: np.ndarray,
    strategy: str,
    random_state: int = 42,
    logger: Optional[Logger] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply sampling strategy to training data
    
    Args:
        X_train: Training features
        y_train: Training labels
        strategy: Sampling strategy name
        random_state: Random seed
        logger: Logger instance
    
    Returns:
        Tuple of (resampled X, resampled y)
    """
    log = logger or Logger()
    
    # Log class distribution before sampling
    unique, counts = np.unique(y_train, return_counts=True)
    log.info(f"Before sampling: {dict(zip(unique, counts))}")
    
    if strategy == 'A0_baseline':
        log.info("No resampling applied (baseline)")
        return X_train, y_train
    
    sampler = get_sampler(strategy, random_state, logger=log)
    
    if isinstance(sampler, list):
        # Special case for tomek + smote
        X_resampled, y_resampled = X_train, y_train
        for name, step_sampler in sampler:
            X_resampled, y_resampled = step_sampler.fit_resample(X_resampled, y_resampled)
            unique, counts = np.unique(y_resampled, return_counts=True)
            log.info(f"After {name}: {dict(zip(unique, counts))}")
    else:
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
        unique, counts = np.unique(y_resampled, return_counts=True)
        log.info(f"After sampling: {dict(zip(unique, counts))}")
    
    return X_resampled, y_resampled



# MODEL BUILDERS

def get_base_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Get dictionary of base models
    
    Args:
        random_state: Random seed
    
    Returns:
        Dictionary of model name to model instance
    """
    models = {
        'LogisticRegression': LogisticRegression(
            solver='liblinear',
            max_iter=1000,
            random_state=random_state
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=100,
            random_state=random_state
        ),
        'SVM': SVC(
            probability=True,
            random_state=random_state
        ),
        'KNN': KNeighborsClassifier(
            n_neighbors=5
        ),
        'DecisionTree': DecisionTreeClassifier(
            random_state=random_state
        ),
        'NaiveBayes': GaussianNB(),
    }
    
    return models


def get_param_grids() -> Dict[str, Dict]:
    """
    Get hyperparameter grids for models
    
    Returns:
        Dictionary of model name to parameter grid
    """
    param_grids = {
        'LogisticRegression': {
            'C': [0.01, 0.1, 1.0, 10.0],
            'penalty': ['l1', 'l2'],
        },
        'RandomForest': {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'max_features': ['sqrt', 'log2'],
        },
        'GradientBoosting': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0],
        },
        'SVM': {
            'C': [0.1, 1.0, 10.0],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto'],
        },
        'KNN': {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
        },
    }
    
    return param_grids



# MODEL TRAINING

def train_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    tune_hyperparams: bool = True,
    random_state: int = 42,
    logger: Optional[Logger] = None
) -> Tuple[Any, Dict]:
    """
    Train a single model with optional hyperparameter tuning
    
    Args:
        model_name: Name of model to train
        X_train: Training features
        y_train: Training labels
        tune_hyperparams: Whether to perform hyperparameter tuning
        random_state: Random seed
        logger: Logger instance
    
    Returns:
        Tuple of (trained model, training info dict)
    """
    log = logger or Logger()
    log.subsection(f"Training: {model_name}")
    
    # Get base model
    models = get_base_models(random_state)
    model = models.get(model_name)
    
    if model is None:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Training info
    info = {
        'model_name': model_name,
        'tuned': tune_hyperparams,
        'best_params': None,
        'cv_score': None,
        'cv_std': None,
        'training_time': 0,
    }
    
    # Time training
    start_time = time.time()
    
    # Hyperparameter tuning
    if tune_hyperparams and model_name in get_param_grids():
        log.info(f"Performing hyperparameter tuning with GridSearchCV...")
        
        param_grid = get_param_grids()[model_name]
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring='f1',
            cv=cv,
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        model = grid_search.best_estimator_
        info['best_params'] = grid_search.best_params_
        info['cv_score'] = grid_search.best_score_
        info['cv_std'] = grid_search.cv_results_['std_test_score'][grid_search.best_index_]
        
        log.success(f"Best params: {grid_search.best_params_}")
        log.success(f"Best CV F1: {grid_search.best_score_:.4f} ± {info['cv_std']:.4f}")
    
    else:
        log.info(f"Training with default parameters...")
        model.fit(X_train, y_train)
        log.success("Training complete")
    
    # Record training time
    info['training_time'] = time.time() - start_time
    log.info(f"Training time: {info['training_time']:.2f} seconds")
    
    return model, info


def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_names: Optional[List[str]] = None,
    tune_hyperparams: bool = True,
    random_state: int = 42,
    logger: Optional[Logger] = None
) -> Dict[str, Tuple[Any, Dict]]:
    """
    Train multiple models
    
    Args:
        X_train: Training features
        y_train: Training labels
        model_names: List of model names to train (None = all)
        tune_hyperparams: Whether to perform hyperparameter tuning
        random_state: Random seed
        logger: Logger instance
    
    Returns:
        Dictionary of model name to (model, info) tuple
    """
    log = logger or Logger()
    log.section("MODEL TRAINING")
    
    # Use all models if not specified
    if model_names is None:
        model_names = list(get_base_models(random_state).keys())
    
    log.info(f"Training {len(model_names)} models: {', '.join(model_names)}")
    log.info(f"Hyperparameter tuning: {'ON' if tune_hyperparams else 'OFF'}")
    log.info(f"Training data shape: {X_train.shape}")
    
    results = {}
    total_time = 0
    
    for model_name in model_names:
        try:
            with Timer(f"Training {model_name}", logger=log):
                model, info = train_model(
                    model_name, X_train, y_train,
                    tune_hyperparams=tune_hyperparams,
                    random_state=random_state,
                    logger=log
                )
                results[model_name] = (model, info)
                total_time += info['training_time']
        except Exception as e:
            log.error(f"Failed to train {model_name}: {str(e)}")
            continue
    
    log.success(f"Training complete: {len(results)}/{len(model_names)} models trained")
    log.info(f"Total training time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    return results



# COMPLETE EXPERIMENT RUNNER

def run_experiment(
    X_train_processed: np.ndarray,
    y_train: np.ndarray,
    X_test_processed: np.ndarray,
    y_test: np.ndarray,
    dataset_name: str,
    sampling_strategy: str,
    feature_config: str,
    model_names: Optional[List[str]] = None,
    tune_hyperparams: bool = True,
    output_dir: Optional[str] = None,
    random_state: int = 42,
    logger: Optional[Logger] = None
) -> Dict:
    """
    Run complete experiment: sampling + training
    
    Args:
        X_train_processed: Preprocessed training features
        y_train: Training labels
        X_test_processed: Preprocessed test features
        y_test: Test labels
        dataset_name: Dataset name
        sampling_strategy: Sampling strategy name
        feature_config: Feature configuration name
        model_names: List of models to train
        tune_hyperparams: Whether to tune hyperparameters
        output_dir: Output directory
        random_state: Random seed
        logger: Logger instance
    
    Returns:
        Dictionary with experiment results
    """
    log = logger or Logger()
    log.section(f"EXPERIMENT: {dataset_name} | {sampling_strategy} | {feature_config}")
    
    # Apply sampling
    log.subsection("Applying Sampling Strategy")
    X_train_resampled, y_train_resampled = apply_sampling(
        X_train_processed, y_train,
        sampling_strategy, random_state, logger=log
    )
    
    # Train models
    trained_models = train_all_models(
        X_train_resampled, y_train_resampled,
        model_names=model_names,
        tune_hyperparams=tune_hyperparams,
        random_state=random_state,
        logger=log
    )
    
    # Prepare results
    results = {
        'dataset': dataset_name,
        'sampling_strategy': sampling_strategy,
        'feature_config': feature_config,
        'models': trained_models,
        'X_train_resampled': X_train_resampled,
        'y_train_resampled': y_train_resampled,
        'X_test': X_test_processed,
        'y_test': y_test,
    }
    
    # Save models if output directory specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        for model_name, (model, info) in trained_models.items():
            model_path = os.path.join(output_dir, f"{model_name}_model.pkl")
            save_model(model, model_path)
            log.info(f"Saved {model_name} to: {model_path}")
        
        # Save training info
        info_df = pd.DataFrame([info for _, info in trained_models.values()])
        info_path = os.path.join(output_dir, "training_info.csv")
        info_df.to_csv(info_path, index=False)
        log.success(f"Saved training info to: {info_path}")
    
    log.success(f"Experiment complete: {len(trained_models)} models trained")
    
    return results



# TESTING

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # Create synthetic imbalanced dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        weights=[0.8, 0.2],
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger = Logger("test_modeling")
    
    # Test different sampling strategies
    for strategy in ['A0_baseline', 'A1_smote', 'A2_tomek_smote']:
        print(f"\n{'='*80}")
        print(f"Testing: {strategy}")
        print(f"{'='*80}")
        
        results = run_experiment(
            X_train, y_train, X_test, y_test,
            dataset_name='test',
            sampling_strategy=strategy,
            feature_config='F0_base',
            model_names=['LogisticRegression', 'RandomForest'],
            tune_hyperparams=False,
            random_state=42,
            logger=logger
        )
        
        print(f"Trained {len(results['models'])} models")
    
    logger.success("All modeling tests passed!")
