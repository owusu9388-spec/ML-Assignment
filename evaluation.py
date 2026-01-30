"""
Evaluation Module with Comprehensive Metrics
Includes: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrix, Threshold Tuning
"""

import os
import warnings
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_curve,
)

warnings.filterwarnings('ignore')

from utils import Logger, save_dataframe



# PROBABILITY/SCORE EXTRACTION

def get_prediction_scores(model: Any, X: np.ndarray, logger: Optional[Logger] = None) -> np.ndarray:
    """
    Get prediction scores/probabilities from model
    
    Args:
        model: Trained model
        X: Features
        logger: Logger instance
    
    Returns:
        Array of prediction scores for positive class
    """
    log = logger or Logger()
    
    try:
        if hasattr(model, 'predict_proba'):
            scores = model.predict_proba(X)[:, 1]
        elif hasattr(model, 'decision_function'):
            scores = model.decision_function(X)
            # Normalize to [0, 1] range
            min_score, max_score = scores.min(), scores.max()
            if max_score > min_score:
                scores = (scores - min_score) / (max_score - min_score)
        else:
            log.warning("Model does not support probability/scoring. Using predictions.")
            scores = model.predict(X).astype(float)
        
        return scores
    
    except Exception as e:
        log.error(f"Error getting prediction scores: {str(e)}")
        return model.predict(X).astype(float)



# THRESHOLD TUNING

def find_optimal_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    method: str = 'f1_max',
    recall_target: float = 0.80,
    logger: Optional[Logger] = None
) -> Tuple[float, Dict]:
    """
    Find optimal classification threshold
    
    Args:
        y_true: True labels
        y_scores: Prediction scores
        method: 'f1_max' or 'recall_target'
        recall_target: Target recall if using 'recall_target' method
        logger: Logger instance
    
    Returns:
        Tuple of (optimal threshold, metrics at threshold)
    """
    log = logger or Logger()
    
    # Compute precision-recall curve
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    
    if method == 'f1_max':
        # Find threshold that maximizes F1
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        optimal_idx = np.argmax(f1_scores[:-1])  # Exclude last element
        optimal_threshold = thresholds[optimal_idx]
        
        metrics = {
            'threshold': optimal_threshold,
            'precision': precisions[optimal_idx],
            'recall': recalls[optimal_idx],
            'f1': f1_scores[optimal_idx],
        }
        
        log.info(f"Optimal threshold (F1-max): {optimal_threshold:.4f}")
        log.info(f"  Precision: {metrics['precision']:.4f}")
        log.info(f"  Recall: {metrics['recall']:.4f}")
        log.info(f"  F1: {metrics['f1']:.4f}")
    
    elif method == 'recall_target':
        # Find threshold that achieves target recall
        valid_idx = recalls[:-1] >= recall_target
        
        if not valid_idx.any():
            log.warning(f"Cannot achieve recall >= {recall_target}. Using threshold with highest recall.")
            optimal_idx = 0
        else:
            # Among thresholds achieving target recall, pick one with highest precision
            optimal_idx = np.where(valid_idx)[0][np.argmax(precisions[:-1][valid_idx])]
        
        optimal_threshold = thresholds[optimal_idx]
        
        metrics = {
            'threshold': optimal_threshold,
            'precision': precisions[optimal_idx],
            'recall': recalls[optimal_idx],
            'f1': 2 * precisions[optimal_idx] * recalls[optimal_idx] / (precisions[optimal_idx] + recalls[optimal_idx] + 1e-10),
        }
        
        log.info(f"Optimal threshold (recall >= {recall_target}): {optimal_threshold:.4f}")
        log.info(f"  Precision: {metrics['precision']:.4f}")
        log.info(f"  Recall: {metrics['recall']:.4f}")
        log.info(f"  F1: {metrics['f1']:.4f}")
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return optimal_threshold, metrics



# EVALUATION METRICS

def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    threshold: Optional[float] = None,
    tune_threshold: bool = True,
    threshold_method: str = 'f1_max',
    logger: Optional[Logger] = None
) -> Dict:
    """
    Comprehensive model evaluation
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: True test labels
        model_name: Model name
        threshold: Custom classification threshold (None = use 0.5 or tune)
        tune_threshold: Whether to tune threshold
        threshold_method: Method for threshold tuning
        logger: Logger instance
    
    Returns:
        Dictionary with evaluation metrics
    """
    log = logger or Logger()
    log.subsection(f"Evaluating: {model_name}")
    
    # Get prediction scores
    y_scores = get_prediction_scores(model, X_test, logger=log)
    
    # Tune threshold if requested and no custom threshold provided
    if tune_threshold and threshold is None:
        threshold, threshold_metrics = find_optimal_threshold(
            y_test, y_scores,
            method=threshold_method,
            logger=log
        )
    elif threshold is None:
        threshold = 0.5
    
    # Make predictions with threshold
    y_pred = (y_scores >= threshold).astype(int)
    
    # Compute metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # ROC-AUC
    try:
        roc_auc = roc_auc_score(y_test, y_scores)
    except Exception as e:
        log.warning(f"Cannot compute ROC-AUC: {str(e)}")
        roc_auc = np.nan
    
    # PR-AUC
    try:
        pr_auc = average_precision_score(y_test, y_scores)
    except Exception as e:
        log.warning(f"Cannot compute PR-AUC: {str(e)}")
        pr_auc = np.nan
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    # Log results
    log.info(f"Threshold: {threshold:.4f}")
    log.info(f"Accuracy:  {acc:.4f}")
    log.info(f"Precision: {prec:.4f}")
    log.info(f"Recall:    {rec:.4f}")
    log.info(f"F1-Score:  {f1:.4f}")
    log.info(f"ROC-AUC:   {roc_auc:.4f}")
    log.info(f"PR-AUC:    {pr_auc:.4f}")
    log.info(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    
    # Return comprehensive results
    results = {
        'model': model_name,
        'threshold': threshold,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'confusion_matrix': cm,
        'classification_report': classification_report(y_test, y_pred, zero_division=0),
        'y_pred': y_pred,
        'y_scores': y_scores,
    }
    
    return results


def evaluate_all_models(
    trained_models: Dict[str, Tuple[Any, Dict]],
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: Optional[List[str]] = None,
    tune_threshold: bool = True,
    threshold_method: str = 'f1_max',
    logger: Optional[Logger] = None
) -> pd.DataFrame:
    """
    Evaluate all trained models
    
    Args:
        trained_models: Dictionary of model name to (model, info) tuple
        X_test: Test features
        y_test: True test labels
        feature_names: Optional list of feature names (kept for API compatibility)
        tune_threshold: Whether to tune threshold
        threshold_method: Method for threshold tuning
        logger: Logger instance
    
    Returns:
        DataFrame with evaluation results
    """
    log = logger or Logger()
    log.section("MODEL EVALUATION")
    
    log.info(f"Evaluating {len(trained_models)} models")
    log.info(f"Test data shape: {X_test.shape}")
    log.info(f"Threshold tuning: {'ON' if tune_threshold else 'OFF'}")
    
    all_results = []
    
    for model_name, (model, training_info) in trained_models.items():
        try:
            # Evaluate model
            eval_results = evaluate_model(
                model, X_test, y_test,
                model_name=model_name,
                tune_threshold=tune_threshold,
                threshold_method=threshold_method,
                logger=log
            )
            
            # Merge with training info
            eval_results.update({
                'training_time': training_info.get('training_time', 0),
                'cv_score': training_info.get('cv_score'),
                'cv_std': training_info.get('cv_std'),
                'best_params': str(training_info.get('best_params')),
            })
            
            all_results.append(eval_results)
        
        except Exception as e:
            log.error(f"Error evaluating {model_name}: {str(e)}")
            continue
    
    # Create DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Sort by F1 score
    if 'f1' in results_df.columns:
        results_df = results_df.sort_values('f1', ascending=False)
    
    log.success(f"Evaluation complete for {len(results_df)} models")
    
    return results_df



# RESULTS EXPORT

def save_evaluation_results(
    results_df: pd.DataFrame,
    dataset_name: str,
    sampling_strategy: str,
    feature_config: str,
    output_dir: str,
    logger: Optional[Logger] = None
):
    """
    Save evaluation results to files
    
    Args:
        results_df: Evaluation results DataFrame
        dataset_name: Dataset name
        sampling_strategy: Sampling strategy name
        feature_config: Feature configuration name
        output_dir: Output directory
        logger: Logger instance
    """
    log = logger or Logger()
    log.subsection("Saving Evaluation Results")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Select columns for export (exclude complex objects)
    export_cols = [
        'model', 'threshold', 'accuracy', 'precision', 'recall', 'f1',
        'roc_auc', 'pr_auc', 'tn', 'fp', 'fn', 'tp',
        'training_time', 'cv_score', 'cv_std'
    ]
    export_cols = [c for c in export_cols if c in results_df.columns]
    
    # Save model comparison
    comparison_path = os.path.join(output_dir, "model_comparison.csv")
    results_df[export_cols].to_csv(comparison_path, index=False)
    log.success(f"Saved model comparison to: {comparison_path}")
    
    # Save detailed classification reports
    for _, row in results_df.iterrows():
        model_name = row['model']
        if 'classification_report' in row:
            report_path = os.path.join(output_dir, f"{model_name}_classification_report.txt")
            with open(report_path, 'w') as f:
                f.write(f"Model: {model_name}\n")
                f.write(f"Dataset: {dataset_name}\n")
                f.write(f"Sampling: {sampling_strategy}\n")
                f.write(f"Features: {feature_config}\n")
                f.write(f"\n{row['classification_report']}\n")
            log.info(f"Saved classification report for {model_name}")
    
    log.success(f"All results saved to: {output_dir}")



# MASTER RESULTS AGGREGATION

def aggregate_master_results(
    results_df: pd.DataFrame,
    dataset_name: str,
    sampling_strategy: str,
    feature_config: str,
    master_file: str,
    logger: Optional[Logger] = None
):
    """
    Append results to master results file
    
    Args:
        results_df: Evaluation results DataFrame
        dataset_name: Dataset name
        sampling_strategy: Sampling strategy name
        feature_config: Feature configuration name
        master_file: Path to master results file
        logger: Logger instance
    """
    log = logger or Logger()
    
    # Add experiment info columns
    results_df_copy = results_df.copy()
    results_df_copy.insert(0, 'dataset', dataset_name)
    results_df_copy.insert(1, 'sampling_strategy', sampling_strategy)
    results_df_copy.insert(2, 'feature_config', feature_config)
    
    # Select columns
    master_cols = [
        'dataset', 'sampling_strategy', 'feature_config', 'model',
        'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc',
        'tn', 'fp', 'fn', 'tp', 'threshold', 'training_time',
        'cv_score', 'cv_std'
    ]
    master_cols = [c for c in master_cols if c in results_df_copy.columns]
    
    # Load existing master results or create new
    if os.path.exists(master_file):
        master_df = pd.read_csv(master_file)
        master_df = pd.concat([master_df, results_df_copy[master_cols]], ignore_index=True)
        log.info("Appended to existing master results")
    else:
        master_df = results_df_copy[master_cols]
        log.info("Created new master results")
    
    # Save
    os.makedirs(os.path.dirname(master_file), exist_ok=True)
    master_df.to_csv(master_file, index=False)
    log.success(f"Master results saved to: {master_file}")



# TESTING

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    
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
    
    # Train models
    models = {
        'RandomForest': RandomForestClassifier(random_state=42),
        'LogisticRegression': LogisticRegression(random_state=42)
    }
    
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = (model, {'training_time': 1.0})
    
    # Test evaluation
    logger = Logger("test_evaluation")
    
    results_df = evaluate_all_models(
        trained_models,
        X_test, y_test,
        tune_threshold=True,
        threshold_method='f1_max',
        logger=logger
    )
    
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(results_df[['model', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']])
    
    logger.success("All evaluation tests passed!")
