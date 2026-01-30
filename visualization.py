"""
Visualization Module
ROC curves, PR curves, confusion matrices, feature importance, comparison plots
"""

import os
import warnings
from typing import Dict, List, Optional, Any

import matplotlib
# Use non-interactive backend to avoid Tkinter thread issues when saving plots
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, ConfusionMatrixDisplay

warnings.filterwarnings('ignore')

from config import VIZ_CONFIG
from utils import Logger

# Set style
sns.set_style(VIZ_CONFIG['style'])
sns.set_context(VIZ_CONFIG['context'])
sns.set_palette(VIZ_CONFIG['palette'])



# ROC AND PR CURVES


def plot_roc_curves(
    results_df: pd.DataFrame,
    y_test: np.ndarray,
    title: str,
    save_path: str,
    logger: Optional[Logger] = None
):
    """Plot ROC curves for all models in results_df using stored y_scores and provided y_test."""
    log = logger or Logger()
    try:
        plt.figure()
        for _, row in results_df.iterrows():
            y_scores = row.get("y_scores", None)
            if y_scores is None:
                continue
            fpr, tpr, _ = roc_curve(y_test, y_scores)
            auc = row.get("roc_auc", np.nan)
            plt.plot(fpr, tpr, label=f"{row.get('model','model')} (AUC={auc:.3f})")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.title(title)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        log.info(f"Saved ROC curves: {save_path}")
    except Exception as e:
        log.warning(f"Failed ROC curve plot: {e}")


def plot_pr_curves(
    results_df: pd.DataFrame,
    y_test: np.ndarray,
    title: str,
    save_path: str,
    logger: Optional[Logger] = None
):
    """Plot Precision-Recall curves for all models in results_df using stored y_scores and provided y_test."""
    log = logger or Logger()
    try:
        plt.figure()
        for _, row in results_df.iterrows():
            y_scores = row.get("y_scores", None)
            if y_scores is None:
                continue
            precision, recall, _ = precision_recall_curve(y_test, y_scores)
            ap = row.get("pr_auc", np.nan)
            plt.plot(recall, precision, label=f"{row.get('model','model')} (AP={ap:.3f})")
        plt.title(title)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend(loc="lower left")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        log.info(f"Saved PR curves: {save_path}")
    except Exception as e:
        log.warning(f"Failed PR curve plot: {e}")

def plot_confusion_matrices(
    results_df: pd.DataFrame,
    output_dir: str,
    logger: Optional[Logger] = None
):
    """Plot confusion matrix for each model"""
    log = logger or Logger()
    
    cm_dir = os.path.join(output_dir, "confusion_matrices")
    os.makedirs(cm_dir, exist_ok=True)
    
    for _, row in results_df.iterrows():
        if 'confusion_matrix' not in row or row['confusion_matrix'] is None:
            continue
        
        model_name = row['model']
        cm = row['confusion_matrix']
        
        fig, ax = plt.subplots(figsize=VIZ_CONFIG['figure_sizes']['small'])
        
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
        disp.plot(ax=ax, cmap='Blues', values_format='d')

        # Ensure annotation text is always readable by setting contrast against cell intensity.
        # 'Blues' colormap: higher values are darker -> use white text there.
        if hasattr(disp, "text_") and disp.text_ is not None:
            thresh = cm.max() / 2.0 if cm.size else 0
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    t = disp.text_[i, j]
                    if t is None:
                        continue
                    val = cm[i, j]
                    t.set_color("black" if val > thresh else "white")
                    t.set_fontweight("bold")
                    t.set_fontsize(12)

        
        plt.title(f'Confusion Matrix - {model_name}')
        plt.tight_layout()
        
        save_path = os.path.join(cm_dir, f"{model_name}_confusion_matrix.png")
        plt.savefig(save_path, dpi=VIZ_CONFIG['dpi'], bbox_inches='tight')
        plt.close()
        
        log.info(f"Saved confusion matrix for {model_name}")
    
    log.success(f"All confusion matrices saved to: {cm_dir}")



# FEATURE IMPORTANCE

def plot_feature_importance(
    model: Any,
    model_name: str,
    feature_names: List[str],
    save_path: str,
    top_n: int = 20,
    logger: Optional[Logger] = None
):
    """Plot feature importance for models that support it"""
    log = logger or Logger()
    
    importance = None
    
    # Tree-based models
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    
    # Linear models
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
    
    if importance is None:
        log.warning(f"Model {model_name} does not support feature importance")
        return
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names[:len(importance)],
        'importance': importance
    }).sort_values('importance', ascending=False).head(top_n)
    
    # Plot
    plt.figure(figsize=VIZ_CONFIG['figure_sizes']['medium'])
    sns.barplot(data=importance_df, x='importance', y='feature', orient='h')
    plt.title(f'Top {top_n} Feature Importances - {model_name}')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(save_path, dpi=VIZ_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    
    log.success(f"Feature importance saved to: {save_path}")



# COMPARISON PLOTS

def plot_metric_comparison(
    results_df: pd.DataFrame,
    metric: str,
    title: str,
    save_path: str,
    logger: Optional[Logger] = None
):
    """Plot bar chart comparing models on a specific metric"""
    log = logger or Logger()
    
    if metric not in results_df.columns:
        log.warning(f"Metric '{metric}' not in results")
        return
    
    
    # Sort by metric
    plot_df = results_df.sort_values(metric, ascending=False)

    # Dynamically scale height to the number of bars actually plotted
    n_bars = max(1, len(plot_df))
    base_w, _ = VIZ_CONFIG['figure_sizes']['medium']
    fig_h = max(3.5, 0.65 * n_bars + 1.5)  # keeps 3 bars compact, scales up 
    plt.figure(figsize=(base_w, fig_h))

    
    sns.barplot(data=plot_df, x=metric, y='model', orient='h')
    plt.title(title)
    plt.xlabel(metric.replace('_', ' ').title())
    plt.ylabel('Model')
    
    # Add value labels
    for i, row in enumerate(plot_df.itertuples()):
        value = getattr(row, metric)
        if pd.notna(value):
            plt.text(value, i, f'{value:.3f}', va='center', ha='left', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=VIZ_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    
    log.success(f"Metric comparison saved to: {save_path}")


def plot_metrics_heatmap(
    results_df: pd.DataFrame,
    metrics: List[str],
    title: str,
    save_path: str,
    logger: Optional[Logger] = None
):
    """Plot heatmap of multiple metrics across models"""
    log = logger or Logger()
    
    # Select available metrics
    available_metrics = [m for m in metrics if m in results_df.columns]
    
    if not available_metrics:
        log.warning("No metrics available for heatmap")
        return
    
    # Create pivot table
    data = results_df[['model'] + available_metrics].set_index('model')
    
    plt.figure(figsize=VIZ_CONFIG['figure_sizes']['wide'])
    sns.heatmap(data.T, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0, vmax=1, cbar_kws={'label': 'Score'})
    plt.title(title)
    plt.ylabel('Metric')
    plt.xlabel('Model')
    plt.tight_layout()
    plt.savefig(save_path, dpi=VIZ_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    
    log.success(f"Metrics heatmap saved to: {save_path}")



# EDA VISUALIZATIONS

def plot_class_distribution(
    y: pd.Series,
    title: str,
    save_path: str,
    logger: Optional[Logger] = None
):
    """Plot class distribution"""
    log = logger or Logger()
    
    plt.figure(figsize=VIZ_CONFIG['figure_sizes']['small'])
    
    counts = y.value_counts()
    total = len(y)
    
    sns.countplot(x=y, palette='Set2')
    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Count')
    
    # Add percentage labels
    for i, (label, count) in enumerate(counts.items()):
        pct = 100 * count / total
        plt.text(i, count, f'{count}\n({pct:.1f}%)', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=VIZ_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    
    log.success(f"Class distribution saved to: {save_path}")


def plot_missing_values(
    df: pd.DataFrame,
    title: str,
    save_path: str,
    logger: Optional[Logger] = None
):
    """Plot missing values summary"""
    log = logger or Logger()
    
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    
    if len(missing) == 0:
        log.info("No missing values to plot")
        return
    
    plt.figure(figsize=VIZ_CONFIG['figure_sizes']['medium'])
    
    missing_pct = 100 * missing / len(df)
    
    sns.barplot(x=missing_pct.values, y=missing_pct.index, orient='h')
    plt.title(title)
    plt.xlabel('Missing (%)')
    plt.ylabel('Feature')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=VIZ_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    
    log.success(f"Missing values plot saved to: {save_path}")


def plot_correlation_heatmap(
    df: pd.DataFrame,
    title: str,
    save_path: str,
    method: str = 'pearson',
    logger: Optional[Logger] = None
):
    """Plot correlation heatmap for numeric features"""
    log = logger or Logger()
    
    numeric_df = df.select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) == 0:
        log.warning("No numeric features to plot correlation")
        return
    
    corr = numeric_df.corr(method=method)
    
    plt.figure(figsize=VIZ_CONFIG['figure_sizes']['large'])
    sns.heatmap(corr, cmap='coolwarm', center=0, annot=False, fmt='.2f', square=True, linewidths=0.5)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=VIZ_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    
    log.success(f"Correlation heatmap saved to: {save_path}")



# TESTING

if __name__ == "__main__":
    # Create synthetic results for testing
    results_data = {
        'model': ['RandomForest', 'LogisticRegression', 'SVM'],
        'accuracy': [0.85, 0.82, 0.80],
        'precision': [0.78, 0.75, 0.73],
        'recall': [0.72, 0.70, 0.68],
        'f1': [0.75, 0.72, 0.70],
        'roc_auc': [0.88, 0.85, 0.83],
        'pr_auc': [0.76, 0.73, 0.71],
    }
    
    results_df = pd.DataFrame(results_data)
    
    logger = Logger("test_viz")
    
    # Test plots
    os.makedirs('/home/claude/test_viz', exist_ok=True)
    
    plot_metric_comparison(
        results_df, 'f1', 'F1 Score Comparison',
        '/home/claude/test_viz/f1_comparison.png', logger
    )
    
    plot_metrics_heatmap(
        results_df, ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc'],
        'Model Performance Heatmap',
        '/home/claude/test_viz/metrics_heatmap.png', logger
    )
    
    logger.success("Visualization tests completed!")


# SET A-STYLE EDA PLOTS (Merged)

def plot_numeric_distributions(df: pd.DataFrame, numeric_cols: List[str], out_dir: str, logger: Optional[Logger] = None):
    """Save histogram for each numeric feature."""
    log = logger or Logger()
    os.makedirs(out_dir, exist_ok=True)
    for c in numeric_cols:
        try:
            s = pd.to_numeric(df[c], errors="coerce").dropna()
            if s.empty:
                continue
            plt.figure()
            plt.hist(s.values, bins=30)
            plt.title(f"Distribution: {c}")
            plt.xlabel(c)
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"hist_{c}.png"), dpi=300)
            plt.close()
        except Exception as e:
            log.warning(f"Failed histogram for {c}: {e}")


def plot_boxplots(df: pd.DataFrame, numeric_cols: List[str], out_dir: str, logger: Optional[Logger] = None):
    """Save a boxplot for each numeric feature."""
    log = logger or Logger()
    os.makedirs(out_dir, exist_ok=True)
    for c in numeric_cols:
        try:
            s = pd.to_numeric(df[c], errors="coerce").dropna()
            if s.empty:
                continue
            plt.figure()
            plt.boxplot(s.values, vert=True)
            plt.title(f"Boxplot: {c}")
            plt.ylabel(c)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"box_{c}.png"), dpi=300)
            plt.close()
        except Exception as e:
            log.warning(f"Failed boxplot for {c}: {e}")


def plot_skewness(df: pd.DataFrame, numeric_cols: List[str], out_dir: str, logger: Optional[Logger] = None):
    """Plot skewness values for numeric columns."""
    log = logger or Logger()
    os.makedirs(out_dir, exist_ok=True)
    try:
        sk = {}
        for c in numeric_cols:
            s = pd.to_numeric(df[c], errors="coerce").dropna()
            sk[c] = float(s.skew()) if not s.empty else np.nan
        sk_ser = pd.Series(sk).dropna()
        if sk_ser.empty:
            return
        sk_ser = sk_ser.sort_values(key=lambda s: s.abs(), ascending=False)
        plt.figure(figsize=(12, 5))
        plt.bar(sk_ser.index.astype(str), sk_ser.values)
        plt.xticks(rotation=90)
        plt.title("Skewness by numeric feature")
        plt.ylabel("Skewness")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "skewness.png"), dpi=300)
        plt.close()
    except Exception as e:
        log.warning(f"Failed skewness plot: {e}")


def plot_outlier_counts(outlier_summary_df: pd.DataFrame, out_dir: str, logger: Optional[Logger] = None):
    """Plot mild/extreme outlier counts by column."""
    log = logger or Logger()
    os.makedirs(out_dir, exist_ok=True)
    try:
        dfp = outlier_summary_df.copy()
        if dfp.empty:
            return
        cols = dfp["column"].astype(str).tolist()
        mild = dfp["mild_outliers"].values
        extreme = dfp["extreme_outliers"].values
        x = np.arange(len(cols))
        width = 0.4
        plt.figure(figsize=(12, 5))
        plt.bar(x - width/2, mild, width, label="mild")
        plt.bar(x + width/2, extreme, width, label="extreme")
        plt.xticks(x, cols, rotation=90)
        plt.title("IQR Outlier Counts by Feature")
        plt.ylabel("Count")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "outlier_counts.png"), dpi=300)
        plt.close()
    except Exception as e:
        log.warning(f"Failed outlier counts plot: {e}")


def plot_confusion_matrix(cm: np.ndarray, title: str, output_path: str, logger: Optional[Logger] = None):
    """Backward-compatible single confusion matrix plot."""
    log = logger or Logger()
    try:
        plt.figure()
        plt.imshow(cm, interpolation='nearest')
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(cm.shape[0])
        plt.xticks(tick_marks, tick_marks)
        plt.yticks(tick_marks, tick_marks)
        thresh = cm.max() / 2.0 if cm.size else 0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(
                    j, i, format(int(cm[i, j]), 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                )
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
    except Exception as e:
        log.warning(f"Failed confusion matrix plot: {e}")