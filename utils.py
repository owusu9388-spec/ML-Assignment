"""
Utility Functions for Capstone Project
Logging, file I/O, data validation, and helper functions
"""

import os
import sys
import time
import pickle
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib



# LOGGING UTILITIES

class Logger:
    """
    Custom logger with colored output and file logging
    Fallback to simple print if color support unavailable
    """
    
    # ANSI color codes
    COLORS = {
        'INFO': '\033[94m',     # Blue
        'SUCCESS': '\033[92m',  # Green
        'WARNING': '\033[93m',  # Yellow
        'ERROR': '\033[91m',    # Red
        'RESET': '\033[0m',     # Reset
    }
    
    def __init__(self, name: str = "logger", log_file: Optional[str] = None):
        self.name = name
        self.log_file = log_file
        self.start_time = time.time()
        
        # Check if colors are supported
        self.use_colors = self._supports_color()
        
        # Create log file if specified
        if self.log_file:
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            # Write header
            with open(self.log_file, 'a') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"Log started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"{'='*80}\n\n")
    
    def _supports_color(self) -> bool:
        """Check if terminal supports ANSI colors"""
        return (
            hasattr(sys.stdout, 'isatty') and sys.stdout.isatty() and
            os.getenv('TERM') != 'dumb'
        )
    
    def _format_message(self, level: str, msg: str) -> str:
        """Format message with timestamp and level"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return f"[{timestamp}] [{level:8s}] {msg}"
    
    def _log(self, level: str, msg: str):
        """Internal logging method"""
        formatted_msg = self._format_message(level, msg)
        
        # Console output with colors
        if self.use_colors and level in self.COLORS:
            color_msg = f"{self.COLORS[level]}{formatted_msg}{self.COLORS['RESET']}"
            print(color_msg)
        else:
            print(formatted_msg)
        
        # File output without colors
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(formatted_msg + '\n')
    
    def info(self, msg: str):
        """Log info message"""
        self._log('INFO', msg)
    
    def success(self, msg: str):
        """Log success message"""
        self._log('SUCCESS', msg)
    
    def warning(self, msg: str):
        """Log warning message"""
        self._log('WARNING', msg)
    
    def error(self, msg: str):
        """Log error message"""
        self._log('ERROR', msg)
    
    def section(self, title: str):
        """Log section header"""
        separator = '=' * 80
        self._log('INFO', separator)
        self._log('INFO', title.center(80))
        self._log('INFO', separator)
    
    def subsection(self, title: str):
        """Log subsection header"""
        separator = '-' * 80
        self._log('INFO', separator)
        self._log('INFO', title)
        self._log('INFO', separator)
    
    def elapsed_time(self):
        """Log elapsed time since logger creation"""
        elapsed = time.time() - self.start_time
        self.info(f"Elapsed time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")


def setup_logger(name: str = "experiment", log_file: Optional[str] = None) -> Logger:
    """
    Setup and return a logger instance
    
    Args:
        name: Logger name
        log_file: Optional log file path
    
    Returns:
        Logger instance
    """
    return Logger(name=name, log_file=log_file)



# FILE I/O UTILITIES

def ensure_dir(directory: str) -> str:
    """
    Ensure directory exists, create if not
    
    Args:
        directory: Directory path
    
    Returns:
        Directory path
    """
    os.makedirs(directory, exist_ok=True)
    return directory


def save_pickle(obj: Any, filepath: str):
    """
    Save object as pickle file
    
    Args:
        obj: Object to save
        filepath: Output file path
    """
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filepath: str) -> Any:
    """
    Load object from pickle file
    
    Args:
        filepath: Input file path
    
    Returns:
        Loaded object
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_json(obj: Dict, filepath: str, indent: int = 2):
    """
    Save dictionary as JSON file
    
    Args:
        obj: Dictionary to save
        filepath: Output file path
        indent: JSON indentation
    """
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w') as f:
        json.dump(obj, f, indent=indent, default=str)


def load_json(filepath: str) -> Dict:
    """
    Load dictionary from JSON file
    
    Args:
        filepath: Input file path
    
    Returns:
        Loaded dictionary
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def save_dataframe(df: pd.DataFrame, filepath: str, index: bool = False):
    """
    Save DataFrame to CSV
    
    Args:
        df: DataFrame to save
        filepath: Output file path
        index: Whether to save index
    """
    ensure_dir(os.path.dirname(filepath))
    df.to_csv(filepath, index=index)


def load_dataframe(filepath: str) -> pd.DataFrame:
    """
    Load DataFrame from CSV
    
    Args:
        filepath: Input file path
    
    Returns:
        Loaded DataFrame
    """
    return pd.read_csv(filepath)


def save_model(model: Any, filepath: str):
    """
    Save model using joblib
    
    Args:
        model: Model to save
        filepath: Output file path
    """
    ensure_dir(os.path.dirname(filepath))
    joblib.dump(model, filepath)


def load_model(filepath: str) -> Any:
    """
    Load model using joblib
    
    Args:
        filepath: Input file path
    
    Returns:
        Loaded model
    """
    return joblib.load(filepath)



# DATA VALIDATION UTILITIES

def validate_dataframe(
    df: pd.DataFrame,
    required_cols: Optional[List[str]] = None,
    logger: Optional[Logger] = None
) -> bool:
    """
    Validate DataFrame structure and content
    
    Args:
        df: DataFrame to validate
        required_cols: List of required column names
        logger: Logger instance
    
    Returns:
        True if valid, False otherwise
    """
    log = logger or Logger()
    
    if df is None or df.empty:
        log.error("DataFrame is None or empty")
        return False
    
    if required_cols:
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            log.error(f"Missing required columns: {missing_cols}")
            return False
    
    log.success("DataFrame validation passed")
    return True


def check_class_balance(y: pd.Series, logger: Optional[Logger] = None) -> Dict[str, float]:
    """
    Check class balance and compute imbalance ratio
    
    Args:
        y: Target variable
        logger: Logger instance
    
    Returns:
        Dictionary with class distribution info
    """
    log = logger or Logger()
    
    counts = y.value_counts()
    total = len(y)
    
    info = {
        'total_samples': total,
        'class_counts': counts.to_dict(),
        'class_proportions': (counts / total).to_dict(),
        'imbalance_ratio': counts.max() / counts.min(),
        'minority_class': counts.idxmin(),
        'majority_class': counts.idxmax(),
    }
    
    log.info("Class Distribution:")
    for cls, count in counts.items():
        pct = 100 * count / total
        log.info(f"  Class {cls}: {count:,} ({pct:.2f}%)")
    log.info(f"Imbalance Ratio: {info['imbalance_ratio']:.2f}:1")
    
    return info


def check_missing_values(df: pd.DataFrame, logger: Optional[Logger] = None) -> pd.DataFrame:
    """
    Check for missing values and return summary
    
    Args:
        df: DataFrame to check
        logger: Logger instance
    
    Returns:
        DataFrame with missing value summary
    """
    log = logger or Logger()
    
    missing = df.isnull().sum()
    missing_pct = 100 * missing / len(df)
    
    summary = pd.DataFrame({
        'missing_count': missing,
        'missing_percent': missing_pct
    })
    summary = summary[summary['missing_count'] > 0].sort_values('missing_count', ascending=False)
    
    if len(summary) > 0:
        log.warning(f"Found missing values in {len(summary)} columns")
        log.info("\n" + str(summary))
    else:
        log.success("No missing values found")
    
    return summary



# DATA TRANSFORMATION UTILITIES

def reduce_memory_usage(df: pd.DataFrame, logger: Optional[Logger] = None) -> pd.DataFrame:
    """
    Reduce memory usage by downcasting numeric types
    
    Args:
        df: DataFrame to optimize
        logger: Logger instance
    
    Returns:
        Optimized DataFrame
    """
    log = logger or Logger()
    
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    end_mem = df.memory_usage().sum() / 1024**2
    reduction = 100 * (start_mem - end_mem) / start_mem
    
    log.info(f"Memory usage reduced from {start_mem:.2f} MB to {end_mem:.2f} MB ({reduction:.1f}% reduction)")
    
    return df


def encode_target(y: pd.Series, positive_class: Any, logger: Optional[Logger] = None) -> Tuple[pd.Series, Dict]:
    """
    Encode target variable to 0/1
    
    Args:
        y: Target variable
        positive_class: Value to encode as 1
        logger: Logger instance
    
    Returns:
        Tuple of (encoded target, encoding mapping)
    """
    log = logger or Logger()
    
    unique_vals = y.unique()
    
    if len(unique_vals) != 2:
        log.error(f"Target variable must be binary. Found {len(unique_vals)} unique values: {unique_vals}")
        raise ValueError("Target variable must be binary")
    
    # Create mapping
    other_class = [v for v in unique_vals if v != positive_class][0]
    mapping = {positive_class: 1, other_class: 0}
    
    # Apply encoding
    y_encoded = y.map(mapping)
    
    if y_encoded.isnull().any():
        log.error("Encoding produced null values")
        raise ValueError("Encoding failed")
    
    log.success(f"Target encoded: {positive_class} -> 1, {other_class} -> 0")
    
    return y_encoded, mapping



# PERFORMANCE UTILITIES

class Timer:
    """Context manager for timing code execution"""
    
    def __init__(self, name: str = "Operation", logger: Optional[Logger] = None):
        self.name = name
        self.logger = logger or Logger()
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.logger.info(f"{self.name} started...")
        return self
    
    def __exit__(self, *args):
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        self.logger.success(f"{self.name} completed in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    
    def elapsed(self) -> float:
        """Get elapsed time"""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time if self.start_time else 0



# EXPERIMENT UTILITIES

def save_experiment_config(config: Dict, filepath: str, logger: Optional[Logger] = None):
    """
    Save experiment configuration to file
    
    Args:
        config: Configuration dictionary
        filepath: Output file path
        logger: Logger instance
    """
    log = logger or Logger()
    
    # Add metadata
    config['timestamp'] = datetime.now().isoformat()
    config['python_version'] = sys.version
    
    # Save as JSON
    save_json(config, filepath)
    log.success(f"Experiment config saved to: {filepath}")


def load_experiment_results(results_dir: str, logger: Optional[Logger] = None) -> pd.DataFrame:
    """
    Load and combine experiment results from directory
    
    Args:
        results_dir: Directory containing result files
        logger: Logger instance
    
    Returns:
        Combined results DataFrame
    """
    log = logger or Logger()
    
    results_files = []
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file == 'model_comparison.csv':
                results_files.append(os.path.join(root, file))
    
    if not results_files:
        log.warning(f"No result files found in {results_dir}")
        return pd.DataFrame()
    
    # Load and combine all results
    dfs = []
    for filepath in results_files:
        df = pd.read_csv(filepath)
        # Extract sampling and feature config from path
        path_parts = filepath.split(os.sep)
        if len(path_parts) >= 3:
            df['sampling_strategy'] = path_parts[-3]
            df['feature_config'] = path_parts[-2]
        dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    log.success(f"Loaded {len(results_files)} result files with {len(combined)} total rows")
    
    return combined



# TESTING

if __name__ == "__main__":
    # Test logger
    print("\n=== Testing Logger ===")
    logger = setup_logger("test", log_file="/home/claude/test_utils.log")
    logger.section("Testing Utilities")
    logger.info("This is an info message")
    logger.success("This is a success message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Test timer
    print("\n=== Testing Timer ===")
    with Timer("Test operation", logger):
        time.sleep(1)
    
    # Test data validation
    print("\n=== Testing Data Validation ===")
    df = pd.DataFrame({
        'A': [1, 2, 3, None],
        'B': ['a', 'b', 'c', 'd'],
        'target': [0, 1, 0, 1]
    })
    
    validate_dataframe(df, required_cols=['A', 'B', 'target'], logger=logger)
    check_missing_values(df, logger=logger)
    check_class_balance(df['target'], logger=logger)
    
    logger.elapsed_time()
    logger.success("All utility tests passed!")


# EDA HELPERS 

def get_continuous_numeric_cols(
    df: pd.DataFrame,
    target_col: str,
    min_unique: int = 10,
    logger: Optional[Logger] = None,
) -> List[str]:
    """Return numeric columns (excluding target) with at least `min_unique` unique values."""
    log = logger or Logger()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    cont_cols = [c for c in numeric_cols if df[c].nunique() >= min_unique]
    log.info(f"Identified {len(cont_cols)} continuous numeric columns for outlier/EDA")
    return cont_cols


def iqr_outlier_masks(
    df: pd.DataFrame,
    numeric_cols: List[str],
    k_mild: float = 1.5,
    k_extreme: float = 3.0,
    logger: Optional[Logger] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute mild/extreme outlier masks and summary using IQR for each numeric column.
    Returns (mild_mask_df, extreme_mask_df, summary_df).
    """
    log = logger or Logger()
    n = len(df)
    mild_mask = pd.DataFrame(False, index=df.index, columns=numeric_cols)
    extreme_mask = pd.DataFrame(False, index=df.index, columns=numeric_cols)
    summary_rows = []

    for col in numeric_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        s_valid = s.dropna()
        if s_valid.empty:
            continue
        q1 = s_valid.quantile(0.25)
        q3 = s_valid.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0 or pd.isna(iqr):
            continue

        mild_low = q1 - k_mild * iqr
        mild_high = q3 + k_mild * iqr
        extreme_low = q1 - k_extreme * iqr
        extreme_high = q3 + k_extreme * iqr

        mild_col_mask = (s < mild_low) | (s > mild_high)
        extreme_col_mask = (s < extreme_low) | (s > extreme_high)

        mild_mask[col] = mild_col_mask.fillna(False)
        extreme_mask[col] = extreme_col_mask.fillna(False)

        summary_rows.append({
            "column": col,
            "mild_outliers": int(mild_col_mask.sum()),
            "extreme_outliers": int(extreme_col_mask.sum()),
            "mild_pct": 100.0 * mild_col_mask.sum() / n if n else 0.0,
            "extreme_pct": 100.0 * extreme_col_mask.sum() / n if n else 0.0,
        })

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values("mild_outliers", ascending=False).reset_index(drop=True)
    log.info(f"IQR outlier summary computed for {len(summary_df)} columns")
    return mild_mask, extreme_mask, summary_df


def export_outliers(
    df: pd.DataFrame,
    mild_mask: pd.DataFrame,
    extreme_mask: pd.DataFrame,
    out_dir: str,
    prefix: str = "",
    logger: Optional[Logger] = None,
):
    """Export rows flagged as mild or extreme outliers (union across columns)."""
    log = logger or Logger()
    os.makedirs(out_dir, exist_ok=True)

    mild_any = mild_mask.any(axis=1) if not mild_mask.empty else pd.Series(False, index=df.index)
    extreme_any = extreme_mask.any(axis=1) if not extreme_mask.empty else pd.Series(False, index=df.index)

    mild_df = df[mild_any].copy()
    extreme_df = df[extreme_any].copy()

    if not mild_df.empty:
        path_mild = os.path.join(out_dir, f"{prefix}_mild_outliers_IQR.csv")
        mild_df.to_csv(path_mild, index=False)
        log.info(f"Exported {len(mild_df)} mild outlier rows to: {path_mild}")

    if not extreme_df.empty:
        path_extreme = os.path.join(out_dir, f"{prefix}_extreme_outliers_IQR.csv")
        extreme_df.to_csv(path_extreme, index=False)
        log.info(f"Exported {len(extreme_df)} extreme outlier rows to: {path_extreme}")


def describe_numeric(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    """Return extended descriptive stats for numeric features."""
    sub = df[numeric_cols].copy()
    desc = sub.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
    desc["skew"] = sub.skew(numeric_only=True)
    desc["kurtosis"] = sub.kurt(numeric_only=True)
    return desc


def describe_categorical(
    df: pd.DataFrame,
    cat_cols: List[str],
    top_k: int = 10,
) -> Dict[str, pd.DataFrame]:
    """Return dict of {column_name: top-k frequency table}."""
    out: Dict[str, pd.DataFrame] = {}
    n = len(df) if len(df) else 1
    for c in cat_cols:
        vc = df[c].astype(str).value_counts(dropna=False).head(top_k)
        pct = 100.0 * vc / n
        out[c] = pd.DataFrame({"count": vc, "percent": pct})
    return out
