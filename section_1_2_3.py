import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from logger import setup_logger
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.stats import skew
from sklearn.feature_selection import chi2, VarianceThreshold, mutual_info_classif

# ==========================================================
# Paths, dataset selection, and logger setup
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Choose which dataset to use:
# options: "online", "telco"
DATASET = "online"

if DATASET == "online":
    CSV_FILE = "online_shoppers_intention.csv"
    target_col = "Revenue"

elif DATASET == "telco":
    CSV_FILE = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    target_col = "Churn"

else:
    raise ValueError(f"Unknown DATASET: {DATASET}. Use 'online' or 'telco' only.")

CSV_PATH = os.path.join(BASE_DIR, CSV_FILE)

logger = setup_logger("")
logger.warning(f"Starting the Script for DATASET = {DATASET}")


# ==========================================================
# Helper function
# ==========================================================

def reduce_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns to reduce memory footprint."""
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    return df


# ==========================================================
# Load data and dataset-specific cleaning
# ==========================================================

data_df = pd.read_csv(CSV_PATH)
logger.success("Data loaded successfully")
logger.info(f"Initial data shape: {data_df.shape}")

# Dataset-specific cleaning
if DATASET == "telco":
    # Drop customerID (identifier, not a feature)
    if "customerID" in data_df.columns:
        data_df = data_df.drop(columns=["customerID"])

    # TotalCharges sometimes comes as object with blanks → coerce to numeric
    if "TotalCharges" in data_df.columns:
        data_df["TotalCharges"] = pd.to_numeric(data_df["TotalCharges"], errors="coerce")

# Reduce memory usage
data_df = reduce_memory_usage(data_df)

logger.info("Data info AFTER dataset-specific cleaning and memory reduction:")
data_df.info()
logger.info(f"Missing values per column:\n{data_df.isnull().sum()}")
logger.info(f"Duplicate rows: {data_df.duplicated().sum()}")
print("*" * 100)

# ==========================================================
# Section 1: Visualize initial class imbalance (full dataset)
# ==========================================================

if target_col not in data_df.columns:
    raise ValueError(f"Target column '{target_col}' not found in dataset")

plt.figure(figsize=(6, 4))
sns.countplot(x=data_df[target_col])
plt.title(f"Initial Class Distribution (Full Dataset: {DATASET})")
plt.tight_layout()
plt.show()
logger.info("Plotted initial class distribution on full dataset")

# ==========================================================
# Basic cleaning: duplicates, categorical inspection
# ==========================================================

# Inspect categorical/object columns for unusual values
for col in data_df.select_dtypes(include="object").columns:
    logger.success(f"Value counts in '{col}':\n{data_df[col].value_counts()}")
    logger.info(f"Unique values in '{col}': {data_df[col].unique()}")

# Drop duplicate rows (if any)
duplicates_before = data_df.duplicated().sum()
logger.info(f"Duplicate rows before dropping: {duplicates_before}")

if duplicates_before > 0:
    data_df = data_df.drop_duplicates()
    logger.success(f"Dropped {duplicates_before} duplicate rows.")
    logger.info(f"Shape after dropping duplicates: {data_df.shape}")
else:
    logger.info("No duplicate rows found.")


# ==========================================================
# Normalize categorical formatting 
# - trim spaces
# - lowercase
# - dataset-specific semantic cleanup (e.g., 'no internet service' → 'no')
# ==========================================================

cat_cols = data_df.select_dtypes(include=["object"]).columns.tolist()

for col in cat_cols:
    # Convert to string, strip whitespace, and lowercase for consistency
    data_df[col] = (
        data_df[col]
        .astype(str)
        .str.strip()
        .str.lower()
    )
    logger.info(f"Standardized formatting for categorical column '{col}'")
    logger.info(f"Unique values after basic formatting: {data_df[col].unique()}")

# Telco-specific category normalization:
# Map 'no internet service' and 'no phone service' to 'no'
if DATASET == "telco":
    service_replacements = {
        "no internet service": "no",
        "no phone service": "no",
    }
    for col in cat_cols:
        data_df[col] = data_df[col].replace(service_replacements)
        logger.info(f"Applied telco service replacements in column '{col}'")
        logger.info(f"Unique values after telco replacements: {data_df[col].unique()}")


# ==========================================================
# Outlier detection (IQR-based) and visualization for continuous features
# - Do NOT remove outliers, just detect and visualize
# - Mild outliers: outside Q1 ± 1.5 * IQR
# - Extreme outliers: outside Q1 ± 3 * IQR
# ==========================================================

numeric_cols_full = data_df.select_dtypes(include=["number"]).columns.tolist()

# Define "continuous" numeric columns (skip binary / near-binary like 0/1)
continuous_cols = [
    col for col in numeric_cols_full
    if data_df[col].nunique() > 10 and col != target_col
]

logger.info(f"Continuous numeric features for outlier analysis: {continuous_cols}")

mild_outlier_masks = []
extreme_outlier_masks = []

for col in continuous_cols:
    col_series = data_df[col].dropna()
    if col_series.empty:
        continue

    Q1 = col_series.quantile(0.25)
    Q3 = col_series.quantile(0.75)
    IQR = Q3 - Q1

    if IQR == 0:
        logger.info(f"Column '{col}' has IQR=0 (no spread). Skipping outlier detection.")
        continue

    mild_lower = Q1 - 1.5 * IQR
    mild_upper = Q3 + 1.5 * IQR

    extreme_lower = Q1 - 3 * IQR
    extreme_upper = Q3 + 3 * IQR

    mild_mask = (data_df[col] < mild_lower) | (data_df[col] > mild_upper)
    extreme_mask = (data_df[col] < extreme_lower) | (data_df[col] > extreme_upper)

    mild_count = mild_mask.sum()
    extreme_count = extreme_mask.sum()

    mild_outlier_masks.append(mild_mask)
    extreme_outlier_masks.append(extreme_mask)

    logger.info(
        f"Column '{col}' — mild outliers (|beyond 1.5 IQR|): {mild_count}, "
        f"extreme outliers (|beyond 3 IQR|): {extreme_count}"
    )

    # Visualization: boxplot for each continuous feature
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=data_df[col])
    plt.title(f"Boxplot highlighting IQR-based outliers: {col}")
    plt.tight_layout()
    plt.show()

# Combine and export mild outliers across all continuous columns
if mild_outlier_masks:
    combined_mild_mask = mild_outlier_masks[0]
    for m in mild_outlier_masks[1:]:
        combined_mild_mask = combined_mild_mask | m

    mild_outliers_df = data_df[combined_mild_mask]
    mild_outlier_filename = f"{DATASET}_mild_outliers_IQR.csv"
    mild_outlier_path = os.path.join(BASE_DIR, mild_outlier_filename)
    mild_outliers_df.to_csv(mild_outlier_path, index=False)
    logger.success(
        f"Extracted {mild_outliers_df.shape[0]} mild outlier rows (IQR-based) to: {mild_outlier_path}"
    )
else:
    logger.info("No mild outliers detected based on IQR across continuous features.")

# Combine and export extreme outliers across all continuous columns
if extreme_outlier_masks:
    combined_extreme_mask = extreme_outlier_masks[0]
    for m in extreme_outlier_masks[1:]:
        combined_extreme_mask = combined_extreme_mask | m

    extreme_outliers_df = data_df[combined_extreme_mask]
    extreme_outlier_filename = f"{DATASET}_extreme_outliers_IQR.csv"
    extreme_outlier_path = os.path.join(BASE_DIR, extreme_outlier_filename)
    extreme_outliers_df.to_csv(extreme_outlier_path, index=False)
    logger.success(
        f"Extracted {extreme_outliers_df.shape[0]} extreme outlier rows (IQR-based) to: {extreme_outlier_path}"
    )
else:
    logger.info("No extreme outliers detected based on IQR across continuous features.")


# ==========================================================
# Export cleaned template CSV for feature engineering (Section 3)
# ==========================================================

template_filename = f"{DATASET}_clean_template.csv"
template_path = os.path.join(BASE_DIR, template_filename)
data_df.to_csv(template_path, index=False)
logger.success(f"Exported cleaned template CSV for feature engineering: {template_path}")


# ==========================================================
# Exploratory Data Analysis (EDA) – focused on continuous features
# ==========================================================

logger.success(f"Exploratory data summary:\n{data_df.describe().T}")
'''
# Histograms for continuous numeric features only
for col in continuous_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(data=data_df, x=col, kde=True)
    plt.title(f"Histogram of {col} (continuous)")
    plt.tight_layout()
    plt.show()
'''

# Correlation heatmap for all numeric features
corr_matrix = data_df.select_dtypes(include="number").corr()
plt.figure(figsize=(15, 15))
sns.heatmap(corr_matrix, annot=True)
plt.title("Correlation Heatmap (All Numeric Features)")
plt.show()

# ==========================================================
# SECTION 3 – FEATURE ENGINEERING (PART 1: Target-Independent)
#   - Log transforms for skewed numeric features
#   - Interaction features
#   - Binning / grouping
# ==========================================================
log_features = []
interaction_features = []
binned_features = []

# Base feature frame (no target)
feature_df = data_df.drop(columns=[target_col])
numeric_cols = feature_df.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = feature_df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

logger.info(f"[FE] (Part 1) Numeric columns: {numeric_cols}")
logger.info(f"[FE] (Part 1) Categorical columns: {categorical_cols}")

# -----------------------------
# 3.2 Handling Skewed Distributions (log1p)
# -----------------------------
if len(numeric_cols) > 0:
    skewness = feature_df[numeric_cols].apply(lambda x: skew(x.dropna()))
    logger.info(f"[FE] Skewness of numeric features:\n{skewness}")

    skewed_feats = skewness[skewness.abs() > 1].index.tolist()
    logger.info(f"[FE] Highly skewed numeric features (|skew| > 1): {skewed_feats}")

    for col in skewed_feats:
        if (feature_df[col] >= 0).all():
            new_col = f"{col}_log1p"
            data_df[new_col] = np.log1p(data_df[col])
            log_features.append(new_col)
            logger.info(f"[FE] Created log-transformed feature '{new_col}' from '{col}'")
        else:
            logger.info(f"[FE] Skipping log1p transform for '{col}' (contains negative values).")
else:
    logger.info("[FE] No numeric features available for skewness transformation.")

# -----------------------------
# 3.3 Interaction Features (dataset-specific)
# -----------------------------
if DATASET == "telco":
    base_interact = []
    for c in ["tenure", "monthlycharges", "totalcharges", "TotalCharges"]:
        if c in data_df.columns:
            base_interact.append(c)

    for i in range(len(base_interact)):
        for j in range(i + 1, len(base_interact)):
            f1 = base_interact[i]
            f2 = base_interact[j]
            new_col = f"{f1}_x_{f2}"
            data_df[new_col] = data_df[f1] * data_df[f2]
            interaction_features.append(new_col)
            logger.info(f"[FE] Created interaction feature '{new_col}' = {f1} * {f2}")

elif DATASET == "online":
    base_interact = []
    for c in ["pagevalues", "productrelated", "exitrate"]:
        if c in data_df.columns:
            base_interact.append(c)

    for i in range(len(base_interact)):
        for j in range(i + 1, len(base_interact)):
            f1 = base_interact[i]
            f2 = base_interact[j]
            new_col = f"{f1}_x_{f2}"
            data_df[new_col] = data_df[f1] * data_df[f2]
            interaction_features.append(new_col)
            logger.info(f"[FE] Created interaction feature '{new_col}' = {f1} * {f2}")

# -----------------------------
# 3.3 Binning / Grouping
# -----------------------------
# Telco: tenure bins
if "tenure" in data_df.columns:
    tenure_bin_col = "tenure_group"
    data_df[tenure_bin_col] = pd.cut(
        data_df["tenure"],
        bins=[0, 12, 36, 72],
        labels=["0-1 year", "1-3 years", "3+ years"],
        include_lowest=True
    )
    binned_features.append(tenure_bin_col)
    logger.info("[FE] Created binned feature 'tenure_group' from 'tenure'")

# Telco: total charges bins
charge_col = None
for c in ["totalcharges", "TotalCharges"]:
    if c in data_df.columns:
        charge_col = c
        break

if charge_col is not None:
    charge_bin_col = "charges_group"
    try:
        data_df[charge_bin_col] = pd.qcut(
            data_df[charge_col],
            q=4,
            labels=["low", "medium-low", "medium-high", "high"]
        )
        binned_features.append(charge_bin_col)
        logger.info(f"[FE] Created binned feature '{charge_bin_col}' from '{charge_col}' using quartiles")
    except ValueError:
        logger.info(f"[FE] Skipping qcut binning for '{charge_col}' (insufficient unique values).")

# Online: pagevalues bins
if "pagevalues" in data_df.columns:
    page_bin_col = "pagevalues_group"
    try:
        data_df[page_bin_col] = pd.qcut(
            data_df["pagevalues"],
            q=4,
            labels=["low", "medium-low", "medium-high", "high"]
        )
        binned_features.append(page_bin_col)
        logger.info("[FE] Created binned feature 'pagevalues_group' from 'pagevalues'")
    except ValueError:
        logger.info("[FE] Skipping qcut binning for 'pagevalues' (insufficient unique values).")

logger.success(
    f"[FE] (Part 1) Created engineered features – "
    f"log: {log_features}, interactions: {interaction_features}, bins: {binned_features}"
)



# ==========================================================
# Separate features and target (AFTER cleaning + FE Part 1)
# ==========================================================

if target_col not in data_df.columns:
    raise ValueError(f"Target column '{target_col}' not found in dataset")

X = data_df.drop(columns=[target_col])
y = data_df[target_col]

# For the online shoppers dataset, Revenue is often boolean-like → convert to int
if target_col == "Revenue":
    if y.dtype == "object":
        y = y.map({"true": 1, "false": 0}).astype(int)

logger.info(f"Target value counts after cleaning:\n{y.value_counts()}")


# ==========================================================
# Train-test split
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

logger.info("Train-test split completed")
logger.info(f"Training set shape: {X_train.shape}")
logger.info(f"Test set shape: {X_test.shape}")

# ==========================================================
# SECTION 3 – FEATURE ENGINEERING (PART 2: Train-Only Selection )
# ==========================================================

# Re-detect types from X_train (includes engineered features)
numeric_cols_train = X_train.select_dtypes(include=["number"]).columns.tolist()
categorical_cols_train = X_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

logger.info(f"[FE] (Part 2) Train numeric columns: {numeric_cols_train}")
logger.info(f"[FE] (Part 2) Train categorical columns: {categorical_cols_train}")

# Numeric y for selection
y_train_fe = y_train.copy()
if y_train_fe.dtype == "object":
    y_train_fe_num = y_train_fe.map({
        "yes": 1, "no": 0,
        "true": 1, "false": 0,
        "1": 1, "0": 0
    })
else:
    y_train_fe_num = y_train_fe

valid_mask_train = ~y_train_fe_num.isna()
X_train_sel = X_train.loc[valid_mask_train]
y_train_sel = y_train_fe_num.loc[valid_mask_train]

logger.info(f"[FE] (Part 2) Target value counts used for selection:\n{y_train_sel.value_counts()}")

# ---------- Numeric feature selection: corr + MI ----------
top_numeric_base = []
if len(numeric_cols_train) > 0 and y_train_sel.dtype != "O":
    # Work on a numeric-only copy and fill NaNs for selection purposes
    X_num_sel = X_train_sel[numeric_cols_train].copy()

    # Simple imputation for selection only (does NOT change main data_df)
    X_num_sel = X_num_sel.fillna(X_num_sel.median(numeric_only=True))

    # Correlation with target (based on imputed copy)
    corr_with_target = X_num_sel.corrwith(y_train_sel)
    corr_abs = corr_with_target.abs().sort_values(ascending=False)

    logger.info("[FE] Train numeric corr with target (after NaN imputation for selection):")
    logger.info(f"\n{corr_with_target.sort_values(ascending=False)}")

    # Mutual information on NaN-free numeric data
    mi_scores = mutual_info_classif(
        X_num_sel,
        y_train_sel,
        discrete_features=False,
        random_state=42,
    )
    mi_series = pd.Series(mi_scores, index=numeric_cols_train).sort_values(ascending=False)
    logger.info(f"[FE] Train numeric mutual information scores:\n{mi_series}")

    # Choose top K numeric based on |corr| – A2: allow ~10
    K_NUMERIC = min(10, len(numeric_cols_train))
    top_numeric_base = list(corr_abs.head(K_NUMERIC).index)
    logger.info(f"[FE] Initially selected top numeric features (by |corr|): {top_numeric_base}")

    # Redundancy removal: drop one of each highly correlated pair (|corr| > 0.9)
    if len(top_numeric_base) > 1:
        corr_matrix_num = X_num_sel[top_numeric_base].corr().abs()
        upper_tri = corr_matrix_num.where(
            np.triu(np.ones(corr_matrix_num.shape), k=1).astype(bool)
        )
        to_drop_redundant = [
            column for column in upper_tri.columns
            if any(upper_tri[column] > 0.9)
        ]
        logger.info(f"[FE] Numeric features dropped due to high inter-correlation (>0.9): {to_drop_redundant}")
        top_numeric_base = [c for c in top_numeric_base if c not in to_drop_redundant]

    logger.info(f"[FE] Final selected base numeric features (train-based): {top_numeric_base}")
else:
    logger.info("[FE] Skipping numeric selection (no numeric features or non-numeric target).")

# ---------- Categorical feature selection: chi-square ----------
top_categorical_base = []
if len(categorical_cols_train) > 0:
    X_cat_train = pd.get_dummies(X_train_sel[categorical_cols_train], drop_first=True)
    chi_scores, p_values = chi2(X_cat_train, y_train_sel)

    chi_df = pd.DataFrame({
        "dummy_feature": X_cat_train.columns,
        "chi2_score": chi_scores,
        "p_value": p_values,
    })
    chi_df["orig_feature"] = chi_df["dummy_feature"].apply(lambda x: x.split("_")[0])
    chi_agg = chi_df.groupby("orig_feature")["chi2_score"].max().sort_values(ascending=False)

    logger.info(f"[FE] Aggregated chi-square scores per categorical feature (train):\n{chi_agg}")

    K_CAT = min(10, len(categorical_cols_train))
    top_categorical_base = list(chi_agg.head(K_CAT).index)
    logger.info(f"[FE] Selected top categorical features (train-based): {top_categorical_base}")
else:
    logger.info("[FE] No categorical features available for chi-square selection on train.")

# ---------- Variance threshold (logging only) ----------
X_train_all_encoded = pd.get_dummies(X_train_sel, drop_first=True)
if X_train_all_encoded.shape[1] > 0:
    vt_selector = VarianceThreshold(threshold=0.0)
    vt_selector.fit(X_train_all_encoded)

    low_variance_mask = vt_selector.get_support()
    kept_vt = X_train_all_encoded.columns[low_variance_mask]
    dropped_vt = X_train_all_encoded.columns[~low_variance_mask]

    logger.info(f"[FE] Train features that would be dropped by VarianceThreshold: {list(dropped_vt)}")
    logger.info(f"[FE] Train features kept by VarianceThreshold: {len(kept_vt)}")
else:
    logger.info("[FE] Skipping VarianceThreshold (no encoded train features).")


# ---------- Final feature lists for ColumnTransformer ----------
numeric_features = []

for col in top_numeric_base:
    if col in X_train.columns:
        numeric_features.append(col)

for col in log_features + interaction_features:
    if col in X_train.columns:
        numeric_features.append(col)

numeric_features = list(dict.fromkeys(numeric_features))
logger.info(f"[FE] Final numeric_features list (used for modeling): {numeric_features}")

categorical_features = []

for col in top_categorical_base:
    if col in X_train.columns:
        categorical_features.append(col)

for col in binned_features:
    if col in X_train.columns:
        categorical_features.append(col)

categorical_features = list(dict.fromkeys(categorical_features))
logger.info(f"[FE] Final categorical_features list (used for modeling): {categorical_features}")

logger.success(
    f"[FE] TOTAL selected features for modeling: "
    f"{len(numeric_features)} numeric + {len(categorical_features)} categorical"
)



# ==========================================================
# Preprocessing pipelines (numeric + categorical)
# ==========================================================

# Numeric: KNNImputer + scaling
numeric_transformer = Pipeline(
    steps=[
        ("imputer", KNNImputer(n_neighbors=5)),
        ("scaler", StandardScaler()),
    ]
)


# Categorical: most_frequent + one-hot
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(
                drop="first",
                sparse_output=False,
                handle_unknown="ignore",
            ),
        ),
    ]
)

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

logger.info("Preprocessing pipeline created")


# ==========================================================
# Apply preprocessing to train and test
# ==========================================================

logger.info("Fitting preprocessing on training data...")
X_train_processed = preprocessor.fit_transform(X_train)
logger.success("Training data preprocessing completed")

logger.info("Transforming test data...")
X_test_processed = preprocessor.transform(X_test)
logger.success("Test data preprocessing completed")

# Get feature names after preprocessing
feature_names = []
for name, transformer, features in preprocessor.transformers_:
    if name == "num":
        feature_names.extend(features)
    elif name == "cat":
        if len(features) > 0:
            cat_features = transformer.named_steps["onehot"].get_feature_names_out(features)
            feature_names.extend(cat_features)

logger.info(f"Final processed feature count: {len(feature_names)}")


# ==========================================================
# Apply Tomek Links + SMOTE on training data (Section 2)
# ==========================================================

logger.info("Class distribution BEFORE Tomek Links / SMOTE:")
logger.info(f"\n{y_train.value_counts()}")

# First: Tomek Links to reduce class overlap / noise
tl = TomekLinks(sampling_strategy="auto")
X_clean, y_clean = tl.fit_resample(X_train_processed, y_train)
logger.info("Applied Tomek Links under-sampling to reduce class overlap/noise.")
logger.info("Class distribution AFTER Tomek Links:")
logger.info(f"\n{pd.Series(y_clean).value_counts()}")

# Then: SMOTE to oversample minority class
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_clean, y_clean)

logger.success("SMOTE oversampling complete.")
logger.info("Class distribution AFTER SMOTE:")
logger.info(f"\n{pd.Series(y_resampled).value_counts()}")

# Convert processed arrays back to DataFrames (optional, but nice for inspection)
X_resampled_df = pd.DataFrame(X_resampled, columns=feature_names)
X_test_processed_df = pd.DataFrame(X_test_processed, columns=feature_names)

logger.success(f"Final resampled training data shape: {X_resampled_df.shape}")
logger.success(f"Final test data shape: {X_test_processed_df.shape}")


# ==========================================================
# Visualize the effect of balancing (Tomek + SMOTE)
# ==========================================================

plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
sns.countplot(x=y_train)
plt.title("Class Distribution Before Tomek/SMOTE")

plt.subplot(1, 3, 2)
sns.countplot(x=y_clean)
plt.title("After Tomek Links")

plt.subplot(1, 3, 3)
sns.countplot(x=y_resampled)
plt.title("After Tomek + SMOTE")

plt.tight_layout()
plt.show()

logger.success("Data preprocessing and balancing completed successfully!")
logger.info("Use X_resampled, y_resampled for model training.")
logger.info("Use X_test_processed, y_test for model evaluation.")
