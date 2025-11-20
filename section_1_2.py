import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from logger import setup_logger
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ==========================================================
# Paths, dataset selection, and logger setup
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Choose which dataset to use:
# options: "online", "telco"
DATASET = "telco"

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
# Normalize categorical formatting (Section 2 requirement)
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
# Separate features and target (AFTER cleaning)
# ==========================================================

if target_col not in data_df.columns:
    raise ValueError(f"Target column '{target_col}' not found in dataset")

X = data_df.drop(columns=[target_col])
y = data_df[target_col]

# For the online shoppers dataset, Revenue is often boolean-like → convert to int
if target_col == "Revenue":
    # After earlier processing, Revenue will be 'true'/'false' as strings if it was boolean.
    if y.dtype == "object":
        y = y.map({"true": 1, "false": 0}).astype(int)

logger.info(f"Target value counts after cleaning:\n{y.value_counts()}")

# Define feature types based on X
numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

logger.info(f"Numeric features: {numeric_features}")
logger.info(f"Categorical features: {categorical_features}")

# Sanity check: all columns accounted for
all_features = numeric_features + categorical_features + [target_col]
missing_cols = set(data_df.columns) - set(all_features)
extra_cols = set(all_features) - set(data_df.columns)

if missing_cols:
    logger.warning(f"Columns in data but not in feature lists: {missing_cols}")
if extra_cols:
    logger.warning(f"Columns in feature lists but not in data: {extra_cols}")

# High-cardinality categorical features
high_cardinality_threshold = 20
for col in categorical_features:
    unique_count = data_df[col].nunique()
    if unique_count > high_cardinality_threshold:
        logger.warning(
            f"High cardinality feature '{col}': {unique_count} unique values. "
            f"Consider alternative encodings for some models."
        )


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
# Preprocessing pipelines (numeric + categorical)
# ==========================================================

# Numeric: KNNImputer + scaling
numeric_transformer = Pipeline(
    steps=[
        ("imputer", KNNImputer(n_neighbors=5)),
        ("scpler", StandardScaler()),
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
