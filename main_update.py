import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
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
# options: "heart", "bank", "online", "telco", "stroke"
DATASET = "stroke"

if DATASET == "heart":
    CSV_FILE = "heart.csv"
    target_col = "target"

elif DATASET == "bank":
    CSV_FILE = "bank.csv"
    target_col = "deposit"

elif DATASET == "online":
    CSV_FILE = "online_shoppers_intention.csv"
    target_col = "Revenue"

elif DATASET == "telco":
    CSV_FILE = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    target_col = "Churn"

elif DATASET == "stroke":
    CSV_FILE = "healthcare-dataset-stroke-data.csv"
    target_col = "stroke"

else:
    raise ValueError(f"Unknown DATASET: {DATASET}")

CSV_PATH = os.path.join(BASE_DIR, CSV_FILE)

logger = setup_logger("")
logger.warning("Starting the Script")


# Helper function

def reduce_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns to reduce memory footprint."""
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    return df


# Load data and dataset-specific cleaning

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

elif DATASET == "stroke":
    # Drop id (identifier)
    if "id" in data_df.columns:
        data_df = data_df.drop(columns=["id"])

# Reduce memory usage
data_df = reduce_memory_usage(data_df)

logger.info("Data info:")
data_df.info()
logger.info(f"Missing values per column:\n{data_df.isnull().sum()}")
logger.info(f"Duplicate rows: {data_df.duplicated().sum()}")
print("*" * 100)

# Basic cleaning: duplicates, object columns, age sanity checks

# Inspect categorical/object columns for unusual values
for col in data_df.select_dtypes(include="object").columns:
    logger.success(f"Garbage values in '{col}':\n{data_df[col].value_counts()}")
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

# Age sanity check (if applicable)
if "age" in data_df.columns:
    logger.info(f"Age stats: min={data_df['age'].min()}, max={data_df['age'].max()}")

    invalid_ages = data_df[(data_df["age"] < 18) | (data_df["age"] > 100)]
    if not invalid_ages.empty:
        logger.warning(f"Found {invalid_ages.shape[0]} rows with suspicious ages.")
        data_df = data_df[(data_df["age"] >= 18) & (data_df["age"] <= 100)]
        logger.success("Dropped rows with invalid ages.")
    else:
        logger.info("No suspicious ages found.")
else:
    logger.info("No 'age' column in this dataset; skipping age stats and age filtering.")


# Exploratory Data Analysis (EDA)

logger.success(f"Exploratory data summary:\n{data_df.describe().T}")

'''
# Histograms for numeric features
numeric_cols_full = data_df.select_dtypes(include=["number"]).columns.tolist()

for col in numeric_cols_full:
    plt.figure(figsize=(6, 4))
    sns.histplot(data=data_df, x=col, kde=True)
    plt.title(f"Histogram of {col}")
    plt.tight_layout()
    plt.show()

# Boxplots for numeric features (skip target column)
for col in numeric_cols_full:
    if col == target_col:
        continue
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=data_df, x=col)
    plt.title(f"Boxplot of {col}")
    plt.tight_layout()
    plt.show()
'''

# Correlation heatmap for numeric features
corr_matrix = data_df.select_dtypes(include="number").corr()
plt.figure(figsize=(15, 15))
sns.heatmap(corr_matrix, annot=True)
plt.title("Correlation Heatmap")
plt.show()


# Separate features and target (AFTER cleaning)

if target_col not in data_df.columns:
    raise ValueError(f"Target column '{target_col}' not found in dataset")

X = data_df.drop(columns=[target_col])
y = data_df[target_col]

# For the online shoppers dataset, Revenue is often boolean → convert to int
if target_col == "Revenue":
    y = y.astype(int)

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


# Train-test split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

logger.info("Train-test split completed")
logger.info(f"Training set shape: {X_train.shape}")
logger.info(f"Test set shape: {X_test.shape}")


# Preprocessing pipelines (numeric + categorical)

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


# Apply preprocessing to train and test

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


# Apply SMOTE on training data

logger.info("Class distribution BEFORE SMOTE:")
logger.info(f"\n{y_train.value_counts()}")

sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_train_processed, y_train)

logger.success("SMOTE oversampling complete.")
logger.info("Class distribution AFTER SMOTE:")
logger.info(f"\n{pd.Series(y_resampled).value_counts()}")

# Convert processed arrays back to DataFrames (optional, but nice for inspection)
X_resampled_df = pd.DataFrame(X_resampled, columns=feature_names)
X_test_processed_df = pd.DataFrame(X_test_processed, columns=feature_names)

logger.success(f"Final resampled training data shape: {X_resampled_df.shape}")
logger.success(f"Final test data shape: {X_test_processed_df.shape}")


# Visualize the effect of SMOTE

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.countplot(x=y_train)
plt.title("Class Distribution Before SMOTE")

plt.subplot(1, 2, 2)
sns.countplot(x=y_resampled)
plt.title("Balanced Class Distribution After SMOTE")

plt.tight_layout()
plt.show()

logger.success("Data preprocessing completed successfully!")
logger.info("Next steps: Use X_resampled, y_resampled for model training")
logger.info("Use X_test_processed, y_test for model evaluation")



'''
Following the rules of Machine Learning and Algorithm Development, we will split the data into training, validation, and test sets.
1. Data discretization
2. Data cleaning 
3. Data integration
4. Data transformation
5. Data reduction
'''