"""
Section 4 – Supervised Learning

This script:
- Loads the preprocessed & balanced datasets exported from Section 1–3
- Trains multiple supervised learning models
- Performs hyperparameter tuning for key models
- Evaluates models on a held-out test set
- Compares models using multiple metrics
- Plots ROC curves and feature importance (where applicable)
- Saves a comparison table and the best model for later reuse

Assumptions:
- section_1_2_3.py has already been run
- Outputs exist in: outputs_section4/
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # for prettier feature-importance plots
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay,
)


# Optional: use your custom logger if available
try:
    from logger import logger
except ImportError:  # fallback to simple print-based logger

    class SimpleLogger:
        def info(self, msg): print(f"[INFO] {msg}")
        def warning(self, msg): print(f"[WARN] {msg}")
        def success(self, msg): print(f"[OK]   {msg}")
        def error(self, msg): print(f"[ERR]  {msg}")

    logger = SimpleLogger()

warnings.filterwarnings("ignore")



# CONFIGURATION
# Choose which dataset to model: "telco" or "online"
DATASET = "online"        # change to "online" when needed
OUTPUT_DIR = "outputs_section4"
RESULTS_DIR = "outputs_section4_results"
os.makedirs(RESULTS_DIR, exist_ok=True)



# UTILITY FUNCTIONS
def load_data(dataset_name: str):
    """
    Load preprocessed train/test data and labels exported from Section 1–3.
    """
    logger.info(f"Loading preprocessed data for dataset: {dataset_name!r}")

    X_train_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_X_train_bal.csv")
    y_train_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_y_train_bal.csv")
    X_test_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_X_test_processed.csv")
    y_test_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_y_test.csv")
    feat_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_final_feature_names.csv")

    # Basic existence checks (lightweight robustness)
    for p in [X_train_path, y_train_path, X_test_path, y_test_path, feat_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required file not found: {p}")

    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)["target"]
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path)["target"]

    feature_names = pd.read_csv(feat_path)["feature_name"].tolist()

    # Sanity checks
    assert X_train.shape[1] == len(feature_names), "Train X and feature names length mismatch."
    assert X_test.shape[1] == len(feature_names), "Test X and feature names length mismatch."

    logger.success(
        f"Loaded X_train: {X_train.shape}, y_train: {y_train.shape}, "
        f"X_test: {X_test.shape}, y_test: {y_test.shape}"
    )

    return X_train, y_train, X_test, y_test, feature_names


def get_models():
    """
    Define a set of candidate classification models.
    Some will be tuned via GridSearchCV.
    """
    models = {
        "LogisticRegression": LogisticRegression(
            solver="liblinear",  # good for small/medium datasets & L1/L2
            max_iter=1000,
        ),
        "RandomForest": RandomForestClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "KNN": KNeighborsClassifier(),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "NaiveBayes": GaussianNB(),
    }
    return models


def get_param_grids():
    """
    Hyperparameter grids for some models.
    Keep grids modest so that GridSearchCV is not too slow.
    """
    param_grids = {
        "LogisticRegression": {
            "C": [0.01, 0.1, 1.0, 10.0],
            "penalty": ["l1", "l2"],
        },
        "RandomForest": {
            "n_estimators": [100, 200],
            "max_depth": [None, 5, 10],
            "max_features": ["sqrt", "log2"],
        },
        "SVM": {
            "C": [0.1, 1.0, 10.0],
            "kernel": ["rbf", "linear"],
            "gamma": ["scale", "auto"],
        },
        "KNN": {
            "n_neighbors": [3, 5, 7, 9],
            "weights": ["uniform", "distance"],
        },
        # DecisionTree and NaiveBayes we keep with default or light config
    }
    return param_grids


def get_y_proba(model, X_test):
    """
    Get prediction probabilities for ROC-AUC.
    Falls back to decision_function if predict_proba is not available.
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        # Convert decision scores to [0,1] via min-max for ROC-AUC
        min_s, max_s = scores.min(), scores.max()
        if max_s > min_s:
            proba = (scores - min_s) / (max_s - min_s)
        else:
            proba = np.zeros_like(scores)
    else:
        # Worst case: no probability-like output
        proba = None
    return proba


def evaluate_model(name, model, X_test, y_test):
    """
    Compute evaluation metrics (accuracy, precision, recall, F1, ROC-AUC).
    Also returns confusion matrix and classification report as text.
    """
    y_pred = model.predict(X_test)
    y_proba = get_y_proba(model, X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    if y_proba is not None:
        auc = roc_auc_score(y_test, y_proba)
    else:
        auc = np.nan  # cannot compute AUC without scores

    cm = confusion_matrix(y_test, y_pred)
    cls_report = classification_report(y_test, y_pred, digits=4)

    logger.info(f"=== {name} – Evaluation on Test Set ===")
    logger.info(f"Accuracy:  {acc:.4f}")
    logger.info(f"Precision: {prec:.4f}")
    logger.info(f"Recall:    {rec:.4f}")
    logger.info(f"F1-score:  {f1:.4f}")
    if not np.isnan(auc):
        logger.info(f"ROC-AUC:   {auc:.4f}")
    logger.info("Confusion Matrix:")
    logger.info(f"\n{cm}")
    logger.info("Classification Report:")
    logger.info(f"\n{cls_report}")

    return {
        "model": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
    }


def plot_roc_curves(models, X_test, y_test, dataset_name: str):
    """
    Plot ROC curves for models that provide probability/score outputs.
    """
    plt.figure(figsize=(8, 6))
    for name, model in models.items():
        y_proba = get_y_proba(model, X_test)
        if y_proba is None:
            continue
        RocCurveDisplay.from_predictions(
            y_test, y_proba, name=name, ax=plt.gca()
        )

    plt.title(f"ROC Curves – {dataset_name}")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.legend()
    plt.tight_layout()

    fig_path = os.path.join(RESULTS_DIR, f"{dataset_name}_roc_curves.png")
    plt.savefig(fig_path, dpi=300)
    logger.success(f"Saved ROC curve plot to: {fig_path}")
    plt.close()


def analyze_feature_importance(best_models, feature_names, dataset_name: str):
    """
    Analyze and plot feature importance for models that support it.

    - Tree-based models: use feature_importances_
    - LogisticRegression: use absolute value of coefficients
    """
    for name, model in best_models.items():
        importance = None

        # Tree-based models
        if hasattr(model, "feature_importances_"):
            importance = np.array(model.feature_importances_)

        # Logistic Regression coefficients (single output)
        elif isinstance(model, LogisticRegression) and hasattr(model, "coef_"):
            # coef_: shape (1, n_features) for binary classification
            importance = np.abs(model.coef_[0])

        if importance is None:
            logger.info(f"[Feature Importance] Skipping {name}: no importance/coefs available.")
            continue

        if len(importance) != len(feature_names):
            logger.warning(
                f"[Feature Importance] Length mismatch for {name}: "
                f"{len(importance)} importances vs {len(feature_names)} feature names."
            )
            continue

        imp_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importance,
        }).sort_values("importance", ascending=False)

        # Save top 20 as CSV
        csv_path = os.path.join(
            RESULTS_DIR, f"{dataset_name}_{name}_feature_importance_top20.csv"
        )
        imp_df.head(20).to_csv(csv_path, index=False)
        logger.success(f"[Feature Importance] Saved top-20 importance for {name} to: {csv_path}")

        # Plot top 20
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=imp_df.head(20),
            x="importance",
            y="feature",
            orient="h"
        )
        plt.title(f"Top 20 Feature Importances – {name} ({dataset_name})")
        plt.tight_layout()

        fig_path = os.path.join(
            RESULTS_DIR, f"{dataset_name}_{name}_feature_importance_top20.png"
        )
        plt.savefig(fig_path, dpi=300)
        logger.success(f"[Feature Importance] Saved plot for {name} to: {fig_path}")
        plt.close()


def save_best_model(results_df: pd.DataFrame, best_models: dict, dataset_name: str):
    """
    Save the best model (by F1-score, with ROC-AUC as tiebreaker) to disk.
    """
    # Primary metric: F1 (good for imbalanced classification)
    # If tie, use ROC-AUC if available.
    # We'll just pick the row with the highest F1; if multiple, pandas idxmax will pick the first.
    best_model_name = results_df["f1"].idxmax()
    best_model = best_models[best_model_name]

    model_path = os.path.join(RESULTS_DIR, f"{dataset_name}_best_model.pkl")
    joblib.dump(best_model, model_path)
    logger.success(
        f"[Model Persistence] Saved best model ({best_model_name}) to: {model_path}"
    )

    return best_model_name, model_path



# MAIN PIPELINE
def main():
    # 1) Load data
    X_train, y_train, X_test, y_test, feature_names = load_data(DATASET)

    # 2) Define models and hyperparameters
    base_models = get_models()
    param_grids = get_param_grids()

    results = []
    best_models = {}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 3) For each model: either tune with GridSearch or fit directly
    for name, model in base_models.items():
        logger.info(f"\n{'=' * 60}\nTraining model: {name}\n{'=' * 60}")

        if name in param_grids:
            logger.info(f"Using GridSearchCV for {name}...")
            grid = GridSearchCV(
                estimator=model,
                param_grid=param_grids[name],
                scoring="f1",
                cv=cv,
                n_jobs=-1,
                verbose=0,
            )
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
            logger.success(f"Best params for {name}: {grid.best_params_}")
            logger.success(f"Best CV F1-score for {name}: {grid.best_score_:.4f}")
        else:
            logger.info(f"Fitting {name} with default parameters...")
            model.fit(X_train, y_train)
            best_model = model

        best_models[name] = best_model

        # Evaluate on test set
        metrics = evaluate_model(name, best_model, X_test, y_test)
        results.append(metrics)

    # 4) Collect results into a DataFrame
    results_df = pd.DataFrame(results).set_index("model")
    logger.info("\n=== Model Comparison (Test Set) ===")
    logger.info(f"\n{results_df}")

    # Save results to CSV
    res_path = os.path.join(RESULTS_DIR, f"{DATASET}_model_comparison.csv")
    results_df.to_csv(res_path)
    logger.success(f"Saved model comparison table to: {res_path}")

    # 5) Plot ROC curves
    plot_roc_curves(best_models, X_test, y_test, DATASET)

    # 6) Feature Importance Analysis (RandomForest, DecisionTree, LogisticRegression)
    analyze_feature_importance(best_models, feature_names, DATASET)

    # 7) Save best model for future use
    best_model_name, model_path = save_best_model(results_df, best_models, DATASET)
    logger.info(
        f"Best model according to F1-score: {best_model_name} "
        f"(saved at {model_path})"
    )

    logger.success("Section 4 supervised learning pipeline completed.")


if __name__ == "__main__":
    main()
