
import os
import warnings
from datetime import datetime

import pandas as pd
import numpy as np

from config import *
from utils import (
    Logger, reduce_memory_usage, check_class_balance, check_missing_values,
    get_continuous_numeric_cols, iqr_outlier_masks, export_outliers,
    describe_numeric, describe_categorical
)
from preprocessing import preprocess_data
from modeling import run_experiment
from evaluation import evaluate_all_models, save_evaluation_results, aggregate_master_results
from visualization import (
    plot_class_distribution, plot_missing_values, plot_correlation_heatmap,
    plot_numeric_distributions, plot_boxplots, plot_skewness, plot_outlier_counts,
    plot_roc_curves, plot_pr_curves, plot_confusion_matrix, plot_metric_comparison
)

warnings.filterwarnings("ignore")


def get_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_dataset(dataset_name: str, logger: Logger) -> pd.DataFrame:
    cfg = get_dataset_config(dataset_name)
    path = os.path.join(BASE_DIR, cfg["file"])
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")
    df = pd.read_csv(path)
    logger.success(f"Loaded {dataset_name}: {df.shape[0]} rows, {df.shape[1]} cols")
    return df


def clean_dataset(df: pd.DataFrame, dataset_name: str, logger: Logger) -> pd.DataFrame:
    # mirror Set B (already close to Set A cleaning)
    if dataset_name == "telco":
        if "customerID" in df.columns:
            df = df.drop(columns=["customerID"])
            logger.info("Dropped customerID")
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
            logger.info("Converted TotalCharges to numeric")

    before = len(df)
    df = df.drop_duplicates()
    if len(df) != before:
        logger.info(f"Dropped {before-len(df)} duplicates")

    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for c in cat_cols:
        df[c] = df[c].astype(str).str.strip().str.lower()

    if dataset_name == "telco":
        repl = {"no internet service": "no", "no phone service": "no"}
        for c in cat_cols:
            df[c] = df[c].replace(repl)

    df = reduce_memory_usage(df, logger=logger)
    return df


def ensure_dirs(*paths: str):
    for p in paths:
        os.makedirs(p, exist_ok=True)


def run_eda(df: pd.DataFrame, dataset_name: str, target_col: str, out_root: str, logger: Logger):
    eda_dir = os.path.join(out_root, "eda")
    tables_dir = os.path.join(eda_dir, "tables")
    plots_dir = os.path.join(eda_dir, "plots")
    ensure_dirs(eda_dir, tables_dir, plots_dir)

    logger.section("EDA (Merged: Set A spirit)")
    # Basic checks
    _ = check_missing_values(df, logger=logger)
    _ = check_class_balance(df[target_col], logger=logger)

    if SAVE_EDA_TABLES:
        # report-friendly tables
        num_cols = df.drop(columns=[target_col]).select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.drop(columns=[target_col]).select_dtypes(include=["object", "category"]).columns.tolist()

        describe_numeric(df, num_cols).to_csv(os.path.join(tables_dir, "numeric_describe.csv"))
        cat_desc = describe_categorical(df, cat_cols, top_k=10)
        # write one file per categorical (keeps it readable)
        for c, tbl in cat_desc.items():
            safe = c.replace("/", "_")
            tbl.to_csv(os.path.join(tables_dir, f"categorical_{safe}_top10.csv"))

    if SAVE_EDA_PLOTS:
        plot_class_distribution(df[target_col], f"{dataset_name}: class distribution", os.path.join(plots_dir, "class_dist.png"), logger=logger)
        plot_missing_values(df, f"{dataset_name}: missing values", os.path.join(plots_dir, "missing.png"), logger=logger)
        plot_correlation_heatmap(df, f"{dataset_name}: correlation heatmap", os.path.join(plots_dir, "corr.png"), logger=logger)

        # Set A style
        cont_cols = get_continuous_numeric_cols(df, target_col=target_col, min_unique=OUTLIER_CONFIG.get("min_unique_continuous", 10))
        if cont_cols:
            plot_numeric_distributions(df, cont_cols, plots_dir, logger=logger)
            plot_boxplots(df, cont_cols, plots_dir, logger=logger)
            plot_skewness(df, cont_cols, plots_dir, logger=logger)

            mild_mask, extreme_mask, summary = iqr_outlier_masks(
                df, cont_cols,
                k_mild=OUTLIER_CONFIG.get("iqr_k_mild", 1.5),
                k_extreme=OUTLIER_CONFIG.get("iqr_k_extreme", 3.0),
            )
            summary.to_csv(os.path.join(tables_dir, "outlier_summary_iqr.csv"), index=False)
            plot_outlier_counts(summary, plots_dir, logger=logger)

            if EXPORT_OUTLIERS:
                export_outliers(df, mild_mask, extreme_mask, tables_dir, prefix=f"{dataset_name}")


def main():
    logger = Logger()
    logger.section("MERGED SET RUNNER (Set B backbone + Set A spirit)")

    dataset_name = DATASET
    cfg = get_dataset_config(dataset_name)
    target_col = cfg["target"]

    ts = get_ts()
    out_root = os.path.join(RESULTS_DIR, f"{dataset_name}_{ts}")
    ensure_dirs(out_root)

    # Load + clean
    df = load_dataset(dataset_name, logger)
    df = clean_dataset(df, dataset_name, logger)

    # EDA
    if RUN_EDA:
        run_eda(df, dataset_name, target_col, out_root, logger)

    # Experiments
    master_rows = []
    for feature_config in FEATURE_CONFIGS:
        logger.section(f"FEATURE CONFIG: {feature_config}")
        feat_dir = os.path.join(out_root, feature_config)
        ensure_dirs(feat_dir)

        prep = preprocess_data(
            df, target_col,
            feature_config=feature_config,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            logger=logger,
            export_dir=os.path.join(feat_dir, "eda_exports"),
            dataset_name=dataset_name
        )

        if EXPORT_PROCESSED_DATASETS:
            # Save processed arrays (npz) + labels (csv) for inspection
            np.savez_compressed(
                os.path.join(feat_dir, f"{dataset_name}_{feature_config}_processed.npz"),
                X_train=prep["X_train"], X_test=prep["X_test"]
            )
            pd.Series(prep["y_train"]).to_csv(os.path.join(feat_dir, f"{dataset_name}_{feature_config}_y_train.csv"), index=False)
            pd.Series(prep["y_test"]).to_csv(os.path.join(feat_dir, f"{dataset_name}_{feature_config}_y_test.csv"), index=False)

        for sampling_strategy in SAMPLING_STRATEGIES:
            logger.section(f"SAMPLING: {sampling_strategy}")
            run_dir = os.path.join(feat_dir, sampling_strategy)
            ensure_dirs(run_dir)

            exp = run_experiment(
                prep["X_train"], prep["y_train"],
                prep["X_test"], prep["y_test"],
                dataset_name=dataset_name,
                sampling_strategy=sampling_strategy,
                feature_config=feature_config,
                model_names=MODELS_TO_TRAIN,
                tune_hyperparams=TUNE_HYPERPARAMS,
                output_dir=run_dir,
                random_state=RANDOM_STATE,
                logger=logger
            )

            trained_models = exp["models"]

            results_df = evaluate_all_models(
                trained_models,
                prep["X_test"],
                prep["y_test"],
                tune_threshold=THRESHOLD_TUNING.get('enabled', True),
                threshold_method=THRESHOLD_TUNING.get('method', 'f1_max'),
                feature_names=prep.get("feature_names"),
                logger=logger
            )

            # Save eval table + per-run plots
            save_evaluation_results(
                results_df,
                dataset_name=dataset_name,
                sampling_strategy=sampling_strategy,
                feature_config=feature_config,
                output_dir=run_dir,
                logger=logger
            )

            # Curves + confusion
            try:
                plot_roc_curves(results_df, prep["y_test"], f"ROC: {dataset_name}/{feature_config}/{sampling_strategy}", os.path.join(run_dir, "roc.png"), logger=logger)
                plot_pr_curves(results_df, prep["y_test"], f"PR: {dataset_name}/{feature_config}/{sampling_strategy}", os.path.join(run_dir, "pr.png"), logger=logger)
                # confusion matrices per model (best threshold in results_df)
                for _, row in results_df.iterrows():
                    cm = row.get("confusion_matrix", None)
                    if cm is None:
                        continue
                    plot_confusion_matrix(cm, f"Confusion: {row['model']}", os.path.join(run_dir, f"cm_{row['model']}.png"), logger=logger)
            except Exception as e:
                logger.warning(f"Plotting error: {str(e)}")

            # Master rows
            for _, r in results_df.iterrows():
                master_rows.append({
                    "dataset": dataset_name,
                    "sampling_strategy": sampling_strategy,
                    "feature_config": feature_config,
                    "model": r.get("model"),
                    "precision": r.get("precision"),
                    "recall": r.get("recall"),
                    "f1": r.get("f1"),
                    "roc_auc": r.get("roc_auc"),
                    "pr_auc": r.get("pr_auc"),
                    "threshold": r.get("threshold"),
                })

    master = pd.DataFrame(master_rows)
    master_path = os.path.join(out_root, "master_results.csv")
    master.to_csv(master_path, index=False)
    logger.success(f"MASTER RESULTS saved to: {master_path}")

    # Optional: metric comparison plot across all runs
    try:
        # choose one metric view (f1)
        best = master.sort_values("f1", ascending=False).head(20)
        plot_metric_comparison(best.rename(columns={"model":"model","f1":"f1"}), metric="f1",
                               title="Top 20 model configs by F1",
                               save_path=os.path.join(out_root, "top20_f1.png"),
                               logger=logger)
    except Exception as e:
        logger.warning(f"Master comparison plot skipped: {str(e)}")


if __name__ == "__main__":
    main()