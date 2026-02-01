**Machine Learning Experimentation Framework**

A comprehensive, modular Python framework for conducting machine learning experiments with multiple sampling strategies, feature engineering approaches, and model comparisons. Designed for research and academic projects requiring systematic model benchmarking and performance analysis.

**Table of Contents**

- Overview
- Project Structure
- Key Features
- Installation
- Configuration
- Usage
- Module Descriptions
- Experiment Workflow
- Output Structure
- Extending the Framework
- Troubleshooting

**Overview**

This framework provides a complete pipeline for machine learning experimentation, from data loading and exploratory data analysis (EDA) through preprocessing, model training, evaluation, and visualization. It was developed to support systematic comparison of:

- Multiple sampling strategies for class imbalance (SMOTE, Tomek Links, undersampling, hybrid methods)
- Feature engineering configurations (baseline, engineered features, RFE selection, PCA reduction)
- Various classification models (Logistic Regression, Random Forest, Gradient Boosting, SVM, KNN, Decision Tree, Naive Bayes)

**Primary Use Case:** Benchmarking model performance across different data preprocessing and sampling configurations to identify optimal approaches for imbalanced classification problems.

**Project Structure**

project_root/  
├── [config.py](http://config.py) # Central configuration file  
├── [utils.py](http://utils.py) # Utility functions (logging, I/O, validation)  
├── [preprocessing.py](http://preprocessing.py) # Data preprocessing and feature engineering  
├── [modeling.py](http://modeling.py) # Model training and sampling strategies  
├── [evaluation.py](http://evaluation.py) # Model evaluation and metrics  
├── [visualization.py](http://visualization.py) # Plotting and visualization functions  
├── run_experiments_modular.py # Main experiment runner script  
├── online_shoppers_intention.csv # Dataset file (example)  
├── WA_Fn-UseC_-Telco-Customer-Churn.csv # Dataset file (example)  
└── results/ # Output directory (auto-created)  
└── \[dataset\]\_\[timestamp\]/  
├── eda/  
├── \[feature_config\]/  
│ └── \[sampling_strategy\]/  
└── master_results.csv

**Key Features**

**Modular Design**

- **Separation of Concerns:** Each module handles a specific aspect (preprocessing, modeling, evaluation)
- **Reusable Components:** Functions can be imported and used independently
- **Easy Extension:** Add new models, sampling strategies, or feature configurations by updating [config.py](http://config.py)

**Comprehensive EDA**

- Class distribution analysis
- Missing value detection and reporting
- Outlier detection using IQR method
- Correlation heatmaps
- Numeric feature distributions (histograms, boxplots)
- Skewness analysis
- Categorical feature frequency tables

**Flexible Preprocessing**

- Multiple feature configurations (F0_base, F1_engineered, F2_reduced_rfe, F3_reduced_pca)
- Custom transformers for log transformation, binning, and feature interactions
- Automatic identification of skewed features and interaction candidates
- KNN or simple imputation for missing values
- StandardScaler for numeric features
- OneHotEncoder for categorical features

**Robust Sampling Strategies**

- **A0_baseline:** No resampling
- **A1_smote:** SMOTE oversampling
- **A2_tomek_smote:** Tomek Links undersampling + SMOTE oversampling
- **A3_undersample:** Random undersampling
- **A4_smoteenn:** SMOTE + Edited Nearest Neighbors (hybrid)

**Model Training and Tuning**

- Automated hyperparameter tuning with GridSearchCV
- Stratified K-fold cross-validation
- Multiple models supported out-of-the-box
- Training time tracking
- Best parameter logging

**Comprehensive Evaluation**

- Accuracy, Precision, Recall, F1-Score
- ROC-AUC and PR-AUC
- Confusion matrices
- Threshold tuning (F1-max or recall-target methods)
- Classification reports
- Per-model and aggregated master results

**Rich Visualizations**

- ROC curves (all models)
- Precision-Recall curves (all models)
- Confusion matrices (per model)
- Feature importance plots (tree-based models)
- Metric comparison bar charts
- Multi-metric heatmaps

**Logging and Tracking**

- Colored console output with timestamps
- File-based logging
- Progress tracking and elapsed time reporting
- Structured section/subsection formatting

**Installation**

**Prerequisites**

- Python 3.8 or higher
- pip package manager

**Required Packages**

pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn joblib scipy

**Specific package list:**

- pandas
- numpy
- scikit-learn
- imbalanced-learn
- matplotlib
- seaborn
- joblib
- scipy

**Installation Steps**

- Clone or download the repository to your local machine
- Navigate to the project directory
- Install dependencies:  
    pip install -r requirements.txt
- Place your dataset files in the project root directory
- Verify installation by running:  
    python [config.py](http://config.py)

**Configuration**

All experiment parameters are centralized in config.py. Key configuration sections:

**Dataset Selection**

DATASET = "online" # Options: "online", "telco"

Define dataset-specific settings in DATASET_CONFIG dictionary (file path, target column, positive class).

**Experiment Parameters**

- **RANDOM_STATE:** Random seed for reproducibility (default: 42)
- **SAMPLING_STRATEGIES:** List of sampling methods to test
- **FEATURE_CONFIGS:** List of feature engineering approaches
- **MODELS_TO_TRAIN:** List of models to train and evaluate
- **TEST_SIZE:** Train-test split ratio (default: 0.2)
- **TUNE_HYPERPARAMS:** Enable/disable hyperparameter tuning (default: True)

**Feature Engineering**

- **skewness_threshold:** Threshold for applying log transformation (default: 1.0)
- **n_bins:** Number of bins for quantile binning (default: 5)
- **rfe.n_features_to_select:** Features to keep after RFE (default: 20)
- **pca.n_components:** Variance to retain with PCA (default: 0.95)

**Model Hyperparameters**

Each model has a parameter grid defined in PARAM_GRIDS. Modify these to adjust search space for GridSearchCV.

**Threshold Tuning**

THRESHOLD_TUNING = {  
'enabled': True,  
'method': 'f1_max', # Options: 'f1_max', 'recall_target'  
'recall_target': 0.80,  
}

**EDA and Output Toggles**

- **RUN_EDA:** Enable exploratory data analysis (default: True)
- **SAVE_EDA_TABLES:** Save numeric/categorical summaries (default: True)
- **SAVE_EDA_PLOTS:** Save distribution plots (default: True)
- **EXPORT_OUTLIERS:** Export outlier data to CSV (default: True)
- **EXPORT_PROCESSED_DATASETS:** Save preprocessed arrays (default: True)

**Usage**

**Basic Execution**

- Configure config.py with desired dataset and parameters
- Run the main experiment script:  
    python run_experiments_modular.py
- Monitor console output for progress and results
- Find outputs in results/\[dataset\]\_\[timestamp\]/ directory

**Execution Flow**

The main script (run_experiments_modular.py) follows this workflow:

- Load dataset specified in [config.py](http://config.py)
- Clean dataset (remove duplicates, handle missing values, normalize text)
- Run EDA if enabled (plots, tables, outlier detection)
- For each feature configuration:
  - Preprocess data (imputation, scaling, encoding, feature engineering)
  - For each sampling strategy:
    - Apply sampling to training data
    - Train all specified models (with hyperparameter tuning if enabled)
    - Evaluate on test set
    - Generate visualizations (ROC, PR, confusion matrices)
    - Save results to CSV
- Aggregate all results into master CSV
- Generate comparison plots

**Example Workflow**

**Configure experiment in** [**config.py**](http://config.py)

DATASET = "online"  
SAMPLING_STRATEGIES = \["A0_baseline", "A1_smote"\]  
FEATURE_CONFIGS = \["F0_base", "F1_engineered"\]  
MODELS_TO_TRAIN = \["LogisticRegression", "RandomForest"\]

**Run experiments**

python run_experiments_modular.py

**Results will be saved to:**

**results/online_\[timestamp\]/master_results.csv**

**Module Descriptions**

[**config.py**](http://config.py)

**Purpose:** Centralized configuration for all experiment parameters

**Key components:**

- Dataset configurations
- Sampling strategy definitions
- Feature engineering settings
- Model hyperparameter grids
- Output directories and file naming conventions
- Visualization settings

**Usage:** Import and access settings  
from config import DATASET, RANDOM_STATE, get_dataset_config

[**utils.py**](http://utils.py)

**Purpose:** Utility functions for logging, I/O, validation, and timing

**Key classes and functions:**

- **Logger:** Custom logger with colored console output and file logging
- **Timer:** Context manager for timing operations
- **File I/O:** save_pickle, load_pickle, save_model, load_model
- **Data validation:** validate_dataframe, check_class_balance, check_missing_values
- **Memory optimization:** reduce_memory_usage

**Usage:**  
from utils import Logger, Timer

logger = Logger("experiment")  
[logger.info](http://logger.info)("Starting experiment...")

with Timer("Data loading", logger):  
df = pd.read_csv("data.csv")

[**preprocessing.py**](http://preprocessing.py)

**Purpose:** Data preprocessing and feature engineering

**Key components:**

- **Custom transformers:** BinningTransformer, LogTransformer, InteractionTransformer
- **Feature identification:** identify_skewed_features, identify_interaction_pairs, identify_binning_candidates
- **Preprocessor builders:** build_base_preprocessor, build_engineered_preprocessor, build_rfe_preprocessor, build_pca_preprocessor
- **Main function:** preprocess_data (handles complete preprocessing pipeline)

**Feature configurations:**

- **F0_base:** Raw features with standard scaling and encoding
- **F1_engineered:** Adds log transforms, binning, and feature interactions
- **F2_reduced_rfe:** F1 + Recursive Feature Elimination
- **F3_reduced_pca:** F1 + Principal Component Analysis

**Usage:**  
from preprocessing import preprocess_data

preprocessed = preprocess_data(  
df, target_col='Revenue',  
feature_config='F1_engineered',  
test_size=0.2, random_state=42  
)

[**modeling.py**](http://modeling.py)

**Purpose:** Model training and sampling strategy application

**Key components:**

- **Sampling:** get_sampler, apply_sampling
- **Model setup:** get_base_models, get_param_grids
- **Training:** train_model, train_all_models
- **Experiment runner:** run_experiment (complete modeling pipeline)

**Supported models:**

- Logistic Regression
- Random Forest
- Gradient Boosting
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Decision Tree
- Naive Bayes

**Usage:**  
from modeling import run_experiment

results = run_experiment(  
X_train, y_train, X_test, y_test,  
dataset_name='online',  
sampling_strategy='A1_smote',  
feature_config='F1_engineered',  
model_names=\['RandomForest', 'LogisticRegression'\]  
)

[**evaluation.py**](http://evaluation.py)

**Purpose:** Model evaluation with comprehensive metrics

**Key components:**

- **Score extraction:** get_prediction_scores
- **Threshold tuning:** find_optimal_threshold
- **Evaluation:** evaluate_model, evaluate_all_models
- **Results export:** save_evaluation_results, aggregate_master_results

**Metrics computed:**

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- PR-AUC (Average Precision)
- Confusion matrix (TN, FP, FN, TP)
- Classification report

**Usage:**  
from evaluation import evaluate_all_models

results_df = evaluate_all_models(  
trained_models, X_test, y_test,  
tune_threshold=True,  
threshold_method='f1_max'  
)

[**visualization.py**](http://visualization.py)

**Purpose:** Generate plots and visualizations for analysis

**Key functions:**

- **Performance curves:** plot_roc_curves, plot_pr_curves
- **Confusion matrices:** plot_confusion_matrices, plot_confusion_matrix
- **Feature analysis:** plot_feature_importance
- **Comparisons:** plot_metric_comparison, plot_metrics_heatmap
- **EDA plots:** plot_class_distribution, plot_missing_values, plot_correlation_heatmap, plot_numeric_distributions, plot_boxplots, plot_skewness, plot_outlier_counts

**Usage:**  
from visualization import plot_roc_curves, plot_metric_comparison

plot_roc_curves(results_df, y_test, "ROC Curves", "roc.png")  
plot_metric_comparison(results_df, 'f1', "F1 Score Comparison", "f1_comparison.png")

**run_experiments_modular.py**

**Purpose:** Main experiment orchestration script

**Workflow:**

- Initialize logger and create output directories
- Load and clean dataset
- Run EDA (if enabled)
- Iterate through feature configurations
- Iterate through sampling strategies
- Train models and evaluate
- Generate visualizations
- Aggregate results into master CSV

**Usage:**  
python run_experiments_modular.py

**Experiment Workflow**

**Step-by-Step Process**

**1\. Data Loading and Cleaning**

- Load dataset from CSV
- Drop unnecessary columns (e.g., customer IDs)
- Convert data types
- Remove duplicates
- Normalize categorical values (lowercase, strip whitespace)
- Reduce memory usage with downcasting

**2\. Exploratory Data Analysis (EDA)**

- Generate class distribution plots
- Create missing value summary tables and plots
- Compute correlation heatmaps
- Generate histograms and boxplots for numeric features
- Calculate skewness statistics
- Detect outliers using IQR method
- Export outlier data if enabled

**3\. Preprocessing**

- Separate features and target
- Encode target variable (0/1 binary)
- Split into train/test sets (stratified)
- Identify numeric and categorical features
- Build appropriate preprocessor based on feature configuration
- Fit preprocessor on training data only
- Transform both training and test sets
- Store feature names and preprocessor for later use

**4\. Sampling**

- Apply sampling strategy to training data only
- Log class distribution before and after sampling
- Handle multiple sampling steps (e.g., Tomek + SMOTE)

**5\. Model Training**

- Initialize base models with default parameters
- If hyperparameter tuning enabled:
  - Set up GridSearchCV with stratified K-fold
  - Search parameter grid
  - Select best estimator based on F1 score
  - Log best parameters and cross-validation scores
- If tuning disabled, train with default parameters
- Track training time for each model
- Save trained models to disk

**6\. Evaluation**

- Extract prediction scores (probabilities or decision functions)
- Tune classification threshold if enabled
- Compute all evaluation metrics
- Generate confusion matrices
- Create classification reports
- Save evaluation results to CSV

**7\. Visualization**

- Plot ROC curves (all models on one plot)
- Plot Precision-Recall curves (all models on one plot)
- Generate individual confusion matrix plots
- Create metric comparison bar charts
- Generate multi-metric heatmaps

**8\. Results Aggregation**

- Collect results from all experiment runs
- Combine into master results DataFrame
- Save master_results.csv with complete experiment metadata
- Generate top-performing model comparison plots

**Output Structure**

results/  
└── \[dataset\]  
<br/><br/><br/><br/><br/><br/><br/>_\[timestamp\]/├── master_results.csv # Aggregated results from all experiments├── top20_f1.png # Top 20 model configurations by F1├── eda/ # Exploratory data analysis outputs│ ├── plots/│ │ ├── class_dist.png│ │ ├── missing.png│ │ ├── corr.png│ │ ├── hist_  
_.png # Histograms per numeric feature│ │ ├── box__.png # Boxplots per numeric feature  
│ │ ├── skewness.png  
│ │ └── outlier_counts.png  
│ └── tables/  
│ ├── numeric_describe.csv  
│ ├── categorical_  
<br/>_\_top10.csv│ ├── outlier_summary_iqr.csv│ └── \[dataset\]outliers_.csv  
├── F0_base/ # Feature configuration directory  
│ ├── \[dataset\]\_F0_base_processed.npz # Processed arrays  
│ ├── \[dataset\]\_F0_base_y_train.csv  
│ ├── \[dataset\]  
<br/><br/><br/><br/>_F0_base_y_test.csv│ ├── A0_baseline/ # Sampling strategy directory│ │ ├── model_comparison.csv # Results for this run│ │ ├── roc.png # ROC curves│ │ ├── pr.png # Precision-Recall curves│ │ ├── cm_\*.png # Confusion matrices per model  
│ │ ├── \*\_model.pkl # Saved model files  
│ │ ├── \*\_classification_report.txt  
│ │ └── training_info.csv  
│ └── A1_smote/  
│ └── \[same structure as A0_baseline\]  
├── F1_engineered/  
│ └── \[same structure as F0_base\]  
├── F2_reduced_rfe/  
│ └── \[same structure as F0_base\]  
└── F3_reduced_pca/  
└── \[same structure as F0_base\]

**Key Output Files**

| File | Description |
| --- | --- |
| master_results.csv | All experiment results aggregated |
| model_comparison.csv | Results for specific experiment run |
| \*\_model.pkl | Trained model objects (joblib format) |
| \*\_classification_report.txt | Detailed per-class metrics |
| roc.png | ROC curves for all models |
| pr.png | Precision-Recall curves |
| cm_\*.png | Confusion matrix per model |
| training_info.csv | Hyperparameters and training metadata |

Table 1: Key output files and their descriptions

**Extending the Framework**

**Adding a New Model**

- Define model in modeling.py:  
    def get_base_models(random_state=42):  
    models = {  
    \# ... existing models ...  
    'YourModel': YourModelClassifier(random_state=random_state)  
    }  
    return models
- Add hyperparameter grid in modeling.py:  
    def get_param_grids():  
    param_grids = {  
    \# ... existing grids ...  
    'YourModel': {  
    'param1': \[value1, value2\],  
    'param2': \[value3, value4\]  
    }  
    }  
    return param_grids
- Update config.py:  
    MODELS_TO_TRAIN = \[  
    'LogisticRegression',  
    'RandomForest',  
    'YourModel' # Add here  
    \]

**Adding a New Sampling Strategy**

- Define sampler in modeling.py:  
    def get_sampler(strategy, random_state=42):  
    samplers = {  
    \# ... existing samplers ...  
    'A5_your_strategy': YourSampler(random_state=random_state)  
    }  
    return samplers.get(strategy)
- Update config.py:  
    SAMPLING_STRATEGIES = \[  
    'A0_baseline',  
    'A1_smote',  
    'A5_your_strategy' # Add here  
    \]

**Adding a New Feature Configuration**

- Define builder function in preprocessing.py:  
    def build_your_preprocessor(...):

**Your preprocessing logic**

return preprocessor

- Update preprocess_data function to handle new config
- Update config.py:  
    FEATURE_CONFIGS = \[  
    'F0_base',  
    'F1_engineered',  
    'F4_your_config' # Add here  
    \]

**Adding a New Dataset**

- Update config.py:  
    DATASET_CONFIG = {

**... existing datasets ...**

'your_dataset': {  
'file': 'your_data.csv',  
'target': 'target_column_name',  
'positive_class': 1  
}  
}

- Add dataset-specific cleaning logic to clean_dataset function in run_experiments_modular.py
- Place dataset file in project root directory

**Troubleshooting**

**Common Issues and Solutions**

**Issue: "Dataset file not found"**

**Solution:** Ensure dataset CSV file is in project root and filename matches config.py setting

**Issue: "No module named 'imbalanced_learn'"**

**Solution:** Install imbalanced-learn package  
pip install imbalanced-learn

**Issue: Memory errors during large experiments**

**Solutions:**

- Reduce number of models trained simultaneously
- Disable hyperparameter tuning temporarily (TUNE_HYPERPARAMS = False)
- Reduce GridSearchCV parameter grid size
- Process one feature configuration at a time
- Use reduce_memory_usage utility on input DataFrame

**Issue: GridSearchCV taking too long**

**Solutions:**

- Reduce parameter grid size in [config.py](http://config.py)
- Reduce CV_CONFIG\['n_splits'\] (default: 5)
- Set GRID_SEARCH_CONFIG\['n_jobs'\] = -1 to use all CPU cores
- Disable tuning for certain models

**Issue: Plots not displaying correctly**

**Solution:** Check that matplotlib backend is set correctly. The framework uses 'Agg' backend for non-interactive saving. If you want interactive plots, modify visualization.py:

**matplotlib.use('Agg') # Comment this out for interactive plots**

**Issue: Sklearn warnings about binary classification**

**Solution:** The preprocessing module automatically encodes target to 0/1. Check that your target column contains exactly two unique values.

**Issue: Feature names mismatch after preprocessing**

**Solution:** Feature names change after one-hot encoding and feature engineering. Use the returned feature_names from preprocess_data for correct mapping.

**Best Practices**

- **Start small:** Test with one feature configuration and one sampling strategy before running full experiments
- **Monitor logs:** Check console output and log files for warnings and errors
- **Verify results:** Always inspect master_results.csv to ensure experiments completed successfully
- **Version control:** Track changes to [config.py](http://config.py) and custom modifications
- **Document experiments:** Use descriptive comments in [config.py](http://config.py) for non-standard settings
- **Resource management:** Monitor CPU and memory usage during large experiments
- **Reproducibility:** Always set and document RANDOM_STATE
- **Backup results:** Copy results directory before making major changes

**Performance Tips**

- Hyperparameter tuning is the most time-consuming step. Consider:
  - Using smaller parameter grids initially
  - Reducing CV folds (e.g., 3 instead of 5)
  - Disabling tuning for baseline comparisons
- Feature engineering (F1) is slower than baseline (F0). Test with F0 first.
- RFE (F2) requires training a model multiple times. Expect longer preprocessing.
- Use EXPORT_PROCESSED_DATASETS = False to skip saving large numpy arrays
- Reduce EDA plots if disk space is limited: SAVE_EDA_PLOTS = False

**License**

This framework is provided as-is for academic and research purposes. Modify and extend as needed for your specific use case.

**Citation**

If you use this framework in your research, please cite appropriately according to your institution's guidelines.

**Contact**

For questions, issues, or contributions related to this framework, refer to your project documentation or contact your project supervisor.