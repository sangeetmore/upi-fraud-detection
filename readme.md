# Credit Card Fraud Detection Model

![Fraud Detection Banner](https://img.shields.io/badge/ML-Fraud%20Detection-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.0-orange?style=flat&logo=scikit-learn&logoColor=white)
![RandomForest](https://img.shields.io/badge/Model-RandomForest-green?style=flat)

A robust machine learning system for detecting fraudulent credit card transactions while minimizing false positives. This model implements intelligent feature engineering, anomaly detection, and optimized classification thresholds.

## 📋 Table of Contents

- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Data Requirements](#data-requirements)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Feature Engineering](#feature-engineering)
- [Customization](#customization)
- [Contributing](#contributing)

## ✨ Key Features

- **Robust Fraud Detection:** Optimized for precision to minimize false positives
- **Intelligent Feature Engineering:** Extracts meaningful patterns from transaction data
- **Anomaly Detection:** Identifies and removes potentially mislabeled training data
- **Optimized Thresholds:** Fine-tunes decision thresholds for real-world application
- **Memory Efficiency:** Handles large transaction datasets with intelligent sampling
- **Visualization:** Generates insightful plots and metrics for model evaluation

## 📂 Project Structure

```
fraud-detection-v2/
├── data/                             # Data directory
│   ├── card_transaction.v1.csv       # Your transaction data goes here
│   └── mcc_codes.json                # Merchant Category Code descriptions
├── fraud_analysis_plots/             # Generated visualizations
├── fraud_detection_results/          # Model results and metrics
├── venv/                             # Python virtual environment
├── data_analysis_script.py           # Data preprocessing and analysis
├── fraud_detection_complete.py       # Main fraud detection model
├── requirements.txt                  # Project dependencies
├── smart_split_data.csv              # Generated train/test split data
└── README.md                         # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- pip (Python package installer)

### Setup Instructions

1. **Clone this repository**

2. **Create and activate a virtual environment**

   ```bash
   # For macOS/Linux
   python -m venv venv
   source venv/bin/activate

   # For Windows
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Upgrade pip and install dependencies**

   ```bash
   pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt
   ```

   > **Note for macOS users**: If you encounter issues with XGBoost, you need to install OpenMP:
   >
   > ```bash
   > brew install libomp
   > pip uninstall -y xgboost
   > pip install xgboost
   > ```

4. **Verify installation**

   ```bash
   python -c "import pandas, sklearn, matplotlib; print('Installation successful!')"
   ```

## 📊 Data Requirements

### Transaction Data Format

Place your transaction dataset as `card_transaction.v1.csv` in the `data/` directory. The model expects the following key columns:

| Column                    | Description            | Type       |
| ------------------------- | ---------------------- | ---------- |
| `User`                    | Unique user identifier | String/Int |
| `Amount`                  | Transaction amount     | Numeric    |
| `Time`                    | Transaction timestamp  | Datetime   |
| `MCC`                     | Merchant Category Code | Int        |
| `Merchant Name`           | Name of merchant       | String     |
| `Is Fraud?` or `is_fraud` | Fraud indicator (1/0)  | Int        |

> **Note**: The model handles various column naming formats, but exact matches are preferred for optimal performance.

### MCC Codes

The repository includes a standard `mcc_codes.json` file in the `data/` directory with Merchant Category Code descriptions. No action is required unless you have custom MCC codes.

## 🔍 Usage

### Data Analysis and Preparation

Run the data analysis script to explore your data and create an intelligent train/test split:

```bash
python data_analysis_script.py
```

This will:

- Load and analyze your transaction data
- Generate exploratory visualizations
- Create a train/test split respecting user boundaries
- Save the processed data as `smart_split_data.csv`

### Fraud Detection Model

Run the main fraud detection script:

```bash
python fraud_detection_complete.py
```

This will:

- Load the split dataset
- Perform feature engineering
- Train a RandomForest model
- Evaluate model performance
- Save results to `fraud_detection_results/`

### Customizing the Run

You can modify parameters in the scripts or import them as modules for custom usage:

```python
from fraud_detection_complete import run_fraud_detection

# Run with custom parameters
model, metrics = run_fraud_detection(
    data_file='path/to/your/data.csv',
    output_dir='custom_results',
    focus_on_precision=True,      # Prioritize reducing false positives
    sample_size=500000,           # Use a subset for training
    clean_training_data=True      # Remove potential mislabeling
)

print(f"Model AUC: {metrics['roc_auc']:.4f}")
```

## 📈 Model Performance

The model is evaluated using multiple metrics, with a focus on minimizing false positives:

- **ROC AUC Score**: Measures overall discrimination ability
- **Precision**: Percentage of true fraud cases among predicted fraud
- **Recall**: Percentage of fraud cases correctly identified
- **F1 Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Visualization of prediction errors

### Interpreting Results

After running the model, check these key files in the `fraud_detection_results/` directory:

- `metrics.json`: Contains all performance metrics
- `confusion_matrix.png`: Visualization of true vs. predicted fraud
- `roc_curve.png`: Model's performance across thresholds
- `feature_importance.png`: Key factors in fraud detection

## 🛠️ Feature Engineering

The model automatically engineers various features from raw transaction data:

- **Time-based features**: Hour of day, day of week, weekend flags
- **Amount-based features**: Log transformation, z-scores, outlier detection
- **User behavior**: Transaction frequency, average amounts, fraud history
- **Merchant insights**: Category risk scores, frequency analysis
- **Combination patterns**: User-merchant relationship analysis

## ⚙️ Customization

### Parameter Tuning

Key parameters that can be adjusted in `fraud_detection_complete.py`:

- `focus_on_precision`: Set to `True` to prioritize fewer false positives
- `sample_size`: Adjust training sample size for memory/performance balance
- `clean_training_data`: Enable/disable anomaly detection for training data
- `RandomForestClassifier` parameters: Change model complexity via `n_estimators`, `max_depth`, etc.

### Adding Custom Features

To add custom features, modify the feature engineering section in `fraud_detection_complete.py`:

```python
# Add your custom feature
train_df['my_feature'] = train_df['Amount'] / train_df['user_avg_amount']
test_df['my_feature'] = test_df['Amount'] / test_df['user_avg_amount']

# Add to feature list
features.append('my_feature')
```
