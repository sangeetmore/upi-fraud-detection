import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc, roc_curve, roc_auc_score
from xgboost import XGBClassifier
import xgboost as xgb
import json
from datetime import datetime
import os
import warnings

# Import the analysis and splitting function
from data_analysis_script import analyze_and_split_data

# Suppress warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def run_fraud_detection(data_file='smart_split_data.csv', 
                        output_dir='fraud_detection_results',
                        use_existing_split=True):
    """
    Run the complete fraud detection model pipeline
    
    Parameters:
    -----------
    data_file : str
        Path to the data file (with train/test split)
    output_dir : str
        Directory to save results and plots
    use_existing_split : bool
        Whether to use the existing train/test split in the data
    """
    print(f"{'='*50}\nSTART FRAUD DETECTION MODEL\n{'='*50}")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 1. Load data
    print(f"\n[1] Loading data from {data_file}...")
    df = pd.read_csv(data_file)
    print(f"Loaded {len(df):,} transactions")
    
    # Check for fraud column and rename if necessary
    if 'Is Fraud?' in df.columns:
        print("Renaming 'Is Fraud?' column to 'Is_Fraud'")
        df.rename(columns={'Is Fraud?': 'Is_Fraud'}, inplace=True)
    
    # Check for split column if using existing split
    if use_existing_split:
        # Look for split column
        split_cols = [col for col in df.columns if col.lower() in 
                      ['split', 'train_test_flag', 'is_train', 'dataset']]
        
        if split_cols:
            split_col = split_cols[0]
            print(f"Using existing split column: '{split_col}'")
            
            # Identify values for train and test
            split_values = df[split_col].unique()
            print(f"Split values: {split_values}")
            
            if len(split_values) == 2:
                # Determine which value corresponds to train
                # Common patterns: train/test, 1/0, True/False
                train_indicators = ['train', '1', 1, 'tr', True, 'true']
                
                for val in split_values:
                    if str(val).lower() in [str(ind).lower() for ind in train_indicators]:
                        train_value = val
                        break
                else:
                    # If no match found, default to first value
                    train_value = split_values[0]
                    print(f"Could not determine train value, defaulting to: {train_value}")
                
                # Split the data
                train_df = df[df[split_col] == train_value]
                test_df = df[df[split_col] != train_value]
                
                print(f"Train set: {len(train_df):,} transactions")
                print(f"Test set: {len(test_df):,} transactions")
            else:
                print(f"Found {len(split_values)} split values instead of 2, creating new split")
                use_existing_split = False
        else:
            print("No split column found, creating new split")
            use_existing_split = False
    
    # Create a new split if needed
    if not use_existing_split:
        print("\nCreating new user-based train/test split")
        # Sort by time if available
        if 'Time' in df.columns:
            df['Time'] = pd.to_datetime(df['Time'])
            df = df.sort_values(['User', 'Time'])
        
        # Split users, not transactions
        users = df['User'].unique()
        np.random.shuffle(users)
        train_users = users[:int(len(users) * 0.7)]  # 70% for training
        test_users = users[int(len(users) * 0.7):]   # 30% for testing
        
        train_df = df[df['User'].isin(train_users)]
        test_df = df[df['User'].isin(test_users)]
        
        print(f"Train set: {len(train_df):,} transactions ({len(train_df)/len(df)*100:.1f}%)")
        print(f"Test set: {len(test_df):,} transactions ({len(test_df)/len(df)*100:.1f}%)")
    
    # 2. Data preprocessing
    print("\n[2] Preprocessing data...")
    
    # Convert amount to numeric if needed
    if 'Amount' in df.columns and not pd.api.types.is_numeric_dtype(df['Amount']):
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
        train_df['Amount'] = pd.to_numeric(train_df['Amount'], errors='coerce')
        test_df['Amount'] = pd.to_numeric(test_df['Amount'], errors='coerce')
    
    # Create log transformed amount
    train_df['amount_log'] = np.log1p(train_df['Amount'])
    test_df['amount_log'] = np.log1p(test_df['Amount'])
    
    # Extract hour from transaction time if available
    if 'Time' in train_df.columns:
        if not pd.api.types.is_datetime64_dtype(train_df['Time']):
            train_df['Time'] = pd.to_datetime(train_df['Time'])
            test_df['Time'] = pd.to_datetime(test_df['Time'])
        
        train_df['hour'] = train_df['Time'].dt.hour
        test_df['hour'] = test_df['Time'].dt.hour
    elif 'transaction_hour' in train_df.columns:
        train_df['hour'] = train_df['transaction_hour']
        test_df['hour'] = test_df['transaction_hour']
    else:
        # Create a placeholder if not available
        train_df['hour'] = 0
        test_df['hour'] = 0
        print("Warning: No time information found. Using placeholder hour value.")
    
    # 3. Feature Engineering
    print("\n[3] Engineering features...")
    
    # Check for required columns and adapt
    required_columns = ['MCC', 'Merchant', 'Location', 'Transaction_ID']
    missing_columns = [col for col in required_columns if col not in train_df.columns]
    
    if missing_columns:
        print(f"Warning: Missing required columns: {missing_columns}")
        print("Adapting feature engineering to available columns...")
        
        # Map column names if alternatives exist
        column_alternatives = {
            'MCC': ['Merchant Category', 'Category', 'MerchantCategory'],
            'Merchant': ['Merchant Name', 'MerchantName', 'Vendor'],
            'Location': ['Merchant City', 'City', 'MerchantCity'],
            'Transaction_ID': ['Transaction ID', 'TransactionID', 'ID']
        }
        
        for missing_col in missing_columns:
            alternatives = column_alternatives.get(missing_col, [])
            found = False
            
            for alt in alternatives:
                if alt in train_df.columns:
                    print(f"Using '{alt}' as alternative for '{missing_col}'")
                    # Create a copy with the expected column name
                    train_df[missing_col] = train_df[alt]
                    test_df[missing_col] = test_df[alt]
                    found = True
                    break
            
            if not found:
                print(f"No alternative found for '{missing_col}', creating placeholder")
                # Create placeholder with unique values per transaction
                train_df[missing_col] = range(len(train_df))
                test_df[missing_col] = range(len(train_df), len(train_df) + len(test_df))
    
    # 3.1. MCC risk score - mean transaction amount per MCC
    mcc_risk = train_df.groupby('MCC')['Amount'].mean().to_dict()
    train_df['mcc_risk_score'] = train_df['MCC'].map(mcc_risk)
    test_df['mcc_risk_score'] = test_df['MCC'].map(mcc_risk)
    
    # Fill missing values with overall mean
    mean_amount = train_df['Amount'].mean()
    train_df['mcc_risk_score'] = train_df['mcc_risk_score'].fillna(mean_amount)
    test_df['mcc_risk_score'] = test_df['mcc_risk_score'].fillna(mean_amount)
    
    # 3.2. Merchant frequency
    merchant_freq = train_df.groupby('Merchant')['User'].count().to_dict()
    train_df['merchant_freq'] = train_df['Merchant'].map(merchant_freq)
    test_df['merchant_freq'] = test_df['Merchant'].map(merchant_freq)
    
    # Fill with minimum frequency for unseen merchants
    min_freq = 1
    train_df['merchant_freq'] = train_df['merchant_freq'].fillna(min_freq)
    test_df['merchant_freq'] = test_df['merchant_freq'].fillna(min_freq)
    
    # 3.3. Online transaction flag
    train_df['is_online'] = train_df['Merchant'].str.contains('online|web|internet|digital', case=False).astype(int)
    test_df['is_online'] = test_df['Merchant'].str.contains('online|web|internet|digital', case=False).astype(int)
    
    train_df['is_online'] = train_df['is_online'].fillna(0)
    test_df['is_online'] = test_df['is_online'].fillna(0)
    
    # 3.4. Location-based transaction frequency
    loc_freq = train_df.groupby(['User', 'Location'])['Transaction_ID'].count().reset_index()
    loc_freq_dict = {}
    for _, row in loc_freq.iterrows():
        loc_freq_dict[(row['User'], row['Location'])] = row['Transaction_ID']
    
    train_df['location_freq'] = train_df.apply(lambda x: loc_freq_dict.get((x['User'], x['Location']), 0), axis=1)
    test_df['location_freq'] = test_df.apply(lambda x: loc_freq_dict.get((x['User'], x['Location']), 0), axis=1)
    
    # 3.5. Hour-based transaction frequency
    hour_freq = train_df.groupby(['User', 'hour'])['Transaction_ID'].count().reset_index()
    hour_freq_dict = {}
    for _, row in hour_freq.iterrows():
        hour_freq_dict[(row['User'], row['hour'])] = row['Transaction_ID']
    
    train_df['transaction_hour_freq'] = train_df.apply(lambda x: hour_freq_dict.get((x['User'], x['hour']), 0), axis=1)
    test_df['transaction_hour_freq'] = test_df.apply(lambda x: hour_freq_dict.get((x['User'], x['hour']), 0), axis=1)
    
    # 3.6. Fraud rate by merchant category
    mcc_fraud_rate = train_df.groupby('MCC')['Is_Fraud'].mean().to_dict()
    train_df['mcc_fraud_rate'] = train_df['MCC'].map(mcc_fraud_rate)
    test_df['mcc_fraud_rate'] = test_df['MCC'].map(mcc_fraud_rate)
    
    # Fill missing with overall fraud rate
    overall_fraud_rate = train_df['Is_Fraud'].mean()
    train_df['mcc_fraud_rate'] = train_df['mcc_fraud_rate'].fillna(overall_fraud_rate)
    test_df['mcc_fraud_rate'] = test_df['mcc_fraud_rate'].fillna(overall_fraud_rate)
    
    # 3.7. User average transaction amount
    user_avg_amount = train_df.groupby('User')['Amount'].mean().to_dict()
    train_df['user_avg_amount'] = train_df['User'].map(user_avg_amount)
    test_df['user_avg_amount'] = test_df['User'].map(user_avg_amount)
    
    # 3.8. Amount deviation from user average
    train_df['amount_deviation'] = (train_df['Amount'] - train_df['user_avg_amount']) / (train_df['user_avg_amount'] + 1)
    test_df['amount_deviation'] = (test_df['Amount'] - test_df['user_avg_amount']) / (test_df['user_avg_amount'] + 1)
    
    # Handle missing values in all features
    train_df = train_df.fillna(0)
    test_df = test_df.fillna(0)
    
    # 4. Feature Selection
    print("\n[4] Selecting features for model...")
    features = [
        'Amount', 'amount_log', 'hour', 'merchant_freq', 'is_online',
        'location_freq', 'MCC', 'transaction_hour_freq', 'mcc_risk_score',
        'mcc_fraud_rate', 'amount_deviation'
    ]
    
    # Verify features exist in the data
    features = [f for f in features if f in train_df.columns]
    print(f"Using {len(features)} features: {', '.join(features)}")
    
    # Prepare training and test data
    X_train = train_df[features]
    y_train = train_df['Is_Fraud']
    
    X_test = test_df[features]
    y_test = test_df['Is_Fraud']
    
    # 5. Feature Scaling
    print("\n[5] Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 6. Model Training
    print("\n[6] Training XGBoost model...")
    
    # Calculate class weight for imbalance
    pos_weight = len(y_train[y_train==0]) / max(1, len(y_train[y_train==1]))
    print(f"Class imbalance ratio (non-fraud to fraud): {pos_weight:.2f}")
    
    # Define model parameters
    model = XGBClassifier(
        max_depth=4,
        learning_rate=0.1,
        n_estimators=100,
        scale_pos_weight=pos_weight,
        random_state=42,
        eval_metric='auc'
    )
    
    # Train with early stopping
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        early_stopping_rounds=20,
        verbose=False
    )
    
    best_iteration = model.best_iteration
    print(f"Model trained, best iteration: {best_iteration}")
    
    # 7. Make predictions
    print("\n[7] Making predictions...")
    
    # Predict probabilities for ROC and threshold analysis
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Default threshold (0.5) predictions
    y_pred = model.predict(X_test_scaled)
    
    # 8. Evaluation
    print("\n[8] Evaluating model performance...")
    
    # Basic classification report
    cr = classification_report(y_test, y_pred, output_dict=True)
    cr_df = pd.DataFrame(cr).transpose()
    
    print("\nClassification Report:")
    print(cr_df.round(3))
    
    # Calculate ROC AUC
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC AUC Score: {roc_auc:.4f}")
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Calculate precision, recall and F1 score at different thresholds
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    # Find optimal threshold for F1
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    
    print(f"Optimal threshold (max F1): {optimal_threshold:.4f}")
    print(f"At this threshold - Precision: {precision[optimal_idx]:.4f}, Recall: {recall[optimal_idx]:.4f}, F1: {f1_scores[optimal_idx]:.4f}")
    
    # Apply optimal threshold
    y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
    cm_optimal = confusion_matrix(y_test, y_pred_optimal)
    
    # Print metrics with optimal threshold
    cr_optimal = classification_report(y_test, y_pred_optimal, output_dict=True)
    cr_optimal_df = pd.DataFrame(cr_optimal).transpose()
    
    print("\nClassification Report with Optimal Threshold:")
    print(cr_optimal_df.round(3))
    
    # 9. Visualizations
    print("\n[9] Creating visualizations...")
    
    # 9.1. Confusion Matrix (default threshold)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Fraud', 'Fraud'],
                yticklabels=['Not Fraud', 'Fraud'])
    plt.title('Confusion Matrix (Default Threshold)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrix.png')
    plt.close()
    
    # 9.2. Confusion Matrix (optimal threshold)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_optimal, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Fraud', 'Fraud'],
                yticklabels=['Not Fraud', 'Fraud'])
    plt.title(f'Confusion Matrix (Optimal Threshold: {optimal_threshold:.4f})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrix_optimal.png')
    plt.close()
    
    # 9.3. ROC Curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_dir}/roc_curve.png')
    plt.close()
    
    # 9.4. Precision-Recall Curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'F1 Max = {f1_scores[optimal_idx]:.4f}')
    plt.axvline(x=recall[optimal_idx], color='r', linestyle='--', 
                label=f'Optimal threshold = {optimal_threshold:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_dir}/precision_recall_curve.png')
    plt.close()
    
    # 9.5. Feature Importance
    plt.figure(figsize=(12, 8))
    xgb.plot_importance(model, max_num_features=len(features), importance_type='weight')
    plt.title('Feature Importance (Weight)')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_importance_weight.png')
    plt.close()
    
    plt.figure(figsize=(12, 8))
    xgb.plot_importance(model, max_num_features=len(features), importance_type='gain')
    plt.title('Feature Importance (Gain)')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_importance_gain.png')
    plt.close()
    
    # 9.6. Fraud Probability Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(y_pred_proba[y_test == 0], bins=50, alpha=0.5, label='Not Fraud', color='blue')
    sns.histplot(y_pred_proba[y_test == 1], bins=50, alpha=0.5, label='Fraud', color='red')
    plt.axvline(x=0.5, color='gray', linestyle='--', label='Default Threshold (0.5)')
    plt.axvline(x=optimal_threshold, color='green', linestyle='--', 
                label=f'Optimal Threshold ({optimal_threshold:.4f})')
    plt.xlabel('Predicted Fraud Probability')
    plt.ylabel('Count')
    plt.title('Distribution of Fraud Probabilities')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fraud_probability_distribution.png')
    plt.close()
    
    # 10. Save model and predictions
    print("\n[10] Saving model and results...")
    
    # Save model
    model_path = f'{output_dir}/fraud_detection_model.json'
    model.save_model(model_path)
    print(f"Model saved to {model_path}")
    
    # Save feature names
    with open(f'{output_dir}/features.json', 'w') as f:
        json.dump(features, f)
    
    # Save predictions
    predictions_df = test_df[['User', 'Transaction_ID', 'Is_Fraud']].copy()
    predictions_df['predicted_prob'] = y_pred_proba
    predictions_df['predicted_fraud'] = y_pred
    predictions_df['predicted_fraud_optimal'] = y_pred_optimal
    predictions_df.to_csv(f'{output_dir}/predictions.csv', index=False)
    
    # Save metrics
    metrics = {
        'roc_auc': roc_auc,
        'optimal_threshold': float(optimal_threshold),
        'precision_optimal': float(precision[optimal_idx]),
        'recall_optimal': float(recall[optimal_idx]),
        'f1_optimal': float(f1_scores[optimal_idx]),
        'class_imbalance_ratio': float(pos_weight),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(f'{output_dir}/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"{'='*50}\nFRAUD DETECTION MODEL COMPLETE\n{'='*50}")
    print(f"Results saved to {output_dir}/")
    return model, metrics

if __name__ == "__main__":
    # First run the analysis and data splitting if needed
    if not os.path.exists('smart_split_data.csv'):
        print("Split dataset not found. Running data analysis and splitting...")
        split_data = analyze_and_split_data(
            input_file='./data/card_transaction.v1.csv',
            mcc_file='./data/mcc_codes.json',
            output_split_file='smart_split_data.csv'
        )
    
    # Run the fraud detection model
    model, metrics = run_fraud_detection(
        data_file='smart_split_data.csv',
        output_dir='fraud_detection_results',
        use_existing_split=True
    )
    
    print("\nFraud Detection Summary:")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"Optimal Threshold: {metrics['optimal_threshold']:.4f}")
    print(f"F1 Score at Optimal Threshold: {metrics['f1_optimal']:.4f}")