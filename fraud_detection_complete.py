import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import json
from datetime import datetime
import os
import warnings
import gc  # Garbage collector

# Import the analysis and splitting function
from data_analysis_script import analyze_and_split_data

# Suppress warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def run_fraud_detection(data_file='smart_split_data.csv', 
                        output_dir='fraud_detection_results',
                        use_existing_split=True,
                        sample_size=None):  # Optional sampling for very large datasets
    """
    Run the optimized fraud detection model pipeline using RandomForest
    
    Parameters:
    -----------
    data_file : str
        Path to the data file (with train/test split)
    output_dir : str
        Directory to save results and plots
    use_existing_split : bool
        Whether to use the existing train/test split in the data
    sample_size : int or None
        If provided, will use a random sample of this size for training
    """
    print(f"{'='*50}\nSTART FRAUD DETECTION MODEL\n{'='*50}")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 1. Load data
    print(f"\n[1] Loading data from {data_file}...")
    
    # For very large datasets, consider using chunking
    df = pd.read_csv(data_file)
    print(f"Loaded {len(df):,} transactions")
    
    # Check for fraud column and ensure proper name
    if 'is_fraud' not in df.columns:
        if 'Is Fraud?' in df.columns:
            print("Renaming 'Is Fraud?' column to 'is_fraud'")
            df['is_fraud'] = df['Is Fraud?'].astype(int)
        elif 'Is_Fraud' in df.columns:
            print("Renaming 'Is_Fraud' column to 'is_fraud'")
            df['is_fraud'] = df['Is_Fraud'].astype(int)
        else:
            fraud_columns = [col for col in df.columns if 'fraud' in col.lower()]
            if fraud_columns:
                print(f"Using {fraud_columns[0]} as fraud indicator")
                df['is_fraud'] = df[fraud_columns[0]].astype(int)
            else:
                raise ValueError("No fraud column found in the dataset")
    
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
                train_indicators = ['train', '1', 1, 'tr', True, 'true']
                
                for val in split_values:
                    if str(val).lower() in [str(ind).lower() for ind in train_indicators]:
                        train_value = val
                        break
                else:
                    # Default to first value
                    train_value = split_values[0]
                    print(f"Could not determine train value, defaulting to: {train_value}")
                
                # Split the data
                train_df = df[df[split_col] == train_value].copy()
                test_df = df[df[split_col] != train_value].copy()
                
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
        # Split users, not transactions
        users = df['User'].unique()
        np.random.shuffle(users)
        train_users = users[:int(len(users) * 0.7)]  # 70% for training
        test_users = users[int(len(users) * 0.7):]   # 30% for testing
        
        train_df = df[df['User'].isin(train_users)].copy()
        test_df = df[df['User'].isin(test_users)].copy()
        
        print(f"Train set: {len(train_df):,} transactions ({len(train_df)/len(df)*100:.1f}%)")
        print(f"Test set: {len(test_df):,} transactions ({len(test_df)/len(df)*100:.1f}%)")
    
    # Free up memory
    del df
    gc.collect()
    
    # Sample if needed for very large datasets
    if sample_size is not None and sample_size < len(train_df):
        print(f"\nSampling {sample_size:,} rows from training set for model building...")
        
        # Make sure we include enough fraud cases
        fraud_train = train_df[train_df['is_fraud'] == 1]
        non_fraud_train = train_df[train_df['is_fraud'] == 0]
        
        # Keep all fraud or sample if too many
        if len(fraud_train) > sample_size // 10:
            fraud_sample = fraud_train.sample(sample_size // 10, random_state=42)
        else:
            fraud_sample = fraud_train
            
        # Sample remaining from non-fraud
        non_fraud_sample_size = min(sample_size - len(fraud_sample), len(non_fraud_train))
        non_fraud_sample = non_fraud_train.sample(non_fraud_sample_size, random_state=42)
        
        # Combine and shuffle
        train_df_sampled = pd.concat([fraud_sample, non_fraud_sample])
        train_df_sampled = train_df_sampled.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"Sampled training set: {len(train_df_sampled):,} transactions with {len(fraud_sample):,} fraud cases")
        train_df = train_df_sampled
    
    # 2. Data preprocessing and feature engineering (combined for efficiency)
    print("\n[2] Preprocessing data and engineering features...")
    
    # ---------- Numeric features ----------
    # Convert amount to numeric if needed
    if 'Amount' in train_df.columns and not pd.api.types.is_numeric_dtype(train_df['Amount']):
        train_df['Amount'] = pd.to_numeric(train_df['Amount'], errors='coerce')
        test_df['Amount'] = pd.to_numeric(test_df['Amount'], errors='coerce')
    
    # Log transform amount - useful for skewed monetary values
    train_df['amount_log'] = np.log1p(train_df['Amount'])
    test_df['amount_log'] = np.log1p(test_df['Amount'])
    
    # ---------- Time-based features ----------
    # Extract hour from transaction time if available
    if 'Time' in train_df.columns:
        if not pd.api.types.is_datetime64_dtype(train_df['Time']):
            train_df['Time'] = pd.to_datetime(train_df['Time'], errors='coerce')
            test_df['Time'] = pd.to_datetime(test_df['Time'], errors='coerce')
        
        # Extract hour
        train_df['hour'] = train_df['Time'].dt.hour
        test_df['hour'] = test_df['Time'].dt.hour
        
        # Day of week (0=Monday, 6=Sunday)
        train_df['day_of_week'] = train_df['Time'].dt.dayofweek
        test_df['day_of_week'] = test_df['Time'].dt.dayofweek
        
        # Is weekend
        train_df['is_weekend'] = (train_df['day_of_week'] >= 5).astype(int)
        test_df['is_weekend'] = (test_df['day_of_week'] >= 5).astype(int)
    elif 'transaction_hour' in train_df.columns:
        train_df['hour'] = train_df['transaction_hour']
        test_df['hour'] = test_df['transaction_hour']
    else:
        # Create a placeholder if not available
        train_df['hour'] = 0
        test_df['hour'] = 0
        print("Warning: No time information found. Using placeholder hour value.")
    
    # ---------- MCC-based features ----------
    if 'MCC' in train_df.columns:
        # MCC risk score - mean transaction amount per MCC
        mcc_stats = train_df.groupby('MCC')['Amount'].agg(['mean', 'std', 'count']).reset_index()
        mcc_stats.columns = ['MCC', 'mcc_avg_amount', 'mcc_std_amount', 'mcc_count']
        
        # Convert to dictionaries for faster mapping
        mcc_avg_dict = dict(zip(mcc_stats['MCC'], mcc_stats['mcc_avg_amount']))
        mcc_std_dict = dict(zip(mcc_stats['MCC'], mcc_stats['mcc_std_amount']))
        mcc_count_dict = dict(zip(mcc_stats['MCC'], mcc_stats['mcc_count']))
        
        # Apply mappings
        train_df['mcc_avg_amount'] = train_df['MCC'].map(mcc_avg_dict)
        test_df['mcc_avg_amount'] = test_df['MCC'].map(mcc_avg_dict)
        
        train_df['mcc_count'] = train_df['MCC'].map(mcc_count_dict)
        test_df['mcc_count'] = test_df['MCC'].map(mcc_count_dict)
        
        # Fraud rate by MCC
        if 'is_fraud' in train_df.columns:
            mcc_fraud = train_df.groupby('MCC')['is_fraud'].mean().reset_index()
            mcc_fraud.columns = ['MCC', 'mcc_fraud_rate']
            mcc_fraud_dict = dict(zip(mcc_fraud['MCC'], mcc_fraud['mcc_fraud_rate']))
            
            train_df['mcc_fraud_rate'] = train_df['MCC'].map(mcc_fraud_dict)
            test_df['mcc_fraud_rate'] = test_df['MCC'].map(mcc_fraud_dict)
    else:
        print("Warning: No MCC column found")
    
    # ---------- User-based features ----------
    # User average transaction amount
    user_stats = train_df.groupby('User')['Amount'].agg(['mean', 'std', 'count']).reset_index()
    user_stats.columns = ['User', 'user_avg_amount', 'user_std_amount', 'user_txn_count']
    
    # Convert to dictionaries for faster mapping
    user_avg_dict = dict(zip(user_stats['User'], user_stats['user_avg_amount']))
    user_std_dict = dict(zip(user_stats['User'], user_stats['user_std_amount']))
    user_count_dict = dict(zip(user_stats['User'], user_stats['user_txn_count']))
    
    # Apply mappings
    train_df['user_avg_amount'] = train_df['User'].map(user_avg_dict)
    test_df['user_avg_amount'] = test_df['User'].map(user_avg_dict)
    
    train_df['user_txn_count'] = train_df['User'].map(user_count_dict)
    test_df['user_txn_count'] = test_df['User'].map(user_count_dict)
    
    # Amount deviation from user average (z-score)
    # Avoid division by zero
    train_df['user_std_amount'] = train_df['User'].map(user_std_dict)
    test_df['user_std_amount'] = test_df['User'].map(user_std_dict)
    
    # Replace zero std with median std
    median_std = train_df['user_std_amount'].median()
    train_df['user_std_amount'] = train_df['user_std_amount'].replace(0, median_std)
    test_df['user_std_amount'] = test_df['user_std_amount'].replace(0, median_std)
    
    # Calculate z-score
    train_df['amount_zscore'] = (train_df['Amount'] - train_df['user_avg_amount']) / train_df['user_std_amount']
    test_df['amount_zscore'] = (test_df['Amount'] - test_df['user_avg_amount']) / test_df['user_std_amount']
    
    # ---------- Location-based features ----------
    # Check if we have location-related columns
    location_columns = [col for col in train_df.columns if col in 
                      ['Merchant City', 'Location', 'Merchant State', 'Zip']]
    
    if location_columns:
        # Use the first available location column
        location_col = location_columns[0]
        print(f"Using '{location_col}' for location-based features")
        
        # Location frequency
        location_counts = train_df.groupby(location_col)['User'].count().reset_index()
        location_counts.columns = [location_col, 'location_freq']
        loc_freq_dict = dict(zip(location_counts[location_col], location_counts['location_freq']))
        
        train_df['location_freq'] = train_df[location_col].map(loc_freq_dict)
        test_df['location_freq'] = test_df[location_col].map(loc_freq_dict)
        
        # Fill missing with 0
        train_df['location_freq'] = train_df['location_freq'].fillna(0)
        test_df['location_freq'] = test_df['location_freq'].fillna(0)
        
        # User-location pairs
        if len(train_df) < 5000000:  # Only do this for smaller datasets as it's memory intensive
            user_loc_pairs = train_df.groupby(['User', location_col]).size().reset_index()
            user_loc_pairs.columns = ['User', location_col, 'user_loc_freq']
            user_loc_pairs['user_loc_key'] = user_loc_pairs['User'].astype(str) + '_' + user_loc_pairs[location_col].astype(str)
            
            user_loc_dict = dict(zip(user_loc_pairs['user_loc_key'], user_loc_pairs['user_loc_freq']))
            
            # Apply to train set
            train_df['user_loc_key'] = train_df['User'].astype(str) + '_' + train_df[location_col].astype(str)
            train_df['user_loc_freq'] = train_df['user_loc_key'].map(user_loc_dict)
            train_df.drop(columns=['user_loc_key'], inplace=True)
            
            # Apply to test set
            test_df['user_loc_key'] = test_df['User'].astype(str) + '_' + test_df[location_col].astype(str)
            test_df['user_loc_freq'] = test_df['user_loc_key'].map(user_loc_dict)
            test_df.drop(columns=['user_loc_key'], inplace=True)
            
            # Fill missing with 0
            train_df['user_loc_freq'] = train_df['user_loc_freq'].fillna(0)
            test_df['user_loc_freq'] = test_df['user_loc_freq'].fillna(0)
        else:
            print("Skipping user-location pairs calculation for large dataset")
    else:
        print("Warning: No location columns found")
        train_df['location_freq'] = 0
        test_df['location_freq'] = 0
    
    # Handle missing values in all features
    train_df = train_df.fillna(0)
    test_df = test_df.fillna(0)
    
    # 4. Feature Selection
    print("\n[3] Selecting features for model...")
    # Choose features that exist in the dataframe
    all_features = [
        'Amount', 'amount_log', 'hour', 'day_of_week', 'is_weekend',
        'mcc_avg_amount', 'mcc_count', 'mcc_fraud_rate',
        'user_avg_amount', 'user_txn_count', 'amount_zscore',
        'location_freq', 'user_loc_freq'
    ]
    
    # Filter to only include features that exist
    features = [f for f in all_features if f in train_df.columns]
    print(f"Using {len(features)} features: {', '.join(features)}")
    
    # Prepare training and test data
    X_train = train_df[features]
    y_train = train_df['is_fraud']
    
    X_test = test_df[features]
    y_test = test_df['is_fraud']
    
    # 5. Feature Scaling
    print("\n[4] Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 6. Model Training
    print("\n[5] Training RandomForest model...")
    
    # Calculate class weights
    n_samples = len(y_train)
    n_fraud = y_train.sum()
    n_normal = n_samples - n_fraud
    
    # Handle class imbalance
    class_weight = {
        0: 1,
        1: n_normal / max(1, n_fraud)  # Adjust weight for fraud class
    }
    print(f"Class imbalance ratio (non-fraud to fraud): {n_normal / max(1, n_fraud):.2f}")
    
    # Define model parameters - optimized for memory efficiency
    model = RandomForestClassifier(
        n_estimators=50,  # Reduced for memory
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features='sqrt',
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1  # Use all available cores
    )
    
    # Train the model
    print("Training model (this may take a while with large datasets)...")
    model.fit(X_train_scaled, y_train)
    print("Model training complete!")
    
    # 7. Make predictions
    print("\n[6] Making predictions...")
    
    # Predict probabilities for ROC and threshold analysis
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Default threshold (0.5) predictions
    y_pred = model.predict(X_test_scaled)
    
    # 8. Evaluation
    print("\n[7] Evaluating model performance...")
    
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
    print("\n[8] Creating visualizations...")
    
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
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.title('Feature Importance')
    plt.bar(range(len(indices)), importances[indices], align='center')
    plt.xticks(range(len(indices)), [features[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_importance.png')
    plt.close()
    
    # 10. Save predictions and metrics
    print("\n[9] Saving results...")
    
    # Take a sample of predictions for saving (to avoid memory issues)
    if len(test_df) > 1000000:
        print(f"Sampling {1000000:,} predictions for saving...")
        sample_indices = np.random.choice(len(test_df), size=1000000, replace=False)
        
        # Select the test samples
        test_sample = test_df.iloc[sample_indices].copy()
        sample_y_test = y_test.iloc[sample_indices]
        sample_y_pred = y_pred[sample_indices]
        sample_y_pred_proba = y_pred_proba[sample_indices]
        sample_y_pred_optimal = y_pred_optimal[sample_indices]
        
        # Create prediction dataframe
        predictions_df = test_sample[['User']].copy()
        predictions_df['is_fraud'] = sample_y_test
        predictions_df['predicted_prob'] = sample_y_pred_proba
        predictions_df['predicted_fraud'] = sample_y_pred
        predictions_df['predicted_fraud_optimal'] = sample_y_pred_optimal
    else:
        # Save all predictions
        predictions_df = test_df[['User']].copy()
        predictions_df['is_fraud'] = y_test
        predictions_df['predicted_prob'] = y_pred_proba
        predictions_df['predicted_fraud'] = y_pred
        predictions_df['predicted_fraud_optimal'] = y_pred_optimal
    
    # Save predictions
    predictions_df.to_csv(f'{output_dir}/predictions.csv', index=False)
    
    # Save feature names
    with open(f'{output_dir}/features.json', 'w') as f:
        json.dump(features, f)
    
    # Save metrics
    metrics = {
        'roc_auc': float(roc_auc),
        'optimal_threshold': float(optimal_threshold),
        'precision_optimal': float(precision[optimal_idx]),
        'recall_optimal': float(recall[optimal_idx]),
        'f1_optimal': float(f1_scores[optimal_idx]),
        'class_imbalance_ratio': float(n_normal / max(1, n_fraud)),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(f'{output_dir}/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"{'='*50}\nFRAUD DETECTION MODEL COMPLETE\n{'='*50}")
    print(f"Results saved to {output_dir}/")
    return model, metrics

if __name__ == "__main__":
    # First check if split dataset exists
    if not os.path.exists('smart_split_data.csv'):
        print("Split dataset not found. Running data analysis and splitting...")
        split_data = analyze_and_split_data(
            input_file='./data/card_transaction.v1.csv',
            mcc_file='./data/mcc_codes.json',
            output_split_file='smart_split_data.csv'
        )
    
    # Run the fraud detection model with sampling for very large datasets
    # You can adjust the sample_size parameter based on your system's memory
    sample_size = 1000000  # Use None to process the entire dataset
    
    model, metrics = run_fraud_detection(
        data_file='smart_split_data.csv',
        output_dir='fraud_detection_results',
        use_existing_split=True,
        sample_size=sample_size
    )
    
    print("\nFraud Detection Summary:")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"Optimal Threshold: {metrics['optimal_threshold']:.4f}")
    print(f"F1 Score at Optimal Threshold: {metrics['f1_optimal']:.4f}")