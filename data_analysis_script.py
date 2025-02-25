import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
import os

# Setting display options for better readability
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
np.random.seed(42)  # For reproducibility

def analyze_and_split_data(input_file='./data/card_transaction.v1.csv',
                           mcc_file='./data/mcc_codes.json',
                           output_split_file='smart_split_data.csv',
                           test_size=0.3,
                           fraud_oversampling=True):
    """
    Analyzes credit card transaction data and creates intelligent train/test splits
    
    Parameters:
    -----------
    input_file : str
        Filename of the transaction data
    mcc_file : str
        Filename of the MCC codes JSON
    output_split_file : str
        Filename to save the split dataset
    test_size : float
        Proportion of data to use for testing (default 0.3)
    fraud_oversampling : bool
        Whether to ensure balanced fraud representation in train/test sets
    
    Returns:
    --------
    DataFrame with added 'split' column indicating 'train' or 'test'
    """
    print(f"{'='*50}\nSTART DATA ANALYSIS AND SMART SPLITTING\n{'='*50}")
    
    # 1. Load and examine data
    print(f"\n[1] Loading transaction data from {input_file}...")
    try:
        df = pd.read_csv(input_file)
        print(f"Successfully loaded {len(df):,} transactions")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    # 2. Basic data exploration
    print("\n[2] Performing basic data exploration...")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check for missing values
    missing_vals = df.isnull().sum()
    if missing_vals.sum() > 0:
        print("\nMissing values by column:")
        print(missing_vals[missing_vals > 0])
    else:
        print("\nNo missing values found")
    
    # 3. Load MCC codes
    print(f"\n[3] Loading MCC codes from {mcc_file}...")
    try:
        with open(mcc_file, 'r') as f:
            mcc_data = json.load(f)
        
        # Handle both dictionary and list formats
        if isinstance(mcc_data, dict):
            mcc_dict = {int(k): v['edited_description'] for k, v in mcc_data.items() if k.isdigit()}
        elif isinstance(mcc_data, list):
            # If it's a list, try to convert to dictionary format
            mcc_dict = {}
            for item in mcc_data:
                if isinstance(item, dict) and 'mcc' in item and 'edited_description' in item:
                    mcc_dict[int(item['mcc'])] = item['edited_description']
        else:
            mcc_dict = {}
            
        print(f"Successfully loaded {len(mcc_dict):,} MCC codes")
    except Exception as e:
        print(f"Error loading MCC codes: {e}. Will proceed without MCC descriptions.")
        mcc_dict = {}
    
    # 4. Data preprocessing
    print("\n[4] Preprocessing data...")

    # Ensure proper data types
    if 'Time' in df.columns:
        df['Time'] = pd.to_datetime(df['Time'])
        print("Converted Time column to datetime")
        
        # Extract date components
        df['transaction_date'] = df['Time'].dt.date
        df['transaction_hour'] = df['Time'].dt.hour
        df['transaction_day'] = df['Time'].dt.day_name()
        df['transaction_month'] = df['Time'].dt.month
        
        # Sort by user and time
        df = df.sort_values(['User', 'Time'])
        print("Sorted transactions by User and Time")

    # Convert Amount to numeric if needed
    if 'Amount' in df.columns and not pd.api.types.is_numeric_dtype(df['Amount']):
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
        print("Converted Amount to numeric")

    # Check for fraud column, rename and convert to boolean format
    if 'Is Fraud?' in df.columns:
        print("Renaming 'Is Fraud?' column to 'is_fraud' and converting to boolean (0/1)")
        
        # First, make a copy of the column with the new name
        df['is_fraud'] = df['Is Fraud?']
        
        # Convert various formats to boolean where True=1 and False=0
        if df['is_fraud'].dtype == 'object':
            # For text-based boolean indicators
            true_values = ['yes', 'true', '1', 't', 'y', 'True', 'TRUE', 'Yes', 'YES']
            df['is_fraud'] = df['is_fraud'].astype(str).str.strip().isin(true_values).astype(int)
        else:
            # For numeric indicators, ensure they're properly converted to 0/1
            df['is_fraud'] = df['is_fraud'].astype(int)
            
        # Drop the original column
        df.drop(columns=['Is Fraud?'], inplace=True)
        
        print(f"Converted 'is_fraud' to binary format with {df['is_fraud'].sum()} fraud cases")
    elif 'Is_Fraud' in df.columns:
        print("Renaming 'Is_Fraud' column to 'is_fraud' and converting to boolean (0/1)")
        
        # First, make a copy of the column with the new name
        df['is_fraud'] = df['Is_Fraud']
        
        # Convert various formats to boolean where True=1 and False=0
        if df['is_fraud'].dtype == 'object':
            # For text-based boolean indicators
            true_values = ['yes', 'true', '1', 't', 'y', 'True', 'TRUE', 'Yes', 'YES']
            df['is_fraud'] = df['is_fraud'].astype(str).str.strip().isin(true_values).astype(int)
        else:
            # For numeric indicators, ensure they're properly converted to 0/1
            df['is_fraud'] = df['is_fraud'].astype(int)
            
        # Drop the original column
        df.drop(columns=['Is_Fraud'], inplace=True)
        
        print(f"Converted 'is_fraud' to binary format with {df['is_fraud'].sum()} fraud cases")
    else:
        print("No fraud column found. Checking for similar columns...")
        fraud_columns = [col for col in df.columns if 'fraud' in col.lower()]
        if fraud_columns:
            print(f"Found potential fraud columns: {fraud_columns}")
            # Use the first matching column
            fraud_col = fraud_columns[0]
            print(f"Converting '{fraud_col}' to 'is_fraud' and formatting as boolean")
            
            # First, make a copy of the column with the new name
            df['is_fraud'] = df[fraud_col]
            
            # Convert various formats to boolean where True=1 and False=0
            if df['is_fraud'].dtype == 'object':
                # For text-based boolean indicators
                true_values = ['yes', 'true', '1', 't', 'y', 'True', 'TRUE', 'Yes', 'YES']
                df['is_fraud'] = df['is_fraud'].astype(str).str.strip().isin(true_values).astype(int)
            else:
                # For numeric indicators, ensure they're properly converted to 0/1
                df['is_fraud'] = df['is_fraud'].astype(int)
                
            # Drop the original column if different
            if fraud_col != 'is_fraud':
                df.drop(columns=[fraud_col], inplace=True)
                
            print(f"Converted 'is_fraud' to binary format with {df['is_fraud'].sum()} fraud cases")
        else:
            print("No fraud-related columns found. Add an 'is_fraud' column to enable fraud analysis.")

    # Calculate total users here to avoid UnboundLocalError
    total_users = df['User'].nunique()
    
    # 5. Fraud analysis
    print("\n[5] Analyzing fraud distribution...")
    if 'is_fraud' in df.columns:
        fraud_count = df['is_fraud'].sum()
        total_count = len(df)
        fraud_pct = (fraud_count / total_count) * 100
        
        print(f"Fraud transactions: {fraud_count:,} ({fraud_pct:.2f}%)")
        print(f"Non-fraud transactions: {total_count - fraud_count:,} ({100 - fraud_pct:.2f}%)")
        
        # Fraud by user
        fraud_by_user = df.groupby('User')['is_fraud'].sum()
        users_with_fraud = fraud_by_user[fraud_by_user > 0].count()
        
        print(f"Users with at least one fraudulent transaction: {users_with_fraud:,} ({(users_with_fraud/total_users)*100:.2f}%)")
    else:
        print("No 'is_fraud' column found. Fraud analysis will be skipped.")
    
    # 6. User transaction patterns
    print("\n[6] Analyzing user transaction patterns...")
    user_txn_counts = df.groupby('User').size()
    print(f"Average transactions per user: {user_txn_counts.mean():.1f}")
    print(f"Median transactions per user: {user_txn_counts.median():.1f}")
    print(f"Min transactions per user: {user_txn_counts.min()}")
    print(f"Max transactions per user: {user_txn_counts.max()}")
    
    # Identify users with few transactions
    low_activity_users = user_txn_counts[user_txn_counts < 5].count()
    print(f"Users with fewer than 5 transactions: {low_activity_users:,} ({(low_activity_users/total_users)*100:.2f}%)")
    
    # 7. Time-based patterns
    if 'Time' in df.columns:
        print("\n[7] Analyzing time-based patterns...")
        
        # Transactions by hour
        hour_counts = df.groupby('transaction_hour').size()
        peak_hour = hour_counts.idxmax()
        print(f"Peak transaction hour: {peak_hour} (with {hour_counts[peak_hour]:,} transactions)")
        
        # Transactions by day
        if 'transaction_day' in df.columns:
            day_counts = df.groupby('transaction_day').size()
            print("\nTransactions by day of week:")
            for day, count in day_counts.items():
                print(f"  {day}: {count:,}")
    
    # 8. Create intelligent train/test split
    print(f"\n[8] Creating train/test split (test size: {test_size})...")
    
    # Strategy: Split by users, not by individual transactions
    # This prevents data leakage where a user's behavior patterns appear in both train and test
    
    user_list = df['User'].unique()
    np.random.shuffle(user_list)  # Shuffle users
    
    if fraud_oversampling and 'is_fraud' in df.columns:
        print("Using stratified sampling to ensure balanced fraud representation...")
        
        # Get users with fraud
        fraud_users = df[df['is_fraud'] == 1]['User'].unique()
        non_fraud_users = np.setdiff1d(user_list, fraud_users)
        
        # Split fraud users
        test_size_fraud = int(len(fraud_users) * test_size)
        test_fraud_users = fraud_users[:test_size_fraud]
        train_fraud_users = fraud_users[test_size_fraud:]
        
        # Split non-fraud users
        test_size_non_fraud = int(len(non_fraud_users) * test_size)
        test_non_fraud_users = non_fraud_users[:test_size_non_fraud]
        train_non_fraud_users = non_fraud_users[test_size_non_fraud:]
        
        # Combine
        test_users = np.concatenate([test_fraud_users, test_non_fraud_users])
        train_users = np.concatenate([train_fraud_users, train_non_fraud_users])
        
        print(f"Train set: {len(train_users):,} users with {len(train_fraud_users):,} fraud users")
        print(f"Test set: {len(test_users):,} users with {len(test_fraud_users):,} fraud users")
    else:
        # Simple random split
        test_size_int = int(len(user_list) * test_size)
        test_users = user_list[:test_size_int]
        train_users = user_list[test_size_int:]
        
        print(f"Train set: {len(train_users):,} users")
        print(f"Test set: {len(test_users):,} users")
    
    # Create the split column
    df['split'] = 'train'  # Default
    df.loc[df['User'].isin(test_users), 'split'] = 'test'
    
    # Verify split
    train_df = df[df['split'] == 'train']
    test_df = df[df['split'] == 'test']
    
    print(f"\nFinal split:")
    print(f"  Train: {len(train_df):,} transactions ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Test: {len(test_df):,} transactions ({len(test_df)/len(df)*100:.1f}%)")
    
    if 'is_fraud' in df.columns:
        train_fraud_rate = train_df['is_fraud'].mean() * 100
        test_fraud_rate = test_df['is_fraud'].mean() * 100
        print(f"  Train fraud rate: {train_fraud_rate:.2f}%")
        print(f"  Test fraud rate: {test_fraud_rate:.2f}%")
    
    # 9. Save the split dataset
    print(f"\n[9] Saving split dataset to {output_split_file}...")
    df.to_csv(output_split_file, index=False)
    print(f"Dataset saved successfully")
    
    # 10. Generate simple visualizations
    print("\n[10] Generating exploratory visualizations...")
    
    # Create output directory for plots
    plots_dir = 'fraud_analysis_plots'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    # 10.1. Fraud distribution
    if 'is_fraud' in df.columns:
        plt.figure(figsize=(10, 6))
        df['is_fraud'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
        plt.title('Distribution of Fraud vs Non-Fraud Transactions')
        plt.xlabel('Is Fraud')
        plt.ylabel('Number of Transactions')
        plt.xticks([0, 1], ['Non-Fraud', 'Fraud'], rotation=0)
        plt.savefig(f'{plots_dir}/fraud_distribution.png')
        plt.close()
    
    # 10.2. Transaction amount distribution
    if 'Amount' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x='Amount', kde=True, bins=50)
        plt.title('Transaction Amount Distribution')
        plt.xlabel('Amount')
        plt.ylabel('Frequency')
        plt.savefig(f'{plots_dir}/amount_distribution.png')
        plt.close()
        
        # Amount distribution by fraud/non-fraud
        if 'is_fraud' in df.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(data=df, x='Amount', hue='is_fraud', element='step', bins=50, log_scale=True)
            plt.title('Transaction Amount by Fraud Status (Log Scale)')
            plt.xlabel('Amount (Log Scale)')
            plt.ylabel('Frequency')
            plt.legend(['Non-Fraud', 'Fraud'])
            plt.savefig(f'{plots_dir}/amount_by_fraud_status.png')
            plt.close()
    
    # 10.3. Transactions by hour
    if 'transaction_hour' in df.columns:
        plt.figure(figsize=(12, 6))
        hour_counts = df.groupby('transaction_hour').size()
        hour_counts.plot(kind='line', marker='o')
        plt.title('Transactions by Hour of Day')
        plt.xlabel('Hour')
        plt.ylabel('Number of Transactions')
        plt.xticks(range(0, 24))
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(f'{plots_dir}/transactions_by_hour.png')
        plt.close()
        
        # Fraud rate by hour
        if 'is_fraud' in df.columns:
            plt.figure(figsize=(12, 6))
            fraud_by_hour = df.groupby('transaction_hour')['is_fraud'].mean() * 100
            fraud_by_hour.plot(kind='line', marker='o', color='crimson')
            plt.title('Fraud Rate by Hour of Day')
            plt.xlabel('Hour')
            plt.ylabel('Fraud Rate (%)')
            plt.xticks(range(0, 24))
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(f'{plots_dir}/fraud_rate_by_hour.png')
            plt.close()
    
    print(f"\nAnalysis complete! Plots saved to '{plots_dir}/' directory.")
    print(f"{'='*50}\nEND DATA ANALYSIS AND SMART SPLITTING\n{'='*50}")
    
    return df

if __name__ == "__main__":
    # Run the analysis and splitting
    split_data = analyze_and_split_data(
        input_file='./data/card_transaction.v1.csv',  # Original data
        mcc_file='./data/mcc_codes.json',            # MCC codes
        output_split_file='smart_split_data.csv',    # Output file
        test_size=0.3,                               # 70/30 train/test split
        fraud_oversampling=True                      # Ensure balanced fraud representation
    )
    
    if split_data is not None:
        print("Analysis and splitting completed successfully.")
        print("You can now run the fraud detection model using 'smart_split_data.csv'")