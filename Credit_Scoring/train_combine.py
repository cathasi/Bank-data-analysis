import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score, 
                             classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay)
from xgboost import XGBRegressor, XGBClassifier
import joblib
import json
import time
import matplotlib
matplotlib.use('Agg') # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import os

# Create directories if they don't exist
os.makedirs('./models', exist_ok=True)
os.makedirs('./data', exist_ok=True)

start_time = time.perf_counter()

# ==========================================
# CONFIGURATION
# ==========================================
DATA_FILEPATH = './data/train.csv'

# Model Paths
REGRESSOR_PATH = './models/credit_score_regressor.joblib'
CLASSIFIER_PATH = './models/credit_score_classifier.joblib'
ARTIFACTS_PATH = './models/preprocessing_artifacts.json'

# Hyperparameters (Shared)
NUM_TREES = 10000 # Reduced slightly for combined run speed, adjust to 10000 for production
LEARNING_RATE = 0.01
EARLY_STOPPING_ROUNDS = 10

# Columns to drop (Common ID columns)
COMMON_DROP_COLS = ['ID', 'Customer_ID', 'Month', 'Name', 'SSN', 'Type_of_Loan']

# Features Lists
NUMERICAL_FEATURES_RAW = [
    'Age', 'Annual_Income', 'Num_Bank_Accounts', 'Num_Credit_Card', 
    'Interest_Rate', 'Delay_from_due_date', 'Num_of_Delayed_Payment', 
    'Changed_Credit_Limit', 'Num_Credit_Inquiries', 'Outstanding_Debt', 
    'Credit_Utilization_Ratio', 'Total_EMI_per_month', 'Amount_invested_monthly', 
    'Monthly_Balance', 'Num_of_Loan'
]

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def clean_non_numeric(series):
    """Cleans currency symbols and underscores from string columns."""
    series = series.astype(str).str.replace(r'[$,_!#]', '', regex=True)
    return pd.to_numeric(series, errors='coerce')

def convert_age_to_months(age_str):
    """Parses 'X Years and Y Months' string."""
    if pd.isna(age_str):
        return np.nan
    try:
        parts = str(age_str).split(' and ')
        years = int(parts[0].split(' ')[0])
        months = int(parts[1].split(' ')[0])
        return years * 12 + months
    except:
        return np.nan

def load_and_clean_base_data(filepath):
    """Loads data and performs cleaning common to both models."""
    print(f"Loading data from {filepath}...")
    try:
        # Added low_memory=False to suppress DtypeWarning for mixed types
        df = pd.read_csv(filepath, low_memory=False)
    except FileNotFoundError:
        print(f"Error: {filepath} not found.")
        exit()
        
    # Drop IDs
    df = df.drop(columns=COMMON_DROP_COLS, axis=1)
    
    # Clean Numerical Columns
    print("Cleaning numerical features...")
    for col in NUMERICAL_FEATURES_RAW:
        if col in df.columns:
            df[col] = clean_non_numeric(df[col])

    # Handle Age outliers (Common logic)
    df['Age'] = df['Age'].clip(lower=18, upper=100)

    counts = df['Credit_Score'].value_counts()
    print(counts)

    plt.figure(figsize=(8, 6)) # Adjust figure size for better viewing
    plt.bar(counts.index, counts.values, color=['gold', 'red', 'green'])

    # Add labels and title
    plt.title('Distribution of Quality Groups')
    plt.xlabel('Quality Group')
    plt.ylabel('Count')

    # Show the plot
    plt.grid(axis='y', linestyle='--') # Add horizontal grid lines
    plt.show()
    plt.plot()
    plt.savefig('graphics/counts.png')
    
    return df

# ==========================================
# REGRESSION PIPELINE
# ==========================================

def train_regressor(df_base):
    print("\n" + "="*40)
    print("STARTING REGRESSION MODEL TRAINING")
    print("="*40)
    
    df = df_base.copy()
    
    # 1. Feature Engineering Specific to Regression
    df['Credit_History_Months'] = df['Credit_History_Age'].apply(convert_age_to_months)
    df.drop(columns=['Credit_History_Age'], inplace=True) # Drop original after conversion
    
    # Encode Credit_Mix (Label Encoding as per original script)
    df['Credit_Mix'] = df['Credit_Mix'].replace({'_': np.nan})
    le = LabelEncoder()
    non_null_mix = df['Credit_Mix'].dropna().unique()
    le.fit(non_null_mix.astype(str))
    
    # Handle NaN for Credit Mix before transform to avoid errors
    df['Credit_Mix'] = df['Credit_Mix'].astype(str)
    df['Credit_Mix_Encoded'] = le.fit_transform(df['Credit_Mix'])
    
    # Impute missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    # 2. Target Generation (Numerical Score)
    print("Generating numerical target values...")
    df['Numerical_Credit_Score'] = df['Credit_Score'].apply(
        lambda x: int((300 + 580)/2) if x == 'Poor' 
        else int((580 + 740)/2) if x == 'Standard'
        else int((670 + 851)/2) if x == 'Good'
        else 500
    )
    
    # Select features
    # FIX: numeric_cols ALREADY includes 'Credit_History_Months' and 'Credit_Mix_Encoded' because they are numeric.
    # We remove the explicit addition of these columns to avoid Duplicates.
    features = [c for c in numeric_cols if c not in ['Numerical_Credit_Score']]
    
    # Filter to ensure columns exist (redundant if using numeric_cols directly, but safe)
    features = [c for c in features if c in df.columns]
    
    # Explicitly remove duplicates to prevent XGBoost "DataFrame has no attribute dtype" error
    features = list(dict.fromkeys(features))

    X = df[features]
    y = df['Numerical_Credit_Score']
    
    print(f"Training Regressor with {X.shape[1]} features: {list(X.columns)}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 3. Train XGBRegressor
    xgb_reg = XGBRegressor(
        objective='reg:squarederror',
        n_estimators=NUM_TREES,
        learning_rate=LEARNING_RATE,
        max_depth=10,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        tree_method='hist',
        random_state=404
    )
    
    print("Training Regressor...")
    xgb_reg.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False
    )
    
    # 4. Evaluation
    y_pred = xgb_reg.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"Regressor RMSE: {rmse:.2f}")
    print(f"Regressor R2: {r2:.4f}")
    
    # 5. Save
    joblib.dump(xgb_reg, REGRESSOR_PATH)
    print(f"Regressor saved to {REGRESSOR_PATH}")

# ==========================================
# CLASSIFICATION PIPELINE
# ==========================================

def train_classifier(df_base):
    print("\n" + "="*40)
    print("STARTING CLASSIFICATION MODEL TRAINING")
    print("="*40)
    
    df = df_base.copy()
    
    # 1. Feature Engineering Specific to Classification
    # Drop Credit_History_Age as per original classification script
    if 'Credit_History_Age' in df.columns:
        df = df.drop(columns=['Credit_History_Age'])
        
    # Ordinal Encoding for Payment_of_Min_Amount
    payment_mapping = {'No': 0, 'NM': 1, 'Yes': 2}
    # Handle NaN or unseen labels safely
    df['Payment_of_Min_Amount'] = df['Payment_of_Min_Amount'].map(payment_mapping).fillna(0)
    
    # Target Mapping
    score_mapping = {'Poor': 0, 'Standard': 1, 'Good': 2}
    df['Credit_Score_Encoded'] = df['Credit_Score'].map(score_mapping)
    df = df.drop(columns=['Credit_Score']) # Drop original string target
    
    # Impute missing values (Median strategy from original script)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    imputation_values = {}
    for col in numeric_cols:
        med_val = df[col].median()
        df[col] = df[col].fillna(med_val)
        imputation_values[col] = med_val

    # One-Hot Encoding for Nominal Columns
    nominal_cols = ['Occupation', 'Credit_Mix', 'Payment_Behaviour']
    # Filter valid columns
    nominal_cols = [c for c in nominal_cols if c in df.columns]
    df_encoded = pd.get_dummies(df, columns=nominal_cols, drop_first=True)
    
    X = df_encoded.drop('Credit_Score_Encoded', axis=1)
    y = df_encoded['Credit_Score_Encoded']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 2. Train XGBClassifier
    xgb_clf = XGBClassifier(
        objective='multi:softmax',
        num_class=3,
        n_estimators=NUM_TREES,
        learning_rate=LEARNING_RATE,
        max_depth=10,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        tree_method='hist',
        random_state=404
    )
    
    print("Training Classifier...")
    xgb_clf.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False
    )
    
    # 3. Evaluation
    y_pred = xgb_clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Classifier Accuracy: {acc:.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Poor', 'Standard', 'Good'])
    print("\nClassification Report:\n", report)
    
    # Save Confusion Matrix Plot
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Poor', 'Standard', 'Good'])
    cm_display.plot()
    plt.title('Classification Confusion Matrix')
    plt.savefig('graphics/confusion_matrix_results.png')
    
    # 4. Save Artifacts & Model
    artifacts = {
        'feature_names': list(X.columns),
        'imputation_values': imputation_values,
        'payment_mapping': payment_mapping,
        'target_mapping': score_mapping
    }
    
    with open(ARTIFACTS_PATH, 'w') as f:
        json.dump(artifacts, f, indent=4)
        
    joblib.dump(xgb_clf, CLASSIFIER_PATH)
    print(f"Classifier saved to {CLASSIFIER_PATH}")
    print(f"Artifacts saved to {ARTIFACTS_PATH}")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Load Data Once
    df_base = load_and_clean_base_data(DATA_FILEPATH)
    
    # 2. Train Regressor
    train_regressor(df_base)
    
    # 3. Train Classifier
    train_classifier(df_base)
    
    # 4. Timing
    end_time = time.perf_counter()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")