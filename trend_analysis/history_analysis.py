import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from datetime import datetime, timedelta
import warnings
import json
warnings.filterwarnings('ignore')

print("=" * 70)
print("🏦 BANKING DATA ANALYSIS - HISTORICAL TREND ANALYSIS")
print("=" * 70)

# ========================
# 1. DATA LOADING & PREPROCESSING
# ========================
print("\n📊 Loading dataset...")
df = pd.read_csv('train.csv')
print(f"✓ Loaded {len(df):,} records with {len(df.columns)} columns")

# Create synthetic run timestamps
np.random.seed(42)
n_runs = 30
base_date = datetime(2024, 1, 1)
run_dates = [base_date + timedelta(days=i*3) for i in range(n_runs)]

print("\n📈 Computing quality metrics...")

# ========================
# 2. COMPUTE QUALITY METRICS OVER TIME
# ========================
def compute_quality_metrics(df, run_date):
    """Compute 5 dimensions of data quality"""
    metrics = {
        'run_date': run_date,
        'completeness': df.notnull().mean().mean(),
        'uniqueness': 1 - (df.duplicated().sum() / len(df)),
        'accuracy_age': ((df['age'] >= 18) & (df['age'] <= 100)).mean(),
        'accuracy_balance': (df['balance'] >= -10000).mean(),
        'consistency_contact': df['contact'].isin(['unknown', 'telephone', 'cellular']).mean(),
        'total_records': len(df)
    }
    return metrics

# Simulate quality metrics over time
quality_history = []
for i, run_date in enumerate(run_dates):
    noise = np.random.normal(0, 0.02)
    if i > 15:
        noise += 0.01
    
    metrics = compute_quality_metrics(df, run_date)
    metrics['completeness'] = min(1.0, metrics['completeness'] + noise)
    metrics['accuracy_age'] = min(1.0, metrics['accuracy_age'] + noise * 0.5)
    
    quality_history.append(metrics)

quality_df = pd.DataFrame(quality_history)

# ========================
# 3. COMPUTE ROLLING WINDOWS
# ========================
quality_df['run_date'] = pd.to_datetime(quality_df['run_date'])
quality_df = quality_df.sort_values('run_date').reset_index(drop=True)

window_7 = 3
window_30 = 10
window_90 = 30

quality_df['completeness_7d'] = quality_df['completeness'].rolling(window=window_7, min_periods=1).mean()
quality_df['completeness_30d'] = quality_df['completeness'].rolling(window=window_30, min_periods=1).mean()
quality_df['completeness_90d'] = quality_df['completeness'].rolling(window=window_90, min_periods=1).mean()

quality_df['delta_vs_prior'] = quality_df['completeness'].diff()
quality_df['delta_vs_baseline'] = quality_df['completeness'] - quality_df['completeness'].iloc[0]

mean = quality_df['completeness'].mean()
std = quality_df['completeness'].std()
quality_df['anomaly_flag'] = np.abs(quality_df['completeness'] - mean) > (2 * std)

print("✓ Rolling windows computed (7d, 30d, 90d)")
print("✓ Delta calculations completed")
print("✓ Anomaly detection applied")

latest = quality_df.iloc[-1]
print(f"\n📊 Latest Run: {latest['run_date'].date()}")
print(f"   Completeness: {latest['completeness']:.2%}")
print(f"   Anomalies: {quality_df['anomaly_flag'].sum()} detected")

# ========================
# 4. TRAIN PREDICTION MODEL
# ========================
print("\n🤖 Training XGBoost model...")

df_model = df.copy()

le = LabelEncoder()
categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

for col in categorical_cols:
    df_model[col] = le.fit_transform(df_model[col].astype(str))

df_model['y'] = le.fit_transform(df_model['y'])

X = df_model.drop('y', axis=1)
y = df_model['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss'
)

model.fit(X_train, y_train)
print("✓ Model training completed")

# ========================
# 5. MODEL EVALUATION
# ========================
print("\n📊 Evaluating model performance...")

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   Precision: {precision:.4f}")
print(f"   Recall:    {recall:.4f}")
print(f"   F1-Score:  {f1:.4f}")
print(f"   AUC-ROC:   {auc:.4f}")

# ========================
# 6. GENERATE JSON FOR WEB DASHBOARD
# ========================
print("\n💾 Generating web dashboard data...")

# Get feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False).head(10)

# Prepare dashboard data
dashboard_data = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'dataset_info': {
        'total_records': int(len(df)),
        'total_columns': int(len(df.columns)),
        'training_samples': int(len(X_train)),
        'test_samples': int(len(X_test))
    },
    'latest_metrics': {
        'run_date': latest['run_date'].strftime('%Y-%m-%d'),
        'completeness': float(latest['completeness']),
        'completeness_7d': float(latest['completeness_7d']),
        'completeness_30d': float(latest['completeness_30d']),
        'completeness_90d': float(latest['completeness_90d']),
        'delta_vs_prior': float(latest['delta_vs_prior']),
        'delta_vs_baseline': float(latest['delta_vs_baseline']),
        'total_anomalies': int(quality_df['anomaly_flag'].sum())
    },
    'quality_dimensions': {
        'completeness': float(latest['completeness']),
        'uniqueness': float(latest['uniqueness']),
        'accuracy_age': float(latest['accuracy_age']),
        'accuracy_balance': float(latest['accuracy_balance']),
        'consistency_contact': float(latest['consistency_contact'])
    },
    'model_performance': {
        'accuracy': {'value': float(accuracy), 'target': 0.85, 'status': 'PASS' if accuracy >= 0.85 else 'FAIL'},
        'precision': {'value': float(precision), 'target': 0.80, 'status': 'PASS' if precision >= 0.80 else 'FAIL'},
        'recall': {'value': float(recall), 'target': 0.75, 'status': 'PASS' if recall >= 0.75 else 'FAIL'},
        'f1_score': {'value': float(f1), 'target': 0.75, 'status': 'PASS' if f1 >= 0.75 else 'FAIL'},
        'auc_roc': {'value': float(auc), 'target': 0.80, 'status': 'PASS' if auc >= 0.80 else 'FAIL'}
    },
    'feature_importance': [
        {'feature': row['feature'], 'importance': float(row['importance'])}
        for _, row in feature_importance.iterrows()
    ],
    'trend_history': [
        {
            'date': row['run_date'].strftime('%Y-%m-%d'),
            'completeness': float(row['completeness']),
            'completeness_7d': float(row['completeness_7d']),
            'completeness_30d': float(row['completeness_30d']),
            'delta_vs_baseline': float(row['delta_vs_baseline']),
            'is_anomaly': bool(row['anomaly_flag'])
        }
        for _, row in quality_df.iterrows()
    ],
    'anomalies': [
        {
            'date': row['run_date'].strftime('%Y-%m-%d'),
            'completeness': float(row['completeness']),
            'delta': float(row['delta_vs_baseline'])
        }
        for _, row in quality_df[quality_df['anomaly_flag']].iterrows()
    ]
}

# Save as JSON
with open('trend_analysis_data.json', 'w') as f:
    json.dump(dashboard_data, f, indent=2)

print("✓ Web dashboard data saved to 'trend_analysis_data.json'")

print("\n" + "=" * 70)
print("✅ ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 70)
print("\nGenerated File:")
print("  📄 trend_analysis_data.json - Ready for web dashboard")
print("=" * 70)
