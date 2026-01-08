from flask import Flask, render_template, jsonify, request, send_file
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
import pickle
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
import csv
import json
import pandas as pd
import numpy as np
import joblib
import io
import traceback

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///fraud_alerts.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Load creditcard.csv data into memory
CREDITCARD_DATA = []
CSV_LOADED = False

# Load churn_test.csv data into memory
CHURN_DATA = []
CHURN_CSV_LOADED = False

# Load credit_test.csv data into memory
CREDIT_TEST_DATA = []
CREDIT_TEST_CSV_LOADED = False

# Load cards_data.csv data into memory for quality check
CARDS_DATA = []
CARDS_CSV_LOADED = False

# Load train.csv data into memory for trend analysis (Bank Marketing Dataset)
BANK_MARKETING_DATA = []
BANK_MARKETING_CSV_LOADED = False

def load_creditcard_csv():
    """Load fraud_test.csv file into memory"""
    global CREDITCARD_DATA, CSV_LOADED
    try:
        print("[INFO] Loading fraud_test.csv...")
        with open('fraud_test.csv', 'r') as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                # Convert row to proper types
                transaction = {
                    'index': idx,
                    'Time': float(row['Time']),
                    'Amount': float(row['Amount']),
                    'Class': int(row['Class'])
                }
                # Add V1-V28 features
                for i in range(1, 29):
                    transaction[f'V{i}'] = float(row[f'V{i}'])

                CREDITCARD_DATA.append(transaction)

        CSV_LOADED = True
        print(f"[OK] Loaded {len(CREDITCARD_DATA)} transactions from fraud_test.csv")
    except Exception as e:
        CSV_LOADED = False
        print(f"[ERROR] Failed to load fraud_test.csv: {e}")

# Load churn CSV data
def load_churn_csv():
    """Load churn_test.csv file into memory"""
    global CHURN_DATA, CHURN_CSV_LOADED
    try:
        print("[INFO] Loading churn_test.csv...")
        with open('churn_test.csv', 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert row to proper types (handle float strings like '23.0')
                customer = {
                    'id': int(float(row['id'])),
                    'CustomerId': int(float(row['CustomerId'])),
                    'Surname': row['Surname'],
                    'CreditScore': int(float(row['CreditScore'])),
                    'Geography': row['Geography'],
                    'Gender': row['Gender'],
                    'Age': int(float(row['Age'])),
                    'Tenure': int(float(row['Tenure'])),
                    'Balance': float(row['Balance']),
                    'NumOfProducts': int(float(row['NumOfProducts'])),
                    'HasCrCard': int(float(row['HasCrCard'])),
                    'IsActiveMember': int(float(row['IsActiveMember'])),
                    'EstimatedSalary': float(row['EstimatedSalary'])
                }
                CHURN_DATA.append(customer)

        CHURN_CSV_LOADED = True
        print(f"[OK] Loaded {len(CHURN_DATA)} customers from churn_test.csv")
    except Exception as e:
        CHURN_CSV_LOADED = False
        print(f"[ERROR] Failed to load churn_test.csv: {e}")

# Load credit_test CSV data
def load_credit_test_csv():
    """Load credit_test.csv file into memory"""
    global CREDIT_TEST_DATA, CREDIT_TEST_CSV_LOADED
    try:
        print("[INFO] Loading credit_test.csv...")
        with open('credit_test.csv', 'r') as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                # Store the row as dict with index
                credit_record = {'index': idx}
                credit_record.update(row)
                CREDIT_TEST_DATA.append(credit_record)

        CREDIT_TEST_CSV_LOADED = True
        print(f"[OK] Loaded {len(CREDIT_TEST_DATA)} records from credit_test.csv")
    except Exception as e:
        CREDIT_TEST_CSV_LOADED = False
        print(f"[ERROR] Failed to load credit_test.csv: {e}")

# Load cards_data CSV data
def load_cards_csv():
    """Load cards_data.csv file into memory for quality check"""
    global CARDS_DATA, CARDS_CSV_LOADED
    try:
        print("[INFO] Loading quality_check/cards_data.csv...")
        with open('quality_check/cards_data.csv', 'r') as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                # Store the row as dict with index
                card_record = {'index': idx}
                card_record.update(row)
                CARDS_DATA.append(card_record)

        CARDS_CSV_LOADED = True
        print(f"[OK] Loaded {len(CARDS_DATA)} card records from quality_check/cards_data.csv")
    except Exception as e:
        CARDS_CSV_LOADED = False
        print(f"[ERROR] Failed to load quality_check/cards_data.csv: {e}")

# Load Bank Marketing CSV data
def load_bank_marketing_csv():
    """Load trend_analysis/train.csv file into memory (Bank Marketing Dataset)"""
    global BANK_MARKETING_DATA, BANK_MARKETING_CSV_LOADED
    try:
        print("[INFO] Loading trend_analysis/train.csv...")
        with open('trend_analysis/train.csv', 'r') as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                # Store the row as dict with index
                record = {'index': idx}
                record.update(row)
                BANK_MARKETING_DATA.append(record)

        BANK_MARKETING_CSV_LOADED = True
        print(f"[OK] Loaded {len(BANK_MARKETING_DATA)} bank marketing records from trend_analysis/train.csv")
    except Exception as e:
        BANK_MARKETING_CSV_LOADED = False
        print(f"[ERROR] Failed to load trend_analysis/train.csv: {e}")

# Credit Scoring Models (XGBoost from Credit Scoring folder)
credit_regressor = None
credit_classifier = None
credit_artifacts = None
CREDIT_MODELS_LOADED = False

def load_credit_scoring_models():
    """Load Credit Scoring XGBoost models"""
    global credit_regressor, credit_classifier, credit_artifacts, CREDIT_MODELS_LOADED
    try:
        print("[INFO] Loading Credit Scoring models...")
        credit_regressor = joblib.load('./Credit_Scoring/models/credit_score_regressor.joblib')
        credit_classifier = joblib.load('./Credit_Scoring/models/credit_score_classifier.joblib')
        with open('./Credit_Scoring/models/preprocessing_artifacts.json', 'r') as f:
            credit_artifacts = json.load(f)
        CREDIT_MODELS_LOADED = True
        print(f"[OK] Credit Scoring models loaded successfully")
    except Exception as e:
        CREDIT_MODELS_LOADED = False
        print(f"[ERROR] Failed to load Credit Scoring models: {e}")

# Load CSV data on startup (only once, not on reloader)
if os.environ.get('WERKZEUG_RUN_MAIN') == 'true' or not app.debug:
    load_creditcard_csv()
    load_churn_csv()
    load_credit_test_csv()
    load_cards_csv()
    load_bank_marketing_csv()
    load_credit_scoring_models()

# Try to load trained models (optional)
try:
    with open('fraud_detection_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    MODEL_LOADED = True
    print("[OK] Fraud Detection Model loaded successfully")
except Exception as e:
    MODEL_LOADED = False
    print(f"[WARNING] Could not load fraud model ({e}). Using simulated predictions.")

try:
    with open('churn_xgboost_model.pkl', 'rb') as f:
        churn_model = pickle.load(f)
    CHURN_MODEL_LOADED = True
    print("[OK] Churn Prediction Model loaded successfully")
except Exception as e:
    CHURN_MODEL_LOADED = False
    print(f"[WARNING] Could not load churn model ({e}). Using simulated predictions.")

# Database Model (RTFD-2.2)
class FraudAlert(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    transaction_id = db.Column(db.String(50), unique=True, nullable=False)
    fraud_score = db.Column(db.Float, nullable=False)
    amount = db.Column(db.Float, nullable=False)
    risk_level = db.Column(db.String(20), nullable=False)  # High, Medium, Low
    review_status = db.Column(db.String(20), default='Pending')  # Pending, Reviewed, Resolved
    email_sent = db.Column(db.Boolean, default=False)

    # Store feature values (V1-V28 from PCA)
    v1 = db.Column(db.Float)
    v2 = db.Column(db.Float)
    v3 = db.Column(db.Float)
    v4 = db.Column(db.Float)
    v5 = db.Column(db.Float)
    v6 = db.Column(db.Float)
    v7 = db.Column(db.Float)
    v8 = db.Column(db.Float)
    v9 = db.Column(db.Float)
    v10 = db.Column(db.Float)
    v11 = db.Column(db.Float)
    v12 = db.Column(db.Float)
    v13 = db.Column(db.Float)
    v14 = db.Column(db.Float)
    v15 = db.Column(db.Float)
    v16 = db.Column(db.Float)
    v17 = db.Column(db.Float)
    v18 = db.Column(db.Float)
    v19 = db.Column(db.Float)
    v20 = db.Column(db.Float)
    v21 = db.Column(db.Float)
    v22 = db.Column(db.Float)
    v23 = db.Column(db.Float)
    v24 = db.Column(db.Float)
    v25 = db.Column(db.Float)
    v26 = db.Column(db.Float)
    v27 = db.Column(db.Float)
    v28 = db.Column(db.Float)

# Initialize database
with app.app_context():
    db.create_all()
    print("[OK] Database initialized")

# Email notification function
def send_email_alert(transaction_id, fraud_score, amount, risk_level):
    """Send email notification for high-risk fraud cases"""
    try:
        sender_email = os.getenv('EMAIL_SENDER', 'fraud-alert@bank.com')
        receiver_email = os.getenv('EMAIL_RECEIVER', 'security@bank.com')
        password = os.getenv('EMAIL_PASSWORD', '')

        if not password:
            print("[WARNING] Email password not configured. Skipping email notification.")
            return False

        message = MIMEMultipart()
        message['From'] = sender_email
        message['To'] = receiver_email
        message['Subject'] = f'HIGH RISK FRAUD ALERT - {transaction_id}'

        body = f"""
        FRAUD ALERT NOTIFICATION

        Transaction ID: {transaction_id}
        Fraud Score: {fraud_score:.4f}
        Amount: ${amount:,.2f}
        Risk Level: {risk_level}
        Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

        IMMEDIATE ACTION REQUIRED
        Please review this transaction immediately.
        """

        message.attach(MIMEText(body, 'plain'))

        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, password)
            server.send_message(message)

        return True
    except Exception as e:
        print(f"[ERROR] Error sending email: {e}")
        return False

# Alert Generation System (RTFD-2.1)
def generate_fraud_alert(transaction_id, features, amount):
    """Generate fraud alert with scoring"""
    if MODEL_LOADED:
        try:
            # Use real ML model
            features_array = [[features[i] for i in range(28)]]
            features_scaled = scaler.transform(features_array)
            fraud_prob = model.predict_proba(features_scaled)[0][1]
        except Exception as e:
            print(f"[ERROR] Model prediction error: {e}. Using simulated prediction.")
            fraud_prob = simulate_fraud_score(features, amount)
    else:
        # Simulated prediction for demo
        fraud_prob = simulate_fraud_score(features, amount)

    # Determine risk level
    if fraud_prob >= 0.7:
        risk_level = 'High'
    elif fraud_prob >= 0.4:
        risk_level = 'Medium'
    else:
        risk_level = 'Low'

    # Create alert in database
    alert = FraudAlert(
        transaction_id=transaction_id,
        fraud_score=fraud_prob,
        amount=amount,
        risk_level=risk_level,
        **{f'v{i+1}': features[i] for i in range(28)}
    )

    db.session.add(alert)
    db.session.commit()

    # Send email for high-risk cases
    if risk_level == 'High' and not alert.email_sent:
        email_sent = send_email_alert(transaction_id, fraud_prob, amount, risk_level)
        if email_sent:
            alert.email_sent = True
            db.session.commit()

    return alert

def simulate_fraud_score(features, amount):
    """Simulate fraud score for demo purposes"""
    import random
    # Simple heuristic based on amount and feature variance
    score = 0.0

    # High amounts are more suspicious
    if amount > 5000:
        score += 0.3
    elif amount > 2000:
        score += 0.2
    elif amount > 1000:
        score += 0.1

    # Check for unusual feature patterns
    feature_sum = sum(abs(f) for f in features)
    if feature_sum > 20:
        score += 0.3
    elif feature_sum > 10:
        score += 0.2

    # Add some randomness
    score += random.uniform(0, 0.3)

    return min(score, 1.0)

def predict_churn_probability(customer_data):
    """Predict churn probability for a customer"""
    if CHURN_MODEL_LOADED:
        try:
            # Prepare features in the correct order
            features = [
                float(customer_data['CreditScore']),
                float(customer_data['Age']),
                int(customer_data['Tenure']),
                float(customer_data['Balance']),
                int(customer_data['NumOfProducts']),
                float(customer_data['HasCrCard']),
                float(customer_data['IsActiveMember']),
                float(customer_data['EstimatedSalary']),
                1.0 if customer_data['Geography'] == 'Germany' else 0.0,
                1.0 if customer_data['Geography'] == 'Spain' else 0.0,
                1.0 if customer_data['Gender'] == 'Male' else 0.0
            ]

            # Predict using loaded model
            churn_prob = churn_model.predict_proba([features])[0][1]
            return churn_prob
        except Exception as e:
            print(f"[ERROR] Churn model prediction error: {e}. Using simulated prediction.")
            return simulate_churn_score(customer_data)
    else:
        # Simulated prediction for demo
        return simulate_churn_score(customer_data)

def simulate_churn_score(customer_data):
    """Simulate churn score for demo purposes"""
    import random
    score = 0.0

    # Age factor: very young or old customers might churn more
    age = customer_data['Age']
    if age < 25 or age > 60:
        score += 0.2

    # Balance factor: low balance = higher churn risk
    balance = customer_data['Balance']
    if balance < 50000:
        score += 0.2
    elif balance == 0:
        score += 0.3

    # Product count: having only 1 product = higher risk
    if customer_data['NumOfProducts'] == 1:
        score += 0.15

    # Activity: inactive members = higher risk
    if customer_data['IsActiveMember'] == 0:
        score += 0.25

    # Credit score: low score = higher risk
    if customer_data['CreditScore'] < 600:
        score += 0.2

    # Add some randomness
    score += random.uniform(0, 0.2)

    return min(score, 1.0)

# ==========================================
# CREDIT SCORING HELPER FUNCTIONS
# ==========================================

def clean_non_numeric(value):
    """Cleans currency symbols and special chars from a single value or Series."""
    if isinstance(value, pd.Series):
        return pd.to_numeric(value.astype(str).str.replace(r'[$,_!#]', '', regex=True), errors='coerce')

    # For single scalar values
    if value is None:
        return np.nan
    clean_str = str(value).replace('_', '').replace('$', '').replace(',', '').replace('!', '').replace('#', '')
    try:
        return float(clean_str)
    except ValueError:
        return 0.0

def convert_age_to_months(age_str):
    """Parses 'X Years and Y Months' string."""
    if pd.isna(age_str):
        return 0
    try:
        parts = str(age_str).split(' and ')
        years = int(parts[0].split(' ')[0])
        months = int(parts[1].split(' ')[0])
        return years * 12 + months
    except:
        return 0

def preprocess_credit_single_input(data_dict, is_regression):
    """Prepares a single JSON input for model inference."""
    df = pd.DataFrame([data_dict])

    # 1. Clean fields with special characters (currency symbols, etc.)
    cols_to_clean = [
        'Age', 'Annual_Income', 'Num_of_Delayed_Payment',
        'Changed_Credit_Limit', 'Outstanding_Debt',
        'Amount_invested_monthly', 'Monthly_Balance'
    ]
    for col in cols_to_clean:
        if col in df.columns:
            df[col] = df[col].apply(clean_non_numeric)
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 2. Ensure all other numeric columns are properly typed
    numeric_cols = [
        'Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan',
        'Delay_from_due_date', 'Num_Credit_Inquiries', 'Credit_Utilization_Ratio',
        'Total_EMI_per_month', 'Monthly_Inhand_Salary'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Remove Monthly_Inhand_Salary if exists (not used in model)
    if 'Monthly_Inhand_Salary' in df.columns:
        df.drop(columns=['Monthly_Inhand_Salary'], inplace=True)

    if 'Age' in df.columns:
        df['Age'] = df['Age'].clip(lower=18, upper=100)

    # 3. Feature Engineering
    if 'Credit_History_Age' in df.columns:
        if is_regression:
            df['Credit_History_Months'] = df['Credit_History_Age'].apply(convert_age_to_months)
        df.drop(columns=['Credit_History_Age'], inplace=True)

    # 4. Specific Path Logic
    if is_regression:
        # Credit Mix Encoding
        mix_mapping = {'Bad': 0, 'Good': 1, 'Standard': 2}
        df['Credit_Mix_Encoded'] = df['Credit_Mix'].map(mix_mapping).fillna(1)

        # Align columns with Regressor
        if hasattr(credit_regressor, 'feature_names_in_'):
            required_cols = credit_regressor.feature_names_in_
        else:
            required_cols = df.select_dtypes(include=[np.number]).columns

        # Ensure all columns exist
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0

        return df[required_cols]

    else: # Classification
        # Payment Min Amount Map
        payment_map = credit_artifacts['payment_mapping']
        df['Payment_of_Min_Amount'] = df['Payment_of_Min_Amount'].map(payment_map).fillna(0)

        # One-Hot Encoding
        nominal_cols = ['Occupation', 'Credit_Mix', 'Payment_Behaviour']
        df_encoded = pd.get_dummies(df, columns=nominal_cols, drop_first=True)

        # Align with Training Features
        train_features = credit_artifacts['feature_names']
        df_final = pd.DataFrame(0, index=np.arange(len(df_encoded)), columns=train_features)

        for col in df_encoded.columns:
            if col in df_final.columns:
                df_final[col] = df_encoded[col].values

        return df_final

def process_credit_batch_dataframe(df_raw):
    """Prepares a bulk DataFrame for inference."""
    df = df_raw.copy()

    # 1. Clean Numeric - Only clean fields with special characters (currency symbols, etc.)
    numeric_features = [
        'Age', 'Annual_Income', 'Num_Bank_Accounts', 'Num_Credit_Card',
        'Interest_Rate', 'Delay_from_due_date', 'Num_of_Delayed_Payment',
        'Changed_Credit_Limit', 'Num_Credit_Inquiries', 'Outstanding_Debt',
        'Credit_Utilization_Ratio', 'Total_EMI_per_month', 'Amount_invested_monthly',
        'Monthly_Balance', 'Num_of_Loan', 'Monthly_Inhand_Salary'
    ]
    for col in numeric_features:
        if col in df.columns:
            # Clean and convert to numeric
            df[col] = clean_non_numeric(df[col])
            # Ensure the column is numeric type
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Remove Monthly_Inhand_Salary if exists (not used in model)
    if 'Monthly_Inhand_Salary' in df.columns:
        df.drop(columns=['Monthly_Inhand_Salary'], inplace=True)

    if 'Age' in df.columns:
        df['Age'] = df['Age'].clip(lower=18, upper=100)

    # --- REGRESSION PREDICTION ---
    df_reg = df.copy()
    if 'Credit_History_Age' in df_reg.columns:
        df_reg['Credit_History_Months'] = df_reg['Credit_History_Age'].apply(convert_age_to_months)
        df_reg.drop(columns=['Credit_History_Age'], inplace=True)

    # Mix Encoding
    mix_map = {'Bad': 0, 'Good': 1, 'Standard': 2}
    if 'Credit_Mix' in df_reg.columns:
        df_reg['Credit_Mix_Encoded'] = df_reg['Credit_Mix'].map(mix_map)
    else:
        df_reg['Credit_Mix_Encoded'] = 0

    # Fill NaNs
    num_cols = df_reg.select_dtypes(include=[np.number]).columns
    df_reg[num_cols] = df_reg[num_cols].fillna(0)

    # Align Regressor Features
    reg_features = credit_regressor.feature_names_in_ if hasattr(credit_regressor, 'feature_names_in_') else []
    for col in reg_features:
        if col not in df_reg.columns:
            df_reg[col] = 0

    if len(reg_features) > 0:
        df_raw['Predicted_Score'] = credit_regressor.predict(df_reg[reg_features])

    # --- CLASSIFICATION PREDICTION ---
    df_clf = df.copy()
    if 'Credit_History_Age' in df_clf.columns:
        df_clf.drop(columns=['Credit_History_Age'], inplace=True)

    if 'Payment_of_Min_Amount' in df_clf.columns:
        df_clf['Payment_of_Min_Amount'] = df_clf['Payment_of_Min_Amount'].map(credit_artifacts['payment_mapping']).fillna(0)

    # Impute
    for col, val in credit_artifacts.get('imputation_values', {}).items():
        if col in df_clf.columns:
            df_clf[col] = df_clf[col].fillna(val)

    # Encoding
    nominal_cols = ['Occupation', 'Credit_Mix', 'Payment_Behaviour']
    present_nominal = [c for c in nominal_cols if c in df_clf.columns]
    df_clf_encoded = pd.get_dummies(df_clf, columns=present_nominal, drop_first=True)

    # Align Classification Features
    clf_features = credit_artifacts['feature_names']
    df_aligned_clf = pd.DataFrame(0, index=df_clf_encoded.index, columns=clf_features)
    common_cols = [c for c in df_clf_encoded.columns if c in clf_features]
    df_aligned_clf[common_cols] = df_clf_encoded[common_cols]

    # Predict
    pred_ids = credit_classifier.predict(df_aligned_clf)

    # Map back
    target_map_inv = {v: k for k, v in credit_artifacts['target_mapping'].items()}
    df_raw['Predicted_Credit_Bracket'] = [target_map_inv.get(x, "Unknown") for x in pred_ids]

    return df_raw

# Routes
@app.route('/')
def index():
    return render_template('fraud_detection.html')

# Dashboard Statistics API (RTFD-2.3)
@app.route('/api/statistics')
def get_statistics():
    """Get real-time fraud detection statistics"""
    # Total transactions (last 24 hours)
    last_24h = datetime.utcnow() - timedelta(hours=24)
    total_transactions = FraudAlert.query.filter(FraudAlert.timestamp >= last_24h).count()

    # Fraud alerts by risk level
    high_risk = FraudAlert.query.filter(
        FraudAlert.risk_level == 'High',
        FraudAlert.timestamp >= last_24h
    ).count()

    medium_risk = FraudAlert.query.filter(
        FraudAlert.risk_level == 'Medium',
        FraudAlert.timestamp >= last_24h
    ).count()

    low_risk = FraudAlert.query.filter(
        FraudAlert.risk_level == 'Low',
        FraudAlert.timestamp >= last_24h
    ).count()

    # Calculate fraud rate
    fraud_count = high_risk + medium_risk
    fraud_rate = (fraud_count / total_transactions * 100) if total_transactions > 0 else 0

    # Get trend data (last 7 days)
    trend_data = []
    for i in range(7):
        date = datetime.utcnow() - timedelta(days=6-i)
        day_start = date.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=1)

        day_fraud = FraudAlert.query.filter(
            FraudAlert.timestamp >= day_start,
            FraudAlert.timestamp < day_end,
            FraudAlert.risk_level.in_(['High', 'Medium'])
        ).count()

        trend_data.append({
            'date': date.strftime('%Y-%m-%d'),
            'count': day_fraud
        })

    return jsonify({
        'total_transactions': total_transactions,
        'fraud_rate': round(fraud_rate, 2),
        'high_risk_alerts': high_risk,
        'medium_risk_alerts': medium_risk,
        'low_risk_alerts': low_risk,
        'trend_data': trend_data
    })

# Alert Filtering API (RTFD-2.4)
@app.route('/api/alerts')
def get_alerts():
    """Get filtered fraud alerts with pagination"""
    # Filters
    risk_level = request.args.get('risk_level', None)
    status = request.args.get('status', None)
    date_from = request.args.get('date_from', None)
    date_to = request.args.get('date_to', None)
    min_amount = request.args.get('min_amount', None)
    max_amount = request.args.get('max_amount', None)

    # Pagination
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)

    # Build query
    query = FraudAlert.query

    if risk_level:
        query = query.filter(FraudAlert.risk_level == risk_level)

    if status:
        query = query.filter(FraudAlert.review_status == status)

    if date_from:
        date_from_obj = datetime.fromisoformat(date_from)
        query = query.filter(FraudAlert.timestamp >= date_from_obj)

    if date_to:
        date_to_obj = datetime.fromisoformat(date_to)
        query = query.filter(FraudAlert.timestamp <= date_to_obj)

    if min_amount:
        query = query.filter(FraudAlert.amount >= float(min_amount))

    if max_amount:
        query = query.filter(FraudAlert.amount <= float(max_amount))

    # Execute with pagination
    alerts = query.order_by(FraudAlert.timestamp.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )

    return jsonify({
        'alerts': [{
            'id': alert.id,
            'transaction_id': alert.transaction_id,
            'timestamp': alert.timestamp.isoformat(),
            'fraud_score': round(alert.fraud_score, 4),
            'amount': alert.amount,
            'risk_level': alert.risk_level,
            'review_status': alert.review_status,
            'email_sent': alert.email_sent
        } for alert in alerts.items],
        'total': alerts.total,
        'pages': alerts.pages,
        'current_page': page
    })

# Fraud Prediction API
@app.route('/api/predict', methods=['POST'])
def predict_fraud():
    """Predict fraud for a new transaction"""
    data = request.json

    # Extract features (28 PCA components + amount)
    features = [
        data.get(f'V{i}', 0.0) for i in range(1, 29)
    ]

    amount = data.get('Amount', 0.0)
    transaction_id = data.get('transaction_id', f'TXN-{int(datetime.now().timestamp())}')

    # Generate alert
    alert = generate_fraud_alert(transaction_id, features, amount)

    return jsonify({
        'transaction_id': alert.transaction_id,
        'fraud_score': round(alert.fraud_score, 4),
        'risk_level': alert.risk_level,
        'amount': alert.amount,
        'timestamp': alert.timestamp.isoformat(),
        'email_sent': alert.email_sent
    })

# Update alert status
@app.route('/api/alerts/<int:alert_id>/status', methods=['PUT'])
def update_alert_status(alert_id):
    """Update review status of an alert"""
    alert = FraudAlert.query.get_or_404(alert_id)
    data = request.json

    new_status = data.get('status')
    if new_status in ['Pending', 'Reviewed', 'Resolved']:
        alert.review_status = new_status
        db.session.commit()
        return jsonify({'success': True, 'status': new_status})

    return jsonify({'success': False, 'error': 'Invalid status'}), 400

# Reset all data
@app.route('/api/reset', methods=['POST'])
def reset_data():
    """Reset all fraud alerts and dashboard data"""
    try:
        # Delete all fraud alerts from database
        FraudAlert.query.delete()
        db.session.commit()
        print("[INFO] All fraud alerts have been deleted")
        return jsonify({'success': True, 'message': 'All data has been reset'})
    except Exception as e:
        print(f"[ERROR] Failed to reset data: {e}")
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

# Get transaction by index from CSV
@app.route('/api/csv/transaction/<int:transaction_index>')
def get_csv_transaction(transaction_index):
    """Get transaction data from CSV by index"""
    if not CSV_LOADED:
        return jsonify({'success': False, 'error': 'CSV data not loaded'}), 500

    if transaction_index < 0 or transaction_index >= len(CREDITCARD_DATA):
        return jsonify({'success': False, 'error': 'Invalid transaction index'}), 404

    transaction = CREDITCARD_DATA[transaction_index]

    # Extract features for prediction
    features = [transaction[f'V{i}'] for i in range(1, 29)]
    amount = transaction['Amount']
    time = transaction['Time']

    # Generate fraud alert using Time as transaction ID (with index for uniqueness)
    transaction_id = f'TIME-{int(time)}-{transaction_index}'
    alert = generate_fraud_alert(transaction_id, features, amount)

    return jsonify({
        'success': True,
        'transaction_index': transaction_index,
        'transaction_id': transaction_id,
        'time': transaction['Time'],
        'amount': amount,
        'actual_class': transaction['Class'],
        'fraud_score': round(alert.fraud_score, 4),
        'risk_level': alert.risk_level,
        'features': {f'V{i}': transaction[f'V{i}'] for i in range(1, 29)}
    })

# Get transaction list for visualization
@app.route('/api/csv/transactions')
def get_csv_transactions():
    """Get paginated transaction list from CSV"""
    if not CSV_LOADED:
        return jsonify({'success': False, 'error': 'CSV data not loaded'}), 500

    # Pagination parameters
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 100, type=int)

    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page

    transactions = CREDITCARD_DATA[start_idx:end_idx]

    return jsonify({
        'success': True,
        'transactions': [{
            'index': t['index'],
            'time': t['Time'],
            'amount': t['Amount'],
            'class': t['Class']
        } for t in transactions],
        'total': len(CREDITCARD_DATA),
        'page': page,
        'per_page': per_page
    })

# Get CSV statistics
@app.route('/api/csv/stats')
def get_csv_stats():
    """Get statistics about the CSV dataset"""
    if not CSV_LOADED:
        return jsonify({'success': False, 'error': 'CSV data not loaded'}), 500

    fraud_count = sum(1 for t in CREDITCARD_DATA if t['Class'] == 1)
    legitimate_count = len(CREDITCARD_DATA) - fraud_count

    # Get time range
    min_time = min(t['Time'] for t in CREDITCARD_DATA) if CREDITCARD_DATA else 0
    max_time = max(t['Time'] for t in CREDITCARD_DATA) if CREDITCARD_DATA else 0

    return jsonify({
        'success': True,
        'total_transactions': len(CREDITCARD_DATA),
        'fraud_transactions': fraud_count,
        'legitimate_transactions': legitimate_count,
        'fraud_rate': round((fraud_count / len(CREDITCARD_DATA)) * 100, 2),
        'min_time': round(min_time),
        'max_time': round(max_time)
    })

# Get transactions by time range
@app.route('/api/csv/transactions_by_time')
def get_transactions_by_time():
    """Get transactions within a time range"""
    if not CSV_LOADED:
        return jsonify({'success': False, 'error': 'CSV data not loaded'}), 500

    start_time = request.args.get('start_time', type=float, default=0)
    end_time = request.args.get('end_time', type=float)
    max_count = request.args.get('max_count', type=int, default=100)

    # Filter transactions by time range
    filtered = [t for t in CREDITCARD_DATA if t['Time'] >= start_time and (end_time is None or t['Time'] <= end_time)]

    # Limit to max_count
    limited = filtered[:max_count]

    return jsonify({
        'success': True,
        'transactions': [{'index': t['index'], 'time': t['Time']} for t in limited],
        'total_in_range': len(filtered),
        'returned_count': len(limited)
    })

# Churn Prediction API Endpoints
@app.route('/api/churn/stats')
def get_churn_stats():
    """Get statistics about churn dataset"""
    if not CHURN_CSV_LOADED:
        return jsonify({'success': False, 'error': 'Churn CSV data not loaded'}), 500

    min_id = min(c['id'] for c in CHURN_DATA) if CHURN_DATA else 0
    max_id = max(c['id'] for c in CHURN_DATA) if CHURN_DATA else 0

    return jsonify({
        'success': True,
        'total_customers': len(CHURN_DATA),
        'min_id': min_id,
        'max_id': max_id
    })

@app.route('/api/churn/customer/<int:customer_id>')
def get_customer_by_id(customer_id):
    """Get customer data by ID and predict churn"""
    if not CHURN_CSV_LOADED:
        return jsonify({'success': False, 'error': 'Churn CSV data not loaded'}), 500

    # Find customer by id
    customer = next((c for c in CHURN_DATA if c['id'] == customer_id), None)

    if not customer:
        return jsonify({'success': False, 'error': 'Customer not found'}), 404

    # Predict churn probability
    churn_prob = predict_churn_probability(customer)

    # Determine risk level
    if churn_prob >= 0.7:
        risk_level = 'High'
    elif churn_prob >= 0.4:
        risk_level = 'Medium'
    else:
        risk_level = 'Low'

    return jsonify({
        'success': True,
        'customer_id': customer['CustomerId'],
        'id': customer['id'],
        'surname': customer['Surname'],
        'geography': customer['Geography'],
        'age': customer['Age'],
        'balance': customer['Balance'],
        'churn_probability': round(churn_prob * 100, 2),
        'risk_level': risk_level
    })

@app.route('/api/churn/predict', methods=['POST'])
def predict_churn():
    """Predict customer churn probability"""
    try:
        data = request.json

        # Prepare customer data
        customer_data = {
            'CreditScore': data.get('CreditScore', 0),
            'Age': data.get('Age', 0),
            'Tenure': data.get('Tenure', 0),
            'Balance': data.get('Balance', 0),
            'NumOfProducts': data.get('NumOfProducts', 0),
            'HasCrCard': data.get('HasCrCard', 0),
            'IsActiveMember': data.get('IsActiveMember', 0),
            'EstimatedSalary': data.get('EstimatedSalary', 0),
            'Geography': data.get('Geography', 'France'),
            'Gender': data.get('Gender', 'Female')
        }

        # Predict churn probability
        churn_prob = predict_churn_probability(customer_data)

        # Determine risk level and strategy
        if churn_prob >= 0.7:
            risk_level = 'High'
            strategy = 'Urgent intervention needed - Offer premium retention package'
        elif churn_prob >= 0.4:
            risk_level = 'Medium'
            strategy = 'Enhanced loyalty program - Schedule customer follow-up'
        else:
            risk_level = 'Low'
            strategy = 'Standard engagement - Maintain regular communication'

        return jsonify({
            'success': True,
            'churn_probability': round(churn_prob * 100, 2),
            'churn_prediction': 1 if churn_prob >= 0.5 else 0,
            'risk_level': risk_level,
            'retention_strategy': strategy
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# Credit Test CSV API Endpoints
@app.route('/api/credit_test/stats')
def get_credit_test_stats():
    """Get statistics about credit_test dataset"""
    if not CREDIT_TEST_CSV_LOADED:
        return jsonify({'success': False, 'error': 'Credit test CSV data not loaded'}), 500

    min_idx = 0
    max_idx = len(CREDIT_TEST_DATA) - 1 if CREDIT_TEST_DATA else 0

    return jsonify({
        'success': True,
        'total_records': len(CREDIT_TEST_DATA),
        'min_index': min_idx,
        'max_index': max_idx
    })

@app.route('/api/credit_test/record/<int:record_index>')
def get_credit_test_record(record_index):
    """Get credit test record by index and predict credit score"""
    if not CREDIT_TEST_CSV_LOADED:
        return jsonify({'success': False, 'error': 'Credit test CSV data not loaded'}), 500

    if record_index < 0 or record_index >= len(CREDIT_TEST_DATA):
        return jsonify({'success': False, 'error': 'Invalid record index'}), 404

    record = CREDIT_TEST_DATA[record_index]

    # Predict using Credit Scoring models if loaded
    if CREDIT_MODELS_LOADED:
        try:
            # Prepare data for prediction
            data_for_prediction = {k: v for k, v in record.items() if k != 'index'}

            # Predict score
            processed_score = preprocess_credit_single_input(data_for_prediction, is_regression=True)
            predicted_score = credit_regressor.predict(processed_score)[0]

            # Predict bracket
            processed_class = preprocess_credit_single_input(data_for_prediction, is_regression=False)
            prediction_idx = credit_classifier.predict(processed_class)[0]
            target_map_inv = {v: k for k, v in credit_artifacts['target_mapping'].items()}
            predicted_bracket = target_map_inv.get(prediction_idx, "Unknown")

            return jsonify({
                'success': True,
                'record_index': record_index,
                'actual_credit_score': record.get('Credit_Score', 'N/A'),
                'predicted_score': float(predicted_score),
                'credit_bracket': predicted_bracket,
                'record_data': record
            })
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"[ERROR] Credit prediction failed for record {record_index}:")
            print(error_details)
            return jsonify({'success': False, 'error': f'Prediction error: {str(e)}'}), 500
    else:
        return jsonify({
            'success': False,
            'error': 'Credit Scoring models not loaded'
        }), 500

# ==========================================
# CREDIT SCORING API ENDPOINTS (XGBoost)
# ==========================================

@app.route('/api/credit/predict/score', methods=['POST'])
def predict_credit_score_single():
    """Predicts numeric credit score for a single user."""
    if not CREDIT_MODELS_LOADED or not credit_regressor:
        return jsonify({'success': False, 'error': 'Credit Scoring models not loaded'}), 500

    try:
        data = request.json
        processed_data = preprocess_credit_single_input(data, is_regression=True)
        prediction = credit_regressor.predict(processed_data)

        return jsonify({
            'success': True,
            'predicted_score': float(prediction[0]),
            'model_type': 'XGBRegressor'
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/credit/predict/class', methods=['POST'])
def predict_credit_class_single():
    """Predicts credit bracket (Good/Standard/Poor) for a single user."""
    if not CREDIT_MODELS_LOADED or not credit_classifier:
        return jsonify({'success': False, 'error': 'Credit Scoring models not loaded'}), 500

    try:
        data = request.json
        processed_data = preprocess_credit_single_input(data, is_regression=False)
        prediction_idx = credit_classifier.predict(processed_data)[0]

        # Map ID back to Label
        target_map = credit_artifacts['target_mapping']
        reverse_map = {v: k for k, v in target_map.items()}
        label = reverse_map.get(prediction_idx, "Unknown")

        # Confidence
        probs = credit_classifier.predict_proba(processed_data)[0]
        confidence = float(np.max(probs))

        return jsonify({
            'success': True,
            'credit_bracket': label,
            'confidence': confidence,
            'model_type': 'XGBClassifier'
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/credit/predict/batch', methods=['POST'])
def predict_credit_batch():
    """Accepts a CSV file, appends predictions, and returns a CSV download."""
    if not CREDIT_MODELS_LOADED or not credit_regressor or not credit_classifier:
        return jsonify({'success': False, 'error': 'Credit Scoring models not loaded'}), 500

    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'}), 400

    file = request.files['file']

    if not file.filename.endswith('.csv'):
        return jsonify({'success': False, 'error': 'Only CSV files are allowed.'}), 400

    try:
        # Read File
        contents = file.read()

        # Try multiple encodings
        df = None
        for enc in ['utf-8', 'latin1', 'cp1252']:
            try:
                df = pd.read_csv(io.BytesIO(contents), encoding=enc, low_memory=False)
                break
            except:
                continue

        if df is None:
            return jsonify({'success': False, 'error': 'Could not decode CSV file.'}), 400

        # Process
        result_df = process_credit_batch_dataframe(df)

        # Return CSV
        stream = io.StringIO()
        result_df.to_csv(stream, index=False)
        stream.seek(0)

        return send_file(
            io.BytesIO(stream.getvalue().encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name='predictions_result.csv'
        )

    except Exception as e:
        print(f"Batch Processing Error: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Processing failed: {str(e)}'}), 500

# Quality Check Endpoints
@app.route('/api/quality_check/stats')
def quality_check_stats():
    """Get statistics about cards_data.csv for quality check"""
    if not CARDS_CSV_LOADED:
        return jsonify({'success': False, 'error': 'Cards data not loaded'}), 500

    try:
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(CARDS_DATA)

        # Calculate statistics
        total_records = len(df)

        # Field validation stats
        valid_records = 0
        missing_values = {}
        invalid_formats = {}

        # Check each field
        required_fields = ['id', 'client_id', 'card_brand', 'card_type', 'card_number',
                          'expires', 'cvv', 'has_chip', 'credit_limit', 'acct_open_date']

        for field in required_fields:
            if field in df.columns:
                missing = df[field].isna().sum()
                missing_values[field] = int(missing)

        # Calculate valid records percentage
        total_missing = sum(missing_values.values())
        valid_records_pct = ((total_records * len(required_fields) - total_missing) /
                            (total_records * len(required_fields))) * 100

        # Consistency checks
        consistency_checks = {
            'schema_validation': 'Pass',
            'data_type_check': 'Pass',
            'range_validation': 'Pass',
            'null_check': 'Pass'
        }

        # Check schema - all required fields exist
        missing_fields = [f for f in required_fields if f not in df.columns]
        if missing_fields:
            consistency_checks['schema_validation'] = 'Fail'

        # Check for nulls
        total_nulls = df.isnull().sum().sum()
        if total_nulls > 0:
            consistency_checks['null_check'] = 'Warning'

        # Validate data types
        try:
            # Check if credit_limit can be converted to numeric
            if 'credit_limit' in df.columns:
                df['credit_limit'].str.replace('$', '').str.replace(',', '').astype(float)
            # Check if cvv is numeric
            if 'cvv' in df.columns:
                df['cvv'].astype(int)
        except:
            consistency_checks['data_type_check'] = 'Warning'

        # Check ranges (CVV should be 3-4 digits, etc)
        try:
            if 'cvv' in df.columns:
                cvv_valid = df['cvv'].astype(str).str.len().between(3, 4).all()
                if not cvv_valid:
                    consistency_checks['range_validation'] = 'Warning'
        except:
            consistency_checks['range_validation'] = 'Warning'

        # Check for data quality issues
        error_rate = 100 - valid_records_pct

        return jsonify({
            'success': True,
            'total_records': total_records,
            'valid_records_pct': round(valid_records_pct, 1),
            'error_rate': round(error_rate, 1),
            'missing_values': missing_values,
            'consistency_checks': consistency_checks,
            'field_quality': {
                'card_number': round(100 - (missing_values.get('card_number', 0) / total_records * 100), 1),
                'client_id': round(100 - (missing_values.get('client_id', 0) / total_records * 100), 1),
                'expires': round(100 - (missing_values.get('expires', 0) / total_records * 100), 1),
                'credit_limit': round(100 - (missing_values.get('credit_limit', 0) / total_records * 100), 1),
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/quality_check/record/<int:record_index>')
def get_quality_check_record(record_index):
    """Get a specific card record by index for quality check"""
    if not CARDS_CSV_LOADED:
        return jsonify({'success': False, 'error': 'Cards data not loaded'}), 500

    if record_index < 0 or record_index >= len(CARDS_DATA):
        return jsonify({'success': False, 'error': 'Invalid record index'}), 400

    return jsonify({
        'success': True,
        'record': CARDS_DATA[record_index]
    })

# Trend Analysis Endpoint
@app.route('/api/trend_analysis/stats')
def trend_analysis_stats():
    """Get trend analysis statistics from train.csv (Bank Marketing Dataset)"""
    if not BANK_MARKETING_CSV_LOADED:
        return jsonify({'success': False, 'error': 'Bank marketing data not loaded'}), 500

    try:
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(BANK_MARKETING_DATA)

        total_contacts = len(df)

        # Analyze success rate (y = yes means customer subscribed to term deposit)
        subscribed = df[df['y'] == 'yes']
        total_subscribed = len(subscribed)
        subscription_rate = (total_subscribed / total_contacts * 100) if total_contacts > 0 else 0

        # Group by month to find peak month
        month_order = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                      'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}

        contacts_by_month = df.groupby('month').size()
        peak_month = contacts_by_month.idxmax() if len(contacts_by_month) > 0 else 'may'
        avg_contacts_per_month = contacts_by_month.mean() if len(contacts_by_month) > 0 else 0

        # Bank Term Deposit Trends - group by month
        months_in_data = sorted(df['month'].unique(), key=lambda x: month_order.get(x, 0))

        # Get last month and second to last month data
        if len(months_in_data) >= 2:
            last_month = months_in_data[-1]
            second_last_month = months_in_data[-2]
            deposits_last_month = len(df[df['month'] == last_month])
            deposits_second_last = len(df[df['month'] == second_last_month])
            subscribed_last_month = len(df[(df['month'] == last_month) & (df['y'] == 'yes')])
            subscribed_second_last = len(df[(df['month'] == second_last_month) & (df['y'] == 'yes')])
        else:
            last_month = months_in_data[0] if len(months_in_data) > 0 else 'may'
            deposits_last_month = len(df[df['month'] == last_month])
            deposits_second_last = deposits_last_month
            subscribed_last_month = len(df[(df['month'] == last_month) & (df['y'] == 'yes')])
            subscribed_second_last = subscribed_last_month

        # Calculate volumes by periods
        total_months = len(months_in_data)
        volume_this_month = deposits_last_month
        volume_last_month = deposits_second_last
        volume_last_3_months = len(df[df['month'].isin(months_in_data[-3:])]) if total_months >= 3 else total_contacts
        volume_last_6_months = len(df[df['month'].isin(months_in_data[-6:])]) if total_months >= 6 else total_contacts
        volume_ytd = total_contacts

        # Success Rate Distribution by job type
        job_success = df.groupby('job').apply(lambda x: (x['y'] == 'yes').sum() / len(x) * 100 if len(x) > 0 else 0)
        high_success_jobs = job_success[job_success > 40].index.tolist()
        medium_success_jobs = job_success[(job_success >= 20) & (job_success <= 40)].index.tolist()
        low_success_jobs = job_success[job_success < 20].index.tolist()

        high_success_count = len(df[df['job'].isin(high_success_jobs)])
        medium_success_count = len(df[df['job'].isin(medium_success_jobs)])
        low_success_count = len(df[df['job'].isin(low_success_jobs)])

        total_analyzed = high_success_count + medium_success_count + low_success_count
        high_success_pct = (high_success_count / total_analyzed * 100) if total_analyzed > 0 else 0
        medium_success_pct = (medium_success_count / total_analyzed * 100) if total_analyzed > 0 else 0
        low_success_pct = (low_success_count / total_analyzed * 100) if total_analyzed > 0 else 0

        # Calculate percentage changes
        change_month = ((volume_this_month - volume_last_month) / volume_last_month * 100) if volume_last_month > 0 else 0
        avg_per_month_3m = volume_last_3_months / min(3, total_months) if volume_last_3_months > 0 else 0
        avg_per_month_6m = volume_last_6_months / min(6, total_months) if volume_last_6_months > 0 else 0
        change_3m = ((avg_per_month_3m - volume_last_month) / volume_last_month * 100) if volume_last_month > 0 else 0
        change_6m = ((avg_per_month_6m - avg_per_month_3m) / avg_per_month_3m * 100) if avg_per_month_3m > 0 else 0
        subscription_trend = ((subscribed_last_month - subscribed_second_last) / subscribed_second_last * 100) if subscribed_second_last > 0 else 0

        return jsonify({
            'success': True,
            'total_contacts': total_contacts,
            'total_subscribed': total_subscribed,
            'subscription_rate': round(subscription_rate, 1),
            'campaign_trends': {
                'peak_month': peak_month.capitalize(),
                'avg_per_month': round(avg_contacts_per_month, 0),
                'subscription_trend': f"{subscription_trend:+.0f}%"
            },
            'deposit_trends': {
                'this_month': volume_this_month,
                'last_month': volume_last_month,
                'last_3_months': volume_last_3_months,
                'last_6_months': volume_last_6_months,
                'ytd': volume_ytd,
                'changes': {
                    'this_month': f"{change_month:+.0f}%",
                    'last_3_months': f"{change_3m:+.0f}%",
                    'last_6_months': f"{change_6m:+.0f}%",
                    'ytd': f"{subscription_trend:+.0f}%"
                }
            },
            'success_distribution': {
                'high_success': {
                    'count': high_success_count,
                    'percentage': round(high_success_pct, 0)
                },
                'medium_success': {
                    'count': medium_success_count,
                    'percentage': round(medium_success_pct, 0)
                },
                'low_success': {
                    'count': low_success_count,
                    'percentage': round(low_success_pct, 0)
                }
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("FRAUD DETECTION DASHBOARD SERVER")
    print("="*60)
    print(f"Database: SQLite (fraud_alerts.db)")
    print(f"Fraud Model: {'Loaded' if MODEL_LOADED else 'Simulated (for demo)'}")
    print(f"Churn Model: {'Loaded' if CHURN_MODEL_LOADED else 'Simulated (for demo)'}")
    print(f"Credit Scoring Model: {'Loaded (XGBoost)' if CREDIT_MODELS_LOADED else 'Not loaded'}")
    print(f"Fraud CSV: {'Loaded ({} txns)'.format(len(CREDITCARD_DATA)) if CSV_LOADED else 'Not loaded'}")
    print(f"Churn CSV: {'Loaded ({} customers)'.format(len(CHURN_DATA)) if CHURN_CSV_LOADED else 'Not loaded'}")
    print(f"Credit Test CSV: {'Loaded ({} records)'.format(len(CREDIT_TEST_DATA)) if CREDIT_TEST_CSV_LOADED else 'Not loaded'}")
    print(f"Cards Data CSV: {'Loaded ({} cards)'.format(len(CARDS_DATA)) if CARDS_CSV_LOADED else 'Not loaded'}")
    print(f"Bank Marketing CSV: {'Loaded ({} contacts)'.format(len(BANK_MARKETING_DATA)) if BANK_MARKETING_CSV_LOADED else 'Not loaded'}")
    print(f"Server: http://localhost:5000")
    print(f"Login: admin / admin123")
    print("="*60 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000)
