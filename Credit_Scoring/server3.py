import json
import pandas as pd
import numpy as np
import joblib
import io
import traceback
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
app = FastAPI(title="Credit Score Prediction API (Unified)")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths (Ensure these match your directory structure)
REGRESSOR_PATH = './models/credit_score_regressor.joblib'
CLASSIFIER_PATH = './models/credit_score_classifier.joblib'
ARTIFACTS_PATH = './models/preprocessing_artifacts.json'

# Global variables for models
regressor = None
classifier = None
artifacts = None

# ==========================================
# 2. DATA MODELS (For Single Prediction)
# ==========================================
class CreditData(BaseModel):
    Age: str
    Annual_Income: str
    Num_Bank_Accounts: int
    Num_Credit_Card: int
    Interest_Rate: int
    Num_of_Loan: int
    Delay_from_due_date: int
    Num_of_Delayed_Payment: str
    Changed_Credit_Limit: str
    Num_Credit_Inquiries: float
    Credit_Mix: str
    Outstanding_Debt: str
    Credit_Utilization_Ratio: float
    Credit_History_Age: str
    Payment_of_Min_Amount: str
    Total_EMI_per_month: float
    Amount_invested_monthly: str
    Payment_Behaviour: str
    Monthly_Balance: str
    Occupation: str

# ==========================================
# 3. HELPER FUNCTIONS (Shared Logic)
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

# ==========================================
# 4. LIFECYCLE: LOAD MODELS
# ==========================================
@app.on_event("startup")
def load_models():
    global regressor, classifier, artifacts
    try:
        print("Loading models...")
        regressor = joblib.load(REGRESSOR_PATH)
        classifier = joblib.load(CLASSIFIER_PATH)
        with open(ARTIFACTS_PATH, 'r') as f:
            artifacts = json.load(f)
        print("✅ Models and artifacts loaded successfully.")
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")
        print("Ensure 'models/' directory contains the .joblib and .json files.")

# ==========================================
# 5. PREPROCESSING LOGIC
# ==========================================

def preprocess_single_input(data: CreditData, is_regression: bool):
    """Prepares a single JSON input for model inference."""
    # Convert Pydantic to DataFrame
    input_dict = data.dict()
    df = pd.DataFrame([input_dict])

    # 1. Clean Numerical Columns
    cols_to_clean = [
        'Age', 'Annual_Income', 'Num_of_Delayed_Payment', 
        'Changed_Credit_Limit', 'Outstanding_Debt', 
        'Amount_invested_monthly', 'Monthly_Balance'
    ]
    for col in cols_to_clean:
        df[col] = df[col].apply(clean_non_numeric)
    
    df['Age'] = df['Age'].clip(lower=18, upper=100)

    # 2. Feature Engineering
    if 'Credit_History_Age' in df.columns:
        if is_regression:
            df['Credit_History_Months'] = df['Credit_History_Age'].apply(convert_age_to_months)
        df.drop(columns=['Credit_History_Age'], inplace=True)

    # 3. Specific Path Logic
    if is_regression:
        # Credit Mix Encoding (Manual Map to match training LabelEncoder)
        # Assuming Alphabetical: Bad=0, Good=1, Standard=2
        mix_mapping = {'Bad': 0, 'Good': 1, 'Standard': 2}
        df['Credit_Mix_Encoded'] = df['Credit_Mix'].map(mix_mapping).fillna(1) # Fill Default

        # Align columns with Regressor
        if hasattr(regressor, 'feature_names_in_'):
            required_cols = regressor.feature_names_in_
        else:
            required_cols = df.select_dtypes(include=[np.number]).columns
            
        # Ensure all columns exist
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0
        
        return df[required_cols]

    else: # Classification
        # Payment Min Amount Map
        payment_map = artifacts['payment_mapping']
        df['Payment_of_Min_Amount'] = df['Payment_of_Min_Amount'].map(payment_map).fillna(0)

        # One-Hot Encoding
        nominal_cols = ['Occupation', 'Credit_Mix', 'Payment_Behaviour']
        df_encoded = pd.get_dummies(df, columns=nominal_cols, drop_first=True)

        # Align with Training Features
        train_features = artifacts['feature_names']
        df_final = pd.DataFrame(0, index=np.arange(len(df_encoded)), columns=train_features)
        
        for col in df_encoded.columns:
            if col in df_final.columns:
                df_final[col] = df_encoded[col].values

        return df_final

def process_batch_dataframe(df_raw):
    """Prepares a bulk DataFrame for inference."""
    df = df_raw.copy()

    # 1. Clean Numeric
    numeric_features = [
        'Age', 'Annual_Income', 'Num_Bank_Accounts', 'Num_Credit_Card', 
        'Interest_Rate', 'Delay_from_due_date', 'Num_of_Delayed_Payment', 
        'Changed_Credit_Limit', 'Num_Credit_Inquiries', 'Outstanding_Debt', 
        'Credit_Utilization_Ratio', 'Total_EMI_per_month', 'Amount_invested_monthly', 
        'Monthly_Balance', 'Num_of_Loan'
    ]
    for col in numeric_features:
        if col in df.columns:
            df[col] = clean_non_numeric(df[col])
            
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
    reg_features = regressor.feature_names_in_ if hasattr(regressor, 'feature_names_in_') else []
    for col in reg_features:
        if col not in df_reg.columns:
            df_reg[col] = 0
    
    if len(reg_features) > 0:
        df_raw['Predicted_Score'] = regressor.predict(df_reg[reg_features])

    # --- CLASSIFICATION PREDICTION ---
    df_clf = df.copy()
    if 'Credit_History_Age' in df_clf.columns:
        df_clf.drop(columns=['Credit_History_Age'], inplace=True)
    
    if 'Payment_of_Min_Amount' in df_clf.columns:
        df_clf['Payment_of_Min_Amount'] = df_clf['Payment_of_Min_Amount'].map(artifacts['payment_mapping']).fillna(0)

    # Impute
    for col, val in artifacts.get('imputation_values', {}).items():
        if col in df_clf.columns:
            df_clf[col] = df_clf[col].fillna(val)

    # Encoding
    nominal_cols = ['Occupation', 'Credit_Mix', 'Payment_Behaviour']
    present_nominal = [c for c in nominal_cols if c in df_clf.columns]
    df_clf_encoded = pd.get_dummies(df_clf, columns=present_nominal, drop_first=True)

    # Align Classification Features
    clf_features = artifacts['feature_names']
    df_aligned_clf = pd.DataFrame(0, index=df_clf_encoded.index, columns=clf_features)
    common_cols = [c for c in df_clf_encoded.columns if c in clf_features]
    df_aligned_clf[common_cols] = df_clf_encoded[common_cols]

    # Predict
    pred_ids = classifier.predict(df_aligned_clf)
    
    # Map back
    target_map_inv = {v: k for k, v in artifacts['target_mapping'].items()}
    df_raw['Predicted_Credit_Bracket'] = [target_map_inv.get(x, "Unknown") for x in pred_ids]

    return df_raw

# ==========================================
# 6. API ENDPOINTS
# ==========================================

@app.get("/")
def home():
    return {"message": "Unified Credit Score API is Running. Use /docs for Swagger UI."}

# --- SINGLE PREDICTION ENDPOINTS ---

@app.post("/predict/score")
def predict_score_single(data: CreditData):
    """Predicts numeric credit score for a single user."""
    if not regressor:
        raise HTTPException(status_code=500, detail="Regressor model not loaded")
    
    try:
        processed_data = preprocess_single_input(data, is_regression=True)
        prediction = regressor.predict(processed_data)
        return {
            "predicted_score": float(prediction[0]),
            "model_type": "XGBRegressor"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/class")
def predict_class_single(data: CreditData):
    """Predicts credit bracket (Good/Standard/Poor) for a single user."""
    if not classifier:
        raise HTTPException(status_code=500, detail="Classifier model not loaded")
    
    try:
        processed_data = preprocess_single_input(data, is_regression=False)
        prediction_idx = classifier.predict(processed_data)[0]
        
        # Map ID back to Label
        target_map = artifacts['target_mapping']
        reverse_map = {v: k for k, v in target_map.items()}
        label = reverse_map.get(prediction_idx, "Unknown")
        
        # Confidence
        probs = classifier.predict_proba(processed_data)[0]
        confidence = float(np.max(probs))

        return {
            "credit_bracket": label,
            "confidence": confidence,
            "model_type": "XGBClassifier"
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))

# --- BATCH PREDICTION ENDPOINT ---

@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    """Accepts a CSV file, appends predictions, and returns a CSV download."""
    if not regressor or not classifier:
        raise HTTPException(status_code=500, detail="Models not loaded")

    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed.")
    
    try:
        # Read File
        contents = await file.read()
        
        # Try multiple encodings
        df = None
        for enc in ['utf-8', 'latin1', 'cp1252']:
            try:
                df = pd.read_csv(io.BytesIO(contents), encoding=enc, low_memory=False)
                break
            except:
                continue
        
        if df is None:
            raise HTTPException(status_code=400, detail="Could not decode CSV file.")

        # Process
        result_df = process_batch_dataframe(df)

        # Return CSV
        stream = io.StringIO()
        result_df.to_csv(stream, index=False)
        
        response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
        response.headers["Content-Disposition"] = "attachment; filename=predictions_result.csv"
        return response

    except Exception as e:
        print(f"Batch Processing Error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)