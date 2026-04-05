## Overview
This is a Flask-based application that provides a web dashboard and REST APIs for multiple banking analytics features: transaction fraud detection, customer churn prediction, credit scoring (classification and regression), data quality checks, and trend analysis.

### Key points
- Main application: `app_simple.py`
- Database: SQLite (`fraud_alerts.db`)
- Optional ML models (pickle/joblib) for fraud, churn and credit scoring; the app falls back to simulated predictions when models are absent.
- Sample data and helpers live under `Credit_Scoring/`, `quality_check/`, and `trend_analysis/`.

## Features
- Score transactions using 28 PCA components + `Amount` and create alerts with `risk_level` (High/Medium/Low).
- Filter and paginate alerts by risk level, status, date range and amount.
- Fetch and evaluate transactions directly from CSV for visualization and testing.
- Churn prediction endpoints and integrations with the `churn_test.csv` sample dataset.
- Credit scoring endpoints (XGBoost) for numeric score prediction and credit bracket classification.
- Data quality checks for card records (`quality_check/cards_data.csv`).
- Trend analysis based on the Bank Marketing dataset (`trend_analysis/train.csv`).

## Requirements
- Python 3.8+
- Install dependencies from `requirements.txt`.

### Quick setup (Windows)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Environment variables
- `EMAIL_SENDER` — sender email address (default `fraud-alert@bank.com`)
- `EMAIL_RECEIVER` — receiver email address (default `security@bank.com`)
- `EMAIL_PASSWORD` — SMTP password; if empty, email sending is disabled

Set them in PowerShell before starting the server:
```powershell
$env:EMAIL_SENDER = 'youremail@example.com'
$env:EMAIL_RECEIVER = 'security@example.com'
$env:EMAIL_PASSWORD = 'yourpassword'
```

## Running
```powershell
python app_simple.py
```
Or use `start_dashboard.bat` if present. The server runs on `http://localhost:5000` by default.

## Main endpoints (summary)
- `GET /` — Dashboard page (template `fraud_detection.html`)
- `GET /api/statistics` — Dashboard statistics
- `GET /api/alerts` — List and filter alerts
- `POST /api/predict` — Submit transaction JSON for scoring and alert creation
- `GET /api/csv/transaction/<index>` — Evaluate CSV transaction by index
- `GET /api/csv/transactions` — Paginated CSV transactions
- `GET /api/csv/stats` — CSV dataset stats

Churn
- `GET /api/churn/stats`
- `GET /api/churn/customer/<id>`
- `POST /api/churn/predict`

Credit Scoring (XGBoost)
- `POST /api/credit/predict/score`
- `POST /api/credit/predict/class`
- `POST /api/credit/predict/batch` (CSV upload)

Quality & Trend
- `GET /api/quality_check/stats`
- `GET /api/quality_check/record/<index>`
- `GET /api/trend_analysis/stats`

## Data & models
- `fraud_test.csv`, `churn_test.csv`, `credit_test.csv` — sample datasets
- `quality_check/cards_data.csv` — card records for quality check
- `Credit_Scoring/models/` — place `credit_score_regressor.joblib`, `credit_score_classifier.joblib`, and `preprocessing_artifacts.json` here
- Optional: `fraud_detection_model.pkl`, `scaler.pkl`, `churn_xgboost_model.pkl` will be loaded if present

