from app_simple import app, predict_churn_probability, CHURN_MODEL_LOADED

sample = {
    'CreditScore': 650,
    'Age': 35,
    'Tenure': 5,
    'Balance': 50000.0,
    'NumOfProducts': 1,
    'HasCrCard': 1,
    'IsActiveMember': 1,
    'EstimatedSalary': 60000.0,
    'Geography': 'Germany',
    'Gender': 'Male'
}

with app.app_context():
    print('CHURN_MODEL_LOADED =', CHURN_MODEL_LOADED)
    p = predict_churn_probability(sample)
    print('Predicted churn probability:', p)
