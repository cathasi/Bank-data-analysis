from app_simple import app, generate_fraud_alert, FraudAlert

with app.app_context():
    print("Before test, total alerts:", FraudAlert.query.count())
    features = [0.0] * 28
    tid = "TEST-123-UPSERT"

    print("Creating first alert...")
    a1 = generate_fraud_alert(tid, features, 100.0)
    print("First alert:", a1.id, a1.transaction_id, a1.amount, a1.fraud_score)

    print("Creating second alert with same transaction_id (should update)...")
    a2 = generate_fraud_alert(tid, features, 200.0)
    print("Second alert:", a2.id, a2.transaction_id, a2.amount, a2.fraud_score)

    print("After test, total alerts:", FraudAlert.query.count())
