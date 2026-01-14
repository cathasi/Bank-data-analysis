from app_simple import app, CHURN_DATA

first_id = CHURN_DATA[0]['id'] if CHURN_DATA else 1
with app.test_client() as c:
    resp = c.get(f'/api/churn/customer/{first_id}')
    print('Status code:', resp.status_code)
    print('JSON:', resp.get_json())
