import pickle

with open('churn_xgboost_model.pkl', 'rb') as f:
    model = pickle.load(f)

print('Model type:', type(model))
# common sklearn attributes
for attr in ('n_features_in_', 'feature_names_in_', 'n_features_'):
    if hasattr(model, attr):
        print(attr, '=', getattr(model, attr))

# If xgboost scikit-learn wrapper
try:
    import xgboost as xgb
    if isinstance(model, xgb.XGBClassifier) or hasattr(model, 'get_booster'):
        print('Has get_booster()')
        try:
            booster = model.get_booster()
            print('Booster feature_names:', booster.feature_names)
        except Exception as e:
            print('Could not get booster feature names:', e)
except Exception as e:
    print('xgboost import error:', e)

# print dir for inspection
print('dir(model):')
print([d for d in dir(model) if not d.startswith('__')])
