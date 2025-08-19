import joblib
model = joblib.load('models/symptoms_model.pkl')
print("Model features:", model['feature_names'])