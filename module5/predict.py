import pickle
from fastapi import FastAPI
import uvicorn
from typing import Dict, Any

model_file = 'model.bin'

app = FastAPI(title = "customer-churn-prediction")

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)


def predict_single(customer):
    
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    return y_pred


@app.post("/predict")
def predict(customer: Dict[str, Any]):
    prediction = predict_single(customer)
    return {
        "churn_probability": float(prediction), 
        "churn": bool(prediction >= 0.5)
        }


# datapoint = {
#     'gender': 'female',
#     'seniorcitizen': 0,
#     'partner': 'yes',
#     'dependents': 'no',
#     'phoneservice': 'no',
#     'multiplelines': 'no_phone_service',
#     'internetservice': 'dsl',
#     'onlinesecurity': 'no',
#     'onlinebackup': 'yes',
#     'deviceprotection': 'no',
#     'techsupport': 'no',
#     'streamingtv': 'no',
#     'streamingmovies': 'no',
#     'contract': 'month-to-month',
#     'paperlessbilling': 'yes',
#     'paymentmethod': 'electronic_check',
#     'tenure': 1,
#     'monthlycharges': 29.85,
#     'totalcharges': 29.85
# }

