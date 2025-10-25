import pickle
from fastapi import FastAPI
import uvicorn
from typing import Dict, Any


model_file = "pipeline_v1.bin"


with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)
    
app = FastAPI(title="model-deployment-homework")

def predict_single(customer):
    X = dv.transform(customer)
    y_pred = model.predict_proba(X)[0,1]
    return y_pred


@app.post("/predict")
def predict(customer: Dict[str, Any]):
    pred = predict_single(customer)
    return {
        "Probability: ", float(pred)
    }
    

# pred = predict_single({
#     "lead_source": "paid_ads",
#     "number_of_courses_viewed": 2,
#     "annual_income": 79276.0
# })

# print(pred)