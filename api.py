
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Load model and feature list
model = joblib.load('fraud_shield_model.pkl')
features = joblib.load('model_features.pkl')

class Transaction(BaseModel):
    step: int
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float
    type_TRANSFER: int
    type_CASH_OUT: int

@app.post("/predict")
def predict_fraud(data: Transaction):
    # Calculate teammate's custom features on the fly!
    amount_orig_ratio = data.amount / data.oldbalanceOrg if data.oldbalanceOrg > 0 else 0
    balance_diff = data.newbalanceOrig + data.amount - data.oldbalanceOrg
    hour_of_day = data.step % 24
    is_night_time = 1 if hour_of_day <= 6 else 0
    
    # We dynamically assign 'ip_risk' and 'device_change' based on the math integrity
    # (If the balance math is broken, we simulate a high-risk IP connection)
    simulated_ip_risk = 0.85 if balance_diff != 0 else 0.15
    simulated_device_change = 4 if balance_diff != 0 else 0
    
    # Build dictionary of all possible inputs
    input_data = {
        'step': data.step,
        'amount': data.amount,
        'oldbalanceOrg': data.oldbalanceOrg,
        'newbalanceOrig': data.newbalanceOrig,
        'oldbalanceDest': data.oldbalanceDest,
        'newbalanceDest': data.newbalanceDest,
        'type_TRANSFER': data.type_TRANSFER,
        'type_CASH_OUT': data.type_CASH_OUT,
        'type_CASH_IN': 0, 'type_DEBIT': 0, 'type_PAYMENT': 0,
        'hour_of_day': hour_of_day,
        'is_night_time': is_night_time,
        'amount_orig_ratio': amount_orig_ratio,
        'balance_diff': balance_diff,
        'ip_risk_score': simulated_ip_risk,
        'device_change_count': simulated_device_change
    }

    # Create DataFrame with EXACT column order as the trained model
    input_df = pd.DataFrame(columns=features)
    input_df.loc[0] = [input_data.get(col, 0) for col in features]

    # Predict using your teammate's model
    probability = model.predict_proba(input_df)[0][1]
    
    return {
        "status": "Blocked" if probability >= 0.5 else "Approved",
        "risk_score": float(probability)
    }
