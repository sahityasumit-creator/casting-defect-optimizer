from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from d3rlpy.algos import BC
from app.hybrid_policy import HybridPolicy

app = FastAPI(title="Casting Defect Optimizer")

# Load models (dummy ones in models/ folder)
rf_shrinkage = joblib.load("app/models/rf_shrinkage_model.pkl")
rf_ceramic   = joblib.load("app/models/rf_ceramic_model.pkl")
scaler       = joblib.load("app/models/scaler.pkl")
state_scaler = joblib.load("app/models/state_scaler.pkl")

bc = BC()
bc.build_with_dataset(None)
bc.load_model("app/models/bc_model.pt")

success_states = np.load("app/models/success_states_scaled.npy")
success_actions = np.load("app/models/success_actions.npy")

policy = HybridPolicy(bc, success_states, success_actions)

class Input(BaseModel):
    RT: float
    RH: float
    WP: float
    LT: float
    PT: float
    PTE: float

@app.post("/predict")
def predict(data: Input):
    x = np.array([[data.RT, data.RH, data.WP, data.LT, data.PT, data.PTE]])
    x_scaled = scaler.transform(x)

    p_shrink = float(rf_shrinkage.predict_proba(x_scaled)[0][1])
    p_ceramic = float(rf_ceramic.predict_proba(x_scaled)[0][1])

    if p_shrink < 0.25 and p_ceramic < 0.25:
        return {
            "shrinkage_risk": round(p_shrink, 4),
            "ceramic_risk": round(p_ceramic, 4),
            "recommended_PT": round(data.PT, 2),
            "recommended_PTE": round(data.PTE, 2),
            "message": "Parameters are safe!"
        }

    state_scaled = state_scaler.transform([[data.RT, data.RH, data.WP, data.LT]])[0]
    opt = policy.get_action(state_scaled)

    return {
        "shrinkage_risk": round(p_shrink, 4),
        "ceramic_risk": round(p_ceramic, 4),
        "recommended_PT": round(opt[0], 2),
        "recommended_PTE": round(opt[1], 2),
        "message": "High risk → Optimized parameters suggested!"
    }

@app.get("/")
def home():
    return {"message": "Casting Defect Optimizer API is LIVE!"}

