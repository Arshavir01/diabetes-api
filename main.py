import pickle
from pathlib import Path

from fastapi import FastAPI, HTTPException

from schemas import PatientData, PredictionResponse

# The model file is in the same folder as this main.py
MODEL_PATH = Path(__file__).parent / "diabetes_trained_model.sav"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

app = FastAPI(title="Diabetes Prediction API", version="1.0.0")


@app.get("/")
def root():
    return {"message": "Diabetes Prediction API is running. Go to /docs to test it."}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: PatientData):
    # Values in the same order the model was trained on
    values = [list(data.model_dump().values())]

    try:
        pred = int(model.predict(values)[0])
        proba = None
        if hasattr(model, "predict_proba"):
            try:
                proba = round(float(model.predict_proba(values)[0][1]), 4)
            except Exception:
                proba = None  # e.g. SVC trained without probability=True
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

    return PredictionResponse(
        prediction=pred,
        result="The person is diabetic" if pred == 1 else "The person is not diabetic",
        probability=proba,
    )
