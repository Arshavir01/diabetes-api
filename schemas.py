from pydantic import BaseModel, Field


class PatientData(BaseModel):
    """Input features - order must match the order used when training the model."""
    Pregnancies: int = Field(..., ge=0, examples=[2])
    Glucose: float = Field(..., ge=0, examples=[138])
    BloodPressure: float = Field(..., ge=0, examples=[62])
    SkinThickness: float = Field(..., ge=0, examples=[35])
    Insulin: float = Field(..., ge=0, examples=[0])
    BMI: float = Field(..., ge=0, examples=[33.6])
    DiabetesPedigreeFunction: float = Field(..., ge=0, examples=[0.127])
    Age: int = Field(..., ge=0, le=120, examples=[47])


class PredictionResponse(BaseModel):
    prediction: int
    result: str
    probability: float | None = None
