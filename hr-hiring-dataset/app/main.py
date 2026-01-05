from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# --------------------------------------------------
# 1. Load Model
# --------------------------------------------------
model = joblib.load("hr_recruitment_model.pkl")

app = FastAPI(
    title="HR Recruitment Prediction API",
    description="AI model to predict candidate employment probability",
    version="1.0"
)

# --------------------------------------------------
# 2. Request Schema
# --------------------------------------------------
class CandidateInput(BaseModel):
    EdLevel: str
    Employment: int
    MainBranch: str
    YearsCodePro: float
    Country: str
    PreviousSalary: float
    ComputerSkills: float
    HaveWorkedWith: str

# --------------------------------------------------
# 3. Prediction Endpoint
# --------------------------------------------------
@app.post("/predict")
def predict(data: CandidateInput):

    try:
        data_dict = data.dict()

        # Safe text normalization
        skills = data_dict.get("HaveWorkedWith") or ""
        data_dict["HaveWorkedWith"] = skills.lower().replace(",", " ")

        df = pd.DataFrame([data_dict])

        probability = model.predict_proba(df)[0][1]
        prediction = int(probability >= 0.3)

        return {
            "prediction": prediction,
            "employment_probability": round(float(probability), 4),
            "decision": "Recommended" if prediction == 1 else "Needs Review"
        }

    except Exception as e:
        return {
            "error": str(e)
        }