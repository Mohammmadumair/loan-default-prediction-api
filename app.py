from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

# Load model
model = joblib.load("loan_default_pipeline.pkl")


# Home route
@app.get("/")
def home():
    return {
        "message": "Loan Default Prediction API is running"
    }


# Input schema
class LoanInput(BaseModel):

    Age: int
    Income: float
    LoanAmount: float
    CreditScore: int
    MonthsEmployed: int
    NumCreditLines: int
    InterestRate: float
    LoanTerm: int
    DTIRatio: float

    Education_High_School: int
    Education_Master: int
    Education_PhD: int

    EmploymentType_Part_time: int
    EmploymentType_Self_employed: int
    EmploymentType_Unemployed: int

    MaritalStatus_Married: int
    MaritalStatus_Single: int

    HasMortgage_Yes: int
    HasDependents_Yes: int

    LoanPurpose_Business: int
    LoanPurpose_Education: int
    LoanPurpose_Home: int
    LoanPurpose_Other: int

    HasCoSigner_Yes: int


@app.post("/predict")
def predict(data: LoanInput):

    input_df = pd.DataFrame([data.dict()])

    # Rename columns to match training data
    input_df = input_df.rename(columns={
        "Education_High_School": "Education_High School",
        "Education_Master": "Education_Master's",
        "EmploymentType_Part_time": "EmploymentType_Part-time",
        "EmploymentType_Self_employed": "EmploymentType_Self-employed",
    })

    prediction = model.predict(input_df)

    probability = model.predict_proba(input_df)

    return {
        "Default_Prediction": int(prediction[0]),
        "Default_Probability": float(probability[0][1])
    }