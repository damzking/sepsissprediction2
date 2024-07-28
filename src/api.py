from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# Define the Pydantic model for patient data
class PatientsFile(BaseModel):
    PRG: int
    PL: int
    PR: int
    SK: int
    TS: int
    M11: float
    BD2: float
    Age: int
    Insurance: int

# Initialize FastAPI app
app = FastAPI()

# Load the models and encoder with error handling
try:
    decision_tree = joblib.load("./models/Decision_Tree_tunedb_pipeline.joblib")
    print("Decision Tree model loaded successfully.")
except Exception as e:
    print(f"Error loading Decision Tree model: {e}")
    decision_tree = None

try:
    logistic_regression = joblib.load("./models/Logistic_Regression_tunedb_pipeline.joblib")
    print("Logistic Regression model loaded successfully.")
except Exception as e:
    print(f"Error loading Logistic Regression model: {e}")
    logistic_regression = None

try:
    encoder = joblib.load("./models/encoder.joblib")
    print("Encoder loaded successfully.")
except Exception as e:
    print(f"Error loading encoder: {e}")
    encoder = None

# Define the root endpoint
@app.get("/")
def read_root():
    return {"Hello": "World"}

# Define an endpoint to make predictions using the Decision Tree model
@app.post("/predict_decision_tree")
def predict_decision_tree(data: PatientsFile):
    # Ensure the model and encoder are loaded
    if decision_tree is None or encoder is None:
        raise HTTPException(status_code=500, detail="Decision Tree model or encoder is not available.")
    
    # Convert Pydantic model to pandas DataFrame
    df = pd.DataFrame([data.model_dump()])
    
    # Make prediction
    try:
        prediction = decision_tree.predict(df)
        # Inverse transform the prediction using the encoder
        prediction_decoded = encoder.inverse_transform(prediction)
        return {"prediction": prediction_decoded.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

# Define an endpoint to make predictions using the Logistic Regression model
@app.post("/predict_logistic_regression")
def predict_logistic_regression(data: PatientsFile):
    # Ensure the model and encoder are loaded
    if logistic_regression is None or encoder is None:
        raise HTTPException(status_code=500, detail="Logistic Regression model or encoder is not available.")
    
    # Convert Pydantic model to pandas DataFrame
    df = pd.DataFrame([data.model_dump()])
    
    # Make prediction
    try:
        prediction = logistic_regression.predict(df)
        # Inverse transform the prediction using the encoder
        prediction_decoded = encoder.inverse_transform(prediction)
        return {"prediction": prediction_decoded.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
