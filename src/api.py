from fastapi import FastAPI
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

# Define the absolute path to the model files
model_path_d = os.path.abspath("../Models/Decision_Tree_tunedb_pipeline.joblib")
model_path_l = os.path.abspath("../Models/Logistic_Regression_tunedb_pipeline.joblib")
encoder_path = os.path.abspath("../Models/label_encoder.joblib")

# Load the models and encoder
decision_tree = joblib.load(model_path_d)
logistic_regression = joblib.load(model_path_l)
encoder = joblib.load(encoder_path)

# Define the root endpoint
@app.get("/")
def read_root():
    return {"Hello": "World"}

# Define an endpoint to make predictions using the Decision Tree model
@app.post("/predict_decision_tree")
def predict_decision_tree(data: PatientsFile):
    # Convert Pydantic model to pandas DataFrame
    df = pd.DataFrame([data.model_dump()])
    
    # Make prediction
    prediction = decision_tree.predict(df)
    
    return {"prediction": encoder.inverse_transform(prediction).tolist()}

# Define an endpoint to make predictions using the Logistic Regression model
@app.post("/predict_logistic_regression")
def predict_logistic_regression(data: PatientsFile):
    # Convert Pydantic model to pandas DataFrame
    df = pd.DataFrame([data.model_dump()])
    
    # Make prediction
    prediction = logistic_regression.predict(df)
    
    return {"prediction": encoder.inverse_transform(prediction).tolist()}
