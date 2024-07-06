from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# Define the Pydantic model for patients' data
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

# Define the absolute path to the model file
model_path = os.path.abspath("../Models/Decision_Tree_best_model.joblib")

encoder_path = os.path.abspath("../Models/label_encoder.joblib")

# Load the decision tree model and encoder
decision_tree = joblib.load(model_path)
encoder = joblib.load(encoder_path)

# Define the root endpoint
@app.get("/")
def read_root():
    return {"Hello": "World"}

# Define an endpoint to get documents
@app.get("/documents")
def read_documents():
    return {"documents": "all_documents"}

# Define an endpoint to make predictions
@app.post("/predict")
def predict(data: PatientsFile):
    # Convert Pydantic model to pandas DataFrame
    df = pd.DataFrame([data.model_dump()])
    
    # Encode categorical variables if needed (example for Sepssis if it's categorical)
    # If you need to encode more variables, add them here
    # df['Sepssis'] = encoder.transform(df['Sepssis'])
    
    # Make prediction
    prediction = decision_tree.predict(df)
    
    return {"prediction": prediction.tolist()}
