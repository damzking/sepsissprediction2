from fastapi import FastAPI
import joblib
from pydantic import BaseModel


app = FastAPI()


class sepssisfeatures(BaseModel):
    PRG: int
    PL: int
    PR: int
    SK: int
    TS: int
    M11: float
    BD2: float
    Age: int
    Insurance: int
    Sepssis: object
    







@app.get('/')
def MySepssisApi():
    return {'Message': 'Welcome to my Sepssis Prediction API'}




Gradient_Boosting_pipeline = joblib.load('../models/GB_pipeline.joblib')
Logiistic_Regressor_pipeline = joblib.load('../models/LogReg_pipeline.joblib')
Naive_Bayes_pipeline = joblib.load('../models/NB_pipeline.joblib')
XGBoost_pipeline = joblib.load('../models/XGB_pipeline.joblib')
encoder = joblib.load('../models/encoder.joblib')

@app.post('/XGBoost_prediction')
def predict_sepssis(data:sepssisfeatures):
    
    df = PD.DataFrames(Data.model_dump())
    prediction = XGBoost_pipeline.predict(df)
    return {'prediction': prediction}


    