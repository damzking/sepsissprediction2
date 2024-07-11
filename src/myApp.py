import pandas as pd
import numpy as np

from fastapi import FastAPI
import joblib
from pydantic import BaseModel

from Custom_Transformers import LogTransformer, SqrtTransformer, BoxCoxTransformer


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
        
    


@app.get('/')
def MySepssisApi():
    return {'Message': 'Welcome to my Sepssis Prediction API'}



GB_pipeline = joblib.load('/Users/mac/Documents/P5-Machine-Learning-API/models/GB_pipeline.joblib')
LogReg_pipeline = joblib.load('/Users/mac/Documents/P5-Machine-Learning-API/models/LogReg_pipeline.joblib')
XGB_pipeline = joblib.load('/Users/mac/Documents/P5-Machine-Learning-API/models/XGB_pipeline.joblib')
encoder = joblib.load('/Users/mac/Documents/P5-Machine-Learning-API/models/encoder.joblib')

@app.post('/XGBoost_prediction')
def predict_sepssis(data: sepssisfeatures):
    
    df = pd.DataFrame([data.model_dump()])
    prediction = XGB_pipeline.predict(df)
    return {'prediction': prediction}


    