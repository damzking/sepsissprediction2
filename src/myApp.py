
from fastapi import FastAPI
import joblib
from pydantic import BaseModel
import pandas as pd

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



GB_pipeline = joblib.load('../models/GB_pipeline.joblib')
LogReg_pipeline = joblib.load('../models/LogReg_pipeline.joblib')
#NB_pipeline = joblib.load('../models/NB_pipeline.joblib')
XGB_pipeline = joblib.load('../models/XGB_pipeline.joblib')
encoder = joblib.load('../models/encoder.joblib')

@app.post('/XGBoost_prediction')
def XGB_predict(data: sepssisfeatures):
    
    df = pd.DataFrame([data.model_dump()])
    prediction = XGB_pipeline.predict(df)
    prediction = int(prediction[0])
    return {'prediction': prediction}

@app.post('/Gradient_Boosting_Predict')
def GB_predict(data: sepssisfeatures):
    
    df = pd.DataFrame([data.model_dump])
    prediction = GB_pipeline.predict(df)
    prediction=int(prediction[0])
    
    return {'prediction': prediction}


@app.post('/LogReg_prediction')
def logReg_predict(data: sepssisfeatures):
    
    df = pd.DataFrame([data.model_dump])
    prediction = LogReg_pipeline.predict(df)
    prediction = int(prediction[0])
    
    return {'prediction': prediction}

    