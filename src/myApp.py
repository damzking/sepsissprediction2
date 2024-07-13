import pandas as pd
import numpy as np
#from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import ( 
    LabelEncoder, 
    RobustScaler, 
)

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ( 
    GradientBoostingClassifier
)


from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.linear_model import HuberRegressor

from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_curve,
    auc, 
    confusion_matrix, 
    classification_report
)

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



GB_pipeline = joblib.load('../models/GB_pipeline.joblib')
LogReg_pipeline = joblib.load('../models/LogReg_pipeline.joblib')
#NB_pipeline = joblib.load('../models/NB_pipeline.joblib')
XGB_pipeline = joblib.load('../models/XGB_pipeline.joblib')
encoder = joblib.load('../models/encoder.joblib')

@app.post('/XGBoost_prediction')
def XGB_predict(data: sepssisfeatures):
    
    df = pd.DataFrame([data.model_dump()])
    predxgb = XGB_pipeline.predict(df)
    predxgb = int(predxgb[0])
    
    predictxgb = encoder.inverse_transform([predxgb])[0]
    
    probabxgb = XGB_pipeline.predict_proba(df)[0]
    
    
    return {'prediction': predictxgb}

@app.post('/Gradient_Boosting_Predict')
def GB_predict(data: sepssisfeatures):
    
    df = pd.DataFrame([data.model_dump])
    predGB = GB_pipeline.predict(df)
    
    predGB = int(predGB[0])
    predictgb = encoder.inverse_transform([predGB])[0]
    probabgb = GB_pipeline.predict_proba(df)[0]
    
    return {'prediction': predictgb}



@app.post('/LogReg_prediction')
def logReg_predict(data: sepssisfeatures):

    df = pd.DataFrame([data.model_dump])
    predlr = LogReg_pipeline.predict(df)
    predlr = int(predlr[0])
    
    predlogreg = encoder.inverse_transform([predlr])[0]
    probablr = LogReg_pipeline.predict_proba(df)[0]

    return {'prediction': predlogreg}

    