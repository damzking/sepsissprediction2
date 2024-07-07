import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

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
    

# Logarithmic Transformer
class LogTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return np.log1p(X) 


@app.get('/')
def MySepssisApi():
    return {'Message': 'Welcome to my Sepssis Prediction API'}



GB_pipeline = joblib.load('C:/Users/user/Documents/LP5-ML-API/P5-Machine-Learning-API/models/GB_pipeline.joblib')
LogReg_pipeline = joblib.load('C:/Users/user/Documents/LP5-ML-API/P5-Machine-Learning-API/models/LogReg_pipeline.joblib')
NB_pipeline = joblib.load('C:/Users/user/Documents/LP5-ML-API/P5-Machine-Learning-API/models/NB_pipeline.joblib')
XGB_pipeline = joblib.load('C:/Users/user/Documents/LP5-ML-API/P5-Machine-Learning-API/models/XGB_pipeline.joblib')
encoder = joblib.load('C:/Users/user/Documents/LP5-ML-API/P5-Machine-Learning-API/models/encoder.joblib')

@app.post('/XGBoost_prediction')
def predict_sepssis(data: sepssisfeatures):
    
    df = pd.DataFrame([data.model_dump()])
    prediction = XGB_pipeline.predict(df)
    return {'prediction': prediction}


    