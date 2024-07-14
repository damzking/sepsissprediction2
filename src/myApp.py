import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Define your features class using Pydantic
class SepsisFeatures(BaseModel):
    PRG: int
    PL: int
    PR: int
    SK: int
    TS: int
    M11: float
    BD2: float
    Age: int
    Insurance: int
    model_name: str

# Initialize FastAPI app
app = FastAPI()


@app.get('/')
async def SepssisAPI():
    return {"message": "SepssisAPI"}

# Load all models and the encoder
try:
    models = {
        'GradientBoosting': joblib.load('../models/GB_pipeline.joblib'),
        'LogisticRegression': joblib.load('../models/LogReg_pipeline.joblib'),
        'SVM': joblib.load('../models/SVM_pipeline.joblib'),
        'XGBoost': joblib.load('../models/XGB_pipeline.joblib')
    }
    encoder = joblib.load('../models/encoder.joblib')

except Exception as e:
    logger.error(f"Error loading models: {e}")

@app.post('/predict/{model_name}')
def predict(model_name: str, data: SepsisFeatures):
    
    
    # Validate model name
    if model_name not in models:
        raise HTTPException(status_code=400, detail="Model not found")
    
    else:    
    # Convert input data to DataFrame
        df = pd.DataFrame([data.model_dump()])
    
    # Get the specified model
        model = models[model_name]
    
    # Make prediction
        pred = model.predict(df)
        pred = int(pred[0])
    
    # Decode the prediction
        decoded_pred = encoder.inverse_transform([pred])[0]
    
    # Get probabilities if supported by the model
        try:
            probab = model.predict_proba(df)[0]
        except AttributeError:
            probab = "Probabilities not available for this model"

    # Return prediction and probabilities
    return {'model': model_name, 'prediction': decoded_pred, 'probabilities': probab}
