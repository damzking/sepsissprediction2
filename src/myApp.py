import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import logging
from typing import Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Defining features class using Pydantic
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
    


# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/')
async def SepssisAPI():
    return {"message": "SepssisAPI"}

# Load all models and the encoder
models = {}
encoder = None
try:
    models = {
        'GradientBoosting': joblib.load('./models/GB_pipeline.joblib'),
        'LogisticRegression': joblib.load('./models/LogReg_pipeline.joblib'),
        'SVM': joblib.load('./models/SVM_pipeline.joblib'),
        'XGBoost': joblib.load('./models/XGB_pipeline.joblib')
    }
    encoder = joblib.load('./models/encoder.joblib')
    logger.info("Models and encoder loaded successfully.")
except Exception as e:
    logger.error(f"Error loading models: {e}")

@app.post('/predict/{model_name}')
async def predict(model_name: str, data: SepsisFeatures):
    try:
        # Validate model name
        if model_name not in models:
            raise HTTPException(status_code=400, detail="Model not found")
        
        # Convert input data to DataFrame
        df = pd.DataFrame([data.dict()])

        # Get the specified model
        model = models[model_name]

        # Make prediction
        pred = model.predict(df)
        pred = int(pred[0])

        # Decode the prediction
        decoded_pred = encoder.inverse_transform([pred])[0]

        # Get probabilities if supported by the model
        try:
            probab = model.predict_proba(df)[0].tolist()[0]  
            #probab2 = [round(p, 2) for p in probab]
        except AttributeError:
            probab = "Probability not available for this model"

        # Return prediction and probabilities
        return {'model': model_name, 'prediction': decoded_pred, 'probability': probab}
    except HTTPException as e:
        logger.error(f"HTTP Exception: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"An error occurred during prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# Main block to start the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
