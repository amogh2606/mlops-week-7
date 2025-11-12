import os
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from joblib import load
import sys
from typing import List
import logging
import time
import uuid # For unique request IDs

# --- 1. Logging Setup ---
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("iris-api")

# --- Pydantic Input Model ---
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# --- FastAPI App Setup ---
app = FastAPI(title="IRIS Prediction API (Docker/K8s)", version="1.0")
MODEL_PATH = "artifacts/model.joblib"
model = None
MODEL_VERSION = "1.0.0" # Hardcoded or read from a version file

def load_model():
    """Loads the model from the local file system inside the container."""
    global model
    logger.info(f"Attempting to load model from path: {MODEL_PATH}")
    try:
        if not os.path.exists(MODEL_PATH):
            # Log critical error if file is missing
            logger.critical(f"Model artifact NOT found at path: {MODEL_PATH}")
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
            
        model = load(MODEL_PATH)
        logger.info("Model loaded successfully. Startup complete.")
        
    except Exception as e:
        # Log error during startup
        logger.error(f"FATAL ERROR during model load: {e}", exc_info=True)
        model = None
        # Raise exception to ensure K8s fails the readiness probe
        raise

# Load model on application startup
try:
    load_model()
except Exception:
    sys.exit(1) # Exit if load_model fails, forcing container restart

# --- Health Check Endpoint ---
@app.get("/health", status_code=200, tags=["Health"])
def health():
    if model is not None:
        return {"status": "healthy", "model_status": "loaded", "version": MODEL_VERSION}
    else:
        logger.warning("Health check failed: Model is None.")
        raise HTTPException(status_code=503, detail="Model load failure")

# --- Prediction Endpoint ---
@app.post("/predict", tags=["Prediction"])
def predict(features: IrisFeatures, request: Request):
    
    # 1. Start Telemetry Timer and Log Request
    start_time = time.time()
    request_id = str(uuid.uuid4())
    logger.info(f"[{request_id}] Received prediction request from {request.client.host}")
    
    if model is None:
        logger.critical(f"[{request_id}] Request failed: Model not initialized.")
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        # 2. Prepare Input Data
        input_data = [
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width
        ]
        input_array = np.array([input_data])
        
        # 3. Predict
        prediction = model.predict(input_array)
        predicted_class = prediction[0] 

        # 4. Log Success and Telemetry
        latency_ms = (time.time() - start_time) * 1000
        logger.info(
            f"[{request_id}] Prediction SUCCESS. "
            f"Input: {input_data} -> Output: {predicted_class}. "
            f"Latency: {latency_ms:.2f}ms"
        )
        
        return {
            "prediction": predicted_class,
            "model_source": "DVC GCS Remote",
            "model_version": MODEL_VERSION,
            "request_id": request_id,
            "latency_ms": round(latency_ms, 2)
        }

    except Exception as e:
        # 5. Log Failure
        latency_ms = (time.time() - start_time) * 1000
        logger.error(
            f"[{request_id}] Prediction FAILED during execution. "
            f"Latency: {latency_ms:.2f}ms. Error: {str(e)}", 
            exc_info=True
        )
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")