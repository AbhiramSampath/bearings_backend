import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # silence TF logs

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Create FastAPI app
app = FastAPI()

# CORS (so frontend can call backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model
MODEL_PATH = "model/bearing_RUL_LSTM.h5"

try:
    model = keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Error loading model:", e)


# Input format
class RULInput(BaseModel):
    sequence: list  # [[ch1, ch2, ch3, ch4], ... ]


@app.get("/")
def home():
    return {"message": "RUL Prediction API is running."}


@app.post("/predict")
def predict_rul(data: RULInput):
    seq = np.array(data.sequence)

    # Validate dimensions
    if seq.ndim != 2:
        return {"error": "Input must be 2D: (sequence_length, 7)"}

    if seq.shape[1] != 7:
        return {"error": "Each timestep must have 7 features"}

    # Reshape for LSTM: (1, seq_len, 7)
    seq = np.expand_dims(seq, axis=0)

    # Prediction
    try:
        pred = model.predict(seq)[0][0]
        return {"predicted_RUL": float(pred)}
    except Exception as e:
        return {"error": str(e)}
