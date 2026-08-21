from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = FastAPI(title="Facial Attribute Analyzer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "celeba_multitask.keras")
model = tf.keras.models.load_model(MODEL_PATH)

ATTRIBUTES = ['Male', 'Smiling', 'Eyeglasses', 'Young']

@app.get("/")
async def serve_frontend():
    return FileResponse("index.html")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    image = image.resize((178, 218))
    
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_batch = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_batch)
    
    results = {}
    for i, attr in enumerate(ATTRIBUTES):
        val = predictions[i][0][0] if isinstance(predictions, list) else predictions[0][i]
        prob = float(val)
        
        results[attr] = {
            "prediction": "Yes" if prob > 0.5 else "No",
            "confidence": round(prob * 100, 2)
        }
        
    return results