from pathlib import Path
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf

from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

import io
from PIL import Image

app = FastAPI(
    title="Skin Lesion Classifier API",
    description="FastAPI service for skin lesion classification",
    version="1.0"
)

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = BASE_DIR / "models" / "mobilenet" / "mobilenetv2_model.keras"

IMAGE_SIZE = (224, 224)

def prepare_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)

    return image_array

CLASS_NAMES = [
    "Actinic Keratoses",
    "Basal Cell Carcinoma",
    "Benign Keratosis-like Lesions",
    "Dermatofibroma",
    "Melanoma",
    "Melanocytic Nevi",
    "Vascular Lesions"
]

model = tf.keras.models.load_model(MODEL_PATH)

@app.get("/")
def home():
    return {
        "message": "Skin Lesion Classifier API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = prepare_image(image_bytes)

    with tf.device("/CPU:0"):
        probabilities = model(image, training=False).numpy()[0]

    top_indices = np.argsort(probabilities)[-3:][::-1]

    top_predictions = []
    for index in top_indices:
        top_predictions.append({
            "class": CLASS_NAMES[index],
            "confidence": round(float(probabilities[index]) * 100, 2)
        })

    return {
        "filename": file.filename,
        "primary_prediction": top_predictions[0],
        "top_3_predictions": top_predictions
    }