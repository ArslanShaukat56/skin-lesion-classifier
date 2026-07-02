# Skin Lesion Classifier

This project is a deep learning based skin lesion classification system using the HAM10000 dataset.
It combines image classification, medical information retrieval, report generation, FastAPI backend serving, Docker containerization, and a Streamlit frontend.

## Project Overview

The system accepts a skin lesion image and predicts the most likely lesion class using a trained MobileNetV2 model.
It also generates an educational medical report using a FAISS-based retrieval system.

The final pipeline is:

```text
Skin Image
+ Optional User Question
+ Optional PDF/DOCX/TXT Document
        ↓
MobileNetV2 Prediction
        ↓
Top-3 Predictions
        ↓
FAISS Medical Knowledge Retrieval
        ↓
Optional User Document Retrieval
        ↓
Structured Educational Medical Report
        ↓
FastAPI Backend
        ↓
Docker Container
        ↓
Streamlit Frontend
```

## Dataset

Dataset used:

```text
HAM10000 Skin Lesion Dataset
```

The dataset contains 10,015 dermatoscopic images from 7 skin lesion classes.

Class labels:

```text
akiec = Actinic Keratoses and Intraepithelial Carcinoma
bcc   = Basal Cell Carcinoma
bkl   = Benign Keratosis-like Lesions
df    = Dermatofibroma
mel   = Melanoma
nv    = Melanocytic Nevi
vasc  = Vascular Lesions
```

A lesion-level split was used to avoid data leakage.
Images from the same lesion were kept in the same train, validation, or test set.

## Models

Two main deep learning models were developed:

1. Baseline CNN
2. MobileNetV2 Transfer Learning

The final selected model is the frozen MobileNetV2 model because it achieved better melanoma recall and better macro F1-score compared to the fine-tuned version.

## Main Technologies

```text
Python
TensorFlow / Keras
MobileNetV2
FAISS
Sentence Transformers
FastAPI
Docker
Streamlit
MLflow
GitHub Actions
```

## Project Structure

```text
skin-lesion-classifier/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   └── processed/
│
├── deployment/
│   └── api/
│       └── main.py
│
├── models/
│   ├── baseline/
│   └── mobilenet/
│
├── notebooks/
│
├── rag_data/
│   ├── documents/
│   └── index/
│
├── results/
│
├── Dockerfile
├── requirements.txt
└── README.md
```

## Run Streamlit Frontend

```bash
streamlit run app/streamlit_app.py
```

## Run FastAPI Backend Locally

```bash
uvicorn deployment.api.main:app --reload --port 8000
```

API endpoints:

```text
GET  /
GET  /health
POST /predict
```

FastAPI documentation:

```text
http://127.0.0.1:8000/docs
```

## Run With Docker

Build the Docker image:

```bash
sudo docker build -t skin-lesion-api .
```

Run the container:

```bash
sudo docker run --name skin-lesion-api-container -p 8001:8000 skin-lesion-api
```

Open in browser:

```text
http://127.0.0.1:8001
http://127.0.0.1:8001/health
http://127.0.0.1:8001/docs
```

## Features

* Skin lesion image classification
* Top-3 prediction probabilities
* Structured educational medical report
* FAISS medical knowledge retrieval
* Optional user question support
* Optional PDF/DOCX/TXT document retrieval
* FastAPI backend
* Docker containerization
* Streamlit frontend
* Downloadable medical report
* MLflow experiment tracking
* GitHub Actions CI/CD

## Disclaimer

This project is for educational purposes only.
It is not a medical diagnosis system and should not replace consultation with a qualified dermatologist or healthcare professional.
