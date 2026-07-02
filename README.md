# Skin Lesion Classifier

AI-based skin lesion classification system using deep learning, transfer learning, FAISS retrieval, FastAPI, Docker, Kubernetes, MLflow, and Streamlit.

This project classifies skin lesion images from the HAM10000 dataset and generates a structured educational medical report using retrieved medical knowledge. The final system includes image prediction, top-3 probabilities, optional user questions, optional document retrieval, API deployment, Docker containerization, Kubernetes deployment, and a Streamlit frontend.

---

## Project Overview

Skin cancer and skin lesion diagnosis is an important healthcare problem. Early detection of dangerous lesions such as melanoma can help support timely medical consultation.

This project uses deep learning to classify skin lesion images into seven categories and combines the prediction with a retrieval-based medical report generator.

The system is designed for educational and academic purposes only. It is not a replacement for a dermatologist or professional medical diagnosis.

---

## Final Pipeline

```text
Skin Image
+ Optional User Question
+ Optional PDF/DOCX/TXT Document
        |
        v
MobileNetV2 Prediction
        |
        v
Top-3 Predictions
        |
        v
FAISS Medical Knowledge Retrieval
        |
        v
Optional User Document Retrieval
        |
        v
Structured Educational Medical Report
        |
        v
FastAPI Backend
        |
        v
Docker Container
        |
        v
Kubernetes Deployment
        |
        v
Streamlit Frontend
```

---

## Dataset

Dataset used:

```text
HAM10000
```

The HAM10000 dataset contains dermatoscopic images of different skin lesions.

Dataset details:

```text
Total images: 10,015
Image size used: 224 x 224
Classes: 7
```

---

## Classes

```text
0 = akiec
1 = bcc
2 = bkl
3 = df
4 = mel
5 = nv
6 = vasc
```

Full class names:

```text
akiec = Actinic Keratoses and Intraepithelial Carcinoma
bcc   = Basal Cell Carcinoma
bkl   = Benign Keratosis-like Lesions
df    = Dermatofibroma
mel   = Melanoma
nv    = Melanocytic Nevi
vasc  = Vascular Lesions
```

---

## Important Dataset Correction

The original dataset split was first done image by image. Later, we discovered that HAM10000 contains multiple images of the same lesion under the same `lesion_id`.

This caused lesion-level data leakage because images of the same lesion could appear in training, validation, and testing sets.

Old leakage result:

```text
Train-Val overlap: 516
Train-Test overlap: 522
Val-Test overlap: 141
```

The issue was corrected by splitting the dataset using unique `lesion_id` values first.

Corrected result:

```text
Train-Val overlap: 0
Train-Test overlap: 0
Val-Test overlap: 0
```

This made the evaluation more realistic and reliable.

---

## Corrected Dataset Split

```text
Training images: 7014
Validation images: 1509
Testing images: 1492
Total images: 10015
```

Corrected CSV files:

```text
data/processed/train.csv
data/processed/validation.csv
data/processed/test.csv
```

---

## Models Used

Two main deep learning models were developed and compared.

### Model 1: Baseline CNN

A custom CNN model was trained from scratch.

Corrected baseline CNN results:

```text
Test accuracy: 45.78%
Weighted F1-score: 49.88%
Macro F1-score: 14.61%
Melanoma recall: 28.30%
Melanoma precision: 16.85%
```

Saved model:

```text
models/baseline/baseline_cnn.keras
```

---

### Model 2: MobileNetV2 Transfer Learning

MobileNetV2 was used as the improved model with ImageNet pretrained weights.

Frozen MobileNetV2 results:

```text
Test accuracy: 69.57%
Weighted F1-score: 71.54%
Macro F1-score: 51.94%
Melanoma recall: 59.75%
Melanoma precision: 40.08%
```

Fine-tuned MobileNetV2 results:

```text
Test accuracy: 71.11%
Weighted F1-score: 72.35%
Macro F1-score: 48.25%
Melanoma recall: 53.46%
Melanoma precision: 45.45%
```

Final selected model:

```text
Frozen MobileNetV2
```

Reason for selection:

```text
The frozen MobileNetV2 model was selected because it had better melanoma recall and better macro F1-score compared to the fine-tuned model.
```

Saved final model:

```text
models/mobilenet/mobilenetv2_model.keras
```

---

## FAISS RAG Medical Report System

The project includes a retrieval-based medical report generator.

A local medical knowledge base was created for all seven HAM10000 classes. The knowledge base contains:

```text
Class name
Description
Typical signs
Risk level
Recommended action
Source
```

The text was converted into embeddings using Sentence Transformers and stored inside a FAISS index.

Saved files:

```text
rag_data/documents/skin_lesion_knowledge.json
rag_data/index/skin_lesion_faiss.index
```

Embedding model:

```text
all-MiniLM-L6-v2
```

The RAG system retrieves relevant medical information based on the predicted lesion and generates a structured educational report.

---

## Optional User Document RAG

The system also supports optional document upload.

Supported document types:

```text
PDF
DOCX
TXT
```

The uploaded document is processed as follows:

```text
Document upload
→ Text extraction
→ Text chunking
→ Embedding generation
→ Temporary FAISS index
→ Relevant text retrieval
→ Added to final medical report
```

Important note:

```text
The uploaded document does not change the image prediction.
It only adds supporting retrieved information to the medical report.
```

---

## Streamlit Frontend

The final frontend is built using Streamlit.

Main features:

```text
Skin image upload
Small image preview
Optional user question
Optional PDF/DOCX/TXT document upload
Analyze Skin Lesion button
Primary prediction display
Top-3 probability bars
Structured educational medical report
Medical disclaimer
Downloadable report
```

Run Streamlit app:

```bash
streamlit run app/streamlit_app.py
```

Download file generated by the app:

```text
skin_lesion_medical_report.txt
```

---

## FastAPI Backend

FastAPI was added to serve the trained model as an API.

FastAPI file:

```text
deployment/api/main.py
```

API endpoints:

```text
GET /
GET /health
POST /predict
```

Run FastAPI locally:

```bash
uvicorn deployment.api.main:app --reload --port 8000
```

Open API:

```text
http://127.0.0.1:8000
```

Open health endpoint:

```text
http://127.0.0.1:8000/health
```

Open Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

The `/predict` endpoint accepts an uploaded skin lesion image and returns:

```text
Filename
Primary prediction
Top-3 predictions
Confidence scores
```

---

## Docker Containerization

The FastAPI backend was containerized using Docker.

Created files:

```text
Dockerfile
requirements.txt
.dockerignore
```

Build Docker image:

```bash
sudo docker build -t skin-lesion-api .
```

Run Docker container:

```bash
sudo docker run --name skin-lesion-api-container -p 8001:8000 skin-lesion-api
```

Open Docker API:

```text
http://127.0.0.1:8001
```

Open Docker health endpoint:

```text
http://127.0.0.1:8001/health
```

Open Docker Swagger docs:

```text
http://127.0.0.1:8001/docs
```

Docker makes the FastAPI backend portable and ready for deployment.

---

## Kubernetes Deployment

The Dockerized FastAPI backend was deployed locally using Kubernetes with Minikube.

Kubernetes files:

```text
kubernetes/
├── deployment.yaml
└── service.yaml
```

The deployment uses the Docker image:

```text
skin-lesion-api:latest
```

Main Kubernetes commands used:

```bash
minikube start --driver=docker
minikube image load skin-lesion-api:latest
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
kubectl get pods
kubectl get services
minikube service skin-lesion-api-service --url
```

Kubernetes deployment verified:

```text
Pod status: Running
Pod ready: 1/1
Service type: NodePort
API home endpoint working
Health endpoint working
Predict endpoint working
```

The `/predict` endpoint successfully returned top-3 skin lesion predictions inside Kubernetes.

---

## MLflow Experiment Tracking

MLflow was used to track model experiments.

Tracked runs:

```text
Baseline CNN
Frozen MobileNetV2
Fine-Tuned MobileNetV2
Final Selected Model Registration
```

Tracked metrics:

```text
Test accuracy
Weighted F1-score
Macro F1-score
Melanoma recall
Melanoma precision
```

Final registered model:

```text
SkinLesionMobileNetV2
```

MLflow was used to compare model runs and register the final selected MobileNetV2 model.

---

## GitHub Actions CI/CD

GitHub Actions was added for CI/CD.

Workflow file:

```text
.github/workflows/ci.yml
```

The workflow runs on:

```text
Push to main
Pull request to main
```

The CI pipeline checks Python syntax in:

```text
app/
deployment/
```

Main CI command:

```bash
python -m compileall app deployment
```

This confirms that the Streamlit frontend and FastAPI backend Python files do not contain syntax errors.

---

## Project Structure

```text
skin-lesion-classifier/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── processed/
│   │   ├── train.csv
│   │   ├── validation.csv
│   │   └── test.csv
│   └── raw/
│
├── deployment/
│   └── api/
│       └── main.py
│
├── docs/
│
├── kubernetes/
│   ├── deployment.yaml
│   └── service.yaml
│
├── models/
│   ├── baseline/
│   │   └── baseline_cnn.keras
│   └── mobilenet/
│       ├── mobilenetv2_model.keras
│       └── mobilenetv2_finetuned.keras
│
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_baseline_cnn.ipynb
│   ├── 03_mobilenetv2_transfer_learning.ipynb
│   ├── 04_nlp_report_faiss_rag.ipynb
│   ├── 05_gradio_integration.ipynb
│   └── 06_mlflow_tracking.ipynb
│
├── rag_data/
│   ├── documents/
│   │   └── skin_lesion_knowledge.json
│   └── index/
│       └── skin_lesion_faiss.index
│
├── results/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── Dockerfile
├── requirements.txt
├── .dockerignore
├── .gitignore
└── README.md
```

---

## Technologies Used

```text
Python
TensorFlow
Keras
MobileNetV2
NumPy
Pandas
Scikit-learn
FAISS
Sentence Transformers
Streamlit
FastAPI
Docker
Kubernetes
Minikube
MLflow
GitHub Actions
```

---

## How To Run The Project

### 1. Activate Virtual Environment

```bash
cd ~/projects/skin-lesion-classifier
source .venv/bin/activate
```

### 2. Run Streamlit Frontend

```bash
streamlit run app/streamlit_app.py
```

### 3. Run FastAPI Backend

```bash
uvicorn deployment.api.main:app --reload --port 8000
```

### 4. Run Docker Container

```bash
sudo docker build -t skin-lesion-api .
sudo docker run --name skin-lesion-api-container -p 8001:8000 skin-lesion-api
```

### 5. Run Kubernetes Deployment

```bash
minikube start --driver=docker
minikube image load skin-lesion-api:latest
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
kubectl get pods
kubectl get services
minikube service skin-lesion-api-service --url
```

---

## Main Features

```text
Skin lesion image classification
Top-3 prediction probabilities
MobileNetV2 transfer learning
Baseline CNN comparison
Lesion-level leakage correction
FAISS medical knowledge retrieval
Structured medical report generation
Optional user question support
Optional PDF/DOCX/TXT document retrieval
Streamlit frontend
Downloadable medical report
FastAPI model serving
Docker containerization
Kubernetes deployment
MLflow experiment tracking
GitHub Actions CI/CD
```

---

## Limitations

```text
The model is trained on HAM10000 and may not generalize perfectly to all real-world images.
The system is not a medical diagnostic tool.
The uploaded document does not change the model prediction.
Scanned PDFs without selectable text are not supported.
The report is structured and retrieval-based, not a full conversational medical chatbot.
The Kubernetes deployment is local using Minikube, not a cloud deployment.
```

---

## Future Work

```text
Improve model performance using more balanced data.
Add OCR support for scanned PDFs.
Add a large language model for more natural report generation.
Deploy the system to a cloud server.
Add user authentication.
Improve frontend analytics and visualizations.
Add more medical sources to the knowledge base.
```

---

## Medical Disclaimer

This project is for educational and academic purposes only.

It does not provide medical diagnosis and should not be used as a replacement for a dermatologist or qualified medical professional.

For any suspicious, changing, painful, bleeding, or unusual skin lesion, consult a dermatologist.

---

## Author

```text
Skin Lesion Classifier Semester Project
Developed by Arslan Shaukat
```
