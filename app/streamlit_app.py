import streamlit as st
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

import json
import tempfile
import numpy as np
import faiss
import streamlit as st

from sentence_transformers import SentenceTransformer
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from pypdf import PdfReader
from docx import Document


project_path = "/home/arslan/projects/skin-lesion-classifier"

model_path = project_path + "/models/mobilenet/mobilenetv2_model.keras"
knowledge_path = project_path + "/rag_data/documents/skin_lesion_knowledge.json"
faiss_index_path = project_path + "/rag_data/index/skin_lesion_faiss.index"

required_files = {
    "MobileNetV2 model": model_path,
    "Knowledge base": knowledge_path,
    "FAISS index": faiss_index_path
}

missing_files = []

for name, path in required_files.items():
    if not os.path.exists(path):
        missing_files.append(name)

if missing_files:
    st.error("Missing files: " + ", ".join(missing_files))




class_names = {
    "akiec": "Actinic Keratoses and Intraepithelial Carcinoma",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis-like Lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevi",
    "vasc": "Vascular Lesions"
}

class_codes = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc"
]

@st.cache_resource
def load_backend_files():
    model = tf.keras.models.load_model(
        model_path,
        compile=False
    )

    with open(knowledge_path, "r", encoding="utf-8") as file:
        knowledge_base = json.load(file)

    faiss_index = faiss.read_index(faiss_index_path)

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    return model, knowledge_base, faiss_index, embedding_model

 

IMAGE_SIZE = (224, 224)

def prepare_image(image_path):
    image = load_img(
        image_path,
        target_size=IMAGE_SIZE
    )

    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    return image



def predict_top3(image_path, model):
    image = prepare_image(image_path)

    with tf.device("/CPU:0"):
        probabilities = model(
            image,
            training=False
        ).numpy()[0]

    top_indices = np.argsort(probabilities)[-3:][::-1]

    top_predictions = []

    for index in top_indices:
        code = class_codes[index]

        top_predictions.append({
            "class_code": code,
            "class_name": class_names[code],
            "probability": float(probabilities[index])
        })

    return top_predictions



def retrieve_knowledge(query, faiss_index, embedding_model, knowledge_base, top_k=1):
    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = faiss_index.search(
        query_embedding,
        k=top_k
    )

    results = []

    for score, index in zip(scores[0], indices[0]):
        result = knowledge_base[index].copy()
        result["similarity_score"] = float(score)
        results.append(result)

    return results


def extract_document_text(file_path):
    lower_path = file_path.lower()

    if lower_path.endswith(".pdf"):
        reader = PdfReader(file_path)

        text = "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )

    elif lower_path.endswith(".docx"):
        document = Document(file_path)

        text = "\n".join(
            paragraph.text
            for paragraph in document.paragraphs
        )

    elif lower_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()

    else:
        raise ValueError("Only PDF, DOCX and TXT files are supported.")

    if not text.strip():
        raise ValueError("No readable text was found in the document.")

    return text


def split_text_into_chunks(text, chunk_size=500):
    words = text.split()
    chunks = []

    for start in range(0, len(words), chunk_size):
        chunk = " ".join(
            words[start:start + chunk_size]
        )

        chunks.append(chunk)

    return chunks


def build_document_index(file_path, embedding_model):
    text = extract_document_text(file_path)
    chunks = split_text_into_chunks(text)

    chunk_embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    document_index = faiss.IndexFlatIP(
        chunk_embeddings.shape[1]
    )

    document_index.add(chunk_embeddings)

    return chunks, document_index


def retrieve_from_document(query, document_chunks, document_index, embedding_model, top_k=3):
    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = document_index.search(
        query_embedding,
        k=top_k
    )

    results = []

    for score, index in zip(scores[0], indices[0]):
        results.append({
            "text": document_chunks[index],
            "similarity_score": float(score)
        })

    return results


def generate_extended_medical_report(
    top_predictions,
    retrieved_information,
    user_question="",
    document_results=None
):
    report = "## AI Skin Lesion Analysis Report\n\n"

    report += "### Primary Prediction\n"
    report += f"**{top_predictions[0]['class_name']}**\n\n"

    report += "### Top-3 Predictions\n"
    for prediction in top_predictions:
        percentage = prediction["probability"] * 100
        report += f"- {prediction['class_name']}: {percentage:.2f}%\n"

    if user_question:
        report += "\n### User Question\n"
        report += f"{user_question}\n"

    report += "\n### Medical Information\n"
    report += f"**Description:** {retrieved_information['description']}\n\n"
    report += f"**Typical Signs:** {retrieved_information['typical_signs']}\n\n"
    report += f"**Risk Level:** {retrieved_information['risk_level']}\n\n"
    report += f"**Recommended Action:** {retrieved_information['recommended_action']}\n\n"

    if document_results:
        report += "### Uploaded Document Information\n"
        for result in document_results:
            report += f"- {result['text']}\n"

    report += "\n---\n"
    report += "**Disclaimer:** This system is for educational purposes only and is not a replacement for professional dermatologist diagnosis."

    return report


def run_pipeline_with_document(
    image_path,
    model,
    knowledge_base,
    faiss_index,
    embedding_model,
    user_question="",
    document_path=None
):
    top_predictions = predict_top3(
        image_path,
        model
    )

    predicted_class = top_predictions[0]["class_name"]

    query = (
        f"Medical information, typical signs, risk level, "
        f"and recommended action for {predicted_class}. "
        f"{user_question}"
    )

    retrieved_information = retrieve_knowledge(
        query,
        faiss_index,
        embedding_model,
        knowledge_base,
        top_k=1
    )[0]

    document_results = []

    if document_path:
        document_chunks, document_index = build_document_index(
            document_path,
            embedding_model
        )

        document_query = f"{predicted_class}. {user_question}"

        document_results = retrieve_from_document(
            document_query,
            document_chunks,
            document_index,
            embedding_model,
            top_k=3
        )

    report = generate_extended_medical_report(
        top_predictions,
        retrieved_information,
        user_question,
        document_results
    )

    return top_predictions, report    



st.set_page_config(
    page_title="AI Skin Lesion Analysis",
    page_icon="🩺",
    layout="wide"
)

st.markdown(
    """
    <style>
    .main {
        background-color: #0b1120;
    }

    .hero {
        padding: 35px;
        border-radius: 22px;
        background: linear-gradient(135deg, #0f172a, #1e3a8a);
        border: 1px solid #334155;
        margin-bottom: 25px;
    }

    .hero h1 {
        color: white;
        font-size: 42px;
        margin-bottom: 10px;
    }

    .hero p {
        color: #cbd5e1;
        font-size: 18px;
    }

    .disclaimer {
        padding: 15px;
        border-radius: 12px;
        background-color: #422006;
        color: #fde68a;
        border: 1px solid #92400e;
        margin-top: 15px;
    }

            div.stButton > button {
        width: 260px;
        height: 54px;
        border-radius: 16px;
        background: linear-gradient(135deg, #38bdf8, #2563eb, #1d4ed8);
        background-size: 200% 200%;
        color: white;
        font-size: 16px;
        font-weight: 700;
        border: none;
        box-shadow: 0px 10px 24px rgba(37, 99, 235, 0.35);
        transition: all 0.25s ease-in-out;
        animation: buttonGlow 3s ease infinite;
    }

    .prediction-panel {
        padding: 24px;
        border-radius: 22px;
        background: rgba(17, 24, 39, 0.72);
        border: 1px solid rgba(148, 163, 184, 0.16);
        box-shadow: 0px 16px 40px rgba(0, 0, 0, 0.28);
        backdrop-filter: blur(14px);
        margin-top: 8px;
    }

    .prediction-title {
        color: #94a3b8;
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .primary-prediction {
        padding: 18px 20px;
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(34,197,94,0.16), rgba(59,130,246,0.12));
        border: 1px solid rgba(74, 222, 128, 0.25);
        margin-bottom: 22px;
    }

    .report-heading {
        margin-top: 28px;
        margin-bottom: 14px;
        font-size: 28px;
        font-weight: 800;
        color: #f8fafc;
    }

    .report-subtitle {
        color: #94a3b8;
        font-size: 14px;
        margin-bottom: 14px;
    }
    
    .primary-name {
        color: #ffffff;
        font-size: 24px;
        font-weight: 800;
        margin-bottom: 4px;
    }

    .primary-confidence {
        color: #86efac;
        font-size: 14px;
        font-weight: 600;
    }

    .prediction-item {
        margin-bottom: 18px;
    }

    .prediction-line {
        display: flex;
        justify-content: space-between;
        align-items: center;
        color: #e5e7eb;
        font-size: 15px;
        font-weight: 600;
        margin-bottom: 7px;
    }

    .percentage-pill {
        color: #bfdbfe;
        font-size: 13px;
        background: rgba(37, 99, 235, 0.16);
        border: 1px solid rgba(96, 165, 250, 0.2);
        padding: 3px 10px;
        border-radius: 999px;
    }

    .progress-track {
        width: 100%;
        height: 9px;
        border-radius: 999px;
        background: rgba(30, 41, 59, 0.95);
        overflow: hidden;
    }

    .progress-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #60a5fa, #2563eb);
    }

    div.stButton > button:hover {
        transform: translateY(-3px) scale(1.03);
        box-shadow: 0px 16px 32px rgba(56, 189, 248, 0.45);
        cursor: pointer;
    }

    div.stButton > button:active {
        transform: scale(0.97);
        box-shadow: 0px 6px 14px rgba(37, 99, 235, 0.35);
    }

    @keyframes buttonGlow {
        0% {
            background-position: 0% 50%;
        }
        50% {
            background-position: 100% 50%;
        }
        100% {
            background-position: 0% 50%;
        }
    }
    </style>

    <div class="hero">
        <h1>🩺 AI Skin Lesion Analysis System</h1>
        <p>
            A deep-learning based educational tool for skin-lesion prediction,
            top-3 probability analysis, and medical knowledge retrieval.
        </p>
        <div class="disclaimer">
            Educational use only — this system does not replace professional dermatologist diagnosis.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

with st.spinner("Loading model and medical knowledge files..."):
    model, knowledge_base, faiss_index, embedding_model = load_backend_files()


left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("Upload And Analyze")

    image_file = st.file_uploader(
        "Upload Skin Lesion Image",
        type=["jpg", "jpeg", "png"],
        key="skin_image_uploader"
    )

    if image_file is not None:
        st.markdown("**Image Preview**")

        preview_col, empty_col = st.columns([1, 2])

        with preview_col:
            st.image(
                image_file,
                caption="Uploaded Image",
                width=280
            )

    user_question = st.text_area(
        "Optional Question",
        placeholder="Example: What is the risk level?"
    )

    document_file = st.file_uploader(
        "Optional Medical Document",
        type=["pdf", "docx", "txt"],
        key="medical_document_uploader"
    )

    analyze_button = st.button("Analyze Skin Lesion")

report_text = None
top_predictions = None

if analyze_button:
    if image_file is None:
        st.error("Please upload a skin lesion image first.")
    else:
        with st.spinner("Analyzing skin lesion..."):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_image:
                    temp_image.write(image_file.getbuffer())
                    image_path = temp_image.name

                document_path = None

                if document_file is not None:
                    file_suffix = "." + document_file.name.split(".")[-1]

                    with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as temp_doc:
                        temp_doc.write(document_file.getbuffer())
                        document_path = temp_doc.name

                top_predictions, report_text = run_pipeline_with_document(
                    image_path=image_path,
                    model=model,
                    knowledge_base=knowledge_base,
                    faiss_index=faiss_index,
                    embedding_model=embedding_model,
                    user_question=user_question,
                    document_path=document_path
                )

            except Exception as error:
                st.error(f"Error: {error}")

with right_col:
    st.subheader("Prediction Results")

    if top_predictions:
        primary_name = top_predictions[0]["class_name"]

        st.success("Primary Prediction")

        st.markdown(
            f"""
            <h1 style="
                font-size: 42px;
                font-weight: 800;
                color: #f8fafc;
                margin-top: 8px;
                margin-bottom: 28px;
            ">
                {primary_name}
            </h1>
            """,
            unsafe_allow_html=True
        )

        st.markdown("### Top-3 Probabilities")

        for prediction in top_predictions:
            name = prediction["class_name"]
            percent = prediction["probability"] * 100

            st.write(f"**{name}**")
            st.progress(percent / 100)
            st.caption(f"{percent:.2f}%")

    else:
        st.info("Upload an image and click Analyze to view prediction results.")

st.markdown(
    """
    <div class="report-heading">Educational Medical Report</div>
    <div class="report-subtitle">
        Structured educational summary generated from the model prediction and medical knowledge retrieval.
    </div>
    """,
    unsafe_allow_html=True
)

if report_text:
    with st.container(border=True):
        st.markdown(report_text)

    st.download_button(
        label="Download Medical Report",
        data=report_text,
        file_name="skin_lesion_medical_report.txt",
        mime="text/plain"
    )
else:
    with st.container(border=True):
        st.info("Medical report will appear here after analysis.")