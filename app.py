import os
import streamlit as st
from dotenv import load_dotenv
import requests
from PIL import Image
import io
import fitz  # PyMuPDF
import joblib
import numpy as np
from huggingface_hub import hf_hub_download

# Load secrets
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

st.set_page_config(page_title="Medical Assistant Chatbot", layout="centered")
st.title("🩺 AI Medical Assistant")
st.markdown("Describe your symptoms, check diabetes risk, or upload medical reports for interpretation.")

# Load Diabetes Prediction Model from Hugging Face
@st.cache_resource
def load_diabetes_model():
    model_path = hf_hub_download(
        repo_id="jaik256/diebateRandomForest1",
        filename="model.pkl",
        token=HF_TOKEN
    )
    return joblib.load(model_path)

diabetes_model = load_diabetes_model()

# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# File Uploads
st.subheader("📑 Upload Medical Report (Image or PDF)")
uploaded_file = st.file_uploader("Supported: Blood report (PDF), ECG/X-ray (Image)", type=["pdf", "png", "jpg", "jpeg"])

# ---------- GROQ Function ----------
def query_groq(message, include_medicine=False, include_doctor=False):
    system_prompt = (
        "You are a professional, safe, and helpful medical assistant. "
        "You explain medical content in simple terms. "
    )
    if include_medicine:
        system_prompt += "You can suggest general over-the-counter medicines. "
    if include_doctor:
        system_prompt += "You can recommend a medical specialist. "

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ],
        "temperature": 0.7
    }

    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# ---------- HUGGING FACE NER ----------
def classify_symptoms(text):
    api_url = "https://api-inference.huggingface.co/models/d4data/biomedical-ner-all"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": text}

    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        predictions = response.json()

        filtered = [ent for ent in predictions if ent["entity_group"] in {"SYMPTOM", "DISEASE", "DRUG"}]
        if not filtered:
            return "🔍 No symptoms or medical terms clearly identified."

        result_lines = [f"• **{ent['word']}** (*{ent['entity_group']}*)" for ent in filtered]
        return "🔍 **Identified Medical Terms:**\n" + "\n".join(result_lines)

    except Exception as e:
        return f"⚠️ HF API Error: {e}"

# ---------- HUGGING FACE IMAGE INTERPRETER ----------
def describe_image(img_bytes):
    api_url = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    files = {"inputs": img_bytes}

    try:
        response = requests.post(api_url, headers=headers, files=files)
        response.raise_for_status()
        caption = response.json()[0]["generated_text"]
        return f"🖼️ **Image Interpretation:** {caption}"
    except Exception as e:
        return f"⚠️ Image interpretation error: {e}"

# ---------- PDF Extractor ----------
def extract_text_from_pdf(uploaded_pdf):
    text = ""
    try:
        doc = fitz.open(stream=uploaded_pdf.read(), filetype="pdf")
        for page in doc:
            text += page.get_text()
        return text.strip()
    except Exception as e:
        return f"⚠️ PDF extract error: {e}"

# ---------- Process Uploaded Report ----------
if uploaded_file:
    if uploaded_file.type.startswith("image/"):
        st.image(uploaded_file, caption="Uploaded Report", use_column_width=True)
        with st.spinner("Analyzing image..."):
            image_result = describe_image(uploaded_file)
            simplified_explanation = query_groq(image_result)
            st.markdown(image_result)
            st.markdown(f"🧾 **Simplified Explanation:**\n{simplified_explanation}")

    elif uploaded_file.type == "application/pdf":
        with st.spinner("Extracting text from blood report..."):
            pdf_text = extract_text_from_pdf(uploaded_file)
            if pdf_text.startswith("⚠️"):
                st.error(pdf_text)
            else:
                st.markdown("🧾 **Extracted Report Text (Preview):**")
                st.text(pdf_text[:1000])  # Show preview
                with st.spinner("Interpreting report..."):
                    simplified_explanation = query_groq(f"Explain this medical report in simple language:\n{pdf_text}")
                    st.markdown(f"🩺 **Simplified Explanation:**\n{simplified_explanation}")

# ---------- Diabetes Prediction UI ----------
st.subheader("🧪 Diabetes Risk Prediction")

pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, step=1)
glucose = st.number_input("Glucose Level", min_value=0.0, max_value=300.0, step=1.0)
blood_pressure = st.number_input("Blood Pressure", min_value=0.0, max_value=200.0, step=1.0)
skin_thickness = st.number_input("Skin Thickness", min_value=0.0, max_value=100.0, step=1.0)
insulin = st.number_input("Insulin", min_value=0.0, max_value=900.0, step=1.0)
bmi = st.number_input("BMI", min_value=0.0, max_value=60.0, step=0.1)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, step=0.01)
age = st.number_input("Age", min_value=0, max_value=120)

if st.button("Predict Diabetes Risk"):
    input_data = [[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        dpf,
        age
    ]]
    try:
        prediction = diabetes_model.predict(input_data)[0]
        if prediction == 1:
            st.error("⚠️ High Risk of Diabetes Detected")
        else:
            st.success("✅ Low Risk of Diabetes")
    except Exception as e:
        st.warning(f"Prediction error: {e}")

# ---------- Main Chatbot UI ----------
st.subheader("💬 Symptom Checker and Medical Q&A")
user_input = st.text_input("Describe your symptoms or ask a medical question:")
include_meds = st.checkbox("💊 Suggest general medicines (OTC)")
include_doctor = st.checkbox("👨‍⚕️ Recommend specialist doctor")

if user_input:
    st.session_state.chat_history.append(("You", user_input))
    with st.spinner("Processing..."):
        ner_result = classify_symptoms(user_input)
        try:
            groq_result = query_groq(user_input, include_meds, include_doctor)
        except Exception as e:
            groq_result = f"⚠️ Groq API Error: {e}"
    final_response = f"{ner_result}\n\n{groq_result}"
    st.session_state.chat_history.append(("MedicalBot", final_response))

# Chat history
for speaker, msg in st.session_state.chat_history:
    if speaker == "You":
        st.markdown(f"**🧑 {speaker}:** {msg}")
    else:
        st.markdown(f"**🤖 {speaker}:** {msg}")
