import os
import streamlit as st
from dotenv import load_dotenv
import requests

# Load secrets
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Streamlit config
st.set_page_config(page_title="Medical Assistant Chatbot", layout="centered")
st.title("🩺 AI Medical Assistant")
st.markdown("Describe symptoms to get help with conditions, medicines, or finding a doctor.")

# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Groq API medical chat
def query_groq(message, include_medicine=False, include_doctor=False):
    system_prompt = "You are a professional, safe, and helpful medical assistant. You give general advice only and do not diagnose."

    if include_medicine:
        system_prompt += " When asked, you can suggest general over-the-counter medicines (with usage warnings)."
    
    if include_doctor:
        system_prompt += " If necessary, suggest that the user consult a doctor and recommend types of specialists based on symptoms."

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

# Hugging Face symptom classifier
def classify_symptoms(text):
    api_url = "https://api-inference.huggingface.co/models/julien-c/bert-symptom-classifier"
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}"
    }
    payload = {"inputs": text}

    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        predictions = response.json()
        if isinstance(predictions, list) and predictions:
            top_label = predictions[0][0]["label"]
            confidence = predictions[0][0]["score"]
            return f"🔍 **Possible Symptom Category**: *{top_label}* (Confidence: {confidence:.2f})"
        else:
            return "⚠️ Unable to classify symptom."
    except Exception as e:
        return f"⚠️ HF API Error: {e}"

# Input
user_input = st.text_input("Describe your symptoms or ask a medical question:")

# Options
include_meds = st.checkbox("💊 Suggest general medicines (OTC)")
include_doctor = st.checkbox("👨‍⚕️ Recommend specialist doctor")

# On submit
if user_input:
    st.session_state.chat_history.append(("You", user_input))
    with st.spinner("Analyzing..."):
        hf_result = classify_symptoms(user_input)
        try:
            groq_result = query_groq(user_input, include_meds, include_doctor)
        except Exception as e:
            groq_result = f"⚠️ Groq API Error: {e}"

    combined_response = f"{hf_result}\n\n{groq_result}"
    st.session_state.chat_history.append(("MedicalBot", combined_response))

# Chat display
for speaker, msg in st.session_state.chat_history:
    if speaker == "You":
        st.markdown(f"**🧑 {speaker}:** {msg}")
    else:
        st.markdown(f"**🤖 {speaker}:** {msg}")
