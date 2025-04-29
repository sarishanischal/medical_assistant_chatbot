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
st.markdown("Describe symptoms to get help with conditions, medicines, or doctor recommendations.")

# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Query Groq API
def query_groq(message, include_medicine=False, include_doctor=False):
    system_prompt = (
        "You are a professional, safe, and helpful medical assistant. "
        "You give general advice only and do not diagnose or prescribe. "
    )
    if include_medicine:
        system_prompt += "You can suggest over-the-counter medicines with clear safety notes. "
    if include_doctor:
        system_prompt += "Recommend a type of doctor if needed, like dermatologist, pediatrician, etc. "

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

# Hugging Face NER for symptom/medical term extraction
def classify_symptoms(text):
    api_url = "https://api-inference.huggingface.co/models/d4data/biomedical-ner-all"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": text}

    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        predictions = response.json()

        # predictions is a flat list of entities
        filtered = [ent for ent in predictions if ent["entity_group"] in {"SYMPTOM", "DISEASE", "DRUG"}]
        if not filtered:
            return "🔍 No symptoms or medical terms clearly identified."
        
        result_lines = [f"• **{ent['word']}** (*{ent['entity_group']}*)" for ent in filtered]
        return "🔍 **Identified Medical Terms:**\n" + "\n".join(result_lines)

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

    with st.spinner("Analyzing your input..."):
        ner_result = classify_symptoms(user_input)
        try:
            groq_result = query_groq(user_input, include_meds, include_doctor)
        except Exception as e:
            groq_result = f"⚠️ Groq API Error: {e}"

    final_response = f"{ner_result}\n\n{groq_result}"
    st.session_state.chat_history.append(("MedicalBot", final_response))

# Chat display
for speaker, msg in st.session_state.chat_history:
    if speaker == "You":
        st.markdown(f"**🧑 {speaker}:** {msg}")
    else:
        st.markdown(f"**🤖 {speaker}:** {msg}")
