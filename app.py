import os
import streamlit as st
from dotenv import load_dotenv
import requests

# Load API keys from .env
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Streamlit setup
st.set_page_config(page_title="Medical Assistant Chatbot", layout="centered")
st.title("🩺 AI Medical Assistant")
st.markdown("Ask about symptoms, diseases, medications, and treatments.")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Groq API Query
def query_groq(message):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "system", "content": "You are a helpful and trustworthy medical assistant who gives accurate, concise, and safe medical advice. You do not diagnose but suggest what a patient might consider."},
            {"role": "user", "content": message}
        ],
        "temperature": 0.7
    }

    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# User input
user_input = st.text_input("Describe your symptoms or ask a medical question:")

if user_input:
    st.session_state.chat_history.append(("You", user_input))
    with st.spinner("Analyzing your query..."):
        try:
            bot_reply = query_groq(user_input)
        except Exception as e:
            bot_reply = f"⚠️ Error: {e}"
    st.session_state.chat_history.append(("MedicalBot", bot_reply))

# Chat history display
for speaker, msg in st.session_state.chat_history:
    if speaker == "You":
        st.markdown(f"**🧑 {speaker}:** {msg}")
    else:
        st.markdown(f"**🤖 {speaker}:** {msg}")
