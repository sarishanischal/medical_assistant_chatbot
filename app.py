import os
import streamlit as st
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

st.set_page_config(page_title="Medical Assistant Chatbot", layout="centered")
st.title("🩺 AI Medical Assistant")
st.markdown("Get instant help for symptoms, medical conditions, and recommended treatments.")

# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Text input
user_input = st.text_input("Describe your symptoms or ask a medical question:")

# Groq LLM API call
def query_groq(message):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "messages": [{"role": "user", "content": message}],
        "model": "mixtral-8x7b-32768"  # or use "llama3-8b-8192" or another supported model
    }

    response = requests.post("https://api.groq.com/openai/v1/chat/completions", json=data, headers=headers)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# Submit logic
if user_input:
    st.session_state.chat_history.append(("You", user_input))
    with st.spinner("Analyzing..."):
        try:
            bot_reply = query_groq(user_input)
        except Exception as e:
            bot_reply = f"⚠️ Error: {e}"
    st.session_state.chat_history.append(("MedicalBot", bot_reply))

# Display chat
for speaker, msg in st.session_state.chat_history:
    if speaker == "You":
        st.markdown(f"**🧑 {speaker}:** {msg}")
    else:
        st.markdown(f"**🤖 {speaker}:** {msg}")
