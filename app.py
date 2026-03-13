import streamlit as st
import json
import random
import os
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# ==============================
# TEXT PREPROCESSING
# ==============================

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text


# ==============================
# LOAD INTENTS + TRAIN MODEL
# ==============================

@st.cache_resource
def load_model():

    path = os.path.join(os.path.dirname(__file__), "intents.json")

    with open(path) as f:
        intents = json.load(f)

    patterns = []
    tags = []

    for intent in intents:
        for pattern in intent["patterns"]:
            patterns.append(preprocess(pattern))
            tags.append(intent["tag"])

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(patterns)

    model = LogisticRegression(max_iter=1000)
    model.fit(X, tags)

    return intents, model, vectorizer


# ==============================
# GET CHATBOT RESPONSE
# ==============================

def get_response(user_input, intents, model, vectorizer):

    processed = preprocess(user_input)

    X = vectorizer.transform([processed])

    tag = model.predict(X)[0]

    for intent in intents:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

    return "Sorry, I didn't understand that."


# ==============================
# MAIN STREAMLIT APP
# ==============================

def main():

    st.title("🤖 AI Intent-Based NLP Chatbot")
    st.caption("Built with Python, Streamlit and Scikit-Learn")

    intents, model, vectorizer = load_model()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "user_input" not in st.session_state:
        st.session_state.user_input = ""

    # User input box
    user_input = st.text_input("You:", key="user_input")

    # Send button
    if st.button("Send"):

        if user_input.strip() != "":

            response = get_response(user_input, intents, model, vectorizer)

            st.session_state.chat_history.append(("You", user_input))
            st.session_state.chat_history.append(("Bot", response))

            st.session_state.user_input = ""

    st.divider()

    # Chat history display
    for speaker, message in st.session_state.chat_history:

        if speaker == "You":
            st.markdown(f"🧑 **You:** {message}")
        else:
            st.markdown(f"🤖 **Bot:** {message}")


# ==============================
# RUN APP
# ==============================

if __name__ == "__main__":
    main()
