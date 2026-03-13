import streamlit as st
import json
import random
import string
import os
import csv
import datetime

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# ==============================
# NLTK SAFE SETUP
# ==============================

@st.cache_resource
def setup_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except:
        nltk.download("punkt")

    try:
        nltk.data.find("corpora/stopwords")
    except:
        nltk.download("stopwords")

    try:
        nltk.data.find("corpora/wordnet")
    except:
        nltk.download("wordnet")


setup_nltk()


lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


# ==============================
# TEXT PREPROCESSING
# ==============================

def preprocess(text):

    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))

    tokens = nltk.word_tokenize(text)

    tokens = [
        lemmatizer.lemmatize(w)
        for w in tokens
        if w not in stop_words
    ]

    return " ".join(tokens)


# ==============================
# LOAD INTENTS
# ==============================

@st.cache_resource
def load_intents():

    with open("intents.json") as f:
        return json.load(f)


# ==============================
# TRAIN MODEL
# ==============================

@st.cache_resource
def train_model(intents):

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

    return model, vectorizer


intents = load_intents()
model, vectorizer = train_model(intents)


# ==============================
# CHATBOT RESPONSE
# ==============================

def get_response(user_input):

    processed = preprocess(user_input)

    X = vectorizer.transform([processed])

    tag = model.predict(X)[0]

    for intent in intents:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

    return "Sorry, I didn't understand that."


# ==============================
# STREAMLIT UI
# ==============================

st.title("Intent-Based NLP Chatbot")

st.write("Ask me a question.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("You")

if user_input:

    response = get_response(user_input)

    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", response))

    log_file = "chat_log.csv"

    if not os.path.exists(log_file):
        with open(log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["User", "Bot", "Timestamp"])

    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            user_input,
            response,
            datetime.datetime.now()
        ])

    st.rerun()


# ==============================
# DISPLAY CHAT HISTORY
# ==============================

for speaker, message in st.session_state.chat_history:

    if speaker == "You":
        st.markdown(f"**You:** {message}")

    else:
        st.markdown(f"**Bot:** {message}")
