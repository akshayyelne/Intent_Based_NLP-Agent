import streamlit as st
import json
import random
import string
import os
import csv
import datetime

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# ======================================
# SAFE NLTK SETUP
# ======================================

@st.cache_resource
def setup_nltk():

    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)

    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    return lemmatizer, stop_words


lemmatizer, stop_words = setup_nltk()


# ======================================
# TEXT PREPROCESSING
# ======================================

def preprocess(text):

    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))

    tokens = nltk.word_tokenize(text)

    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words
    ]

    return " ".join(tokens)


# ======================================
# LOAD INTENTS
# ======================================

@st.cache_resource
def load_intents():

    file_path = os.path.join(
        os.path.dirname(__file__),
        "intents.json"
    )

    with open(file_path, "r") as f:
        return json.load(f)


# ======================================
# TRAIN MODEL
# ======================================

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


# ======================================
# CHATBOT RESPONSE
# ======================================

def get_response(user_input):

    processed = preprocess(user_input)

    X = vectorizer.transform([processed])

    tag = model.predict(X)[0]

    for intent in intents:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

    return "Sorry, I didn't understand that."


# ======================================
# STREAMLIT UI
# ======================================

st.title("Intent-Based NLP Chatbot")

st.write("Ask me a question.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("You:")

if user_input:

    response = get_response(user_input)

    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", response))

    log_file = os.path.join(os.path.dirname(__file__), "chat_log.csv")

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


# ======================================
# DISPLAY CHAT HISTORY
# ======================================

for speaker, message in st.session_state.chat_history:

    if speaker == "You":
        st.markdown(f"**You:** {message}")
    else:
        st.markdown(f"**Bot:** {message}")
