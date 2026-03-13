import os
import json
import random
import string
import datetime
import csv
import ssl

import nltk
import numpy as np
import streamlit as st

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, ne_chunk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================================
# SAFE NLTK SETUP (STREAMLIT CLOUD SAFE)
# ==========================================================

ssl._create_default_https_context = ssl._create_unverified_context


@st.cache_resource
def setup_nltk():
    packages = [
        "punkt",
        "wordnet",
        "stopwords",
        "averaged_perceptron_tagger",
        "maxent_ne_chunker",
        "words"
    ]

    for pkg in packages:
        try:
            nltk.data.find(pkg)
        except LookupError:
            nltk.download(pkg)


setup_nltk()

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


# ==========================================================
# TEXT PREPROCESSING
# ==========================================================

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))

    tokens = nltk.word_tokenize(text)

    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words
    ]

    return " ".join(tokens)


def preprocess_for_similarity(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.strip()


def extract_entities(text):

    tokens = nltk.word_tokenize(text)
    tagged = pos_tag(tokens)
    chunks = ne_chunk(tagged)

    entities = []

    for chunk in chunks:
        if hasattr(chunk, "label"):
            entity = " ".join(c[0] for c in chunk)
            entities.append((entity, chunk.label()))

    return entities


# ==========================================================
# LOAD INTENTS
# ==========================================================

@st.cache_resource
def load_intents():
    file_path = os.path.join(os.path.dirname(__file__), "intents.json")

    with open(file_path) as f:
        return json.load(f)


# ==========================================================
# TRAIN MODELS
# ==========================================================

@st.cache_resource
def train_models(intents):

    tags = []
    clf_patterns = []
    sim_patterns = []

    for intent in intents:
        for pattern in intent["patterns"]:
            tags.append(intent["tag"])
            clf_patterns.append(preprocess_text(pattern))
            sim_patterns.append(preprocess_for_similarity(pattern))

    clf_vectorizer = TfidfVectorizer(ngram_range=(1, 4))
    sim_vectorizer = TfidfVectorizer(ngram_range=(1, 3))

    classifier = LogisticRegression(
        random_state=0,
        max_iter=10000
    )

    X_clf = clf_vectorizer.fit_transform(clf_patterns)
    classifier.fit(X_clf, tags)

    X_sim = sim_vectorizer.fit_transform(sim_patterns)

    intent_vocab = {}

    for i, tag in enumerate(tags):
        intent_vocab.setdefault(tag, set()).update(
            clf_patterns[i].split()
        )

    return classifier, clf_vectorizer, sim_vectorizer, X_sim, tags, intent_vocab


@st.cache_resource
def load_model():

    intents = load_intents()

    model_data = train_models(intents)

    return intents, model_data


intents, (
    clf,
    clf_vectorizer,
    sim_vectorizer,
    X_sim,
    tags,
    intent_vocab
) = load_model()


SIMILARITY_THRESHOLD = 0.15

FALLBACK = (
    "I don't have the required information to answer that. "
    "Please try asking another question."
)


# ==========================================================
# INTENT VALIDATION
# ==========================================================

def intent_matches_content(clf_input, tag):

    user_words = set(clf_input.split())

    if not user_words:
        return True

    return bool(user_words & intent_vocab.get(tag, set()))


# ==========================================================
# DIALOGUE MANAGER
# ==========================================================

class DialogueManager:

    def __init__(self):

        if "dialogue_state" not in st.session_state:
            st.session_state.dialogue_state = {
                "last_intent": None,
                "context": {},
                "turn_count": 0,
            }

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

    def update_state(self, intent, entities, user_input, response):

        st.session_state.dialogue_state["last_intent"] = intent
        st.session_state.dialogue_state["turn_count"] += 1

        for ent, label in entities:
            st.session_state.dialogue_state["context"][label] = ent

        st.session_state.chat_history.append({
            "user": user_input,
            "bot": response,
            "entities": entities
        })

    def contextual_response(self, intent, base_response):

        context = st.session_state.dialogue_state["context"]
        turn = st.session_state.dialogue_state["turn_count"]

        if intent == "greeting" and turn > 1:
            return "Welcome back! How else can I help you today?"

        if intent == "thanks" and "PERSON" in context:
            return f"You're very welcome, {context['PERSON']}! Happy to help."

        return base_response


# ==========================================================
# CHATBOT LOGIC
# ==========================================================

def chatbot(user_input, dm):

    entities = extract_entities(user_input)

    sim_input = preprocess_for_similarity(user_input)

    if not sim_input:
        dm.update_state("unknown", entities, user_input, FALLBACK)
        return FALLBACK, entities

    sim_vec = sim_vectorizer.transform([sim_input])

    similarities = cosine_similarity(sim_vec, X_sim)[0]

    max_sim = float(np.max(similarities))

    if max_sim < SIMILARITY_THRESHOLD:
        dm.update_state("unknown", entities, user_input, FALLBACK)
        return FALLBACK, entities

    clf_input = preprocess_text(user_input)

    if not clf_input:
        tag = tags[int(np.argmax(similarities))]

    else:

        clf_vec = clf_vectorizer.transform([clf_input])

        tag = clf.predict(clf_vec)[0]

        if not intent_matches_content(clf_input, tag):
            dm.update_state("unknown", entities, user_input, FALLBACK)
            return FALLBACK, entities

    response = None

    for intent in intents:
        if intent["tag"] == tag:
            base = random.choice(intent["responses"])
            response = dm.contextual_response(tag, base)
            break

    if response:
        dm.update_state(tag, entities, user_input, response)

    return response, entities


# ==========================================================
# STREAMLIT UI
# ==========================================================

def main():

    st.title("Chatbot using Intent-Based NLP")

    dm = DialogueManager()

    menu = ["Home", "Conversation History", "About"]

    choice = st.sidebar.selectbox("Menu", menu)

    log_file = os.path.join(os.path.dirname(__file__), "chat_log.csv")

    if choice == "Home":

        st.write(
            "Welcome to the chatbot. "
            "Type a message and press Enter to start the conversation."
        )

        for chat in st.session_state.chat_history:
            st.markdown(f"**You:** {chat['user']}")
            st.markdown(f"**Chatbot:** {chat['bot']}")
            st.markdown("---")

        if not os.path.exists(log_file):

            with open(log_file, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(
                    ["User Input", "Chatbot Response", "Timestamp"]
                )

        user_input = st.text_input("You:")

        if user_input:

            response, entities = chatbot(user_input, dm)

            timestamp = datetime.datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            with open(log_file, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(
                    [user_input, response, timestamp]
                )

            st.rerun()

    elif choice == "Conversation History":

        st.header("Conversation History")

        if os.path.exists(log_file):

            with open(log_file, "r", encoding="utf-8") as f:

                reader = csv.reader(f)

                next(reader)

                rows = sorted(
                    list(reader),
                    key=lambda x: x[2],
                    reverse=True
                )

            for row in rows:
                st.text(f"User: {row[0]}")
                st.text(f"Chatbot: {row[1]}")
                st.text(f"Timestamp: {row[2]}")
                st.markdown("---")

        else:
            st.write("No conversation history found.")

    elif choice == "About":

        st.write(
            "This project demonstrates an intent-based chatbot built "
            "using NLP and Streamlit."
        )

        st.subheader("Technology Stack")

        st.write("""
        - Natural Language Processing (NLTK)
        - Logistic Regression
        - TF-IDF Vectorization
        - Streamlit Web App
        """)


# ==========================================================
# RUN APP
# ==========================================================

if __name__ == "__main__":
    main()
