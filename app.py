import os
import json
import random
import string
import datetime
import csv
import ssl
import nltk
import streamlit as st
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, ne_chunk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

# Disable SSL verification and force-download all required NLTK data
ssl._create_default_https_context = ssl._create_unverified_context

for pkg in ["punkt", "punkt_tab", "wordnet", "stopwords",
            "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng",
            "maxent_ne_chunker", "maxent_ne_chunker_tab", "words"]:
    nltk.download(pkg, quiet=True)

# NLP setup
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def preprocess_text(text):
    """Full preprocessing: lowercase, strip punctuation, remove stopwords, lemmatize."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)


def preprocess_for_similarity(text):
    """Light preprocessing: lowercase and strip punctuation only.
    Keeps stopwords so short phrases like 'How are you' still match patterns."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.strip()


def extract_entities(text):
    tokens = nltk.word_tokenize(text)
    tagged = pos_tag(tokens)
    chunks = ne_chunk(tagged)
    extracted = []
    for chunk in chunks:
        if hasattr(chunk, "label"):
            entity = " ".join(c[0] for c in chunk)
            extracted.append((entity, chunk.label()))
    return extracted


@st.cache_resource
def load_intents():
    file_path = os.path.join(os.path.dirname(__file__), "intents.json")
    with open(file_path) as f:
        return json.load(f)


@st.cache_resource
def train_models(_intents):
    tags, clf_pats, sim_pats = [], [], []
    for intent in _intents:
        for pattern in intent["patterns"]:
            tags.append(intent["tag"])
            clf_pats.append(preprocess_text(pattern))
            sim_pats.append(preprocess_for_similarity(pattern))

    clf_vec = TfidfVectorizer(ngram_range=(1, 4))
    sim_vec = TfidfVectorizer(ngram_range=(1, 3))
    clf = LogisticRegression(random_state=0, max_iter=10000)

    X_clf = clf_vec.fit_transform(clf_pats)
    clf.fit(X_clf, tags)
    X_sim = sim_vec.fit_transform(sim_pats)

    # Build intent vocabulary for content-word verification
    intent_vocab = {}
    for i, tag in enumerate(tags):
        if tag not in intent_vocab:
            intent_vocab[tag] = set()
        intent_vocab[tag].update(clf_pats[i].split())

    return clf, clf_vec, sim_vec, X_sim, tags, intent_vocab


intents = load_intents()
clf, clf_vectorizer, sim_vectorizer, X_sim, tags, intent_vocab = train_models(intents)

SIMILARITY_THRESHOLD = 0.15
FALLBACK = "I don't have the required information to answer that. Please try asking another question."


def intent_matches_content(clf_input, tag):
    """Verify user's content words actually appear in the matched intent's vocabulary.
    Prevents common/generic words from accidentally matching unrelated intents."""
    user_words = set(clf_input.split())
    if not user_words:
        return True  # Short phrase already passed similarity gate
    return bool(user_words & intent_vocab.get(tag, set()))


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
        st.session_state.chat_history.append(
            {"user": user_input, "bot": response, "entities": entities}
        )

    def contextual_response(self, intent, base_response):
        context = st.session_state.dialogue_state["context"]
        turn = st.session_state.dialogue_state["turn_count"]
        if intent == "greeting" and turn > 1:
            return "Welcome back! How else can I help you today?"
        if intent == "thanks" and "PERSON" in context:
            return f"You're very welcome, {context['PERSON']}! Happy to help."
        return base_response


def chatbot(input_text, dm):
    entities = extract_entities(input_text)

    # Stage 1 — similarity check (stop words kept so "How are you" matches correctly)
    sim_input = preprocess_for_similarity(input_text)
    if not sim_input:
        dm.update_state("unknown", entities, input_text, FALLBACK)
        return FALLBACK, entities

    sim_vec = sim_vectorizer.transform([sim_input])
    sims = cosine_similarity(sim_vec, X_sim)[0]
    max_sim = float(np.max(sims))

    print(f"DEBUG: '{input_text}' | sim={max_sim:.4f}")

    if max_sim < SIMILARITY_THRESHOLD:
        dm.update_state("unknown", entities, input_text, FALLBACK)
        return FALLBACK, entities

    # Stage 2 — classify intent
    clf_input = preprocess_text(input_text)

    if not clf_input:
        # All words were stopwords (e.g. "How are you") — use best similarity match
        tag = tags[int(np.argmax(sims))]
    else:
        clf_vec = clf_vectorizer.transform([clf_input])
        tag = clf.predict(clf_vec)[0]

        # Stage 3 — content word verification
        # Ensures user's key words actually appear in the classified intent's vocabulary.
        # This stops sentences like "Do we have an appointment today" matching "gardening".
        if not intent_matches_content(clf_input, tag):
            print(f"DEBUG: Content mismatch — '{clf_input}' vs intent '{tag}'")
            dm.update_state("unknown", entities, input_text, FALLBACK)
            return FALLBACK, entities

    response = None
    for intent in intents:
        if intent["tag"] == tag:
            base = random.choice(intent["responses"])
            response = dm.contextual_response(tag, base)
            break

    if response:
        dm.update_state(tag, entities, input_text, response)
    return response, entities


def main():
    st.title("Chatbot using intent based NLP")
    dm = DialogueManager()

    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.write("Welcome to the chatbot. Please type a message and press Enter to start the conversation.")

        # Display chat history
        for chat in st.session_state.chat_history:
            st.markdown(f"**You:** {chat['user']}")
            st.markdown(f"**Chatbot:** {chat['bot']}")
            st.markdown("---")

        log_file = os.path.join(os.path.dirname(__file__), "chat_log.csv")
        if not os.path.exists(log_file):
            with open(log_file, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["User Input", "Chatbot Response", "Timestamp"])

        def clear_input():
            st.session_state.temp_input = st.session_state.user_input_box
            st.session_state.user_input_box = ""

        st.text_input("You:", key="user_input_box", on_change=clear_input)
        actual_input = st.session_state.get("temp_input", "")

        if actual_input and actual_input != st.session_state.get("last_input", ""):
            st.session_state.last_input = actual_input

            response, entities = chatbot(actual_input, dm)

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(log_file, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([actual_input, response, timestamp])

            goodbye_phrases = ["goodbye", "bye", "good bye", "take care", "see you later"]
            if any(p in response.lower() for p in goodbye_phrases):
                st.session_state.chat_history = []
                st.session_state.dialogue_state = {"last_intent": None, "context": {}, "turn_count": 0}
                st.session_state.last_input = ""
                st.session_state.temp_input = ""

            st.rerun()

    elif choice == "Conversation History":
        st.header("Conversation History")
        log_file = os.path.join(os.path.dirname(__file__), "chat_log.csv")
        if os.path.exists(log_file):
            with open(log_file, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader)
                rows = sorted(list(reader), key=lambda x: x[2], reverse=True)
            for row in rows:
                st.text(f"User: {row[0]}")
                st.text(f"Chatbot: {row[1]}")
                st.text(f"Timestamp: {row[2]}")
                st.markdown("---")
        else:
            st.write("No conversation history found.")

    elif choice == "About":
        st.write("The goal of this project is to create a chatbot that can understand and respond to user input based on intents. The chatbot is built using Natural Language Processing (NLP) library and Logistic Regression, to extract the intents and entities from user input. The chatbot is built using Streamlit, a Python library for building interactive web applications.")

        st.subheader("Project Overview:")
        st.write("""
        The project is divided into two parts:
        1. NLP techniques and Logistic Regression algorithm is used to train the chatbot on labeled intents and entities.
        2. For building the Chatbot interface, Streamlit web framework is used to build a web-based chatbot interface.
        """)

        st.subheader("Dataset:")
        st.write("""
        The dataset used in this project is a collection of labelled intents and entities:
        - Intents: The intent of the user input (e.g. "greeting", "budget", "about")
        - Entities: The entities extracted from user input
        - Text: The user input text.
        """)

        st.subheader("Streamlit Chatbot Interface:")
        st.write("The chatbot interface includes a text input box for users and a chat window to display responses. The interface uses the trained model to generate responses to user input.")

        st.subheader("Conclusion:")
        st.write("In this project, a chatbot is built that can understand and respond to user input based on intents. The chatbot was trained using NLP and Logistic Regression, and the interface was built using Streamlit.")


if __name__ == "__main__":
    main()
