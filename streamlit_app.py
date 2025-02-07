import streamlit as st
import spacy
import os

# Download spaCy model if not present
@st.cache_resource
def load_nlp_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        os.system("python -m spacy download en_core_web_sm")
        return spacy.load("en_core_web_sm")

# Initialize model
nlp = load_nlp_model()

# Import and run main app
from app import main
main() 