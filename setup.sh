#!/bin/bash

# Function to install spaCy model
install_spacy_model() {
    echo "Installing spaCy model..."
    python -m pip install --no-deps https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.0/en_core_web_sm-3.7.0.tar.gz
}

# Check if running on Streamlit Cloud
if [ -n "$STREAMLIT_SHARING_MODE" ]; then
    echo "Running on Streamlit Cloud..."
    # Install system dependencies
    apt-get update && apt-get install -y python3-dev build-essential
    
    # Ensure pip is up to date
    python -m pip install --upgrade pip
    
    # Install numpy first
    python -m pip install numpy==1.24.3
fi

# Install spaCy model
install_spacy_model

# Verify installation
python -c "import spacy; spacy.load('en_core_web_sm')" || echo "Failed to verify model installation" 