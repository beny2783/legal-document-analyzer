#!/bin/bash

# Function to install spaCy model
install_spacy_model() {
    echo "Installing spaCy model..."
    python -m pip install --no-cache-dir --no-deps https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.5.0/en_core_web_sm-3.5.0-py3-none-any.whl
}

# Check if running on Streamlit Cloud
if [ -n "$STREAMLIT_SHARING_MODE" ]; then
    echo "Running on Streamlit Cloud..."
    
    # Install system dependencies
    apt-get update && apt-get install -y \
        python3-dev \
        build-essential \
        python3-distutils \
        python3-setuptools

    # Install pip and setuptools first
    python -m pip install --upgrade pip setuptools wheel

    # Install numpy first (required for spacy)
    python -m pip install --no-cache-dir 'numpy>=1.21.0,<1.25.0'
    
    # Install spacy core
    python -m pip install --no-cache-dir 'spacy>=3.5.0,<3.6.0'
fi

# Install spaCy model
install_spacy_model

# Verify installation
python -c "import spacy; spacy.load('en_core_web_sm')" || echo "Failed to verify model installation" 