#!/bin/bash

# Function to install spaCy model
install_spacy_model() {
    echo "Installing spaCy model..."
    python -m pip install --no-cache-dir --no-deps https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.5.0/en_core_web_sm-3.5.0-py3-none-any.whl
}

# Check Python version and set numpy version accordingly
PYTHON_VERSION=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [[ $(echo "$PYTHON_VERSION >= 3.12" | bc -l) -eq 1 ]]; then
    NUMPY_VERSION="numpy>=1.26.0"
else
    NUMPY_VERSION="numpy>=1.21.0,<1.25.0"
fi

# Check if running on Streamlit Cloud
if [ -n "$STREAMLIT_SHARING_MODE" ]; then
    echo "Running on Streamlit Cloud..."
    
    # Install system dependencies
    apt-get update && apt-get install -y \
        python3-dev \
        build-essential \
        python3-distutils \
        python3-setuptools \
        python3-pip

    # Install pip and setuptools first
    python -m pip install --upgrade pip setuptools wheel

    # Install numpy first (required for spacy)
    python -m pip install --no-cache-dir "$NUMPY_VERSION"
    
    # Install spacy core
    python -m pip install --no-cache-dir 'spacy>=3.5.0,<3.6.0'
fi

# Install spaCy model
install_spacy_model

# Verify installation
python -c "import spacy; spacy.load('en_core_web_sm')" || echo "Failed to verify model installation" 