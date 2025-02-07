#!/bin/bash

# Make script executable
chmod +x setup.sh

# Install system dependencies
apt-get update && apt-get install -y python3-dev

# Download spaCy models
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_trf || echo "Failed to download transformer model, will use small model"

# Verify models are installed
python -c "import spacy; spacy.load('en_core_web_sm')" || echo "Failed to verify en_core_web_sm" 