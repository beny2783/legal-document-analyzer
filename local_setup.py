import subprocess
import sys
import os

def setup_local_environment():
    """Set up the local development environment"""
    print("Setting up local environment...")
    
    # Install requirements
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Install spaCy model
    try:
        import spacy
        spacy.load('en_core_web_sm')
        print("spaCy model already installed")
    except:
        print("Installing spaCy model...")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])

if __name__ == "__main__":
    setup_local_environment() 