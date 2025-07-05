"""
Setup script for the Advanced Local RAG System.

This script handles the installation of dependencies, downloading necessary
models, and setting up the environment for the system. It ensures all
requirements are met before the system can be used.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def install_requirements():
    """Install Python package dependencies."""
    print("Installing Python dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing dependencies: {e}")
        return False

def download_spacy_model():
    """Download spaCy language model."""
    print("Downloading spaCy language model...")
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        print("✓ spaCy model downloaded successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error downloading spaCy model: {e}")
        return False

def download_llm_models(models_dir="models"):
    """Download LLM models."""
    print("Setting up LLM models directory...")
    models_path = Path(models_dir)
    models_path.mkdir(exist_ok=True)
    
    print("\nLLM models are not automatically downloaded due to their size.")
    print("Please download the models manually and place them in the 'models' directory.")
    print("Recommended models:")
    print("  1. Phi-2 (https://huggingface.co/TheBloke/phi-2-GGUF)")
    print("  2. Mistral-7B (https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF)")
    print("  3. Llama-3-8B (https://huggingface.co/TheBloke/Llama-3-8B-Instruct-GGUF)")
    print("\nDownload the Q4_K_M.gguf variants for optimal performance on Apple Silicon.")
    
    return True

def setup_directory_structure():
    """Create the necessary directory structure."""
    print("Creating directory structure...")
    directories = [
        "data/documents",
        "data/vectors",
        "data/processed",
        "models"
    ]
    
    for directory in directories:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        # Create .gitkeep files
        keep_file = path / ".gitkeep"
        keep_file.touch(exist_ok=True)
    
    print("✓ Directory structure created")
    return True

def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Setup the Advanced Local RAG System")
    parser.add_argument("--skip-deps", action="store_true", help="Skip installing dependencies")
    parser.add_argument("--skip-models", action="store_true", help="Skip model setup")
    args = parser.parse_args()
    
    print("=== Advanced Local RAG System Setup ===\n")
    
    # Setup directory structure
    setup_directory_structure()
    
    # Install dependencies
    if not args.skip_deps:
        if not install_requirements():
            print("Warning: Some dependencies could not be installed.")
        
        if not download_spacy_model():
            print("Warning: spaCy model could not be downloaded.")
    
    # Download models
    if not args.skip_models:
        download_llm_models()
    
    print("\n=== Setup Complete ===")
    print("You can now start using the Advanced Local RAG System.")
    print("Run 'python -m src.main interactive' to start in interactive mode.")

if __name__ == "__main__":
    main()
