"""Configuration module for examples."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Model configurations
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "bert-base-uncased")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "512"))

# API keys (if needed)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Device configuration
import torch

def get_device():
    """Get the best available device."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"
        
DEVICE = get_device()
