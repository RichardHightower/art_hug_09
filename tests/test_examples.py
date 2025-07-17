"""Unit tests for Chapter 09 examples."""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config import get_device
from quantization import run_quantization_examples

def test_device_detection():
    """Test that device detection works."""
    device = get_device()
    assert device in ["cpu", "cuda", "mps"]
    
def test_quantization_runs():
    """Test that quantization examples run without errors."""
    # This is a basic smoke test
    try:
        run_quantization_examples()
    except Exception as e:
        pytest.fail(f"quantization examples failed: {e}")
        
def test_imports():
    """Test that all required modules can be imported."""
    import transformers
    import torch
    import numpy
    import pandas
    
    assert transformers.__version__
    assert torch.__version__
