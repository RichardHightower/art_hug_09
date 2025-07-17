"""Quantization implementation."""

from transformers import pipeline, AutoTokenizer, AutoModel
import torch
from config import get_device, DEFAULT_MODEL

def run_quantization_examples():
    """Run quantization examples."""
    
    print(f"Loading model: {DEFAULT_MODEL}")
    device = get_device()
    print(f"Using device: {device}")
    
    # Example implementation
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    model = AutoModel.from_pretrained(DEFAULT_MODEL)
    
    # Example text
    text = "Hugging Face Transformers make NLP accessible to everyone!"
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    
    print(f"\nInput text: {text}")
    print(f"Tokens: {tokenizer.convert_ids_to_tokens(inputs['input_ids'][0].tolist())}")
    print(f"Token IDs: {inputs['input_ids'][0].tolist()}")
    
    # Get model outputs
    with torch.no_grad():
        outputs = model(**inputs)
    
    print(f"\nModel output shape: {outputs.last_hidden_state.shape}")
    print("Example completed successfully!")
    
if __name__ == "__main__":
    print("=== Quantization Examples ===\n")
    run_quantization_examples()
