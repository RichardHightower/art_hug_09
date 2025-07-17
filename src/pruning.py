"""Model pruning implementation for optimization."""

import torch
import torch.nn.utils.prune as prune
from transformers import AutoModel, AutoTokenizer
import numpy as np
from config import get_device, DEFAULT_MODEL
import time


def get_model_size(model):
    """Calculate model size in MB."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def count_parameters(model):
    """Count total and non-zero parameters."""
    total_params = 0
    nonzero_params = 0
    for param in model.parameters():
        total_params += param.numel()
        if param.data is not None:
            nonzero_params += torch.count_nonzero(param.data).item()
    return total_params, nonzero_params


def run_pruning_examples():
    """Run model pruning examples."""
    print("=== Model Pruning Examples ===\n")
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    print(f"\nLoading model: {DEFAULT_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    model = AutoModel.from_pretrained(DEFAULT_MODEL)
    
    # Move model to device
    if device != "cpu":
        model = model.to(device)
    
    # Get original model stats
    original_size = get_model_size(model)
    total_params, nonzero_params = count_parameters(model)
    
    print(f"\nOriginal model statistics:")
    print(f"  Size: {original_size:.2f} MB")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Non-zero parameters: {nonzero_params:,}")
    
    # Example text for inference
    text = "Transformers have revolutionized natural language processing."
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    if device != "cpu":
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Benchmark original model
    print("\nBenchmarking original model...")
    torch.cuda.synchronize() if device == "cuda" else None
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(10):
            outputs = model(**inputs)
    
    torch.cuda.synchronize() if device == "cuda" else None
    original_time = (time.time() - start_time) / 10
    
    # Example 1: Unstructured pruning
    print("\n1. Unstructured Pruning (L1 norm)")
    print("-" * 40)
    
    # Apply L1 unstructured pruning to all linear layers
    pruning_amount = 0.3  # Remove 30% of weights
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=pruning_amount)
    
    # Check pruning results
    total_params_pruned, nonzero_params_pruned = count_parameters(model)
    sparsity = 1.0 - (nonzero_params_pruned / total_params_pruned)
    
    print(f"After unstructured pruning:")
    print(f"  Pruning amount: {pruning_amount*100:.0f}%")
    print(f"  Model sparsity: {sparsity*100:.1f}%")
    print(f"  Non-zero parameters: {nonzero_params_pruned:,}")
    
    # Benchmark pruned model
    torch.cuda.synchronize() if device == "cuda" else None
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(10):
            outputs_pruned = model(**inputs)
    
    torch.cuda.synchronize() if device == "cuda" else None
    pruned_time = (time.time() - start_time) / 10
    
    print(f"\nInference time comparison:")
    print(f"  Original: {original_time*1000:.2f} ms")
    print(f"  Pruned: {pruned_time*1000:.2f} ms")
    print(f"  Speedup: {original_time/pruned_time:.2f}x")
    
    # Example 2: Structured pruning
    print("\n2. Structured Pruning (Channel pruning)")
    print("-" * 40)
    
    # Reset model
    model = AutoModel.from_pretrained(DEFAULT_MODEL)
    if device != "cpu":
        model = model.to(device)
    
    # Apply structured pruning to encoder layers
    structured_amount = 0.2  # Remove 20% of channels
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and 'intermediate' in name:
            prune.ln_structured(
                module, 
                name='weight', 
                amount=structured_amount, 
                n=2,  # L2 norm
                dim=0  # Prune output channels
            )
    
    print(f"Applied structured pruning to intermediate layers")
    print(f"  Pruning amount: {structured_amount*100:.0f}% of channels")
    
    # Example 3: Global magnitude pruning
    print("\n3. Global Magnitude Pruning")
    print("-" * 40)
    
    # Reset model
    model = AutoModel.from_pretrained(DEFAULT_MODEL)
    if device != "cpu":
        model = model.to(device)
    
    # Collect all parameters to prune
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            parameters_to_prune.append((module, 'weight'))
    
    # Apply global pruning
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0.4,  # Remove 40% globally
    )
    
    total_params_global, nonzero_params_global = count_parameters(model)
    global_sparsity = 1.0 - (nonzero_params_global / total_params_global)
    
    print(f"After global magnitude pruning:")
    print(f"  Global sparsity: {global_sparsity*100:.1f}%")
    print(f"  Remaining parameters: {nonzero_params_global:,}")
    
    # Example 4: Fine-tuning after pruning (conceptual)
    print("\n4. Fine-tuning After Pruning (Conceptual)")
    print("-" * 40)
    
    print("In practice, after pruning you would:")
    print("  1. Fine-tune the pruned model on your task")
    print("  2. Iteratively prune and fine-tune")
    print("  3. Remove pruning masks permanently")
    print("  4. Export the sparse model")
    
    # Remove pruning reparametrization (make permanent)
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            try:
                prune.remove(module, 'weight')
            except:
                pass
    
    print("\nPruning examples completed!")
    print("\nNote: For production use, consider:")
    print("  - Task-specific fine-tuning after pruning")
    print("  - Hardware-aware pruning patterns")
    print("  - Quantization + pruning combinations")


if __name__ == "__main__":
    run_pruning_examples()