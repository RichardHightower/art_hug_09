"""ONNX export implementation for model deployment."""

import torch
from transformers import AutoModel, AutoTokenizer
from optimum.onnxruntime import ORTModelForFeatureExtraction
import onnx
import onnxruntime as ort
import numpy as np
import time
from pathlib import Path
from config import get_device, DEFAULT_MODEL, MODELS_DIR
import os


def run_onnx_export_examples():
    """Run ONNX export and optimization examples."""
    print("=== ONNX Export Examples ===\n")
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Note: ONNX export typically works best on CPU
    if device != "cpu":
        print("Note: ONNX export will be performed on CPU")
    
    # Load model and tokenizer
    print(f"\nLoading model: {DEFAULT_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    model = AutoModel.from_pretrained(DEFAULT_MODEL)
    model.eval()  # Set to evaluation mode
    
    # Create output directory
    onnx_dir = MODELS_DIR / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    
    # Example 1: Basic ONNX Export
    print("\n1. Basic ONNX Export")
    print("-" * 40)
    
    # Prepare dummy input
    dummy_text = "This is a sample text for ONNX export."
    inputs = tokenizer(
        dummy_text, 
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )
    
    # Export to ONNX
    onnx_path = onnx_dir / f"{DEFAULT_MODEL.replace('/', '_')}.onnx"
    print(f"Exporting to: {onnx_path}")
    
    # Dynamic axes for variable sequence length
    dynamic_axes = {
        'input_ids': {0: 'batch_size', 1: 'sequence'},
        'attention_mask': {0: 'batch_size', 1: 'sequence'},
        'last_hidden_state': {0: 'batch_size', 1: 'sequence'}
    }
    
    try:
        torch.onnx.export(
            model,
            (inputs['input_ids'], inputs['attention_mask']),
            onnx_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input_ids', 'attention_mask'],
            output_names=['last_hidden_state'],
            dynamic_axes=dynamic_axes
        )
        print("✓ Export successful!")
        
        # Verify ONNX model
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model validation passed!")
        
    except Exception as e:
        print(f"Export failed: {e}")
        print("Using alternative export method...")
        
        # Alternative: Use Optimum library
        from optimum.onnxruntime import ORTModelForFeatureExtraction
        
        ort_model = ORTModelForFeatureExtraction.from_pretrained(
            DEFAULT_MODEL,
            export=True,
            cache_dir=str(onnx_dir)
        )
        print("✓ Alternative export successful!")
    
    # Example 2: ONNX Runtime Inference
    print("\n2. ONNX Runtime Inference")
    print("-" * 40)
    
    # Create ONNX Runtime session
    print("Creating ONNX Runtime session...")
    
    # Set providers based on available hardware
    providers = ['CPUExecutionProvider']
    if device == "cuda":
        providers.insert(0, 'CUDAExecutionProvider')
    
    try:
        session = ort.InferenceSession(str(onnx_path), providers=providers)
        print(f"✓ Session created with providers: {providers}")
        
        # Prepare test inputs
        test_texts = [
            "ONNX enables cross-platform deployment.",
            "Transformers can run on edge devices.",
            "Model optimization improves inference speed."
        ]
        
        # Benchmark PyTorch vs ONNX
        print("\nBenchmarking inference speed...")
        
        # PyTorch inference
        pytorch_times = []
        for text in test_texts:
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            
            start = time.time()
            with torch.no_grad():
                pytorch_output = model(**inputs)
            pytorch_times.append(time.time() - start)
        
        avg_pytorch_time = np.mean(pytorch_times) * 1000
        
        # ONNX inference
        onnx_times = []
        for text in test_texts:
            inputs = tokenizer(text, return_tensors="np", padding=True, truncation=True)
            
            ort_inputs = {
                'input_ids': inputs['input_ids'],
                'attention_mask': inputs['attention_mask']
            }
            
            start = time.time()
            ort_output = session.run(None, ort_inputs)
            onnx_times.append(time.time() - start)
        
        avg_onnx_time = np.mean(onnx_times) * 1000
        
        print(f"\nInference time (average over {len(test_texts)} samples):")
        print(f"  PyTorch: {avg_pytorch_time:.2f} ms")
        print(f"  ONNX Runtime: {avg_onnx_time:.2f} ms")
        print(f"  Speedup: {avg_pytorch_time/avg_onnx_time:.2f}x")
        
    except Exception as e:
        print(f"ONNX Runtime inference failed: {e}")
    
    # Example 3: Model Optimization
    print("\n3. ONNX Model Optimization")
    print("-" * 40)
    
    try:
        from onnxruntime.transformers import optimizer
        
        # Optimize ONNX model
        optimized_path = onnx_dir / f"{DEFAULT_MODEL.replace('/', '_')}_optimized.onnx"
        
        print("Applying ONNX optimizations...")
        optimizer.optimize_model(
            str(onnx_path),
            model_type='bert',
            num_heads=12,  # Adjust based on model
            hidden_size=768,  # Adjust based on model
            output=str(optimized_path)
        )
        
        # Compare file sizes
        original_size = os.path.getsize(onnx_path) / (1024 * 1024)
        optimized_size = os.path.getsize(optimized_path) / (1024 * 1024)
        
        print(f"\nModel size comparison:")
        print(f"  Original: {original_size:.2f} MB")
        print(f"  Optimized: {optimized_size:.2f} MB")
        print(f"  Reduction: {(1 - optimized_size/original_size)*100:.1f}%")
        
    except ImportError:
        print("onnxruntime-tools not installed. Skipping optimization.")
        print("Install with: pip install onnxruntime-tools")
    
    # Example 4: Quantization with ONNX
    print("\n4. ONNX Quantization (INT8)")
    print("-" * 40)
    
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        
        quantized_path = onnx_dir / f"{DEFAULT_MODEL.replace('/', '_')}_quantized.onnx"
        
        print("Applying dynamic quantization...")
        quantize_dynamic(
            str(onnx_path),
            str(quantized_path),
            weight_type=QuantType.QInt8
        )
        
        # Compare sizes
        quantized_size = os.path.getsize(quantized_path) / (1024 * 1024)
        print(f"\nQuantized model size: {quantized_size:.2f} MB")
        print(f"Size reduction: {(1 - quantized_size/original_size)*100:.1f}%")
        
        # Test quantized model
        quantized_session = ort.InferenceSession(str(quantized_path), providers=['CPUExecutionProvider'])
        print("✓ Quantized model loaded successfully!")
        
    except Exception as e:
        print(f"Quantization failed: {e}")
    
    print("\n5. Deployment Considerations")
    print("-" * 40)
    print("ONNX models can be deployed on:")
    print("  - Edge devices (mobile, IoT)")
    print("  - Web browsers (ONNX.js)")
    print("  - Cloud services")
    print("  - Different frameworks (TensorFlow, Core ML)")
    
    print("\nONNX export examples completed!")


if __name__ == "__main__":
    run_onnx_export_examples()