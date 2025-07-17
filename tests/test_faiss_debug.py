#!/usr/bin/env python3
"""
FAISS Index Comparison with Debug Info and Memory Management
Tests different FAISS index types with proper error handling
"""

import numpy as np
import faiss
import time
import sys
import gc
import psutil
import os

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def test_faiss_indices(doc_embeddings, max_vectors=None):
    """
    Test different FAISS index types with memory management
    
    Args:
        doc_embeddings: numpy array of document embeddings
        max_vectors: limit number of vectors to test (for debugging)
    """
    print(f"Initial memory usage: {get_memory_usage():.2f} MB")
    
    if max_vectors:
        doc_embeddings = doc_embeddings[:max_vectors]
        print(f"Limited to {max_vectors} vectors for testing")
    
    print(f"Embeddings shape: {doc_embeddings.shape}")
    print(f"Embeddings dtype: {doc_embeddings.dtype}")
    print(f"Memory usage after loading: {get_memory_usage():.2f} MB")
    
    # Ensure float32 dtype
    if doc_embeddings.dtype != np.float32:
        print("Converting embeddings to float32...")
        doc_embeddings = doc_embeddings.astype(np.float32)
    
    dimension = doc_embeddings.shape[1]
    indices = {}
    build_times = {}
    
    # Test query
    query = doc_embeddings[0:1]  # Use first embedding as query
    k = min(5, doc_embeddings.shape[0])  # Number of neighbors
    
    print("\n" + "="*60)
    print("Testing FAISS Index Types")
    print("="*60)
    
    # 1. Exact search (IndexFlatL2)
    print("\n1. Testing IndexFlatL2 (Exact Search)...")
    try:
        start = time.time()
        index_flat = faiss.IndexFlatL2(dimension)
        print(f"   Created index. Memory: {get_memory_usage():.2f} MB")
        
        index_flat.add(doc_embeddings)
        build_times['Flat L2 (Exact)'] = time.time() - start
        indices['Flat L2 (Exact)'] = index_flat
        
        print(f"   Added {index_flat.ntotal} vectors")
        print(f"   Build time: {build_times['Flat L2 (Exact)']:.3f} seconds")
        print(f"   Memory after build: {get_memory_usage():.2f} MB")
        
        # Test search
        D, I = index_flat.search(query, k)
        print(f"   Search test passed. Found {k} neighbors")
        
    except Exception as e:
        print(f"   ERROR: {type(e).__name__}: {e}")
        indices['Flat L2 (Exact)'] = None
    
    # Clean up memory
    gc.collect()
    
    # 2. Approximate search (IndexIVFFlat)
    print("\n2. Testing IndexIVFFlat (Approximate Search)...")
    try:
        start = time.time()
        nlist = min(50, int(np.sqrt(doc_embeddings.shape[0])))  # Adaptive number of clusters
        print(f"   Using {nlist} clusters")
        
        quantizer = faiss.IndexFlatL2(dimension)
        index_ivf = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        print(f"   Created index. Memory: {get_memory_usage():.2f} MB")
        
        # Training requires at least nlist vectors
        if doc_embeddings.shape[0] >= nlist:
            index_ivf.train(doc_embeddings)
            print(f"   Training completed")
        else:
            print(f"   Skipping training (need at least {nlist} vectors)")
        
        index_ivf.add(doc_embeddings)
        build_times['IVF Flat (Approximate)'] = time.time() - start
        indices['IVF Flat (Approximate)'] = index_ivf
        
        print(f"   Added {index_ivf.ntotal} vectors")
        print(f"   Build time: {build_times['IVF Flat (Approximate)']:.3f} seconds")
        print(f"   Memory after build: {get_memory_usage():.2f} MB")
        
        # Test search (need to set nprobe for IVF)
        index_ivf.nprobe = min(10, nlist)
        D, I = index_ivf.search(query, k)
        print(f"   Search test passed. Found {k} neighbors")
        
    except Exception as e:
        print(f"   ERROR: {type(e).__name__}: {e}")
        indices['IVF Flat (Approximate)'] = None
    
    # Clean up memory
    gc.collect()
    
    # 3. Graph-based search (IndexHNSWFlat)
    print("\n3. Testing IndexHNSWFlat (Graph-based Search)...")
    try:
        start = time.time()
        M = 32  # Connectivity parameter
        print(f"   Using M={M} (connectivity parameter)")
        
        index_hnsw = faiss.IndexHNSWFlat(dimension, M)
        index_hnsw.hnsw.efConstruction = 40  # Construction time/accuracy trade-off
        print(f"   Created index. Memory: {get_memory_usage():.2f} MB")
        
        index_hnsw.add(doc_embeddings)
        build_times['HNSW (Graph)'] = time.time() - start
        indices['HNSW (Graph)'] = index_hnsw
        
        print(f"   Added {index_hnsw.ntotal} vectors")
        print(f"   Build time: {build_times['HNSW (Graph)']:.3f} seconds")
        print(f"   Memory after build: {get_memory_usage():.2f} MB")
        
        # Test search
        index_hnsw.hnsw.efSearch = 50  # Search time/accuracy trade-off
        D, I = index_hnsw.search(query, k)
        print(f"   Search test passed. Found {k} neighbors")
        
    except Exception as e:
        print(f"   ERROR: {type(e).__name__}: {e}")
        indices['HNSW (Graph)'] = None
    
    # Summary
    print("\n" + "="*60)
    print("Summary of Index Build Times:")
    print("="*60)
    for name, time_taken in build_times.items():
        if time_taken is not None:
            print(f"  {name}: {time_taken:.3f} seconds")
    
    print(f"\nFinal memory usage: {get_memory_usage():.2f} MB")
    
    return indices, build_times


def main():
    """Main function to test FAISS with sample data"""
    print("FAISS Debug Test Script")
    print(f"FAISS version: {faiss.__version__ if hasattr(faiss, '__version__') else 'Unknown'}")
    print(f"NumPy version: {np.__version__}")
    print(f"Python version: {sys.version}")
    
    # Create sample embeddings for testing
    print("\nCreating sample embeddings...")
    n_samples = 1000
    dimension = 384  # Common embedding dimension
    
    # Generate random embeddings (normally distributed)
    np.random.seed(42)
    doc_embeddings = np.random.randn(n_samples, dimension).astype(np.float32)
    
    # Normalize embeddings (common practice for similarity search)
    norms = np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
    doc_embeddings = doc_embeddings / norms
    
    print(f"Generated {n_samples} embeddings of dimension {dimension}")
    
    # Test FAISS indices
    indices, build_times = test_faiss_indices(doc_embeddings)
    
    print("\nTest completed successfully!")


if __name__ == "__main__":
    main()