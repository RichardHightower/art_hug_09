#!/usr/bin/env python3
"""
Minimal FAISS test to verify installation
Run this first to check if FAISS is working correctly
"""

import sys
print(f"Python version: {sys.version}")

try:
    import numpy as np
    print(f"✓ NumPy {np.__version__} imported successfully")
except ImportError as e:
    print(f"✗ NumPy import failed: {e}")
    sys.exit(1)

try:
    import faiss
    print(f"✓ FAISS imported successfully")
    if hasattr(faiss, '__version__'):
        print(f"  Version: {faiss.__version__}")
except ImportError as e:
    print(f"✗ FAISS import failed: {e}")
    print("\nTo install FAISS:")
    print("  conda install -c pytorch faiss-cpu")
    print("  # or")
    print("  pip install faiss-cpu")
    sys.exit(1)

# Test basic FAISS functionality
print("\nTesting basic FAISS operations...")

try:
    # Create small test data
    d = 64  # dimension
    nb = 100  # database size
    nq = 10  # nb of queries
    
    np.random.seed(1234)
    xb = np.random.random((nb, d)).astype('float32')
    xb[:, 0] += np.arange(nb) / 1000.
    xq = np.random.random((nq, d)).astype('float32')
    xq[:, 0] += np.arange(nq) / 1000.
    
    print(f"✓ Created test data: {nb} vectors of dimension {d}")
    
    # Test 1: Simple flat index
    print("\n1. Testing IndexFlatL2...")
    index = faiss.IndexFlatL2(d)
    print(f"   Index created. is_trained = {index.is_trained}")
    index.add(xb)
    print(f"   Added {index.ntotal} vectors")
    
    k = 4
    D, I = index.search(xq[:5], k)
    print(f"   Search successful. Found {k} neighbors for 5 queries")
    print(f"   First result indices: {I[0]}")
    
    # Test 2: IVF index
    print("\n2. Testing IndexIVFFlat...")
    nlist = 10
    quantizer = faiss.IndexFlatL2(d)
    index_ivf = faiss.IndexIVFFlat(quantizer, d, nlist)
    print(f"   Index created with {nlist} clusters")
    
    index_ivf.train(xb)
    print(f"   Training complete. is_trained = {index_ivf.is_trained}")
    
    index_ivf.add(xb)
    print(f"   Added {index_ivf.ntotal} vectors")
    
    index_ivf.nprobe = 5
    D, I = index_ivf.search(xq[:5], k)
    print(f"   Search successful with nprobe={index_ivf.nprobe}")
    
    print("\n✅ All FAISS tests passed!")
    print("\nYour FAISS installation is working correctly.")
    
except Exception as e:
    print(f"\n❌ FAISS test failed: {type(e).__name__}: {e}")
    print("\nPossible solutions:")
    print("1. Reinstall FAISS: pip install --upgrade faiss-cpu")
    print("2. Check NumPy compatibility: pip install --upgrade numpy")
    print("3. Try conda instead: conda install -c pytorch faiss-cpu")
    sys.exit(1)