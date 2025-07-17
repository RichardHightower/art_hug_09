"""
Complete FAISS Vector Search Notebook Code
Copy each cell into your Jupyter notebook
"""

# ============================================================================
# CELL 1: Imports and Setup
# ============================================================================
import numpy as np
import faiss
import time
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import gc
import warnings
warnings.filterwarnings('ignore')

print("✓ All imports successful")

# ============================================================================
# CELL 2: Create Documents and Embeddings
# ============================================================================
# Create synthetic documents
synthetic_docs = [
    "Introduction to machine learning and artificial intelligence",
    "Deep learning fundamentals with neural networks",
    "Natural language processing for beginners",
    "Computer vision and image recognition techniques",
    "Reinforcement learning in robotics applications",
    "Transfer learning and pre-trained models",
    "Generative AI and large language models",
    "Transformer architecture explained",
    "BERT for natural language understanding",
    "GPT models and text generation",
    "Data preprocessing for machine learning",
    "Feature engineering best practices",
    "Model evaluation and cross-validation",
    "Hyperparameter tuning strategies",
    "Deployment of ML models in production",
    "AI ethics and responsible AI development",
    "Federated learning for privacy preservation",
    "Quantum machine learning basics",
    "Time series analysis with deep learning",
    "Graph neural networks introduction"
] * 5  # Repeat to get 100 documents

print(f"Created {len(synthetic_docs)} synthetic documents")

# Load a lightweight model
print("\nLoading sentence transformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("✓ Model loaded successfully")

# Generate embeddings
print("\nGenerating embeddings...")
doc_embeddings = model.encode(synthetic_docs, 
                             convert_to_numpy=True,
                             show_progress_bar=True)

# Ensure float32 for FAISS
doc_embeddings = doc_embeddings.astype('float32')

print(f"✓ Generated embeddings shape: {doc_embeddings.shape}")
print(f"  Embedding dimension: {doc_embeddings.shape[1]}")
print(f"  Number of documents: {doc_embeddings.shape[0]}")

# ============================================================================
# CELL 3: Build FAISS Indices Safely
# ============================================================================
dimension = doc_embeddings.shape[1]
n_docs = doc_embeddings.shape[0]

# Dictionary to store indices and timing
indices = {}
build_times = {}

print(f"\nBuilding FAISS indices for {n_docs} documents...")
print("="*60)

# 1. Flat L2 Index (Exact Search)
print("\n1. Building IndexFlatL2...")
try:
    start = time.time()
    index_flat = faiss.IndexFlatL2(dimension)
    index_flat.add(doc_embeddings)
    build_times['Flat L2'] = time.time() - start
    indices['Flat L2'] = index_flat
    print(f"   ✓ Success! Time: {build_times['Flat L2']:.3f}s")
    print(f"   Vectors in index: {index_flat.ntotal}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    build_times['Flat L2'] = 0
    indices['Flat L2'] = None

# 2. IVF Flat Index (Approximate Search)
print("\n2. Building IndexIVFFlat...")
try:
    # Use fewer clusters for smaller datasets
    nlist = min(int(np.sqrt(n_docs)), 10)
    print(f"   Using {nlist} clusters")
    
    start = time.time()
    quantizer = faiss.IndexFlatL2(dimension)
    index_ivf = faiss.IndexIVFFlat(quantizer, dimension, nlist)
    
    # Train the index
    index_ivf.train(doc_embeddings)
    index_ivf.add(doc_embeddings)
    
    build_times['IVF Flat'] = time.time() - start
    indices['IVF Flat'] = index_ivf
    print(f"   ✓ Success! Time: {build_times['IVF Flat']:.3f}s")
    print(f"   Vectors in index: {index_ivf.ntotal}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    build_times['IVF Flat'] = 0
    indices['IVF Flat'] = None

# 3. HNSW Index (Graph-based Search)
print("\n3. Building IndexHNSWFlat...")
try:
    # Use conservative parameters
    M = 16  # Connectivity parameter
    
    start = time.time()
    index_hnsw = faiss.IndexHNSWFlat(dimension, M)
    index_hnsw.hnsw.efConstruction = 40
    index_hnsw.add(doc_embeddings)
    
    build_times['HNSW'] = time.time() - start
    indices['HNSW'] = index_hnsw
    print(f"   ✓ Success! Time: {build_times['HNSW']:.3f}s")
    print(f"   Vectors in index: {index_hnsw.ntotal}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    build_times['HNSW'] = 0
    indices['HNSW'] = None

# Clean up indices dict - remove None values
indices = {k: v for k, v in indices.items() if v is not None}
build_times = {k: v for k, v in build_times.items() if v > 0}

print(f"\n✓ Successfully built {len(indices)} indices")

# ============================================================================
# CELL 4: Benchmark Search Performance
# ============================================================================
if not indices:
    print("❌ No indices were successfully built!")
else:
    # Test query
    query = "Looking for AI and machine learning resources"
    print(f"\nBenchmarking search for: '{query}'")
    
    query_embedding = model.encode([query], convert_to_numpy=True).astype('float32')
    k = min(10, n_docs)  # Number of neighbors
    
    search_times = {}
    search_results = {}
    
    print(f"\nSearching for top {k} results...")
    print("-" * 40)
    
    for name, index in indices.items():
        if index is None:
            continue
            
        try:
            # Set search parameters for IVF
            if 'IVF' in name and hasattr(index, 'nprobe'):
                index.nprobe = min(5, index.nlist)  # Search 5 clusters
            
            # Perform search
            start = time.time()
            distances, indices_found = index.search(query_embedding, k)
            search_time = (time.time() - start) * 1000  # Convert to ms
            
            search_times[name] = search_time
            search_results[name] = (distances[0], indices_found[0])
            
            print(f"{name}: {search_time:.2f} ms")
        except Exception as e:
            print(f"{name}: Search failed - {e}")
    
    # ========================================================================
    # CELL 5: Visualize Results
    # ========================================================================
    if search_times:
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Build times
        names = list(build_times.keys())
        build_values = list(build_times.values())
        colors = ['#2ecc71', '#f39c12', '#3498db'][:len(names)]
        
        bars1 = ax1.bar(names, build_values, color=colors)
        ax1.set_title('Index Build Time', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_ylim(0, max(build_values) * 1.2)
        
        # Add value labels on bars
        for bar, val in zip(bars1, build_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{val:.3f}s', ha='center', va='bottom')
        
        # Search times
        search_names = list(search_times.keys())
        search_values = list(search_times.values())
        
        bars2 = ax2.bar(search_names, search_values, color=colors[:len(search_names)])
        ax2.set_title('Search Time', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Time (milliseconds)')
        ax2.set_ylim(0, max(search_values) * 1.2)
        
        # Add value labels on bars
        for bar, val in zip(bars2, search_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.2f}ms', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        # Display search results
        print(f"\n📊 Search Results")
        print("="*60)
        print(f"Query: '{query}'\n")
        
        for name, (distances, indices_found) in search_results.items():
            print(f"{name} - Top 5 results:")
            for i in range(min(5, len(indices_found))):
                idx = indices_found[i]
                if idx < len(synthetic_docs):
                    dist = distances[i]
                    doc_preview = synthetic_docs[idx][:60] + "..."
                    print(f"  {i+1}. {doc_preview}")
                    print(f"     Distance: {dist:.3f}")
            print()

# ============================================================================
# CELL 6: Memory Cleanup
# ============================================================================
print("\n🧹 Cleaning up memory...")
del doc_embeddings
gc.collect()
print("✓ Memory cleaned")

# ============================================================================
# CELL 7: Summary and Recommendations
# ============================================================================
print("\n📈 Performance Summary")
print("="*60)

if indices:
    # Find fastest build and search
    fastest_build = min(build_times.items(), key=lambda x: x[1])
    if search_times:
        fastest_search = min(search_times.items(), key=lambda x: x[1])
        
        print(f"Fastest Build:  {fastest_build[0]} ({fastest_build[1]:.3f}s)")
        print(f"Fastest Search: {fastest_search[0]} ({fastest_search[1]:.2f}ms)")
        
        print("\n💡 Recommendations:")
        print("- Flat L2: Best for small datasets (<10K vectors) with exact results")
        print("- IVF Flat: Good for medium datasets with speed/accuracy trade-off")
        print("- HNSW: Best for large datasets with fast approximate search")
else:
    print("❌ No indices were successfully built. Check your FAISS installation.")

print("\n✅ Notebook completed successfully!")