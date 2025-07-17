"""
Fixed FAISS comparison code with proper imports and error handling
Run this in your Jupyter notebook
"""

# Cell 1: Import all required libraries
import numpy as np
import faiss
import time
from sentence_transformers import SentenceTransformer, util
import torch

# Cell 2: Generate embeddings if not already done
print("Checking for embeddings...")
if 'doc_embeddings' not in globals():
    print("Generating sample embeddings...")
    # Create sample documents
    documents = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing helps computers understand human language.",
        "Computer vision enables machines to interpret visual information.",
        "Reinforcement learning trains agents through rewards and penalties.",
        "Transfer learning reuses pre-trained models for new tasks.",
        "Generative AI creates new content like text and images.",
        "Transformer models revolutionized NLP tasks.",
        "BERT uses bidirectional training for language understanding.",
        "GPT models are powerful for text generation.",
    ] * 10  # Repeat to have more vectors
    
    # Load a model and generate embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    doc_embeddings = model.encode(documents, convert_to_numpy=True)
    print(f"Generated {len(doc_embeddings)} embeddings of dimension {doc_embeddings.shape[1]}")
else:
    print(f"Using existing embeddings: {doc_embeddings.shape}")

# Cell 3: FAISS comparison with safety checks
def safe_faiss_comparison(doc_embeddings):
    """Compare FAISS indices with error handling"""
    
    # Ensure float32
    if doc_embeddings.dtype != np.float32:
        doc_embeddings = doc_embeddings.astype(np.float32)
    
    dimension = doc_embeddings.shape[1]
    n_vectors = doc_embeddings.shape[0]
    indices = {}
    build_times = {}
    
    print(f"\nTesting FAISS indices with {n_vectors} vectors of dimension {dimension}")
    print("="*60)
    
    # 1. Exact search (IndexFlatL2)
    try:
        print("\n1. IndexFlatL2 (Exact Search)")
        start = time.time()
        index_flat = faiss.IndexFlatL2(dimension)
        index_flat.add(doc_embeddings)
        build_times['Flat L2 (Exact)'] = time.time() - start
        indices['Flat L2 (Exact)'] = index_flat
        print(f"   ✓ Success! Build time: {build_times['Flat L2 (Exact)']:.3f}s")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # 2. Approximate search (IndexIVFFlat)
    try:
        print("\n2. IndexIVFFlat (Approximate Search)")
        start = time.time()
        nlist = min(50, max(4, int(np.sqrt(n_vectors))))  # Adaptive clusters
        quantizer = faiss.IndexFlatL2(dimension)
        index_ivf = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        
        # Only train if we have enough vectors
        if n_vectors >= nlist:
            index_ivf.train(doc_embeddings)
            index_ivf.add(doc_embeddings)
            build_times['IVF Flat (Approximate)'] = time.time() - start
            indices['IVF Flat (Approximate)'] = index_ivf
            print(f"   ✓ Success! Build time: {build_times['IVF Flat (Approximate)']:.3f}s")
            print(f"   Used {nlist} clusters")
        else:
            print(f"   ⚠ Skipped: Need at least {nlist} vectors")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # 3. Graph-based search (IndexHNSWFlat)
    try:
        print("\n3. IndexHNSWFlat (Graph-based Search)")
        start = time.time()
        M = min(32, max(2, n_vectors // 10))  # Adaptive connectivity
        index_hnsw = faiss.IndexHNSWFlat(dimension, M)
        index_hnsw.add(doc_embeddings)
        build_times['HNSW (Graph)'] = time.time() - start
        indices['HNSW (Graph)'] = index_hnsw
        print(f"   ✓ Success! Build time: {build_times['HNSW (Graph)']:.3f}s")
        print(f"   Used M={M} (connectivity)")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("Summary of Build Times:")
    for name, time_taken in build_times.items():
        print(f"  {name}: {time_taken:.3f} seconds")
    
    # Test search on successful indices
    print("\nTesting search functionality...")
    query = doc_embeddings[0:1]
    k = min(5, n_vectors)
    
    for name, index in indices.items():
        if index is not None:
            try:
                if 'IVF' in name:
                    index.nprobe = 10  # Set search parameter for IVF
                D, I = index.search(query, k)
                print(f"  {name}: Found {k} nearest neighbors")
            except Exception as e:
                print(f"  {name}: Search error - {e}")
    
    return indices, build_times

# Run the comparison
indices, build_times = safe_faiss_comparison(doc_embeddings)

# Cell 4: Fix for multilingual model
print("\n" + "="*60)
print("Setting up Multilingual Model")
print("="*60)

# Import and load multilingual model
from sentence_transformers import SentenceTransformer, util

print("Loading multilingual model...")
multilingual_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print("✓ Multilingual model loaded!")

# Create multilingual FAQ dataset
multilingual_faqs = [
    # English
    "How do I reset my password?",
    "What are your business hours?",
    "How can I track my order?",
    
    # Spanish  
    "¿Cómo restablezco mi contraseña?",
    "¿Cuáles son sus horarios comerciales?",
    "¿Cómo puedo rastrear mi pedido?",
    
    # French
    "Comment réinitialiser mon mot de passe?",
    "Quels sont vos horaires d'ouverture?",
    "Comment puis-je suivre ma commande?",
    
    # German
    "Wie setze ich mein Passwort zurück?",
    "Was sind Ihre Geschäftszeiten?",
    "Wie kann ich meine Bestellung verfolgen?"
]

# Encode FAQs
print(f"\nEncoding {len(multilingual_faqs)} multilingual FAQs...")
multilingual_embeddings = multilingual_model.encode(multilingual_faqs)
print(f"✓ Generated embeddings of shape: {multilingual_embeddings.shape}")

# Test cross-lingual search
test_queries = [
    ("password reset help", "English"),
    ("horarios de atención", "Spanish"),
    ("suivre commande", "French"),
    ("Passwort vergessen", "German")
]

print("\n🌍 Cross-Lingual Search Results\n" + "="*50)

for query, lang in test_queries:
    query_embedding = multilingual_model.encode(query)
    similarities = util.cos_sim(query_embedding, multilingual_embeddings)[0]
    top_3 = similarities.argsort(descending=True)[:3]
    
    print(f"\nQuery ({lang}): '{query}'")
    print("Top 3 matches:")
    for i, idx in enumerate(top_3):
        print(f"  {i+1}. {multilingual_faqs[idx]} (score: {similarities[idx]:.3f})")

print("\n✓ All tests completed successfully!")