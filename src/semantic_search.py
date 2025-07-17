"""Semantic search implementation with FAISS and sentence transformers."""

from sentence_transformers import SentenceTransformer, util
import numpy as np
import faiss
from typing import List, Tuple, Dict, Any
import time
from config import get_device, DEFAULT_MODEL, DATA_DIR, MODELS_DIR
import json
import os


class SemanticSearchEngine:
    """Production-ready semantic search engine with FAISS backend."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', index_type: str = 'flat'):
        """Initialize search engine with specified model and index type."""
        print(f"Initializing semantic search engine...")
        print(f"Model: {model_name}")
        print(f"Device: {get_device()}")
        
        self.model = SentenceTransformer(model_name)
        self.index_type = index_type
        self.index = None
        self.documents = []
        self.document_embeddings = None
        
    def index_documents(self, documents: List[str], batch_size: int = 32) -> None:
        """Index a list of documents for semantic search."""
        print(f"\nIndexing {len(documents)} documents...")
        start_time = time.time()
        
        # Store documents
        self.documents = documents
        
        # Generate embeddings
        self.document_embeddings = self.model.encode(
            documents, 
            batch_size=batch_size, 
            show_progress_bar=True,
            convert_to_numpy=True
        ).astype('float32')
        
        # Create FAISS index
        dimension = self.document_embeddings.shape[1]
        
        if self.index_type == 'flat':
            self.index = faiss.IndexFlatL2(dimension)
        elif self.index_type == 'ivf':
            # For larger datasets
            nlist = min(100, len(documents) // 10)  # Adaptive clustering
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
            self.index.train(self.document_embeddings)
        
        # Add embeddings to index
        self.index.add(self.document_embeddings)
        
        elapsed_time = time.time() - start_time
        print(f"Indexing completed in {elapsed_time:.2f} seconds")
        print(f"Index contains {self.index.ntotal} vectors")
        
    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Search for top-k most similar documents."""
        if self.index is None:
            raise ValueError("No documents indexed. Call index_documents() first.")
            
        # Encode query
        query_embedding = self.model.encode([query], convert_to_numpy=True).astype('float32')
        
        # Search
        distances, indices = self.index.search(query_embedding, k)
        
        # Format results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.documents):  # Ensure valid index
                results.append((self.documents[idx], float(distance)))
                
        return results
    
    def save_index(self, filepath: str) -> None:
        """Save FAISS index and documents to disk."""
        # Save FAISS index
        faiss.write_index(self.index, filepath)
        
        # Save documents
        doc_path = filepath.replace('.index', '_documents.json')
        with open(doc_path, 'w') as f:
            json.dump(self.documents, f)
            
        print(f"Index saved to {filepath}")
        print(f"Documents saved to {doc_path}")
        
    def load_index(self, filepath: str) -> None:
        """Load FAISS index and documents from disk."""
        # Load FAISS index
        self.index = faiss.read_index(filepath)
        
        # Load documents
        doc_path = filepath.replace('.index', '_documents.json')
        with open(doc_path, 'r') as f:
            self.documents = json.load(f)
            
        print(f"Index loaded from {filepath}")
        print(f"Loaded {len(self.documents)} documents")


def run_semantic_search_examples():
    """Run semantic search examples."""
    print("=== Semantic Search Examples ===\n")
    
    # Example 1: Basic semantic search
    print("1. Basic Semantic Search Demo")
    print("-" * 40)
    
    # Sample FAQ documents
    faqs = [
        "How can I reset my password?",
        "What are the steps for account recovery?",
        "How do I request a refund?",
        "Information about our privacy policy.",
        "How to change billing information?",
        "Steps to delete your account permanently",
        "What payment methods do you accept?",
        "How to enable two-factor authentication?"
    ]
    
    # Initialize search engine
    search_engine = SemanticSearchEngine()
    search_engine.index_documents(faqs)
    
    # Test queries
    test_queries = [
        "I forgot my login credentials",
        "How can I get my money back?",
        "Privacy concerns about my data",
        "Change my credit card"
    ]
    
    print("\nSearching for similar FAQs:")
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = search_engine.search(query, k=2)
        for i, (doc, score) in enumerate(results, 1):
            print(f"  {i}. {doc} (distance: {score:.3f})")
    
    # Example 2: Comparison with keyword search
    print("\n\n2. Keyword vs Semantic Search Comparison")
    print("-" * 40)
    
    query = "I can't remember my password"
    
    # Keyword search
    print(f"\nQuery: '{query}'")
    print("\nKeyword Search Results:")
    keywords = set(query.lower().split())
    keyword_matches = [faq for faq in faqs if any(word in faq.lower() for word in keywords)]
    if keyword_matches:
        for match in keyword_matches:
            print(f"  - {match}")
    else:
        print("  No matches found!")
    
    # Semantic search
    print("\nSemantic Search Results:")
    results = search_engine.search(query, k=3)
    for i, (doc, score) in enumerate(results, 1):
        print(f"  {i}. {doc} (distance: {score:.3f})")
    
    # Example 3: Performance comparison
    print("\n\n3. Performance Benchmarking")
    print("-" * 40)
    
    # Create larger dataset
    large_docs = faqs * 100  # 800 documents
    
    # Benchmark flat index
    search_engine_flat = SemanticSearchEngine(index_type='flat')
    start = time.time()
    search_engine_flat.index_documents(large_docs)
    flat_index_time = time.time() - start
    
    # Benchmark search time
    start = time.time()
    for _ in range(100):
        search_engine_flat.search("password reset", k=5)
    flat_search_time = (time.time() - start) / 100
    
    print(f"\nFlat Index (Exact Search):")
    print(f"  Indexing time: {flat_index_time:.3f}s")
    print(f"  Average search time: {flat_search_time*1000:.2f}ms")
    
    # Save index for later use
    os.makedirs(MODELS_DIR, exist_ok=True)
    index_path = MODELS_DIR / "semantic_search.index"
    search_engine.save_index(str(index_path))
    
    print("\nSemantic search examples completed!")


if __name__ == "__main__":
    run_semantic_search_examples()