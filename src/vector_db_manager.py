"""
Vector Database Manager - FAISS and Chroma Integration

This module provides a unified interface for managing vector databases,
supporting both local (FAISS) and managed (Chroma) solutions. It demonstrates
production patterns for storing, indexing, and querying embeddings at scale.

Topics covered:
- Unified vector database interface
- FAISS local deployment
- Chroma managed deployment
- Switching between backends
- Performance optimization
- Metadata management
"""

import faiss
import numpy as np
try:
    from chromadb import Client
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    Client = None
    Settings = None
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional, Union
import json
import os
import logging
from abc import ABC, abstractmethod

from config import get_device

def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorDBInterface(ABC):
    """Abstract interface for vector databases."""
    
    @abstractmethod
    def add(self, embeddings: np.ndarray, texts: List[str], metadata: Optional[List[Dict]] = None) -> None:
        """Add embeddings with associated texts and metadata."""
        pass
    
    @abstractmethod
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """Search for similar embeddings."""
        pass
    
    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        """Delete embeddings by ID."""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save the database to disk."""
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load the database from disk."""
        pass


class FAISSManager(VectorDBInterface):
    """FAISS-based vector database for local deployment."""
    
    def __init__(self, dimension: int, index_type: str = "flat"):
        """
        Initialize FAISS manager.
        
        Args:
            dimension: Embedding dimension
            index_type: Type of index ("flat", "ivf", "hnsw")
        """
        self.dimension = dimension
        self.index_type = index_type
        
        # Create index based on type
        if index_type == "flat":
            self.index = faiss.IndexFlatL2(dimension)
        elif index_type == "ivf":
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, 100)
            self.needs_training = True
        elif index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(dimension, 32)
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        # Storage for texts and metadata
        self.texts: List[str] = []
        self.metadata: List[Dict] = []
        self.id_map: Dict[str, int] = {}
        
        logger.info(f"Initialized FAISS {index_type} index with dimension {dimension}")
    
    def add(self, embeddings: np.ndarray, texts: List[str], metadata: Optional[List[Dict]] = None) -> None:
        """Add embeddings to FAISS index."""
        # Ensure float32
        embeddings = embeddings.astype(np.float32)
        
        # Train if needed (for IVF indices)
        if hasattr(self, 'needs_training') and self.needs_training:
            logger.info("Training FAISS index...")
            self.index.train(embeddings)
            self.needs_training = False
        
        # Add to index
        start_idx = len(self.texts)
        self.index.add(embeddings)
        
        # Store texts and metadata
        self.texts.extend(texts)
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{}] * len(texts))
        
        # Update ID map
        for i, text in enumerate(texts):
            self.id_map[f"doc_{start_idx + i}"] = start_idx + i
        
        logger.info(f"Added {len(texts)} documents to FAISS index")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """Search FAISS index."""
        query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
        
        # Search
        distances, indices = self.index.search(query_embedding, k)
        
        # Format results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.texts):  # Valid index
                results.append({
                    'text': self.texts[idx],
                    'metadata': self.metadata[idx],
                    'score': float(1 / (1 + dist)),  # Convert distance to similarity
                    'distance': float(dist),
                    'index': int(idx)
                })
        
        return results
    
    def delete(self, ids: List[str]) -> None:
        """Delete not directly supported in FAISS - would need to rebuild index."""
        logger.warning("Delete operation not supported in FAISS. Consider rebuilding index.")
    
    def save(self, path: str) -> None:
        """Save FAISS index and metadata."""
        # Save index
        faiss.write_index(self.index, f"{path}.index")
        
        # Save metadata
        metadata_dict = {
            'texts': self.texts,
            'metadata': self.metadata,
            'id_map': self.id_map,
            'dimension': self.dimension,
            'index_type': self.index_type
        }
        
        with open(f"{path}.meta", 'w') as f:
            json.dump(metadata_dict, f)
        
        logger.info(f"Saved FAISS index to {path}")
    
    def load(self, path: str) -> None:
        """Load FAISS index and metadata."""
        # Load index
        self.index = faiss.read_index(f"{path}.index")
        
        # Load metadata
        with open(f"{path}.meta", 'r') as f:
            metadata_dict = json.load(f)
        
        self.texts = metadata_dict['texts']
        self.metadata = metadata_dict['metadata']
        self.id_map = metadata_dict['id_map']
        
        logger.info(f"Loaded FAISS index from {path}")


class ChromaManager(VectorDBInterface):
    """Chroma-based vector database for managed deployment."""
    
    def __init__(self, collection_name: str = "documents", persist_directory: Optional[str] = None):
        """
        Initialize Chroma manager.
        
        Args:
            collection_name: Name of the collection
            persist_directory: Directory for persistence (None for in-memory)
        """
        if not CHROMA_AVAILABLE:
            raise ImportError(
                "ChromaDB is not installed. Install with: pip install chromadb\n"
                "Note: ChromaDB may have build issues on some systems."
            )
        
        settings = Settings(
            anonymized_telemetry=False,
            persist_directory=persist_directory
        )
        
        self.client = Client(settings)
        self.collection = self.client.create_collection(
            name=collection_name,
            get_or_create=True
        )
        
        self.collection_name = collection_name
        logger.info(f"Initialized Chroma collection: {collection_name}")
    
    def add(self, embeddings: np.ndarray, texts: List[str], metadata: Optional[List[Dict]] = None) -> None:
        """Add embeddings to Chroma."""
        # Generate IDs
        start_idx = self.collection.count()
        ids = [f"doc_{start_idx + i}" for i in range(len(texts))]
        
        # Prepare metadata
        if metadata is None:
            metadata = [{}] * len(texts)
        
        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadata
        )
        
        logger.info(f"Added {len(texts)} documents to Chroma")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """Search Chroma collection."""
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=k
        )
        
        # Format results
        formatted_results = []
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'score': 1 - results['distances'][0][i],  # Convert distance to similarity
                    'distance': results['distances'][0][i],
                    'id': results['ids'][0][i]
                })
        
        return formatted_results
    
    def delete(self, ids: List[str]) -> None:
        """Delete documents from Chroma."""
        self.collection.delete(ids=ids)
        logger.info(f"Deleted {len(ids)} documents from Chroma")
    
    def save(self, path: str) -> None:
        """Chroma persists automatically if persist_directory is set."""
        logger.info("Chroma automatically persists to the configured directory")
    
    def load(self, path: str) -> None:
        """Chroma loads automatically from persist_directory."""
        logger.info("Chroma automatically loads from the configured directory")


class UnifiedVectorDB:
    """Unified interface supporting multiple vector database backends."""
    
    def __init__(self, backend: str = "faiss", **kwargs):
        """
        Initialize unified vector database.
        
        Args:
            backend: Backend to use ("faiss" or "chroma")
            **kwargs: Backend-specific arguments
        """
        self.backend_name = backend
        
        if backend == "faiss":
            dimension = kwargs.get('dimension', 384)
            index_type = kwargs.get('index_type', 'flat')
            self.backend = FAISSManager(dimension, index_type)
        elif backend == "chroma":
            collection_name = kwargs.get('collection_name', 'documents')
            persist_directory = kwargs.get('persist_directory', None)
            self.backend = ChromaManager(collection_name, persist_directory)
        else:
            raise ValueError(f"Unknown backend: {backend}")
        
        self.encoder = None
        logger.info(f"Initialized UnifiedVectorDB with {backend} backend")
    
    def set_encoder(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Set the sentence transformer model for encoding."""
        device = get_device()
        self.encoder = SentenceTransformer(model_name, device=device)
        logger.info(f"Set encoder: {model_name}")
    
    def add_documents(
        self,
        documents: List[str],
        metadata: Optional[List[Dict]] = None,
        batch_size: int = 32
    ) -> None:
        """Add documents with automatic encoding."""
        if self.encoder is None:
            raise ValueError("Encoder not set. Call set_encoder() first.")
        
        # Encode documents
        logger.info(f"Encoding {len(documents)} documents...")
        embeddings = self.encoder.encode(
            documents,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        
        # Add to backend
        self.backend.add(embeddings, documents, metadata)
    
    def search(
        self,
        query: str,
        k: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """Search with automatic query encoding."""
        if self.encoder is None:
            raise ValueError("Encoder not set. Call set_encoder() first.")
        
        # Encode query
        query_embedding = self.encoder.encode(query, convert_to_numpy=True)
        
        # Search
        results = self.backend.search(query_embedding, k)
        
        # Apply metadata filter if specified
        if filter_metadata:
            filtered_results = []
            for result in results:
                match = all(
                    result['metadata'].get(key) == value
                    for key, value in filter_metadata.items()
                )
                if match:
                    filtered_results.append(result)
            results = filtered_results
        
        return results
    
    def switch_backend(self, new_backend: str, **kwargs):
        """Switch to a different backend."""
        # Get current data
        logger.info(f"Switching from {self.backend_name} to {new_backend}")
        
        # Note: This is a simplified example. In production, you'd need to
        # properly migrate data between backends
        self.backend_name = new_backend
        
        if new_backend == "faiss":
            self.backend = FAISSManager(**kwargs)
        elif new_backend == "chroma":
            self.backend = ChromaManager(**kwargs)
        else:
            raise ValueError(f"Unknown backend: {new_backend}")


def run_vector_db_examples():
    """Demonstrate vector database operations."""
    
    print_section("Vector Database Manager Examples")
    
    # Example 1: FAISS Local Deployment
    print("\n=== Example 1: FAISS Local Deployment ===")
    
    # Sample documents
    documents = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing helps computers understand human language.",
        "Computer vision enables machines to interpret visual information.",
        "Reinforcement learning trains agents through rewards and penalties."
    ]
    
    # Initialize FAISS backend
    db = UnifiedVectorDB(backend="faiss", dimension=384)
    db.set_encoder('all-MiniLM-L6-v2')
    
    # Add documents
    metadata = [{"category": "AI", "id": i} for i in range(len(documents))]
    db.add_documents(documents, metadata)
    
    # Search
    queries = [
        "What is deep learning?",
        "How do computers see?",
        "Training AI agents"
    ]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        results = db.search(query, k=2)
        
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['text'][:80]}...")
            print(f"     Score: {result['score']:.3f}, Category: {result['metadata'].get('category', 'N/A')}")
    
    # Save index
    db.backend.save("faiss_example")
    print("\nFAISS index saved to disk")
    
    # Example 2: Chroma Managed Deployment
    print("\n\n=== Example 2: Chroma Managed Deployment ===")
    
    if not CHROMA_AVAILABLE:
        print("ChromaDB is not installed. Skipping Chroma examples.")
        print("To install ChromaDB, run: pip install chromadb")
        return
    
    # Initialize Chroma backend
    chroma_db = UnifiedVectorDB(
        backend="chroma",
        collection_name="tech_docs",
        persist_directory="./chroma_db"
    )
    chroma_db.set_encoder('all-MiniLM-L6-v2')
    
    # Technical documentation
    tech_docs = [
        "REST APIs use HTTP methods for CRUD operations.",
        "GraphQL provides a flexible query language for APIs.",
        "WebSockets enable real-time bidirectional communication.",
        "gRPC uses Protocol Buffers for efficient serialization.",
        "OAuth 2.0 is a standard for access delegation."
    ]
    
    tech_metadata = [
        {"type": "API", "protocol": "HTTP"},
        {"type": "API", "protocol": "HTTP"},
        {"type": "Communication", "protocol": "WebSocket"},
        {"type": "API", "protocol": "gRPC"},
        {"type": "Security", "protocol": "OAuth"}
    ]
    
    chroma_db.add_documents(tech_docs, tech_metadata)
    
    # Search with metadata filtering
    print("\nSearching for API-related documents:")
    api_results = chroma_db.search(
        "API communication protocols",
        k=5,
        filter_metadata={"type": "API"}
    )
    
    for result in api_results:
        print(f"  - {result['text']}")
        print(f"    Type: {result['metadata']['type']}, Protocol: {result['metadata']['protocol']}")
    
    # Example 3: Performance Comparison
    print("\n\n=== Example 3: Performance Comparison ===")
    
    import time
    
    # Create larger dataset
    large_docs = [f"Document {i}: Content about topic {i % 10}" for i in range(1000)]
    
    backends = [
        ("FAISS Flat", {"backend": "faiss", "dimension": 384, "index_type": "flat"}),
        ("FAISS HNSW", {"backend": "faiss", "dimension": 384, "index_type": "hnsw"}),
        ("Chroma", {"backend": "chroma", "collection_name": "benchmark"})
    ]
    
    for name, config in backends:
        print(f"\n{name}:")
        
        db = UnifiedVectorDB(**config)
        db.set_encoder('all-MiniLM-L6-v2')
        
        # Measure indexing time
        start = time.time()
        db.add_documents(large_docs[:100])  # Use subset for demo
        index_time = time.time() - start
        print(f"  Indexing time: {index_time:.3f}s")
        
        # Measure search time
        queries = ["Document about topic 5", "Content related to 7", "Information on 3"]
        
        start = time.time()
        for query in queries:
            db.search(query, k=5)
        search_time = (time.time() - start) / len(queries)
        print(f"  Average search time: {search_time*1000:.1f}ms")


def demonstrate_advanced_features():
    """Demonstrate advanced vector database features."""
    
    print_section("Advanced Vector Database Features")
    
    # Example: Hybrid Index Management
    print("\n=== Hybrid Index Management ===")
    
    class HybridVectorDB:
        """Manages both exact and approximate indices for different use cases."""
        
        def __init__(self, dimension: int = 384):
            self.dimension = dimension
            
            # Exact index for high-precision queries
            self.exact_index = FAISSManager(dimension, "flat")
            
            # Approximate index for fast queries
            self.approx_index = FAISSManager(dimension, "hnsw")
            
            # Encoder
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        def add(self, documents: List[str], important: List[bool]):
            """Add documents, marking some as important for exact search."""
            embeddings = self.encoder.encode(documents, convert_to_numpy=True)
            
            important_docs = []
            important_embs = []
            regular_docs = []
            regular_embs = []
            
            for doc, emb, imp in zip(documents, embeddings, important):
                if imp:
                    important_docs.append(doc)
                    important_embs.append(emb)
                else:
                    regular_docs.append(doc)
                    regular_embs.append(emb)
            
            if important_docs:
                self.exact_index.add(
                    np.array(important_embs),
                    important_docs,
                    [{"important": True}] * len(important_docs)
                )
            
            if regular_docs:
                self.approx_index.add(
                    np.array(regular_embs),
                    regular_docs,
                    [{"important": False}] * len(regular_docs)
                )
        
        def search(self, query: str, precision_mode: bool = False, k: int = 5):
            """Search using exact or approximate index based on mode."""
            query_emb = self.encoder.encode(query, convert_to_numpy=True)
            
            if precision_mode:
                return self.exact_index.search(query_emb, k)
            else:
                # Search both indices and merge results
                exact_results = self.exact_index.search(query_emb, k//2)
                approx_results = self.approx_index.search(query_emb, k//2)
                
                # Merge and sort by score
                all_results = exact_results + approx_results
                all_results.sort(key=lambda x: x['score'], reverse=True)
                
                return all_results[:k]
    
    # Test hybrid system
    hybrid_db = HybridVectorDB()
    
    # Add documents with importance flags
    docs = [
        ("Legal contract terms and conditions", True),
        ("Company policy on remote work", True),
        ("Blog post about productivity tips", False),
        ("News article on tech trends", False),
        ("Financial regulations compliance", True)
    ]
    
    documents, importance = zip(*docs)
    hybrid_db.add(list(documents), list(importance))
    
    # Test different search modes
    query = "compliance and regulations"
    
    print(f"\nQuery: '{query}'")
    print("\nPrecision mode (exact index only):")
    precision_results = hybrid_db.search(query, precision_mode=True, k=2)
    for r in precision_results:
        print(f"  - {r['text']} (Important: {r['metadata'].get('important', 'N/A')})")
    
    print("\nRegular mode (hybrid search):")
    regular_results = hybrid_db.search(query, precision_mode=False, k=3)
    for r in regular_results:
        print(f"  - {r['text']} (Important: {r['metadata'].get('important', 'N/A')})")


if __name__ == "__main__":
    run_vector_db_examples()
    print("\n" + "="*80 + "\n")
    demonstrate_advanced_features()