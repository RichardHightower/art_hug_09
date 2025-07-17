"""
Embedding Generation Module

This module demonstrates various approaches to generating embeddings for
semantic search applications. It covers both local models (sentence-transformers)
and API-based solutions (OpenAI, Cohere), with production best practices.

Topics covered:
- Batch embedding generation
- Multiple model comparison
- API vs local trade-offs
- Embedding storage and management
- Performance optimization
"""

from sentence_transformers import SentenceTransformer
import numpy as np
import openai
from typing import List, Dict, Optional, Union
import time
import json
import os
from dataclasses import dataclass
import logging

from config import get_device, OPENAI_API_KEY

def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EmbeddingModel:
    """Configuration for an embedding model."""
    name: str
    dimension: int
    max_tokens: int
    provider: str  # "local" or "api"
    description: str


# Available embedding models
EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2": EmbeddingModel(
        name="all-MiniLM-L6-v2",
        dimension=384,
        max_tokens=256,
        provider="local",
        description="Fast, lightweight model for general English text"
    ),
    "all-mpnet-base-v2": EmbeddingModel(
        name="all-mpnet-base-v2",
        dimension=768,
        max_tokens=384,
        provider="local",
        description="Higher quality embeddings, slower than MiniLM"
    ),
    "e5-base-v2": EmbeddingModel(
        name="intfloat/e5-base-v2",
        dimension=768,
        max_tokens=512,
        provider="local",
        description="State-of-the-art model, requires 'query:' prefix"
    ),
    "multilingual-e5-base": EmbeddingModel(
        name="intfloat/multilingual-e5-base",
        dimension=768,
        max_tokens=512,
        provider="local",
        description="Supports 100+ languages"
    ),
    "text-embedding-ada-002": EmbeddingModel(
        name="text-embedding-ada-002",
        dimension=1536,
        max_tokens=8191,
        provider="api",
        description="OpenAI's high-quality embeddings"
    )
}


class EmbeddingGenerator:
    """Unified interface for generating embeddings from various sources."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding generator.
        
        Args:
            model_name: Name of the model to use
        """
        if model_name not in EMBEDDING_MODELS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(EMBEDDING_MODELS.keys())}")
        
        self.model_config = EMBEDDING_MODELS[model_name]
        self.model_name = model_name
        self.device = get_device()
        
        # Initialize based on provider
        if self.model_config.provider == "local":
            logger.info(f"Loading local model: {self.model_config.name}")
            self.model = SentenceTransformer(self.model_config.name, device=self.device)
        elif self.model_config.provider == "api":
            if model_name == "text-embedding-ada-002" and OPENAI_API_KEY:
                openai.api_key = OPENAI_API_KEY
                self.model = None  # API-based, no local model
            else:
                raise ValueError(f"API key not configured for {model_name}")
        
        logger.info(f"Initialized {model_name} (dim={self.model_config.dimension})")
    
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = True,
        normalize: bool = False
    ) -> np.ndarray:
        """
        Generate embeddings for texts.
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            normalize: Normalize embeddings to unit length
            
        Returns:
            Embeddings as numpy array
        """
        # Handle single text
        if isinstance(texts, str):
            texts = [texts]
        
        # Generate embeddings based on provider
        if self.model_config.provider == "local":
            embeddings = self._encode_local(texts, batch_size, show_progress, normalize)
        else:
            embeddings = self._encode_api(texts, batch_size)
            if normalize:
                embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        return embeddings
    
    def _encode_local(
        self,
        texts: List[str],
        batch_size: int,
        show_progress: bool,
        normalize: bool
    ) -> np.ndarray:
        """Encode using local sentence transformer."""
        # Add prefix for E5 models
        if "e5" in self.model_name.lower():
            texts = [f"query: {text}" if len(text) < 100 else f"passage: {text}" for text in texts]
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=normalize
        )
        
        return embeddings
    
    def _encode_api(self, texts: List[str], batch_size: int) -> np.ndarray:
        """Encode using API (OpenAI)."""
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                response = openai.Embedding.create(
                    model=self.model_config.name,
                    input=batch
                )
                
                batch_embeddings = [item['embedding'] for item in response['data']]
                all_embeddings.extend(batch_embeddings)
                
            except Exception as e:
                logger.error(f"API error: {e}")
                raise
        
        return np.array(all_embeddings)
    
    def estimate_cost(self, num_texts: int, avg_tokens_per_text: int = 50) -> Dict[str, float]:
        """
        Estimate cost for generating embeddings.
        
        Args:
            num_texts: Number of texts to embed
            avg_tokens_per_text: Average tokens per text
            
        Returns:
            Cost estimation dictionary
        """
        if self.model_config.provider == "local":
            return {
                "cost_usd": 0.0,
                "time_estimate_seconds": num_texts * 0.001,  # Rough estimate
                "note": "Local model - no API costs"
            }
        elif self.model_name == "text-embedding-ada-002":
            # OpenAI pricing (as of 2024)
            cost_per_1k_tokens = 0.0001
            total_tokens = num_texts * avg_tokens_per_text
            cost = (total_tokens / 1000) * cost_per_1k_tokens
            
            return {
                "cost_usd": cost,
                "total_tokens": total_tokens,
                "time_estimate_seconds": num_texts * 0.01,
                "note": f"OpenAI API costs at ${cost_per_1k_tokens}/1K tokens"
            }
        
        return {"error": "Cost estimation not available"}


def generate_embeddings(texts: List[str], use_openai: bool = False) -> np.ndarray:
    """
    Simple function to generate embeddings (used in article examples).
    
    Args:
        texts: List of texts to embed
        use_openai: Whether to use OpenAI API
        
    Returns:
        Embeddings as numpy array
    """
    if use_openai and OPENAI_API_KEY:
        openai.api_key = OPENAI_API_KEY
        # OpenAI embedding code here (from article)
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=texts
        )
        embeddings = [item['embedding'] for item in response['data']]
        return np.array(embeddings)
    else:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model.encode(texts)


def run_embedding_generation_examples():
    """Demonstrate embedding generation with various models."""
    
    print_section("Embedding Generation Examples")
    
    # Example 1: Compare Different Models
    print("\n=== Example 1: Model Comparison ===")
    
    test_texts = [
        "Machine learning transforms data into insights.",
        "Deep neural networks learn hierarchical representations.",
        "Natural language processing enables human-computer interaction."
    ]
    
    models_to_test = ["all-MiniLM-L6-v2", "all-mpnet-base-v2"]
    
    for model_name in models_to_test:
        print(f"\n{model_name}:")
        generator = EmbeddingGenerator(model_name)
        
        # Generate embeddings and measure time
        start_time = time.time()
        embeddings = generator.encode(test_texts, show_progress=False)
        elapsed = time.time() - start_time
        
        print(f"  Dimension: {embeddings.shape[1]}")
        print(f"  Time: {elapsed:.3f}s ({elapsed/len(test_texts)*1000:.1f}ms per text)")
        print(f"  Description: {generator.model_config.description}")
        
        # Calculate similarity matrix
        from sentence_transformers import util
        similarities = util.cos_sim(embeddings, embeddings)
        print(f"  Avg similarity: {similarities.mean():.3f}")
    
    # Example 2: Batch Processing
    print("\n\n=== Example 2: Batch Processing ===")
    
    # Generate larger dataset
    large_dataset = [f"Document {i}: Content about {['AI', 'ML', 'NLP', 'CV'][i % 4]} topic." 
                     for i in range(100)]
    
    generator = EmbeddingGenerator("all-MiniLM-L6-v2")
    
    # Test different batch sizes
    batch_sizes = [8, 32, 64]
    
    print("\nBatch size performance:")
    for batch_size in batch_sizes:
        start_time = time.time()
        embeddings = generator.encode(
            large_dataset,
            batch_size=batch_size,
            show_progress=False
        )
        elapsed = time.time() - start_time
        
        print(f"  Batch size {batch_size}: {elapsed:.2f}s ({len(large_dataset)/elapsed:.1f} texts/sec)")
    
    # Example 3: Multilingual Embeddings
    print("\n\n=== Example 3: Multilingual Embeddings ===")
    
    multilingual_texts = [
        "Hello, how are you?",  # English
        "Bonjour, comment allez-vous?",  # French
        "Hola, ¿cómo estás?",  # Spanish
        "Здравствуйте, как дела?",  # Russian
        "你好，你好吗？",  # Chinese
    ]
    
    # Use multilingual model
    ml_generator = EmbeddingGenerator("multilingual-e5-base")
    ml_embeddings = ml_generator.encode(multilingual_texts, show_progress=False)
    
    # Calculate cross-lingual similarities
    print("\nCross-lingual similarity matrix:")
    similarities = util.cos_sim(ml_embeddings, ml_embeddings)
    
    languages = ["EN", "FR", "ES", "RU", "ZH"]
    print("     " + "  ".join(f"{lang:>4}" for lang in languages))
    for i, lang in enumerate(languages):
        print(f"{lang:>4} " + "  ".join(f"{similarities[i][j]:.2f}" for j in range(len(languages))))
    
    print("\nNote: Similar greetings show high similarity across languages!")
    
    # Example 4: Storage and Retrieval
    print("\n\n=== Example 4: Embedding Storage ===")
    
    documents = [
        "Python is a versatile programming language.",
        "JavaScript powers interactive web applications.",
        "Rust provides memory safety without garbage collection."
    ]
    
    generator = EmbeddingGenerator()
    embeddings = generator.encode(documents)
    
    # Save embeddings and metadata
    storage_data = {
        "model": generator.model_name,
        "dimension": generator.model_config.dimension,
        "documents": documents,
        "embeddings": embeddings.tolist(),
        "metadata": {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_documents": len(documents)
        }
    }
    
    # Save to file
    output_path = "embeddings_cache.json"
    with open(output_path, 'w') as f:
        json.dump(storage_data, f, indent=2)
    
    print(f"\nSaved {len(documents)} embeddings to {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024:.1f} KB")
    
    # Load and verify
    with open(output_path, 'r') as f:
        loaded_data = json.load(f)
    
    loaded_embeddings = np.array(loaded_data['embeddings'])
    print(f"Loaded embeddings shape: {loaded_embeddings.shape}")
    print(f"Embeddings match: {np.allclose(embeddings, loaded_embeddings)}")
    
    # Clean up
    os.remove(output_path)


def benchmark_embedding_models():
    """Benchmark different embedding models on performance and quality."""
    
    print_section("Embedding Model Benchmarks")
    
    # Test dataset
    test_queries = [
        "What is machine learning?",
        "How does neural network training work?",
        "Explain gradient descent optimization."
    ]
    
    test_documents = [
        "Machine learning is a subset of AI that enables systems to learn from data.",
        "Neural networks train by adjusting weights through backpropagation.",
        "Gradient descent minimizes loss by moving parameters in the steepest descent direction.",
        "Python is a popular programming language for data science.",
        "Cloud computing provides scalable infrastructure."
    ]
    
    # Expected relevance (query_idx, doc_idx pairs)
    expected_relevant = {
        0: {0},  # Query 0 relevant to doc 0
        1: {1},  # Query 1 relevant to doc 1
        2: {2}   # Query 2 relevant to doc 2
    }
    
    models = ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "e5-base-v2"]
    
    results = []
    
    for model_name in models:
        print(f"\n\nBenchmarking {model_name}...")
        generator = EmbeddingGenerator(model_name)
        
        # Encode documents and queries
        start_time = time.time()
        doc_embeddings = generator.encode(test_documents, show_progress=False)
        query_embeddings = generator.encode(test_queries, show_progress=False)
        encode_time = time.time() - start_time
        
        # Calculate retrieval accuracy
        correct = 0
        for q_idx, query_emb in enumerate(query_embeddings):
            # Find most similar document
            similarities = util.cos_sim(query_emb, doc_embeddings)[0]
            top_doc_idx = similarities.argmax().item()
            
            if top_doc_idx in expected_relevant[q_idx]:
                correct += 1
        
        accuracy = correct / len(test_queries)
        
        # Store results
        results.append({
            "model": model_name,
            "dimension": generator.model_config.dimension,
            "encode_time": encode_time,
            "accuracy": accuracy,
            "texts_per_second": len(test_documents + test_queries) / encode_time
        })
        
        print(f"  Accuracy: {accuracy:.1%}")
        print(f"  Speed: {results[-1]['texts_per_second']:.1f} texts/sec")
    
    # Summary table
    print("\n\n=== Benchmark Summary ===")
    print(f"{'Model':<25} {'Dim':<6} {'Accuracy':<10} {'Speed (texts/sec)':<20}")
    print("-" * 70)
    
    for r in results:
        print(f"{r['model']:<25} {r['dimension']:<6} {r['accuracy']:<10.1%} {r['texts_per_second']:<20.1f}")
    
    # Cost comparison
    print("\n\n=== Cost Analysis (1M documents) ===")
    
    for model_name in ["all-MiniLM-L6-v2", "text-embedding-ada-002"]:
        if model_name == "text-embedding-ada-002" and not OPENAI_API_KEY:
            continue
            
        generator = EmbeddingGenerator(model_name)
        cost_estimate = generator.estimate_cost(1_000_000, avg_tokens_per_text=50)
        
        print(f"\n{model_name}:")
        for key, value in cost_estimate.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    run_embedding_generation_examples()
    print("\n" + "="*80 + "\n")
    benchmark_embedding_models()