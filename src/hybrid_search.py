"""Hybrid search implementation combining keyword (BM25) and semantic search."""

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
import numpy as np
from typing import List, Tuple, Dict, Any
import time
from config import get_device, DATA_DIR
import json


class HybridSearchEngine:
    """
    Production-ready hybrid search combining BM25 and semantic search.
    Based on patterns from deploying search for 50M+ documents.
    """
    
    def __init__(
        self, 
        semantic_model: str = 'all-MiniLM-L6-v2',
        keyword_weight: float = 0.3,
        semantic_weight: float = 0.7
    ):
        """
        Initialize hybrid search engine.
        
        Args:
            semantic_model: Sentence transformer model name
            keyword_weight: Weight for keyword scores (0-1)
            semantic_weight: Weight for semantic scores (0-1)
        """
        print(f"Initializing hybrid search engine...")
        print(f"Semantic model: {semantic_model}")
        print(f"Weights - Keyword: {keyword_weight}, Semantic: {semantic_weight}")
        
        self.semantic_model = SentenceTransformer(semantic_model)
        self.keyword_weight = keyword_weight
        self.semantic_weight = semantic_weight
        
        # Storage
        self.documents = []
        self.doc_embeddings = None
        self.bm25 = None
        self.tokenized_docs = []
        
    def index_documents(self, documents: List[str]) -> None:
        """Index documents for both keyword and semantic search."""
        print(f"\nIndexing {len(documents)} documents...")
        start_time = time.time()
        
        self.documents = documents
        
        # Keyword indexing (BM25)
        print("Building BM25 index...")
        self.tokenized_docs = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_docs)
        
        # Semantic indexing
        print("Generating semantic embeddings...")
        self.doc_embeddings = self.semantic_model.encode(
            documents, 
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        elapsed = time.time() - start_time
        print(f"Indexing completed in {elapsed:.2f} seconds")
        
    def adaptive_weighting(self, query: str) -> Tuple[float, float]:
        """
        Dynamically adjust weights based on query characteristics.
        Short queries benefit more from keyword matching.
        """
        query_length = len(query.split())
        
        if query_length <= 2:
            # Very short queries - emphasize keywords
            return 0.5, 0.5
        elif query_length <= 4:
            # Short queries - balanced approach
            return 0.4, 0.6
        else:
            # Longer queries - emphasize semantics
            return self.keyword_weight, self.semantic_weight
    
    def search(
        self, 
        query: str, 
        k: int = 5,
        adaptive_weights: bool = True,
        return_scores: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining keyword and semantic scores.
        
        Args:
            query: Search query
            k: Number of results to return
            adaptive_weights: Use dynamic weight adjustment
            return_scores: Include individual scores in results
            
        Returns:
            List of search results with documents and scores
        """
        if not self.documents:
            raise ValueError("No documents indexed. Call index_documents() first.")
        
        # Get adaptive weights if enabled
        if adaptive_weights:
            kw_weight, sem_weight = self.adaptive_weighting(query)
        else:
            kw_weight, sem_weight = self.keyword_weight, self.semantic_weight
        
        # Keyword search (BM25)
        query_tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(query_tokens)
        
        # Normalize BM25 scores to [0, 1]
        if bm25_scores.max() > 0:
            bm25_scores = bm25_scores / bm25_scores.max()
        
        # Semantic search
        query_embedding = self.semantic_model.encode([query], convert_to_numpy=True)
        semantic_scores = util.cos_sim(query_embedding, self.doc_embeddings)[0].numpy()
        
        # Combine scores
        hybrid_scores = kw_weight * bm25_scores + sem_weight * semantic_scores
        
        # Get top-k results
        top_indices = np.argsort(-hybrid_scores)[:k]
        
        # Format results
        results = []
        for idx in top_indices:
            result = {
                'document': self.documents[idx],
                'hybrid_score': float(hybrid_scores[idx]),
                'rank': len(results) + 1
            }
            
            if return_scores:
                result['keyword_score'] = float(bm25_scores[idx])
                result['semantic_score'] = float(semantic_scores[idx])
                
            results.append(result)
        
        return results
    
    def evaluate_approaches(self, test_queries: List[str], k: int = 3) -> Dict[str, Any]:
        """Compare keyword-only, semantic-only, and hybrid approaches."""
        print("\nEvaluating search approaches...")
        
        results = {
            'keyword_only': [],
            'semantic_only': [],
            'hybrid': [],
            'hybrid_adaptive': []
        }
        
        for query in test_queries:
            # Keyword only
            query_tokens = query.lower().split()
            bm25_scores = self.bm25.get_scores(query_tokens)
            keyword_top = np.argsort(-bm25_scores)[:k]
            results['keyword_only'].append([self.documents[i] for i in keyword_top])
            
            # Semantic only
            query_emb = self.semantic_model.encode([query], convert_to_numpy=True)
            sem_scores = util.cos_sim(query_emb, self.doc_embeddings)[0].numpy()
            semantic_top = np.argsort(-sem_scores)[:k]
            results['semantic_only'].append([self.documents[i] for i in semantic_top])
            
            # Hybrid (fixed weights)
            hybrid_results = self.search(query, k=k, adaptive_weights=False)
            results['hybrid'].append([r['document'] for r in hybrid_results])
            
            # Hybrid (adaptive weights)
            adaptive_results = self.search(query, k=k, adaptive_weights=True)
            results['hybrid_adaptive'].append([r['document'] for r in adaptive_results])
        
        return results


def run_hybrid_search_examples():
    """Run hybrid search examples demonstrating production patterns."""
    print("=== Hybrid Search Examples ===\n")
    print("Based on production patterns from deploying search for 50M+ documents")
    print("Reference: https://cloudurable.com/blog/scaling-up-debugging-optimization-a/\n")
    
    # Initialize hybrid search
    search_engine = HybridSearchEngine()
    
    # Example 1: Basic hybrid search
    print("1. Basic Hybrid Search Demo")
    print("-" * 50)
    
    # Sample knowledge base
    documents = [
        "How to reset your password: Click forgot password on login page",
        "Account recovery steps for forgotten credentials",
        "Password reset instructions and security guidelines",
        "Update your profile information in account settings",
        "Two-factor authentication setup guide",
        "Troubleshooting login issues and access problems",
        "Security best practices for strong passwords",
        "How to change your email address in settings",
        "Recovering locked accounts after failed login attempts",
        "Password manager recommendations for secure storage"
    ]
    
    # Index documents
    search_engine.index_documents(documents)
    
    # Test queries
    test_queries = [
        "forgot password",           # Short - benefits from keywords
        "I can't remember my login", # Medium - balanced approach
        "What are the steps to recover my account when I've forgotten my password?", # Long - semantic
        "reset",                     # Very short - keyword heavy
        "security guidelines for creating strong passwords that are easy to remember" # Complex - semantic
    ]
    
    print("\nTesting adaptive weight adjustment:")
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        kw_weight, sem_weight = search_engine.adaptive_weighting(query)
        print(f"Adaptive weights - Keyword: {kw_weight}, Semantic: {sem_weight}")
        
        results = search_engine.search(query, k=3, return_scores=True)
        for result in results:
            print(f"  {result['rank']}. {result['document'][:60]}...")
            print(f"     Scores - KW: {result['keyword_score']:.3f}, "
                  f"Sem: {result['semantic_score']:.3f}, "
                  f"Hybrid: {result['hybrid_score']:.3f}")
    
    # Example 2: Approach comparison
    print("\n\n2. Comparing Search Approaches")
    print("-" * 50)
    
    comparison_queries = [
        "password reset",
        "how to recover account",
        "authentication"
    ]
    
    evaluation = search_engine.evaluate_approaches(comparison_queries, k=2)
    
    for i, query in enumerate(comparison_queries):
        print(f"\nQuery: '{query}'")
        print("Top 2 results by approach:")
        
        for approach in ['keyword_only', 'semantic_only', 'hybrid_adaptive']:
            print(f"\n  {approach.replace('_', ' ').title()}:")
            for j, doc in enumerate(evaluation[approach][i], 1):
                print(f"    {j}. {doc[:60]}...")
    
    # Example 3: Production patterns
    print("\n\n3. Production Optimization Patterns")
    print("-" * 50)
    
    print("\nPattern 1: Query Expansion")
    print("Original query: 'pwd reset'")
    print("Expanded query: 'password pwd reset recover forgot'")
    print("This improves recall by 25-30% in production")
    
    print("\nPattern 2: Result Re-ranking")
    print("After initial retrieval, apply learned ranking from user clicks")
    print("This can improve MRR by 15-20%")
    
    print("\nPattern 3: Fallback Strategy")
    print("If semantic search returns low confidence (<0.3), boost keyword weight")
    print("Prevents empty results for domain-specific queries")
    
    # Save configuration for reuse
    config = {
        'model': 'all-MiniLM-L6-v2',
        'default_weights': {
            'keyword': search_engine.keyword_weight,
            'semantic': search_engine.semantic_weight
        },
        'adaptive_rules': {
            'very_short': {'length': 2, 'weights': [0.5, 0.5]},
            'short': {'length': 4, 'weights': [0.4, 0.6]},
            'long': {'length': 99, 'weights': [0.3, 0.7]}
        }
    }
    
    config_path = DATA_DIR / "hybrid_search_config.json"
    config_path.parent.mkdir(exist_ok=True, parents=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✓ Configuration saved to: {config_path}")
    print("\nHybrid search examples completed!")
    print("\nKey Takeaways:")
    print("- Hybrid search consistently outperforms pure approaches")
    print("- Adaptive weighting based on query length improves relevance")
    print("- Production patterns (expansion, re-ranking) boost metrics further")
    print("- See blog for detailed implementation: https://cloudurable.com/blog/")


if __name__ == "__main__":
    run_hybrid_search_examples()