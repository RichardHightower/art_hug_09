"""Performance benchmarking for semantic search and transformers."""

import time
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import faiss
import torch
from typing import List, Dict, Any
import pandas as pd
from config import get_device, DATA_DIR
import json


class PerformanceBenchmark:
    """Comprehensive performance benchmarking for semantic search."""
    
    def __init__(self):
        """Initialize benchmarking suite."""
        self.device = get_device()
        self.results = {}
        
    def benchmark_embedding_models(self, texts: List[str], models: List[str]) -> Dict[str, Any]:
        """Benchmark different embedding models."""
        print("\nBenchmarking Embedding Models")
        print("=" * 50)
        
        results = {}
        
        for model_name in models:
            print(f"\nTesting: {model_name}")
            
            try:
                # Load model
                load_start = time.time()
                model = SentenceTransformer(model_name)
                load_time = time.time() - load_start
                
                # Get model info
                model_size = sum(p.numel() for p in model.parameters()) / 1e6  # Million params
                
                # Benchmark encoding
                encode_times = []
                for i in range(3):  # Multiple runs
                    start = time.time()
                    embeddings = model.encode(texts, batch_size=32, show_progress_bar=False)
                    encode_times.append(time.time() - start)
                
                avg_encode_time = np.mean(encode_times)
                throughput = len(texts) / avg_encode_time
                
                # Memory usage (approximate)
                embedding_dim = embeddings.shape[1]
                memory_mb = embeddings.nbytes / (1024 * 1024)
                
                results[model_name] = {
                    'load_time': load_time,
                    'model_size_m': model_size,
                    'embedding_dim': embedding_dim,
                    'encode_time': avg_encode_time,
                    'throughput': throughput,
                    'memory_mb': memory_mb
                }
                
                print(f"  ✓ Load time: {load_time:.2f}s")
                print(f"  ✓ Model size: {model_size:.1f}M parameters")
                print(f"  ✓ Embedding dimension: {embedding_dim}")
                print(f"  ✓ Encoding throughput: {throughput:.1f} texts/sec")
                
            except Exception as e:
                print(f"  ✗ Failed: {e}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    def benchmark_faiss_indices(self, embeddings: np.ndarray, queries: np.ndarray) -> Dict[str, Any]:
        """Benchmark different FAISS index types."""
        print("\nBenchmarking FAISS Indices")
        print("=" * 50)
        
        dimension = embeddings.shape[1]
        results = {}
        
        # Index configurations
        indices = {
            'Flat (Exact)': faiss.IndexFlatL2(dimension),
            'IVF100,Flat': None,  # Will create after training
            'HNSW32': faiss.IndexHNSWFlat(dimension, 32),
            'LSH': faiss.IndexLSH(dimension, dimension * 2)
        }
        
        # Create IVF index
        nlist = min(100, len(embeddings) // 10)
        quantizer = faiss.IndexFlatL2(dimension)
        indices['IVF100,Flat'] = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
        
        for index_name, index in indices.items():
            print(f"\nTesting: {index_name}")
            
            try:
                # Training (if required)
                train_start = time.time()
                if hasattr(index, 'train') and not index.is_trained:
                    index.train(embeddings)
                train_time = time.time() - train_start
                
                # Adding vectors
                add_start = time.time()
                index.add(embeddings)
                add_time = time.time() - add_start
                
                # Search benchmark
                k = 10
                search_times = []
                for _ in range(10):
                    start = time.time()
                    D, I = index.search(queries, k)
                    search_times.append(time.time() - start)
                
                avg_search_time = np.mean(search_times)
                qps = len(queries) / avg_search_time  # Queries per second
                
                # Memory usage
                memory_mb = index.ntotal * dimension * 4 / (1024 * 1024)  # Approximate
                
                results[index_name] = {
                    'train_time': train_time,
                    'add_time': add_time,
                    'search_time': avg_search_time,
                    'qps': qps,
                    'memory_mb': memory_mb
                }
                
                print(f"  ✓ Training time: {train_time:.3f}s")
                print(f"  ✓ Indexing time: {add_time:.3f}s")
                print(f"  ✓ Search QPS: {qps:.1f}")
                print(f"  ✓ Memory: ~{memory_mb:.1f} MB")
                
            except Exception as e:
                print(f"  ✗ Failed: {e}")
                results[index_name] = {'error': str(e)}
        
        return results
    
    def benchmark_search_quality(self, embeddings: np.ndarray, queries: np.ndarray, 
                               ground_truth: List[List[int]]) -> Dict[str, Any]:
        """Benchmark search quality metrics."""
        print("\nBenchmarking Search Quality")
        print("=" * 50)
        
        dimension = embeddings.shape[1]
        results = {}
        
        # Create exact search index for ground truth
        exact_index = faiss.IndexFlatL2(dimension)
        exact_index.add(embeddings)
        
        # Test different indices
        test_indices = {
            'Exact': exact_index,
            'IVF50': None,
            'HNSW16': faiss.IndexHNSWFlat(dimension, 16)
        }
        
        # Create IVF index
        nlist = 50
        quantizer = faiss.IndexFlatL2(dimension)
        test_indices['IVF50'] = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
        test_indices['IVF50'].train(embeddings)
        
        for index_name, index in test_indices.items():
            if index is None:
                continue
                
            print(f"\nTesting: {index_name}")
            
            # Add data
            if index.ntotal == 0:
                index.add(embeddings)
            
            # Search
            k = 10
            D, I = index.search(queries, k)
            
            # Calculate metrics
            precision_at_k = []
            recall_at_k = []
            
            for i, (retrieved, relevant) in enumerate(zip(I, ground_truth)):
                if len(relevant) == 0:
                    continue
                    
                retrieved_set = set(retrieved)
                relevant_set = set(relevant[:k])
                
                precision = len(retrieved_set & relevant_set) / k
                recall = len(retrieved_set & relevant_set) / len(relevant_set)
                
                precision_at_k.append(precision)
                recall_at_k.append(recall)
            
            avg_precision = np.mean(precision_at_k)
            avg_recall = np.mean(recall_at_k)
            f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall + 1e-8)
            
            results[index_name] = {
                'precision@10': avg_precision,
                'recall@10': avg_recall,
                'f1_score': f1_score
            }
            
            print(f"  ✓ Precision@10: {avg_precision:.3f}")
            print(f"  ✓ Recall@10: {avg_recall:.3f}")
            print(f"  ✓ F1 Score: {f1_score:.3f}")
        
        return results
    
    def create_visualization(self, results: Dict[str, Any]) -> None:
        """Create performance visualization plots."""
        print("\nCreating Performance Visualizations")
        print("=" * 50)
        
        # Ensure data directory exists
        viz_dir = DATA_DIR / "benchmarks"
        viz_dir.mkdir(exist_ok=True, parents=True)
        
        # Model comparison plot
        if 'embedding_models' in results:
            model_data = results['embedding_models']
            valid_models = {k: v for k, v in model_data.items() if 'error' not in v}
            
            if valid_models:
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                fig.suptitle('Embedding Model Performance Comparison', fontsize=16)
                
                models = list(valid_models.keys())
                
                # Throughput
                throughputs = [valid_models[m]['throughput'] for m in models]
                axes[0, 0].bar(models, throughputs)
                axes[0, 0].set_title('Encoding Throughput')
                axes[0, 0].set_ylabel('Texts/second')
                axes[0, 0].tick_params(axis='x', rotation=45)
                
                # Model size
                sizes = [valid_models[m]['model_size_m'] for m in models]
                axes[0, 1].bar(models, sizes)
                axes[0, 1].set_title('Model Size')
                axes[0, 1].set_ylabel('Million Parameters')
                axes[0, 1].tick_params(axis='x', rotation=45)
                
                # Embedding dimension
                dims = [valid_models[m]['embedding_dim'] for m in models]
                axes[1, 0].bar(models, dims)
                axes[1, 0].set_title('Embedding Dimension')
                axes[1, 0].set_ylabel('Dimensions')
                axes[1, 0].tick_params(axis='x', rotation=45)
                
                # Load time
                load_times = [valid_models[m]['load_time'] for m in models]
                axes[1, 1].bar(models, load_times)
                axes[1, 1].set_title('Model Load Time')
                axes[1, 1].set_ylabel('Seconds')
                axes[1, 1].tick_params(axis='x', rotation=45)
                
                plt.tight_layout()
                plt.savefig(viz_dir / 'model_comparison.png', dpi=150, bbox_inches='tight')
                print("  ✓ Saved model comparison plot")
        
        # FAISS index comparison
        if 'faiss_indices' in results:
            index_data = results['faiss_indices']
            valid_indices = {k: v for k, v in index_data.items() if 'error' not in v}
            
            if valid_indices:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                fig.suptitle('FAISS Index Performance Comparison', fontsize=16)
                
                indices = list(valid_indices.keys())
                
                # QPS comparison
                qps_values = [valid_indices[idx]['qps'] for idx in indices]
                ax1.bar(indices, qps_values)
                ax1.set_title('Search Performance (QPS)')
                ax1.set_ylabel('Queries per Second')
                ax1.tick_params(axis='x', rotation=45)
                
                # Memory usage
                memory_values = [valid_indices[idx]['memory_mb'] for idx in indices]
                ax2.bar(indices, memory_values)
                ax2.set_title('Memory Usage')
                ax2.set_ylabel('MB')
                ax2.tick_params(axis='x', rotation=45)
                
                plt.tight_layout()
                plt.savefig(viz_dir / 'faiss_comparison.png', dpi=150, bbox_inches='tight')
                print("  ✓ Saved FAISS comparison plot")


def run_performance_benchmarking_examples():
    """Run comprehensive performance benchmarking."""
    print("=== Performance Benchmarking Examples ===\n")
    
    benchmark = PerformanceBenchmark()
    all_results = {}
    
    # Generate test data
    print("Generating test data...")
    np.random.seed(42)
    
    # Sample texts
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming industries.",
        "Natural language processing enables human-computer interaction.",
        "Deep learning models require significant computational resources.",
        "Transformers have revolutionized NLP tasks.",
    ] * 20  # 100 texts total
    
    # Example 1: Benchmark embedding models
    print("\n1. Embedding Model Comparison")
    print("-" * 40)
    
    test_models = [
        'all-MiniLM-L6-v2',           # Fast, small
        'all-mpnet-base-v2',          # Balanced
        'paraphrase-MiniLM-L3-v2',    # Very fast
    ]
    
    model_results = benchmark.benchmark_embedding_models(test_texts, test_models)
    all_results['embedding_models'] = model_results
    
    # Use the first model for subsequent tests
    print("\nGenerating embeddings for FAISS tests...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(test_texts, convert_to_numpy=True).astype('float32')
    
    # Generate query embeddings
    query_texts = test_texts[:10]
    query_embeddings = model.encode(query_texts, convert_to_numpy=True).astype('float32')
    
    # Example 2: Benchmark FAISS indices
    print("\n2. FAISS Index Type Comparison")
    print("-" * 40)
    
    faiss_results = benchmark.benchmark_faiss_indices(embeddings, query_embeddings)
    all_results['faiss_indices'] = faiss_results
    
    # Example 3: Search quality comparison
    print("\n3. Search Quality Metrics")
    print("-" * 40)
    
    # Generate synthetic ground truth (for demo purposes)
    ground_truth = []
    for i in range(len(query_embeddings)):
        # Assume top-5 similar items for each query
        relevant = list(range(i, min(i + 5, len(embeddings))))
        ground_truth.append(relevant)
    
    quality_results = benchmark.benchmark_search_quality(embeddings, query_embeddings, ground_truth)
    all_results['search_quality'] = quality_results
    
    # Example 4: Create visualizations
    benchmark.create_visualization(all_results)
    
    # Save results
    results_path = DATA_DIR / "benchmarks" / "benchmark_results.json"
    results_path.parent.mkdir(exist_ok=True, parents=True)
    
    with open(results_path, 'w') as f:
        # Convert numpy types for JSON serialization
        json_results = json.dumps(all_results, indent=2, default=float)
        f.write(json_results)
    
    print(f"\n✓ Results saved to: {results_path}")
    
    # Summary recommendations
    print("\n" + "=" * 50)
    print("PERFORMANCE RECOMMENDATIONS")
    print("=" * 50)
    
    print("\nFor Production Semantic Search:")
    print("1. Model Selection:")
    print("   - Speed priority: all-MiniLM-L6-v2")
    print("   - Quality priority: all-mpnet-base-v2")
    print("   - Multilingual: paraphrase-multilingual-MiniLM-L12-v2")
    
    print("\n2. FAISS Index Selection:")
    print("   - < 10K vectors: IndexFlatL2 (exact search)")
    print("   - 10K-1M vectors: IndexIVFFlat with nlist=sqrt(N)")
    print("   - > 1M vectors: IndexIVFPQ or IndexHNSWFlat")
    
    print("\n3. Optimization Tips:")
    print("   - Batch encode documents for better throughput")
    print("   - Use GPU when available for large-scale encoding")
    print("   - Consider quantization for memory-constrained environments")
    print("   - Implement caching for frequently searched queries")
    
    print("\nPerformance benchmarking completed!")


if __name__ == "__main__":
    run_performance_benchmarking_examples()