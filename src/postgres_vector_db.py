"""
PostgreSQL Vector Database Example

This module demonstrates how to use PostgreSQL with pgvector extension
for semantic search, comparing different index types and their performance.
"""

import os
import time
import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from typing import List, Dict, Any, Tuple
import json

# Load environment variables
load_dotenv()


class PostgresVectorDB:
    """PostgreSQL vector database manager using pgvector extension."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize PostgreSQL connection and embedding model."""
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        # Database connection parameters
        self.conn_params = {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': os.getenv('POSTGRES_PORT', '5433'),
            'dbname': os.getenv('POSTGRES_DB', 'vector_demo'),
            'user': os.getenv('POSTGRES_USER', 'postgres'),
            'password': os.getenv('POSTGRES_PASSWORD', 'postgres')
        }
        
        self.conn = None
        self.cursor = None
        
    def connect(self):
        """Establish database connection."""
        try:
            self.conn = psycopg2.connect(**self.conn_params)
            self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            print("✅ Connected to PostgreSQL with pgvector")
            
            # Register pgvector extension
            from pgvector.psycopg2 import register_vector
            register_vector(self.conn)
            
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            raise
            
    def close(self):
        """Close database connection."""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
            
    def create_table_with_index(self, index_type: str = 'ivfflat'):
        """Create table with specified index type."""
        # Drop existing table
        self.cursor.execute("DROP TABLE IF EXISTS documents CASCADE")
        
        # Create table
        create_table_sql = f"""
        CREATE TABLE documents (
            id SERIAL PRIMARY KEY,
            content TEXT NOT NULL,
            embedding vector({self.dimension}),
            metadata JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        self.cursor.execute(create_table_sql)
        
        # Create index based on type
        if index_type == 'ivfflat':
            # IVFFlat index for approximate search
            index_sql = """
            CREATE INDEX documents_embedding_ivf_idx 
            ON documents USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 10)
            """
        elif index_type == 'hnsw':
            # HNSW index for graph-based search
            index_sql = """
            CREATE INDEX documents_embedding_hnsw_idx 
            ON documents USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
            """
        else:  # exact
            # No index for exact search
            index_sql = None
            
        if index_sql:
            self.cursor.execute(index_sql)
            
        self.conn.commit()
        print(f"✅ Created table with {index_type} index")
        
    def insert_documents(self, documents: List[str], batch_size: int = 100):
        """Insert documents with their embeddings."""
        print(f"Generating embeddings for {len(documents)} documents...")
        embeddings = self.model.encode(documents, batch_size=32, show_progress_bar=True)
        
        print("Inserting documents into PostgreSQL...")
        insert_sql = """
        INSERT INTO documents (content, embedding, metadata)
        VALUES (%s, %s, %s)
        """
        
        # Insert in batches
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_embs = embeddings[i:i + batch_size]
            
            data = []
            for doc, emb in zip(batch_docs, batch_embs):
                metadata = {'length': len(doc), 'category': 'demo'}
                data.append((doc, emb.tolist(), json.dumps(metadata)))
            
            self.cursor.executemany(insert_sql, data)
            self.conn.commit()
            
        print(f"✅ Inserted {len(documents)} documents")
        
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents using cosine similarity."""
        # Generate query embedding
        query_embedding = self.model.encode(query)
        
        # Search query
        search_sql = """
        SELECT 
            id,
            content,
            1 - (embedding <=> %s::vector) AS similarity,
            metadata
        FROM documents
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """
        
        self.cursor.execute(
            search_sql, 
            (query_embedding.tolist(), query_embedding.tolist(), k)
        )
        
        results = self.cursor.fetchall()
        return results
    
    def benchmark_search(self, queries: List[str], k: int = 5) -> Dict[str, float]:
        """Benchmark search performance."""
        times = []
        
        for query in queries:
            start = time.time()
            _ = self.search(query, k)
            times.append(time.time() - start)
            
        return {
            'avg_time': np.mean(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'total_queries': len(queries)
        }


def compare_index_types():
    """Compare different PostgreSQL vector index types."""
    # Generate test data
    np.random.seed(42)
    num_documents = 1000
    
    # Create synthetic documents
    categories = ['tech', 'health', 'finance', 'education', 'travel']
    templates = {
        'tech': ['software', 'hardware', 'programming', 'AI', 'data'],
        'health': ['wellness', 'medicine', 'fitness', 'nutrition', 'mental'],
        'finance': ['investment', 'banking', 'budget', 'savings', 'credit'],
        'education': ['learning', 'teaching', 'courses', 'degree', 'skills'],
        'travel': ['vacation', 'destination', 'flights', 'hotels', 'adventure']
    }
    
    documents = []
    for i in range(num_documents):
        cat = np.random.choice(categories)
        word = np.random.choice(templates[cat])
        documents.append(f"Document about {word} in {cat} category #{i}")
    
    # Test queries
    test_queries = [
        "Looking for AI and machine learning resources",
        "Health and fitness tips",
        "Investment strategies for beginners"
    ]
    
    results = {}
    
    # Test different index types
    for index_type in ['exact', 'ivfflat', 'hnsw']:
        print(f"\n{'='*60}")
        print(f"Testing {index_type.upper()} index")
        print('='*60)
        
        db = PostgresVectorDB()
        db.connect()
        
        # Create table and index
        start = time.time()
        db.create_table_with_index(index_type)
        create_time = time.time() - start
        
        # Insert documents
        start = time.time()
        db.insert_documents(documents)
        insert_time = time.time() - start
        
        # Benchmark search
        benchmark = db.benchmark_search(test_queries * 10)  # Run each query 10 times
        
        # Store results
        results[index_type] = {
            'create_time': create_time,
            'insert_time': insert_time,
            'search_metrics': benchmark
        }
        
        # Show sample search results
        print(f"\nSample search results for: '{test_queries[0]}'")
        results_list = db.search(test_queries[0], k=3)
        for i, result in enumerate(results_list, 1):
            print(f"{i}. {result['content'][:60]}...")
            print(f"   Similarity: {result['similarity']:.3f}")
        
        db.close()
    
    # Display comparison
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    
    for index_type, metrics in results.items():
        print(f"\n{index_type.upper()}:")
        print(f"  Table creation: {metrics['create_time']:.3f}s")
        print(f"  Document insertion: {metrics['insert_time']:.3f}s")
        print(f"  Avg search time: {metrics['search_metrics']['avg_time']*1000:.2f}ms")


def demonstrate_advanced_features():
    """Demonstrate advanced pgvector features."""
    print("\n" + "="*60)
    print("ADVANCED PGVECTOR FEATURES")
    print("="*60)
    
    db = PostgresVectorDB()
    db.connect()
    
    # Create table with HNSW index
    db.create_table_with_index('hnsw')
    
    # Insert sample documents
    documents = [
        "Python is a versatile programming language for data science.",
        "Machine learning models can predict future trends.",
        "Natural language processing enables text understanding.",
        "Deep learning uses neural networks for complex tasks.",
        "Data visualization helps communicate insights effectively."
    ]
    
    db.insert_documents(documents)
    
    print("\n1. Similarity Threshold Search")
    print("-" * 30)
    
    # Search with similarity threshold
    query = "programming and data analysis"
    query_embedding = db.model.encode(query)
    
    threshold_sql = """
    SELECT content, 1 - (embedding <=> %s::vector) AS similarity
    FROM documents
    WHERE 1 - (embedding <=> %s::vector) > %s
    ORDER BY embedding <=> %s::vector
    """
    
    threshold = 0.5
    db.cursor.execute(
        threshold_sql,
        (query_embedding.tolist(), query_embedding.tolist(), 
         threshold, query_embedding.tolist())
    )
    
    results = db.cursor.fetchall()
    print(f"Documents with similarity > {threshold}:")
    for result in results:
        print(f"- {result['content']}")
        print(f"  Similarity: {result['similarity']:.3f}")
    
    print("\n2. Metadata Filtering")
    print("-" * 30)
    
    # Search with metadata filter
    filter_sql = """
    SELECT content, metadata
    FROM documents
    WHERE metadata->>'length' > '50'
    ORDER BY embedding <=> %s::vector
    LIMIT 3
    """
    
    db.cursor.execute(filter_sql, (query_embedding.tolist(),))
    results = db.cursor.fetchall()
    
    print("Documents with length > 50 characters:")
    for result in results:
        print(f"- {result['content']}")
        print(f"  Metadata: {result['metadata']}")
    
    db.close()


def main():
    """Run PostgreSQL vector database demonstrations."""
    print("PostgreSQL Vector Database Demo")
    print("==============================")
    
    # Check if PostgreSQL is accessible
    try:
        db = PostgresVectorDB()
        db.connect()
        db.close()
    except Exception as e:
        print(f"\n❌ Cannot connect to PostgreSQL: {e}")
        print("\nPlease ensure PostgreSQL is running:")
        print("  task postgres-start")
        return
    
    # Run demonstrations
    compare_index_types()
    demonstrate_advanced_features()
    
    print("\n✅ PostgreSQL vector database demo completed!")
    print("\nKey Takeaways:")
    print("- IVFFlat: Good balance of speed and accuracy")
    print("- HNSW: Fastest search, uses more memory")
    print("- Exact: Most accurate, slowest for large datasets")
    print("- pgvector integrates seamlessly with PostgreSQL features")


if __name__ == "__main__":
    main()