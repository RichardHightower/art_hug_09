-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create a table for storing documents with embeddings
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding vector(384),  -- 384 dimensions for all-MiniLM-L6-v2
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for vector similarity search
-- IVF index for approximate search (similar to FAISS IVFFlat)
CREATE INDEX IF NOT EXISTS documents_embedding_ivf_idx 
ON documents USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Create index for exact search (optional, for comparison)
-- CREATE INDEX IF NOT EXISTS documents_embedding_exact_idx 
-- ON documents USING hnsw (embedding vector_cosine_ops);

-- Function to search similar documents
CREATE OR REPLACE FUNCTION search_similar_documents(
    query_embedding vector(384),
    limit_count INTEGER DEFAULT 5
)
RETURNS TABLE(
    id INTEGER,
    content TEXT,
    similarity FLOAT,
    metadata JSONB
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        d.id,
        d.content,
        1 - (d.embedding <=> query_embedding) AS similarity,
        d.metadata
    FROM documents d
    ORDER BY d.embedding <=> query_embedding
    LIMIT limit_count;
END;
$$ LANGUAGE plpgsql;