# Semantic Search and Information Retrieval with Transformers

This project contains working examples for Chapter 09 of the Hugging Face Transformers book, with enhanced examples demonstrating cutting-edge semantic search techniques.

## Overview

Learn how to implement and understand:
- **Embedding Generation**: Create semantic embeddings with multiple models (local and API-based)
- **Hybrid Search**: Combine keyword (BM25) and semantic search for optimal results
- **Vector Databases**: Implement scalable search with FAISS and Chroma
- **RAG Systems**: Build Retrieval-Augmented Generation pipelines
- **Production Patterns**: Real-world deployment strategies and optimizations
- **Model Optimization**: Quantization techniques for efficient deployment

## Prerequisites

- Python 3.12 (managed via pyenv)
- Poetry for dependency management
- Go Task for build automation
- API keys for any required services (see .env.example)

## Setup

1. Clone this repository
2. Run the setup task:
   ```bash
   task setup
   ```
3. Copy `.env.example` to `.env` and configure as needed

## Project Structure

```
.
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration and utilities
│   ├── main.py                # Entry point with all examples
│   ├── embedding_generation.py # Generate embeddings with various models
│   ├── hybrid_search.py       # Hybrid keyword + semantic search
│   ├── vector_db_manager.py   # FAISS and Chroma database management
│   ├── rag_integration.py     # RAG pipeline implementation
│   ├── quantization.py        # Model quantization examples
│   └── utils.py               # Utility functions
├── tests/
│   └── test_examples.py       # Unit tests
├── docs/
│   ├── article9.md            # Original article
│   └── article9i.md           # Enhanced article with additional examples
├── .env.example               # Environment template
├── Taskfile.yml               # Task automation
└── pyproject.toml             # Poetry configuration
```

## Running Examples

Run all examples:
```bash
task run
```

Or run individual modules:
```bash
task run-embeddings       # Generate embeddings with various models
task run-hybrid          # Run hybrid search examples
task run-vector-db       # Vector database management
task run-rag            # Run RAG implementation
task run-quantization   # Run quantization examples
```

Interactive mode:
```bash
python src/main.py      # Choose examples interactively
```

## Interactive Jupyter Notebook Tutorial

For a hands-on learning experience with visualizations and interactive examples:
```bash
task notebook           # Launch in Jupyter Notebook
# or
task notebook-lab       # Launch in JupyterLab
```

The notebook includes:
- Interactive comparisons of keyword vs. semantic search
- Visualizations of embeddings and similarity matrices
- Real-time performance benchmarking
- Hands-on exercises to build your own search systems
- Step-by-step explanations with expected results

## Available Tasks

- `task setup` - Set up Python environment and install dependencies
- `task run` - Run all examples
- `task test` - Run unit tests
- `task format` - Format code with Black and Ruff
- `task clean` - Clean up generated files

## Key Features

### 1. Embedding Generation (`embedding_generation.py`)
- Compare multiple embedding models (all-MiniLM-L6-v2, all-mpnet-base-v2, E5, multilingual)
- Benchmark performance and quality
- Support for both local models and API-based embeddings (OpenAI)
- Batch processing and storage optimization

### 2. Hybrid Search (`hybrid_search.py`)
- Combine BM25 keyword search with semantic embeddings
- Configurable weighting between keyword and semantic components
- Production-ready search engine implementation
- Performance benchmarking against individual approaches

### 3. Vector Database Management (`vector_db_manager.py`)
- Unified interface for FAISS and Chroma
- Support for exact and approximate search indices
- Metadata filtering and advanced querying
- Easy switching between local and managed deployments

### 4. RAG Integration (`rag_integration.py`)
- Complete RAG pipeline from retrieval to generation
- Multi-document context handling
- Advanced filtering and reranking
- Performance benchmarking

## Learn More

- [Hugging Face Documentation](https://huggingface.co/docs)
- [Sentence Transformers](https://www.sbert.net/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Article Series by Rick Hightower](https://cloudurable.com/blog/index.html)
