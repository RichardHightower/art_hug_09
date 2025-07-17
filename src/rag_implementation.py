"""RAG (Retrieval-Augmented Generation) implementation."""

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List, Dict, Any
from config import get_device, DEFAULT_MODEL, HF_TOKEN
import torch


class SimpleRAGSystem:
    """Simple RAG system combining semantic search with text generation."""
    
    def __init__(
        self, 
        embedding_model: str = 'all-MiniLM-L6-v2',
        generation_model: str = 'google/flan-t5-base'
    ):
        """Initialize RAG system with embedding and generation models."""
        print("Initializing RAG system...")
        self.device = get_device()
        
        # Initialize embedding model
        print(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize generation model
        print(f"Loading generation model: {generation_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(generation_model)
        self.generation_model = AutoModelForSeq2SeqLM.from_pretrained(generation_model)
        
        # Move to appropriate device
        if self.device == "cuda":
            self.generation_model = self.generation_model.cuda()
        elif self.device == "mps":
            # MPS support for generation
            self.generation_model = self.generation_model.to("mps")
            
        # Initialize FAISS index
        self.index = None
        self.documents = []
        
    def index_knowledge_base(self, documents: List[str]) -> None:
        """Index documents for retrieval."""
        print(f"\nIndexing {len(documents)} documents for RAG...")
        
        # Store documents
        self.documents = documents
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(
            documents,
            show_progress_bar=True,
            convert_to_numpy=True
        ).astype('float32')
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        
        print(f"Knowledge base indexed with {self.index.ntotal} documents")
        
    def retrieve(self, query: str, k: int = 3) -> List[str]:
        """Retrieve top-k relevant documents for query."""
        if self.index is None:
            raise ValueError("No documents indexed. Call index_knowledge_base() first.")
            
        # Encode query
        query_embedding = self.embedding_model.encode(
            [query], 
            convert_to_numpy=True
        ).astype('float32')
        
        # Search
        distances, indices = self.index.search(query_embedding, k)
        
        # Get documents
        retrieved_docs = [self.documents[idx] for idx in indices[0]]
        return retrieved_docs
    
    def generate_answer(self, query: str, context: List[str]) -> str:
        """Generate answer using retrieved context."""
        # Combine context
        context_text = " ".join(context)
        
        # Create prompt
        prompt = f"Context: {context_text}\n\nQuestion: {query}\n\nAnswer:"
        
        # Tokenize
        inputs = self.tokenizer(
            prompt, 
            max_length=512, 
            truncation=True, 
            return_tensors="pt"
        )
        
        # Move to device
        if self.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        elif self.device == "mps":
            inputs = {k: v.to("mps") for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.generation_model.generate(
                **inputs,
                max_length=150,
                num_beams=4,
                early_stopping=True,
                temperature=0.7
            )
        
        # Decode
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer
    
    def query(self, question: str, k: int = 3) -> Dict[str, Any]:
        """End-to-end RAG query processing."""
        # Retrieve relevant documents
        retrieved_docs = self.retrieve(question, k=k)
        
        # Generate answer
        answer = self.generate_answer(question, retrieved_docs)
        
        return {
            "question": question,
            "answer": answer,
            "retrieved_documents": retrieved_docs
        }


def run_rag_examples():
    """Run RAG implementation examples."""
    print("=== RAG (Retrieval-Augmented Generation) Examples ===\n")
    
    # Initialize RAG system
    rag_system = SimpleRAGSystem()
    
    # Example knowledge base
    knowledge_base = [
        "The transformer architecture was introduced in the paper 'Attention is All You Need' by Vaswani et al. in 2017.",
        "BERT (Bidirectional Encoder Representations from Transformers) was developed by Google in 2018.",
        "GPT (Generative Pre-trained Transformer) was created by OpenAI, with GPT-3 released in 2020.",
        "Transformers use self-attention mechanisms to process sequences in parallel, unlike RNNs.",
        "Fine-tuning is the process of adapting a pre-trained model to a specific task with task-specific data.",
        "The attention mechanism allows models to focus on different parts of the input when generating output.",
        "Hugging Face provides the Transformers library, making it easy to use pre-trained models.",
        "Semantic search uses embeddings to find documents based on meaning rather than keywords.",
        "FAISS is a library for efficient similarity search and clustering of dense vectors.",
        "RAG combines retrieval and generation for more accurate and grounded responses."
    ]
    
    # Index knowledge base
    rag_system.index_knowledge_base(knowledge_base)
    
    # Example queries
    test_queries = [
        "What is the transformer architecture?",
        "Who created BERT and when?",
        "How does semantic search work?",
        "What is the difference between transformers and RNNs?"
    ]
    
    print("\nRAG Question Answering:")
    print("=" * 60)
    
    for query in test_queries:
        result = rag_system.query(query, k=2)
        
        print(f"\nQuestion: {result['question']}")
        print(f"\nRetrieved Context:")
        for i, doc in enumerate(result['retrieved_documents'], 1):
            print(f"  {i}. {doc}")
        print(f"\nGenerated Answer: {result['answer']}")
        print("-" * 60)
    
    # Example: Comparison with direct generation (no retrieval)
    print("\n\nComparison: RAG vs Direct Generation")
    print("=" * 60)
    
    complex_query = "What are the key innovations in the transformer architecture?"
    
    # RAG answer
    rag_result = rag_system.query(complex_query)
    print(f"Question: {complex_query}")
    print(f"\nRAG Answer (with retrieval): {rag_result['answer']}")
    
    # Direct answer (without retrieval)
    direct_answer = rag_system.generate_answer(complex_query, [""])
    print(f"\nDirect Answer (no retrieval): {direct_answer}")
    
    print("\nNotice how RAG provides more grounded, accurate answers!")
    
    print("\nRAG examples completed!")


if __name__ == "__main__":
    run_rag_examples()