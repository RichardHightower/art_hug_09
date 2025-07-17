"""
RAG (Retrieval-Augmented Generation) Integration

This module demonstrates how to combine semantic search with language models
for retrieval-augmented generation. RAG enhances LLM responses by grounding
them in retrieved context, reducing hallucinations and improving accuracy.

Topics covered:
- Basic RAG pipeline implementation
- Semantic retrieval for context
- Prompt engineering for RAG
- Multi-document context handling
- Production considerations
"""

from sentence_transformers import SentenceTransformer, util
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import numpy as np
from typing import List, Dict, Tuple, Optional
import torch
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


class RAGPipeline:
    """Production-ready RAG pipeline combining retrieval and generation."""
    
    def __init__(
        self,
        retriever_model: str = 'all-MiniLM-L6-v2',
        generator_model: str = 'gpt2',  # Use larger models in production
        device: Optional[str] = None
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            retriever_model: Sentence transformer for retrieval
            generator_model: Language model for generation
            device: Device to use (auto-detected if None)
        """
        self.device = device or get_device()
        logger.info(f"Initializing RAG pipeline on device: {self.device}")
        
        # Initialize retriever
        self.retriever = SentenceTransformer(retriever_model, device=self.device)
        
        # Initialize generator
        self.tokenizer = AutoTokenizer.from_pretrained(generator_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.generator = pipeline(
            'text-generation',
            model=generator_model,
            device=0 if self.device == 'cuda' else -1,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Document storage
        self.documents: List[str] = []
        self.doc_embeddings = None
        
    def index_documents(self, documents: List[str]) -> None:
        """
        Index documents for retrieval.
        
        Args:
            documents: List of document texts
        """
        self.documents = documents
        logger.info(f"Indexing {len(documents)} documents...")
        
        self.doc_embeddings = self.retriever.encode(
            documents,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        logger.info("Document indexing complete")
        
    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = 0.3
    ) -> List[Tuple[str, float]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve
            min_score: Minimum similarity score threshold
            
        Returns:
            List of (document, score) tuples
        """
        if not self.documents:
            raise ValueError("No documents indexed. Call index_documents first.")
        
        # Encode query
        query_embedding = self.retriever.encode(query)
        
        # Calculate similarities
        scores = util.cos_sim(query_embedding, self.doc_embeddings)[0].numpy()
        
        # Get top results above threshold
        top_indices = np.argsort(-scores)[:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] >= min_score:
                results.append((self.documents[idx], float(scores[idx])))
        
        return results
        
    def generate(
        self,
        query: str,
        context: List[str],
        max_length: int = 150,
        temperature: float = 0.7
    ) -> str:
        """
        Generate response using retrieved context.
        
        Args:
            query: User question
            context: Retrieved documents
            max_length: Maximum generation length
            temperature: Generation temperature
            
        Returns:
            Generated response
        """
        # Construct prompt with context
        context_text = "\n\n".join(context)
        prompt = f"""Context information:
{context_text}

Based on the above context, please answer the following question:
Question: {query}
Answer:"""
        
        # Generate response
        response = self.generator(
            prompt,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            num_return_sequences=1
        )[0]['generated_text']
        
        # Extract only the answer part
        answer_start = response.find("Answer:") + len("Answer:")
        answer = response[answer_start:].strip()
        
        return answer
        
    def query(
        self,
        question: str,
        top_k: int = 3,
        max_length: int = 150,
        include_sources: bool = True
    ) -> Dict[str, any]:
        """
        End-to-end RAG query.
        
        Args:
            question: User question
            top_k: Number of documents to retrieve
            max_length: Maximum generation length
            include_sources: Whether to include source documents
            
        Returns:
            Dictionary with answer and optional sources
        """
        # Retrieve relevant documents
        retrieved = self.retrieve(question, top_k=top_k)
        
        if not retrieved:
            return {
                'answer': "I couldn't find relevant information to answer your question.",
                'sources': []
            }
        
        # Extract documents for context
        context_docs = [doc for doc, _ in retrieved]
        
        # Generate answer
        answer = self.generate(question, context_docs, max_length=max_length)
        
        result = {'answer': answer}
        
        if include_sources:
            result['sources'] = [
                {'document': doc, 'score': score}
                for doc, score in retrieved
            ]
        
        return result


def run_rag_examples():
    """Demonstrate RAG with various examples."""
    
    print_section("RAG Integration Examples")
    
    # Example 1: Basic Q&A System
    print("\n=== Example 1: Basic Q&A System ===")
    
    # Knowledge base
    knowledge_base = [
        "Our refund policy allows returns within 30 days of purchase. To initiate a refund, contact customer support with your order number.",
        "Account recovery can be done through the 'Forgot Password' link on the login page. You'll receive an email with reset instructions.",
        "Two-factor authentication adds an extra layer of security. Enable it in your account settings under the Security tab.",
        "Our customer support is available 24/7 via email at support@example.com or through the live chat on our website.",
        "System requirements include: Windows 10 or later, 8GB RAM minimum, 2GB free disk space, and an internet connection.",
        "Payment methods accepted include Visa, MasterCard, PayPal, and bank transfers. All transactions are secured with SSL encryption.",
        "Data privacy is our priority. We comply with GDPR and never share your personal information with third parties.",
        "Premium features include unlimited storage, priority support, advanced analytics, and API access."
    ]
    
    # Initialize RAG pipeline
    rag = RAGPipeline()
    rag.index_documents(knowledge_base)
    
    # Test questions
    questions = [
        "How do I get a refund?",
        "What security features do you offer?",
        "Can I pay with PayPal?"
    ]
    
    for question in questions:
        print(f"\nQuestion: {question}")
        result = rag.query(question, top_k=2)
        
        print(f"Answer: {result['answer']}")
        
        if result['sources']:
            print("\nSources used:")
            for i, source in enumerate(result['sources'], 1):
                print(f"  {i}. (Score: {source['score']:.3f}) {source['document'][:100]}...")
    
    # Example 2: Multi-Document RAG
    print("\n\n=== Example 2: Multi-Document Technical RAG ===")
    
    technical_docs = [
        "Python decorators are functions that modify other functions. They use the @decorator syntax and are commonly used for logging, timing, and authentication.",
        "REST APIs use HTTP methods: GET for reading, POST for creating, PUT for updating, and DELETE for removing resources. They return data in JSON format.",
        "Docker containers package applications with their dependencies. Use 'docker build' to create images and 'docker run' to start containers.",
        "Git branching strategies include GitFlow and GitHub Flow. Feature branches are created from main, and changes are merged via pull requests.",
        "Unit tests verify individual components work correctly. Use pytest in Python with fixtures for setup and teardown of test data.",
        "Microservices architecture splits applications into small, independent services. Each service has its own database and communicates via APIs.",
        "SQL JOIN operations combine data from multiple tables. INNER JOIN returns matching records, while LEFT JOIN includes all records from the left table.",
        "Machine learning models require training data, feature engineering, and hyperparameter tuning. Use cross-validation to prevent overfitting."
    ]
    
    tech_rag = RAGPipeline()
    tech_rag.index_documents(technical_docs)
    
    tech_questions = [
        "How do I write tests in Python?",
        "Explain Docker and its benefits",
        "What are the different types of SQL joins?"
    ]
    
    for question in tech_questions:
        print(f"\nQuestion: {question}")
        result = tech_rag.query(question, top_k=3, max_length=200)
        print(f"Answer: {result['answer'][:300]}...")  # Truncate long answers
        print(f"Sources used: {len(result['sources'])}")
    
    # Example 3: Advanced RAG with Filtering
    print("\n\n=== Example 3: Advanced RAG with Context Filtering ===")
    
    class AdvancedRAGPipeline(RAGPipeline):
        """Extended RAG with metadata filtering and reranking."""
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.metadata = []
            
        def index_documents_with_metadata(
            self,
            documents: List[str],
            metadata: List[Dict]
        ) -> None:
            """Index documents with associated metadata."""
            self.index_documents(documents)
            self.metadata = metadata
            
        def retrieve_with_filter(
            self,
            query: str,
            category_filter: Optional[str] = None,
            top_k: int = 3
        ) -> List[Tuple[str, float, Dict]]:
            """Retrieve with optional metadata filtering."""
            
            # Get all similarities
            query_embedding = self.retriever.encode(query)
            scores = util.cos_sim(query_embedding, self.doc_embeddings)[0].numpy()
            
            # Apply category filter if specified
            filtered_indices = []
            for i, meta in enumerate(self.metadata):
                if category_filter is None or meta.get('category') == category_filter:
                    filtered_indices.append(i)
            
            # Get top results from filtered set
            filtered_scores = [(i, scores[i]) for i in filtered_indices]
            filtered_scores.sort(key=lambda x: x[1], reverse=True)
            
            results = []
            for i, (idx, score) in enumerate(filtered_scores[:top_k]):
                results.append((
                    self.documents[idx],
                    float(score),
                    self.metadata[idx]
                ))
            
            return results
    
    # Create categorized knowledge base
    categorized_docs = [
        ("Refunds are processed within 5-7 business days.", {"category": "billing"}),
        ("Premium plan costs $19.99/month.", {"category": "billing"}),
        ("Enable 2FA in security settings.", {"category": "security"}),
        ("Passwords must be at least 8 characters.", {"category": "security"}),
        ("API rate limit is 1000 requests/hour.", {"category": "technical"}),
        ("Webhooks support POST requests only.", {"category": "technical"})
    ]
    
    advanced_rag = AdvancedRAGPipeline()
    docs, metadata = zip(*categorized_docs)
    advanced_rag.index_documents_with_metadata(list(docs), list(metadata))
    
    # Test filtered retrieval
    print("\nFiltered retrieval examples:")
    
    test_cases = [
        ("How much does it cost?", "billing"),
        ("Security best practices?", "security"),
        ("API limitations?", "technical")
    ]
    
    for query, category in test_cases:
        print(f"\nQuery: '{query}' (Category: {category})")
        results = advanced_rag.retrieve_with_filter(query, category_filter=category, top_k=2)
        
        for doc, score, meta in results:
            print(f"  - {doc} (Score: {score:.3f}, Category: {meta['category']})")


def benchmark_rag_performance():
    """Benchmark RAG retrieval and generation performance."""
    
    print_section("RAG Performance Benchmark")
    
    import time
    
    # Create test dataset
    num_docs = 1000
    documents = [
        f"Document {i}: This is test content about {['technology', 'science', 'business'][i % 3]} topic number {i}."
        for i in range(num_docs)
    ]
    
    rag = RAGPipeline()
    
    # Benchmark indexing
    start_time = time.time()
    rag.index_documents(documents)
    index_time = time.time() - start_time
    
    print(f"\nIndexing {num_docs} documents took: {index_time:.2f} seconds")
    print(f"Average time per document: {index_time/num_docs*1000:.2f} ms")
    
    # Benchmark retrieval
    test_queries = [
        "technology advancements",
        "scientific discoveries",
        "business strategies"
    ]
    
    print("\nRetrieval benchmark:")
    for query in test_queries:
        start_time = time.time()
        results = rag.retrieve(query, top_k=5)
        retrieval_time = time.time() - start_time
        
        print(f"  Query: '{query}' - Time: {retrieval_time*1000:.2f} ms, Results: {len(results)}")
    
    # Benchmark end-to-end
    print("\nEnd-to-end RAG benchmark:")
    start_time = time.time()
    result = rag.query("Tell me about technology", top_k=3, max_length=100)
    total_time = time.time() - start_time
    
    print(f"  Total time: {total_time:.2f} seconds")
    print(f"  Answer length: {len(result['answer'])} characters")


if __name__ == "__main__":
    run_rag_examples()
    print("\n" + "="*80 + "\n")
    benchmark_rag_performance()