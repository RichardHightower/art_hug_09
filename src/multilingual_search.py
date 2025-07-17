"""Multilingual semantic search implementation."""

from sentence_transformers import SentenceTransformer, util
import numpy as np
from typing import List, Dict, Tuple
import time
from config import get_device


class MultilingualSearchEngine:
    """Multilingual semantic search supporting 50+ languages."""
    
    def __init__(self):
        """Initialize with multilingual model."""
        print("Initializing multilingual search engine...")
        print(f"Device: {get_device()}")
        
        # Use multilingual model
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.documents = []
        self.languages = []
        self.embeddings = None
        
    def add_documents(self, documents: List[str], languages: List[str]) -> None:
        """Add documents with their language tags."""
        print(f"\nAdding {len(documents)} documents in multiple languages...")
        
        self.documents.extend(documents)
        self.languages.extend(languages)
        
        # Generate embeddings
        new_embeddings = self.model.encode(documents, convert_to_numpy=True)
        
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
            
        print(f"Total documents: {len(self.documents)}")
        
    def cross_lingual_search(self, query: str, k: int = 5) -> List[Dict]:
        """Search across all languages."""
        if self.embeddings is None:
            raise ValueError("No documents indexed.")
            
        # Encode query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        
        # Calculate similarities
        similarities = util.cos_sim(query_embedding, self.embeddings)[0].numpy()
        
        # Get top-k results
        top_indices = np.argsort(-similarities)[:k]
        
        results = []
        for idx in top_indices:
            results.append({
                'document': self.documents[idx],
                'language': self.languages[idx],
                'similarity': float(similarities[idx])
            })
            
        return results
    
    def benchmark_languages(self, test_phrase: str, translations: Dict[str, str]) -> None:
        """Benchmark cross-lingual understanding."""
        print(f"\nBenchmarking cross-lingual search...")
        print(f"Test phrase: '{test_phrase}'")
        
        # Encode all translations
        languages = list(translations.keys())
        phrases = list(translations.values())
        
        embeddings = self.model.encode(phrases, convert_to_numpy=True)
        
        # Calculate similarity matrix
        similarity_matrix = util.cos_sim(embeddings, embeddings).numpy()
        
        print("\nCross-lingual similarity matrix:")
        print(f"{'':10}", end='')
        for lang in languages[:5]:  # Show first 5 for readability
            print(f"{lang:>10}", end='')
        print()
        
        for i, lang1 in enumerate(languages[:5]):
            print(f"{lang1:10}", end='')
            for j in range(min(5, len(languages))):
                print(f"{similarity_matrix[i, j]:10.3f}", end='')
            print()


def run_multilingual_search_examples():
    """Run multilingual search examples."""
    print("=== Multilingual Semantic Search Examples ===\n")
    
    # Initialize search engine
    search_engine = MultilingualSearchEngine()
    
    # Example 1: Basic multilingual search
    print("1. Cross-lingual FAQ Search")
    print("-" * 40)
    
    # Multilingual FAQ documents
    multilingual_docs = [
        # English
        ("How do I reset my password?", "en"),
        ("What is your refund policy?", "en"),
        ("How to contact customer support?", "en"),
        
        # Spanish
        ("¿Cómo puedo restablecer mi contraseña?", "es"),
        ("¿Cuál es su política de reembolso?", "es"),
        ("¿Cómo contactar con atención al cliente?", "es"),
        
        # French
        ("Comment réinitialiser mon mot de passe?", "fr"),
        ("Quelle est votre politique de remboursement?", "fr"),
        ("Comment contacter le support client?", "fr"),
        
        # German
        ("Wie kann ich mein Passwort zurücksetzen?", "de"),
        ("Was ist Ihre Rückerstattungsrichtlinie?", "de"),
        ("Wie kontaktiere ich den Kundensupport?", "de"),
        
        # Chinese
        ("如何重置我的密码？", "zh"),
        ("你们的退款政策是什么？", "zh"),
        ("如何联系客户支持？", "zh"),
    ]
    
    # Add documents
    docs, langs = zip(*multilingual_docs)
    search_engine.add_documents(list(docs), list(langs))
    
    # Test queries in different languages
    test_queries = [
        ("I forgot my password", "en"),
        ("J'ai oublié mon mot de passe", "fr"),
        ("Olvidé mi contraseña", "es"),
        ("Ich habe mein Passwort vergessen", "de"),
        ("我忘记了密码", "zh")
    ]
    
    print("\nCross-lingual search results:")
    for query, query_lang in test_queries:
        print(f"\nQuery: '{query}' (Language: {query_lang})")
        results = search_engine.cross_lingual_search(query, k=3)
        
        for i, result in enumerate(results, 1):
            print(f"  {i}. [{result['language']}] {result['document']}")
            print(f"     Similarity: {result['similarity']:.3f}")
    
    # Example 2: Language similarity analysis
    print("\n\n2. Cross-lingual Similarity Analysis")
    print("-" * 40)
    
    # Same phrase in multiple languages
    test_phrase = "How can I help you?"
    translations = {
        "English": "How can I help you?",
        "Spanish": "¿Cómo puedo ayudarte?",
        "French": "Comment puis-je vous aider?",
        "German": "Wie kann ich Ihnen helfen?",
        "Italian": "Come posso aiutarti?",
        "Portuguese": "Como posso ajudá-lo?",
        "Dutch": "Hoe kan ik u helpen?",
        "Russian": "Как я могу вам помочь?",
        "Japanese": "どうすればお手伝いできますか？",
        "Chinese": "我能帮你什么？"
    }
    
    search_engine.benchmark_languages(test_phrase, translations)
    
    # Example 3: Performance comparison
    print("\n\n3. Multilingual vs Monolingual Performance")
    print("-" * 40)
    
    # Create monolingual engine for comparison
    mono_engine = MultilingualSearchEngine()
    mono_engine.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # English-only documents
    english_docs = [doc for doc, lang in multilingual_docs if lang == "en"]
    
    # Benchmark encoding speed
    start = time.time()
    multilingual_embeddings = search_engine.model.encode(english_docs)
    multi_time = time.time() - start
    
    start = time.time()
    monolingual_embeddings = mono_engine.model.encode(english_docs)
    mono_time = time.time() - start
    
    print(f"\nEncoding {len(english_docs)} English documents:")
    print(f"  Multilingual model: {multi_time:.3f}s")
    print(f"  Monolingual model: {mono_time:.3f}s")
    print(f"  Multilingual overhead: {(multi_time/mono_time - 1)*100:.1f}%")
    
    print("\nMultilingual search examples completed!")


if __name__ == "__main__":
    run_multilingual_search_examples()