"""Main entry point for all examples."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from quantization import run_quantization_examples
# from pruning import run_pruning_examples
# from onnx_export import run_onnx_export_examples
# from performance_benchmarking import run_performance_benchmarking_examples
# from semantic_search import run_semantic_search_examples
# from rag_implementation import run_rag_examples
# from multilingual_search import run_multilingual_search_examples
from hybrid_search import run_hybrid_search_examples
from rag_integration import run_rag_examples
from vector_db_manager import run_vector_db_examples
from embedding_generation import run_embedding_generation_examples

def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")

def main():
    """Run all examples."""
    print_section("CHAPTER 09: SEMANTIC SEARCH AND INFORMATION RETRIEVAL")
    print("Welcome! This script demonstrates the concepts from this chapter.")
    print("Each example builds on the previous concepts.\n")
    
    # Check for command line argument for non-interactive mode
    import sys
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        print("Choose which examples to run:")
        print("1. Embedding Generation")
        print("2. Hybrid Search (Keyword + Semantic)")
        print("3. Vector Database Management")
        print("4. RAG Implementation")
        print("5. Model Optimization (Quantization)")
        print("6. Run All Semantic Search Examples")
        
        try:
            choice = input("\nEnter your choice (1-6): ")
        except EOFError:
            # Non-interactive environment, run option 2 as demo
            print("\nRunning in non-interactive mode. Running Hybrid Search demo...")
            choice = "2"
    
    if choice == "1" or choice == "6":
        print_section("1. EMBEDDING GENERATION")
        run_embedding_generation_examples()
    
    if choice == "2" or choice == "6":
        print_section("2. HYBRID SEARCH")
        run_hybrid_search_examples()
    
    if choice == "3" or choice == "6":
        print_section("3. VECTOR DATABASE MANAGEMENT")
        run_vector_db_examples()
    
    if choice == "4" or choice == "6":
        print_section("4. RAG IMPLEMENTATION")
        run_rag_examples()
    
    if choice == "5" or choice == "6":
        print_section("5. QUANTIZATION")
        run_quantization_examples()
    
    print_section("CONCLUSION")
    print("These examples demonstrate the key concepts from this chapter.")
    print("Try modifying the code to experiment with different approaches!")

if __name__ == "__main__":
    main()
