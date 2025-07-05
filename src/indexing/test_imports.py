"""Test script to verify correct module imports."""

try:
    print("Testing imports...")
    from src.retrieval.retriever import Retriever
    from src.indexing.vector_store import VectorStore
    from src.embeddings.embedder import Embedder
    print("✓ All imports successful!")
except ImportError as e:
    print(f"✗ Import error: {e}")
    import sys
    print(f"Python path: {sys.path}")
