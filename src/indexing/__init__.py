"""
Indexing package for the Advanced RAG System.

This package provides vector storage and retrieval capabilities for 
efficient document indexing and semantic search operations.
"""

from .vector_store import VectorStore, SearchResult

__all__ = ['VectorStore', 'SearchResult']
