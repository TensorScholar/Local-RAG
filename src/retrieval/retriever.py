"""
Retrieval module for the Advanced RAG System.

This module provides document retrieval capabilities for finding relevant
documents based on semantic similarity and other criteria.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result of a document retrieval operation."""
    document: Any
    similarity: float
    metadata: Dict[str, Any]


class Retriever:
    """
    Document retriever for semantic search and filtering.
    
    This class provides advanced document retrieval capabilities with
    configurable similarity thresholds and filtering options.
    """
    
    def __init__(self, vector_store):
        """
        Initialize the retriever with a vector store.
        
        Args:
            vector_store: Vector store instance for document storage and search
        """
        self.vector_store = vector_store
        self.default_limit = 5
        self.similarity_threshold = 0.7
    
    def retrieve(self, query: str, limit: int = None, 
                similarity_threshold: float = None) -> List[RetrievalResult]:
        """
        Retrieve documents based on semantic similarity.
        
        Args:
            query: Search query text
            limit: Maximum number of results to return
            similarity_threshold: Minimum similarity score for results
            
        Returns:
            List of retrieval results with documents and similarity scores
        """
        if limit is None:
            limit = self.default_limit
        if similarity_threshold is None:
            similarity_threshold = self.similarity_threshold
        
        try:
            # Use vector store for similarity search
            search_results = self.vector_store.similarity_search(query, limit)
            
            # Filter by similarity threshold and convert to RetrievalResult
            results = []
            for result in search_results:
                if result['similarity'] >= similarity_threshold:
                    results.append(RetrievalResult(
                        document=result['document'],
                        similarity=result['similarity'],
                        metadata=result.get('metadata', {})
                    ))
            
            logger.info(f"Retrieved {len(results)} documents for query: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return []
    
    def retrieve_by_metadata(self, metadata_filters: Dict[str, Any], 
                           limit: int = None) -> List[RetrievalResult]:
        """
        Retrieve documents based on metadata filters.
        
        Args:
            metadata_filters: Dictionary of metadata key-value pairs to filter by
            limit: Maximum number of results to return
            
        Returns:
            List of retrieval results matching the metadata filters
        """
        if limit is None:
            limit = self.default_limit
        
        try:
            # Use vector store with metadata filtering
            search_results = self.vector_store.similarity_search(
                "", limit, filter_metadata=metadata_filters
            )
            
            results = []
            for result in search_results:
                results.append(RetrievalResult(
                    document=result['document'],
                    similarity=result['similarity'],
                    metadata=result.get('metadata', {})
                ))
            
            logger.info(f"Retrieved {len(results)} documents with metadata filters")
            return results
            
        except Exception as e:
            logger.error(f"Error during metadata retrieval: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get retrieval statistics and performance metrics.
        
        Returns:
            Dictionary containing retrieval statistics
        """
        return {
            'total_documents': self.vector_store.count(),
            'default_limit': self.default_limit,
            'similarity_threshold': self.similarity_threshold
        }
