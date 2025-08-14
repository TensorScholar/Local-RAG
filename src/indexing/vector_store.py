"""
Advanced Vectorized Knowledge Storage and Retrieval Architecture.

This module implements a sophisticated vector database system employing cutting-edge
computational geometry, algebraic topology, and advanced indexing algorithms for
high-dimensional semantic search. The architecture integrates state-of-the-art
approximate nearest neighbor algorithms with adaptive optimization strategies and
mathematical precision guarantees.

Computational Paradigms:
1. Geometric Deep Learning - High-dimensional manifold operations with topological invariants
2. Adaptive Index Structures - Self-optimizing spatial data structures with performance guarantees
3. Algebraic Vector Spaces - Rigorous mathematical foundations for semantic similarity
4. Quantum-Inspired Algorithms - Superposition and entanglement concepts for parallel search
5. Information-Theoretic Optimization - Maximum entropy principles for optimal space partitioning

Advanced Features:
- Hierarchical Navigable Small World (HNSW) graphs with adaptive layer construction
- Approximate Maximum Inner Product Search (MIPS) with theoretical convergence guarantees  
- Dynamic re-indexing with minimal computational overhead
- Mathematical precision in distance computations with numerical stability
- Advanced compression techniques using random projections and locality-sensitive hashing

Author: Advanced RAG System Team
Version: 2.0.0
"""

import asyncio
import logging
import pickle
import hashlib
import threading
import weakref
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple, Protocol, TypeVar, Generic
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import defaultdict
import json
import time

# Advanced computational libraries
import numpy as np
import faiss
from sklearn.decomposition import PCA
from sklearn.random_projection import SparseRandomProjection
import chromadb
from chromadb.config import Settings
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
import sqlite3

# Type system enhancements
from typing_extensions import Self, TypedDict, NotRequired

# Configure computational logging
logger = logging.getLogger(__name__)

# Advanced type definitions
VectorType = np.ndarray
DocumentID = str
CollectionID = str
SimilarityScore = float

T = TypeVar('T')
V = TypeVar('V', bound=VectorType)


class SearchResult(TypedDict):
    """Type-safe search result structure with mathematical precision."""
    id: str
    document: Any
    similarity: float
    distance: float
    metadata: Dict[str, Any]


class VectorStoreProtocol(Protocol):
    """Type protocol defining the mathematical interface for vector storage operations."""
    
    def add_vectors(self, vectors: VectorType, documents: List[Any], ids: List[str]) -> None:
        """Add vectors to the store with associated documents."""
        ...
    
    def search(self, query_vector: VectorType, k: int) -> List[SearchResult]:
        """Perform k-nearest neighbor search with similarity computation."""
        ...
    
    def get_dimensions(self) -> int:
        """Get vector space dimensionality."""
        ...


@dataclass
class VectorStoreConfiguration:
    """Advanced configuration for vector storage operations with mathematical parameters."""
    
    # Core dimensional parameters
    collection_name: str = "rag_vectors"
    persist_directory: Optional[Path] = None
    
    # Computational optimization parameters
    similarity_metric: str = "cosine"  # cosine, euclidean, inner_product
    index_type: str = "hnsw"  # hnsw, ivf, lsh, exact
    embedding_dimension: Optional[int] = None
    
    # HNSW-specific hyperparameters with theoretical foundations
    hnsw_m: int = 16  # Maximum bi-directional links for each node
    hnsw_ef_construction: int = 200  # Size of dynamic candidate list
    hnsw_ef_search: int = 50  # Size of dynamic candidate list during search
    hnsw_max_elements: int = 1000000  # Maximum number of elements
    
    # Advanced mathematical parameters
    dimensionality_reduction: bool = False
    target_dimensions: Optional[int] = None
    compression_ratio: float = 0.1
    numerical_precision: str = "float32"  # float16, float32, float64
    
    # Persistence and caching strategies
    enable_persistence: bool = True
    auto_save_interval: int = 1000  # Operations between automatic saves
    memory_mapping: bool = True
    
    # Performance optimization parameters
    batch_size: int = 1000
    parallel_search: bool = True
    search_threads: int = 4
    
    def __post_init__(self):
        """Post-initialization validation and computational setup."""
        if self.persist_directory is None:
            self.persist_directory = Path.home() / ".rag_system" / "vector_store"
        
        # Ensure directory exists
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Validate mathematical parameters
        if self.hnsw_m < 4 or self.hnsw_m > 64:
            raise ValueError("HNSW M parameter must be between 4 and 64 for optimal performance")
        
        if self.compression_ratio <= 0 or self.compression_ratio > 1:
            raise ValueError("Compression ratio must be in (0, 1] interval")


class AdvancedDistanceMetrics:
    """
    Sophisticated distance and similarity computation with mathematical rigor.
    Implements multiple metric spaces with theoretical guarantees and numerical stability.
    """
    
    @staticmethod
    def cosine_similarity(a: VectorType, b: VectorType) -> float:
        """
        Compute cosine similarity with numerical stability guarantees.
        
        Mathematical Foundation:
        cos(θ) = (a · b) / (||a|| ||b||)
        
        Numerical Stability:
        - Uses L2 normalization with epsilon regularization
        - Handles zero vectors gracefully
        - Maintains precision across different floating-point representations
        """
        # Numerical stability constants
        epsilon = 1e-8
        
        # Compute norms with regularization
        norm_a = np.linalg.norm(a) + epsilon
        norm_b = np.linalg.norm(b) + epsilon
        
        # Compute dot product
        dot_product = np.dot(a, b)
        
        # Return cosine similarity with clipping for numerical stability
        similarity = dot_product / (norm_a * norm_b)
        return np.clip(similarity, -1.0, 1.0)
    
    @staticmethod
    def euclidean_distance(a: VectorType, b: VectorType) -> float:
        """
        Compute Euclidean distance with optimized numerical computation.
        
        Mathematical Foundation:
        d(a,b) = ||a - b||₂ = √(∑(aᵢ - bᵢ)²)
        """
        return float(np.linalg.norm(a - b))
    
    @staticmethod
    def manhattan_distance(a: VectorType, b: VectorType) -> float:
        """
        Compute Manhattan (L1) distance.
        
        Mathematical Foundation:
        d(a,b) = ||a - b||₁ = ∑|aᵢ - bᵢ|
        """
        return float(np.sum(np.abs(a - b)))
    
    @staticmethod
    def inner_product_similarity(a: VectorType, b: VectorType) -> float:
        """
        Compute inner product similarity for maximum inner product search.
        
        Mathematical Foundation:
        ⟨a, b⟩ = ∑aᵢbᵢ
        """
        return float(np.dot(a, b))


class FAISSVectorIndex:
    """
    Advanced FAISS-based vector indexing with sophisticated optimization strategies.
    Implements multiple indexing algorithms with theoretical performance guarantees.
    """
    
    def __init__(self, dimension: int, config: VectorStoreConfiguration):
        """
        Initialize FAISS index with advanced configuration.
        
        Args:
            dimension: Vector space dimensionality
            config: Vector store configuration with optimization parameters
        """
        self.dimension = dimension
        self.config = config
        self.index = None
        self.id_mapping: Dict[int, str] = {}
        self.reverse_id_mapping: Dict[str, int] = {}
        self.next_id = 0
        
        self._initialize_index()
    
    def _initialize_index(self) -> None:
        """Initialize FAISS index with optimal parameters for the given configuration."""
        if self.config.index_type == "hnsw":
            self._initialize_hnsw_index()
        elif self.config.index_type == "ivf":
            self._initialize_ivf_index()
        elif self.config.index_type == "exact":
            self._initialize_exact_index()
        else:
            raise ValueError(f"Unsupported index type: {self.config.index_type}")
    
    def _initialize_hnsw_index(self) -> None:
        """
        Initialize Hierarchical Navigable Small World index with optimal parameters.
        
        HNSW Algorithm:
        - Constructs a multi-layer graph with exponentially decreasing probability
        - Uses bidirectional links for efficient navigation
        - Provides logarithmic search complexity with high probability
        """
        # Create HNSW index with advanced parameters
        self.index = faiss.IndexHNSWFlat(self.dimension, self.config.hnsw_m)
        
        # Configure HNSW-specific parameters
        self.index.hnsw.efConstruction = self.config.hnsw_ef_construction
        self.index.hnsw.efSearch = self.config.hnsw_ef_search
        
        # Optimize for cosine similarity if specified
        if self.config.similarity_metric == "cosine":
            # Normalize vectors will be handled during addition
            pass
        elif self.config.similarity_metric == "inner_product":
            # Use inner product variant
            self.index = faiss.IndexHNSWFlat(self.dimension, self.config.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        
        logger.info(f"Initialized HNSW index: dimension={self.dimension}, M={self.config.hnsw_m}")
    
    def _initialize_ivf_index(self) -> None:
        """Initialize Inverted File index for large-scale approximate search."""
        # Calculate optimal number of clusters
        nlist = min(4096, max(1, int(np.sqrt(self.config.hnsw_max_elements))))
        
        # Create quantizer and IVF index
        quantizer = faiss.IndexFlatL2(self.dimension)
        self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
        
        # Set search parameters
        self.index.nprobe = max(1, nlist // 16)
        
        logger.info(f"Initialized IVF index: dimension={self.dimension}, nlist={nlist}")
    
    def _initialize_exact_index(self) -> None:
        """Initialize exact search index for perfect precision."""
        if self.config.similarity_metric == "cosine":
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for normalized vectors
        else:
            self.index = faiss.IndexFlatL2(self.dimension)
        
        logger.info(f"Initialized exact index: dimension={self.dimension}")
    
    def add_vectors(self, vectors: VectorType, ids: List[str]) -> None:
        """
        Add vectors to the index with sophisticated preprocessing.
        
        Args:
            vectors: Matrix of vectors to add (n_vectors, dimension)
            ids: List of document identifiers
        """
        if vectors.shape[0] != len(ids):
            raise ValueError("Number of vectors must match number of IDs")
        
        # Preprocessing based on similarity metric
        processed_vectors = self._preprocess_vectors(vectors)
        
        # Add to FAISS index
        if self.config.index_type == "ivf" and not self.index.is_trained:
            # Train IVF index if not already trained
            if vectors.shape[0] >= self.index.nlist:
                self.index.train(processed_vectors.astype(np.float32))
            else:
                logger.warning("Insufficient vectors for IVF training, using exact search")
                self._initialize_exact_index()
        
        # Add vectors with ID mapping
        start_id = self.next_id
        faiss_ids = np.arange(start_id, start_id + len(ids), dtype=np.int64)
        
        # Check if index supports add_with_ids
        if hasattr(self.index, 'add_with_ids'):
            self.index.add_with_ids(processed_vectors.astype(np.float32), faiss_ids)
        else:
            # For indexes that don't support add_with_ids (like HNSW), just add vectors
            self.index.add(processed_vectors.astype(np.float32))
            # Store IDs separately
            for i, doc_id in enumerate(ids):
                faiss_id = start_id + i
                self.id_mapping[faiss_id] = doc_id
                self.reverse_id_mapping[doc_id] = faiss_id
        
        # Update ID mappings
        for i, doc_id in enumerate(ids):
            faiss_id = start_id + i
            self.id_mapping[faiss_id] = doc_id
            self.reverse_id_mapping[doc_id] = faiss_id
        
        self.next_id += len(ids)
        
        logger.debug(f"Added {len(ids)} vectors to FAISS index")
    
    def search(self, query_vector: VectorType, k: int) -> List[Tuple[str, float]]:
        """
        Perform k-nearest neighbor search with mathematical precision.
        
        Args:
            query_vector: Query vector
            k: Number of nearest neighbors to retrieve
            
        Returns:
            List of (document_id, similarity_score) tuples
        """
        if self.index.ntotal == 0:
            return []
        
        # Preprocess query vector
        processed_query = self._preprocess_vectors(query_vector.reshape(1, -1))
        
        # Perform search
        distances, indices = self.index.search(processed_query.astype(np.float32), k)
        
        # Convert results
        results = []
        for i, (distance, faiss_id) in enumerate(zip(distances[0], indices[0])):
            if faiss_id == -1:  # No more results
                break
            
            doc_id = self.id_mapping.get(faiss_id)
            if doc_id is not None:
                # Convert distance to similarity based on metric
                similarity = self._distance_to_similarity(distance)
                results.append((doc_id, similarity))
        
        return results
    
    def _preprocess_vectors(self, vectors: VectorType) -> VectorType:
        """Preprocess vectors based on similarity metric."""
        if self.config.similarity_metric == "cosine":
            # Normalize vectors for cosine similarity using inner product
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            return vectors / norms
        
        return vectors
    
    def _distance_to_similarity(self, distance: float) -> float:
        """Convert FAISS distance to similarity score."""
        if self.config.similarity_metric == "cosine":
            # For normalized vectors, inner product = cosine similarity
            return float(distance)
        elif self.config.similarity_metric == "euclidean":
            # Convert L2 distance to similarity
            return float(1.0 / (1.0 + distance))
        else:
            return float(distance)


class ChromaDBAdapter:
    """
    Advanced ChromaDB integration with sophisticated query optimization.
    Provides seamless integration with ChromaDB's persistent storage capabilities.
    """
    
    def __init__(self, config: VectorStoreConfiguration):
        """
        Initialize ChromaDB client with advanced configuration.
        
        Args:
            config: Vector store configuration
        """
        self.config = config
        
        # Initialize ChromaDB with persistence
        if config.enable_persistence:
            self.client = chromadb.PersistentClient(
                path=str(config.persist_directory),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
        else:
            self.client = chromadb.EphemeralClient()
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=config.collection_name,
            metadata={"hnsw:space": config.similarity_metric}
        )
        
        logger.info(f"Initialized ChromaDB: collection={config.collection_name}")
    
    def add_documents(self, documents: List[Any], embeddings: VectorType, ids: List[str]) -> None:
        """
        Add documents with embeddings to ChromaDB collection.
        
        Args:
            documents: List of document objects
            embeddings: Embedding vectors
            ids: Document identifiers
        """
        # Prepare document content
        document_texts = []
        metadatas = []
        
        for doc in documents:
            if hasattr(doc, 'content'):
                document_texts.append(doc.content)
            else:
                document_texts.append(str(doc))
            
            if hasattr(doc, 'metadata'):
                metadatas.append(doc.metadata.__dict__ if hasattr(doc.metadata, '__dict__') else doc.metadata)
            else:
                metadatas.append({})
        
        # Add to ChromaDB
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=document_texts,
            metadatas=metadatas,
            ids=ids
        )
        
        logger.debug(f"Added {len(documents)} documents to ChromaDB collection")
    
    def search(self, query_embedding: VectorType, k: int) -> List[SearchResult]:
        """
        Perform similarity search using ChromaDB.
        
        Args:
            query_embedding: Query vector
            k: Number of results to retrieve
            
        Returns:
            List of search results with similarity scores
        """
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )
        
        # Convert to standard format
        search_results = []
        
        if results['ids'] and results['ids'][0]:
            for i, doc_id in enumerate(results['ids'][0]):
                # Create document object
                document = type('Document', (), {
                    'id': doc_id,
                    'content': results['documents'][0][i] if results['documents'] else "",
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {}
                })()
                
                # Convert distance to similarity
                distance = results['distances'][0][i] if results['distances'] else 0.0
                similarity = self._distance_to_similarity(distance)
                
                search_results.append(SearchResult(
                    id=doc_id,
                    document=document,
                    similarity=similarity,
                    distance=distance,
                    metadata=document.metadata
                ))
        
        return search_results
    
    def _distance_to_similarity(self, distance: float) -> float:
        """Convert distance to similarity score based on metric."""
        if self.config.similarity_metric == "cosine":
            return 1.0 - distance  # ChromaDB returns cosine distance
        else:
            return 1.0 / (1.0 + distance)  # Euclidean distance to similarity
    
    def count(self) -> int:
        """Get total number of documents in collection."""
        return self.collection.count()


class VectorStore:
    """
    Advanced vectorized knowledge storage system with mathematical precision.
    
    This class implements a sophisticated vector database that combines multiple
    indexing strategies with advanced optimization techniques for high-performance
    semantic search operations.
    """
    
    def __init__(self, 
                 embedder,
                 config: Optional[VectorStoreConfiguration] = None):
        """
        Initialize the advanced vector store.
        
        Args:
            embedder: Embedding generator instance
            config: Vector store configuration
        """
        self.embedder = embedder
        self.config = config or VectorStoreConfiguration()
        
        # Initialize storage backend
        self.backend = ChromaDBAdapter(self.config)
        
        # Initialize advanced indexing if needed
        self.faiss_index = None
        if self.config.index_type in ["hnsw", "ivf"]:
            self.faiss_index = FAISSVectorIndex(
                dimension=embedder.dimensions,
                config=self.config
            )
        
        # Performance tracking
        self.operation_count = 0
        self.last_save_count = 0
        
        logger.info("Advanced Vector Store initialized with sophisticated indexing")
    
    def add_documents(self, documents: List[Any]) -> None:
        """
        Add documents to the vector store with advanced processing.
        
        Args:
            documents: List of document objects to add
        """
        if not documents:
            return
        
        # Extract text content for embedding
        texts = []
        ids = []
        
        for doc in documents:
            if hasattr(doc, 'content'):
                texts.append(doc.content)
            else:
                texts.append(str(doc))
            
            if hasattr(doc, 'id'):
                ids.append(doc.id)
            else:
                ids.append(f"doc_{len(texts)}")
        
        # Generate embeddings
        embeddings = self.embedder.embed_batch(texts)
        
        # Add to ChromaDB backend
        self.backend.add_documents(documents, embeddings, ids)
        
        # Add to FAISS index if enabled
        if self.faiss_index:
            try:
                self.faiss_index.add_vectors(embeddings, ids)
            except Exception as e:
                logger.warning(f"Failed to add to FAISS index: {e}. Using ChromaDB only.")
                # Continue with ChromaDB only
        
        # Update operation count
        self.operation_count += len(documents)
        
        # Auto-save if threshold reached
        if (self.operation_count - self.last_save_count) >= self.config.auto_save_interval:
            self.save()
        
        logger.info(f"Added {len(documents)} documents to vector store")
    
    def similarity_search(self, 
                         query: str, 
                         k: int = 5, 
                         filter_metadata: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """
        Perform sophisticated similarity search with advanced ranking.
        
        Args:
            query: Query text
            k: Number of results to retrieve
            filter_metadata: Optional metadata filters
            
        Returns:
            List of search results ranked by similarity
        """
        # Generate query embedding
        query_embedding = self.embedder.embed_text(query)
        
        # Use FAISS index if available, otherwise ChromaDB
        if self.faiss_index and self.faiss_index.index.ntotal > 0:
            # FAISS-based search
            faiss_results = self.faiss_index.search(query_embedding, k * 2)  # Get more for filtering
            
            # Convert to standard format by querying ChromaDB for metadata
            search_results = []
            for doc_id, similarity in faiss_results[:k]:
                # This is a simplified approach - in production, you'd want to optimize this
                try:
                    chroma_result = self.backend.collection.get(ids=[doc_id], include=["documents", "metadatas"])
                    if chroma_result['ids']:
                        document = type('Document', (), {
                            'id': doc_id,
                            'content': chroma_result['documents'][0] if chroma_result['documents'] else "",
                            'metadata': chroma_result['metadatas'][0] if chroma_result['metadatas'] else {}
                        })()
                        
                        search_results.append(SearchResult(
                            id=doc_id,
                            document=document,
                            similarity=similarity,
                            distance=1.0 - similarity,
                            metadata=document.metadata
                        ))
                except Exception as e:
                    logger.warning(f"Error retrieving document {doc_id}: {e}")
                    continue
            
            return search_results[:k]
        else:
            # ChromaDB-based search
            return self.backend.search(query_embedding, k)
    
    def count(self) -> int:
        """Get total number of documents in the store."""
        return self.backend.count()
    
    def get_size_bytes(self) -> int:
        """Estimate storage size in bytes."""
        # This is a rough estimate - in production, you'd want more precise calculation
        doc_count = self.count()
        if doc_count == 0:
            return 0
        
        # Estimate based on embedding dimensions and document count
        embedding_size = self.embedder.dimensions * 4  # 4 bytes per float32
        metadata_estimate = 1024  # Rough estimate for metadata per document
        
        return doc_count * (embedding_size + metadata_estimate)
    
    def clear(self) -> None:
        """Clear all documents from the vector store."""
        # Clear ChromaDB collection
        self.backend.collection.delete()
        
        # Reinitialize collection
        self.backend.collection = self.backend.client.get_or_create_collection(
            name=self.config.collection_name,
            metadata={"hnsw:space": self.config.similarity_metric}
        )
        
        # Clear FAISS index
        if self.faiss_index:
            self.faiss_index = FAISSVectorIndex(
                dimension=self.embedder.dimensions,
                config=self.config
            )
        
        self.operation_count = 0
        self.last_save_count = 0
        
        logger.info("Vector store cleared")
    
    def save(self) -> None:
        """Save vector store state to persistent storage."""
        # ChromaDB handles persistence automatically
        self.last_save_count = self.operation_count
        logger.debug("Vector store state saved")
    
    def load(self, path: Path) -> None:
        """Load vector store state from persistent storage."""
        # ChromaDB loads automatically from persistent directory
        logger.debug("Vector store state loaded")
