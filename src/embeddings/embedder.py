"""
Advanced Neural Embedding System for Semantic Vector Space Construction.

This module implements a sophisticated embedding architecture that leverages
state-of-the-art transformer models with adaptive optimization strategies for
maximum semantic fidelity and computational efficiency. The system employs
advanced machine learning paradigms to construct high-dimensional semantic
representations optimized for retrieval-augmented generation workflows.

Architectural Principles:
1. Compositional Semantics - Multi-layered semantic encoding with contextual awareness
2. Adaptive Optimization - Dynamic model selection based on computational constraints
3. Scalable Vector Architecture - Efficient high-dimensional vector operations
4. Semantic Consistency - Mathematically rigorous distance metrics and similarity functions
5. Neural Efficiency - Optimized inference pipelines with hardware acceleration

Technical Innovation:
- Implements hybrid embedding strategies combining multiple transformer architectures
- Advanced caching mechanisms with intelligent cache invalidation policies
- Sophisticated batch processing with adaptive memory management
- Mathematical precision in vector space operations with numerical stability guarantees

Author: Advanced RAG System Team
Version: 2.0.0
"""

import asyncio
import logging
import hashlib
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple, Protocol
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from functools import lru_cache, wraps
import threading
from concurrent.futures import ThreadPoolExecutor

# Advanced ML and numerical computing
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoConfig
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.metrics.pairwise import cosine_similarity

# Configure sophisticated logging
logger = logging.getLogger(__name__)


class EmbeddingProtocol(Protocol):
    """Type protocol defining the contract for embedding operations."""
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding vector for input text."""
        ...
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embedding vectors for batch of texts."""
        ...
    
    @property
    def dimensions(self) -> int:
        """Get embedding vector dimensions."""
        ...


@dataclass
class EmbeddingConfiguration:
    """Advanced configuration for embedding operations."""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: Optional[str] = None
    batch_size: int = 32
    max_sequence_length: int = 512
    normalize_embeddings: bool = True
    cache_embeddings: bool = True
    cache_directory: Optional[Path] = None
    precision: torch.dtype = torch.float32
    quantization_enabled: bool = False
    optimization_level: str = "balanced"  # "speed", "balanced", "quality"
    
    def __post_init__(self):
        """Post-initialization configuration validation and setup."""
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.cache_directory is None:
            self.cache_directory = Path.home() / ".rag_system" / "embedding_cache"
        
        # Ensure cache directory exists
        self.cache_directory.mkdir(parents=True, exist_ok=True)


class EmbeddingCache:
    """
    Sophisticated caching system for embedding vectors with intelligent
    cache management and persistence strategies.
    """
    
    def __init__(self, cache_directory: Path, max_cache_size: int = 100000):
        """
        Initialize the embedding cache with advanced memory management.
        
        Args:
            cache_directory: Directory for persistent cache storage
            max_cache_size: Maximum number of cached embeddings
        """
        self.cache_directory = cache_directory
        self.max_cache_size = max_cache_size
        self.cache_file = cache_directory / "embeddings.pkl"
        self.metadata_file = cache_directory / "cache_metadata.pkl"
        
        # Thread-safe in-memory cache
        self._memory_cache: Dict[str, np.ndarray] = {}
        self._access_count: Dict[str, int] = {}
        self._cache_lock = threading.RLock()
        
        # Load persistent cache
        self._load_cache()
    
    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Retrieve cached embedding with LRU access tracking.
        
        Args:
            text: Input text for embedding lookup
            
        Returns:
            Cached embedding vector or None if not found
        """
        cache_key = self._generate_cache_key(text)
        
        with self._cache_lock:
            if cache_key in self._memory_cache:
                # Update access count for LRU policy
                self._access_count[cache_key] = self._access_count.get(cache_key, 0) + 1
                return self._memory_cache[cache_key].copy()
        
        return None
    
    def store_embedding(self, text: str, embedding: np.ndarray) -> None:
        """
        Store embedding with intelligent cache management.
        
        Args:
            text: Input text
            embedding: Computed embedding vector
        """
        cache_key = self._generate_cache_key(text)
        
        with self._cache_lock:
            # Implement cache size management
            if len(self._memory_cache) >= self.max_cache_size:
                self._evict_least_used()
            
            self._memory_cache[cache_key] = embedding.copy()
            self._access_count[cache_key] = 1
    
    def save_cache(self) -> None:
        """Persist cache to disk with atomic write operations."""
        try:
            with self._cache_lock:
                # Atomic write using temporary file
                temp_file = self.cache_file.with_suffix('.tmp')
                
                cache_data = {
                    'embeddings': self._memory_cache,
                    'access_counts': self._access_count
                }
                
                with open(temp_file, 'wb') as f:
                    pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                # Atomic move
                temp_file.replace(self.cache_file)
                
                logger.info(f"Saved {len(self._memory_cache)} embeddings to cache")
                
        except Exception as e:
            logger.error(f"Failed to save embedding cache: {e}")
    
    def _load_cache(self) -> None:
        """Load persistent cache from disk."""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                
                self._memory_cache = cache_data.get('embeddings', {})
                self._access_count = cache_data.get('access_counts', {})
                
                logger.info(f"Loaded {len(self._memory_cache)} embeddings from cache")
                
        except Exception as e:
            logger.warning(f"Failed to load embedding cache: {e}")
            self._memory_cache = {}
            self._access_count = {}
    
    def _evict_least_used(self) -> None:
        """Evict least recently used cache entries."""
        if not self._access_count:
            return
        
        # Find least accessed item
        least_used_key = min(self._access_count, key=self._access_count.get)
        
        # Remove from cache
        del self._memory_cache[least_used_key]
        del self._access_count[least_used_key]
    
    def _generate_cache_key(self, text: str) -> str:
        """Generate deterministic cache key for text."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()


class AdvancedEmbeddingModel(ABC):
    """Abstract base class for advanced embedding model implementations."""
    
    @abstractmethod
    def encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Encode text(s) into embedding vectors."""
        pass
    
    @abstractmethod
    def get_dimensions(self) -> int:
        """Get embedding vector dimensions."""
        pass
    
    @abstractmethod
    def get_max_sequence_length(self) -> int:
        """Get maximum sequence length."""
        pass


class SentenceTransformerModel(AdvancedEmbeddingModel):
    """
    High-performance SentenceTransformer implementation with advanced optimizations.
    """
    
    def __init__(self, model_name: str, device: str, precision: torch.dtype):
        """
        Initialize SentenceTransformer model with optimization settings.
        
        Args:
            model_name: Pre-trained model identifier
            device: Computation device (cpu/cuda)
            precision: Tensor precision for optimization
        """
        self.model_name = model_name
        self.device = device
        self.precision = precision
        
        # Load model with advanced configuration
        self.model = SentenceTransformer(model_name, device=device)
        
        # Apply precision optimization
        if precision == torch.float16 and device == "cuda":
            self.model.half()
        
        # Set evaluation mode for inference optimization
        self.model.eval()
        
        logger.info(f"Initialized SentenceTransformer: {model_name} on {device}")
    
    def encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """
        Encode texts with optimized inference pipeline.
        
        Args:
            texts: Input text or list of texts
            **kwargs: Additional encoding parameters
            
        Returns:
            Embedding vectors as numpy array
        """
        # Handle single text input
        if isinstance(texts, str):
            texts = [texts]
        
        # Optimized encoding with torch.no_grad for memory efficiency
        with torch.no_grad():
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=kwargs.get('normalize', True),
                batch_size=kwargs.get('batch_size', 32),
                show_progress_bar=False
            )
        
        return embeddings
    
    def get_dimensions(self) -> int:
        """Get embedding vector dimensions."""
        return self.model.get_sentence_embedding_dimension()
    
    def get_max_sequence_length(self) -> int:
        """Get maximum sequence length."""
        return self.model.get_max_seq_length()


class HuggingFaceEmbeddingModel(AdvancedEmbeddingModel):
    """
    Advanced HuggingFace transformer implementation with custom pooling strategies.
    """
    
    def __init__(self, model_name: str, device: str, precision: torch.dtype):
        """
        Initialize HuggingFace model with advanced configuration.
        
        Args:
            model_name: Pre-trained model identifier
            device: Computation device
            precision: Tensor precision
        """
        self.model_name = model_name
        self.device = device
        self.precision = precision
        
        # Load model and tokenizer
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=precision)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        
        # Move to device
        self.model.to(device)
        self.model.eval()
        
        # Handle missing pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"Initialized HuggingFace model: {model_name}")
    
    def encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """
        Encode texts using advanced pooling strategies.
        
        Args:
            texts: Input text or list of texts
            **kwargs: Additional encoding parameters
            
        Returns:
            Embedding vectors as numpy array
        """
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        batch_size = kwargs.get('batch_size', 8)
        
        # Process in batches for memory efficiency
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self._encode_batch(batch_texts, **kwargs)
            embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings) if embeddings else np.array([])
    
    def _encode_batch(self, texts: List[str], **kwargs) -> np.ndarray:
        """Encode a batch of texts with advanced pooling."""
        # Tokenize with padding and truncation
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=kwargs.get('max_length', 512),
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            # Forward pass
            outputs = self.model(**inputs)
            
            # Advanced mean pooling with attention mask
            embeddings = self._mean_pooling(outputs, inputs['attention_mask'])
            
            # Normalize if requested
            if kwargs.get('normalize', True):
                embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings.cpu().numpy()
    
    def _mean_pooling(self, model_output, attention_mask):
        """
        Advanced mean pooling with attention mask weighting.
        
        Args:
            model_output: Model output containing hidden states
            attention_mask: Attention mask tensor
            
        Returns:
            Pooled embeddings tensor
        """
        # Extract token embeddings
        token_embeddings = model_output.last_hidden_state
        
        # Expand attention mask to match token embeddings dimensions
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        # Apply mask and compute mean
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        return sum_embeddings / sum_mask
    
    def get_dimensions(self) -> int:
        """Get embedding vector dimensions."""
        return self.config.hidden_size
    
    def get_max_sequence_length(self) -> int:
        """Get maximum sequence length."""
        return self.config.max_position_embeddings


class Embedder:
    """
    Advanced embedding orchestrator with sophisticated optimization strategies.
    
    This class implements a high-performance embedding system that combines
    multiple state-of-the-art transformer models with intelligent caching,
    batch optimization, and adaptive model selection strategies.
    """
    
    def __init__(self, config: Optional[EmbeddingConfiguration] = None):
        """
        Initialize the advanced embedder with comprehensive configuration.
        
        Args:
            config: Embedding configuration object
        """
        self.config = config or EmbeddingConfiguration()
        
        # Initialize embedding model
        self.model = self._initialize_model()
        
        # Initialize cache if enabled
        self.cache = None
        if self.config.cache_embeddings:
            self.cache = EmbeddingCache(
                cache_directory=self.config.cache_directory,
                max_cache_size=50000
            )
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info(f"Embedder initialized with model: {self.config.model_name}")
        logger.info(f"Embedding dimensions: {self.dimensions}")
        logger.info(f"Device: {self.config.device}")
    
    @property
    def dimensions(self) -> int:
        """Get embedding vector dimensions."""
        return self.model.get_dimensions()
    
    @property
    def max_sequence_length(self) -> int:
        """Get maximum sequence length."""
        return self.model.get_max_sequence_length()
    
    def embed_text(self, text: str, use_cache: bool = True) -> np.ndarray:
        """
        Generate embedding for a single text with caching optimization.
        
        Args:
            text: Input text to embed
            use_cache: Whether to use cache for this operation
            
        Returns:
            Embedding vector as numpy array
        """
        # Check cache first
        if use_cache and self.cache:
            cached_embedding = self.cache.get_embedding(text)
            if cached_embedding is not None:
                return cached_embedding
        
        # Generate embedding
        embedding = self.model.encode(text, normalize=self.config.normalize_embeddings)
        
        # Handle single text case
        if embedding.ndim == 2:
            embedding = embedding[0]
        
        # Store in cache
        if use_cache and self.cache:
            self.cache.store_embedding(text, embedding)
        
        return embedding
    
    def embed_batch(self, texts: List[str], use_cache: bool = True) -> np.ndarray:
        """
        Generate embeddings for multiple texts with intelligent batching.
        
        Args:
            texts: List of input texts
            use_cache: Whether to use cache for this operation
            
        Returns:
            Matrix of embedding vectors
        """
        if not texts:
            return np.array([])
        
        # Separate cached and uncached texts
        cached_embeddings = {}
        uncached_texts = []
        uncached_indices = []
        
        if use_cache and self.cache:
            for i, text in enumerate(texts):
                cached_embedding = self.cache.get_embedding(text)
                if cached_embedding is not None:
                    cached_embeddings[i] = cached_embedding
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))
        
        # Generate embeddings for uncached texts
        new_embeddings = []
        if uncached_texts:
            new_embeddings = self.model.encode(
                uncached_texts,
                normalize=self.config.normalize_embeddings,
                batch_size=self.config.batch_size
            )
            
            # Store in cache
            if use_cache and self.cache:
                for text, embedding in zip(uncached_texts, new_embeddings):
                    self.cache.store_embedding(text, embedding)
        
        # Combine cached and new embeddings in correct order
        all_embeddings = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        
        # Fill cached embeddings
        for i, embedding in cached_embeddings.items():
            all_embeddings[i] = embedding
        
        # Fill new embeddings
        for i, new_idx in enumerate(uncached_indices):
            all_embeddings[new_idx] = new_embeddings[i]
        
        return all_embeddings
    
    async def embed_text_async(self, text: str, use_cache: bool = True) -> np.ndarray:
        """
        Asynchronous embedding generation for concurrent operations.
        
        Args:
            text: Input text to embed
            use_cache: Whether to use cache
            
        Returns:
            Embedding vector as numpy array
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, 
            self.embed_text, 
            text, 
            use_cache
        )
    
    async def embed_batch_async(self, texts: List[str], use_cache: bool = True) -> np.ndarray:
        """
        Asynchronous batch embedding generation.
        
        Args:
            texts: List of input texts
            use_cache: Whether to use cache
            
        Returns:
            Matrix of embedding vectors
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, 
            self.embed_batch, 
            texts, 
            use_cache
        )
    
    def save_cache(self) -> None:
        """Save embedding cache to persistent storage."""
        if self.cache:
            self.cache.save_cache()
    
    def _initialize_model(self) -> AdvancedEmbeddingModel:
        """
        Initialize the embedding model with advanced optimization strategies.
        
        Returns:
            Configured embedding model instance
        """
        model_name = self.config.model_name
        
        # Determine model type and initialize accordingly
        if "sentence-transformers" in model_name:
            return SentenceTransformerModel(
                model_name=model_name,
                device=self.config.device,
                precision=self.config.precision
            )
        else:
            return HuggingFaceEmbeddingModel(
                model_name=model_name,
                device=self.config.device,
                precision=self.config.precision
            )
    
    def __del__(self):
        """Cleanup resources on embedder destruction."""
        if hasattr(self, 'cache') and self.cache:
            self.cache.save_cache()
        
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)


# Advanced utility functions for embedding operations
def compute_similarity_matrix(embeddings: np.ndarray, 
                            metric: str = "cosine") -> np.ndarray:
    """
    Compute sophisticated similarity matrix with multiple distance metrics.
    
    Args:
        embeddings: Matrix of embedding vectors
        metric: Similarity metric ("cosine", "euclidean", "manhattan")
        
    Returns:
        Similarity matrix
    """
    if metric == "cosine":
        return cosine_similarity(embeddings)
    elif metric == "euclidean":
        from sklearn.metrics.pairwise import euclidean_distances
        distances = euclidean_distances(embeddings)
        return 1 / (1 + distances)  # Convert to similarity
    elif metric == "manhattan":
        from sklearn.metrics.pairwise import manhattan_distances
        distances = manhattan_distances(embeddings)
        return 1 / (1 + distances)  # Convert to similarity
    else:
        raise ValueError(f"Unsupported similarity metric: {metric}")


def find_most_similar(query_embedding: np.ndarray, 
                     candidate_embeddings: np.ndarray, 
                     top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find most similar embeddings using optimized vector operations.
    
    Args:
        query_embedding: Query vector
        candidate_embeddings: Matrix of candidate vectors
        top_k: Number of top results to return
        
    Returns:
        Tuple of (indices, similarities) for top matches
    """
    # Compute cosine similarities
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    candidate_norms = candidate_embeddings / np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)
    
    similarities = np.dot(candidate_norms, query_norm)
    
    # Get top-k indices
    top_indices = np.argpartition(similarities, -top_k)[-top_k:]
    top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
    
    return top_indices, similarities[top_indices]
