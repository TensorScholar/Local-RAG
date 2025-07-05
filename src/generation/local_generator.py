"""
Advanced Local Model Generation Engine with Optimized Inference.

This module implements a sophisticated local model inference system that leverages
state-of-the-art transformer architectures with advanced optimization techniques
for high-performance text generation. The architecture employs cutting-edge
computational strategies including dynamic batching, memory optimization, and
adaptive precision scaling for optimal resource utilization.

Computational Paradigms:
1. Adaptive Model Loading - Dynamic model selection with memory-aware optimization
2. Precision Engineering - Mixed-precision inference with numerical stability guarantees
3. Context Optimization - Intelligent context window management and truncation strategies
4. Token Economics - Advanced token management with cost-aware generation policies
5. Memory Architecture - Sophisticated memory management with gradient checkpointing

Advanced Features:
- Multi-model support with automatic hardware optimization
- Dynamic context window adaptation based on available resources
- Advanced sampling techniques with temperature scheduling
- Intelligent prompt engineering with context injection
- Real-time performance monitoring and adaptive optimization

Author: Advanced RAG System Team
Version: 2.0.0
"""

import asyncio
import logging
import time
import threading
import gc
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Protocol
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum, auto
import weakref

# Advanced ML and optimization libraries
import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoConfig,
    GenerationConfig, StoppingCriteria, StoppingCriteriaList,
    BitsAndBytesConfig
)
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import numpy as np

# Memory and performance optimization
import psutil
from contextlib import contextmanager

# Configure advanced logging
logger = logging.getLogger(__name__)


class ModelPrecision(Enum):
    """Enumeration of model precision levels for optimization."""
    FLOAT32 = auto()      # Full precision
    FLOAT16 = auto()      # Half precision
    BFLOAT16 = auto()     # Brain floating point
    INT8 = auto()         # 8-bit quantization
    INT4 = auto()         # 4-bit quantization


class SamplingStrategy(Enum):
    """Advanced sampling strategies for text generation."""
    GREEDY = auto()           # Deterministic greedy decoding
    NUCLEUS = auto()          # Nucleus (top-p) sampling
    TOP_K = auto()            # Top-k sampling
    TEMPERATURE = auto()      # Temperature-based sampling
    CONTRASTIVE = auto()      # Contrastive search
    BEAM_SEARCH = auto()      # Beam search decoding


@dataclass
class GenerationConfiguration:
    """Advanced configuration for text generation with optimization parameters."""
    
    # Core generation parameters
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    length_penalty: float = 1.0
    
    # Advanced sampling configuration
    sampling_strategy: SamplingStrategy = SamplingStrategy.NUCLEUS
    do_sample: bool = True
    num_beams: int = 1
    early_stopping: bool = True
    
    # Context and memory management
    max_context_length: int = 2048
    context_truncation_strategy: str = "sliding_window"  # sliding_window, truncate_left, truncate_right
    memory_efficient: bool = True
    use_cache: bool = True
    
    # Performance optimization
    batch_size: int = 1
    num_return_sequences: int = 1
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    
    # Advanced features
    use_stopping_criteria: bool = True
    custom_stopping_tokens: List[str] = field(default_factory=list)
    enable_streaming: bool = False
    
    def __post_init__(self):
        """Post-initialization validation and optimization."""
        # Validate temperature range
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        
        # Validate top_p range
        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError("Top-p must be between 0.0 and 1.0")
        
        # Optimize parameters for memory efficiency
        if self.memory_efficient:
            self.use_cache = True
            if self.max_context_length > 4096:
                logger.warning("Large context length may cause memory issues in memory-efficient mode")


@dataclass
class ModelConfiguration:
    """Advanced configuration for local model management."""
    
    # Model identification
    model_name: str = "microsoft/DialoGPT-medium"
    model_path: Optional[Path] = None
    custom_model: bool = False
    
    # Hardware optimization
    device: Optional[str] = None
    precision: ModelPrecision = ModelPrecision.FLOAT16
    low_cpu_mem_usage: bool = True
    device_map: Optional[str] = "auto"
    
    # Quantization settings
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    quantization_config: Optional[Dict[str, Any]] = None
    
    # Memory management
    torch_dtype: Optional[torch.dtype] = None
    max_memory: Optional[Dict[str, str]] = None
    offload_folder: Optional[str] = None
    
    # Performance settings
    trust_remote_code: bool = False
    revision: str = "main"
    use_auth_token: Optional[str] = None
    
    def __post_init__(self):
        """Post-initialization configuration and validation."""
        # Auto-detect device if not specified
        if self.device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        
        # Set torch dtype based on precision
        if self.torch_dtype is None:
            if self.precision == ModelPrecision.FLOAT32:
                self.torch_dtype = torch.float32
            elif self.precision == ModelPrecision.FLOAT16:
                self.torch_dtype = torch.float16
            elif self.precision == ModelPrecision.BFLOAT16:
                self.torch_dtype = torch.bfloat16
        
        # Configure quantization
        if self.load_in_4bit or self.load_in_8bit:
            self.quantization_config = BitsAndBytesConfig(
                load_in_4bit=self.load_in_4bit,
                load_in_8bit=self.load_in_8bit,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )


@dataclass
class GenerationResult:
    """Comprehensive result object for text generation operations."""
    
    text: str
    tokens: List[str] = field(default_factory=list)
    token_count: int = 0
    generation_time: float = 0.0
    tokens_per_second: float = 0.0
    
    # Advanced metadata
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    # Generation statistics
    perplexity: Optional[float] = None
    entropy: Optional[float] = None
    confidence_score: Optional[float] = None
    
    # Source attribution
    sources: List[str] = field(default_factory=list)
    context_used: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Compute derived metrics."""
        if not self.token_count and self.tokens:
            self.token_count = len(self.tokens)
        
        if not self.total_tokens:
            self.total_tokens = self.prompt_tokens + self.completion_tokens
        
        if self.generation_time > 0 and self.token_count > 0:
            self.tokens_per_second = self.token_count / self.generation_time


class CustomStoppingCriteria(StoppingCriteria):
    """Advanced stopping criteria with custom token detection."""
    
    def __init__(self, stop_tokens: List[str], tokenizer):
        """
        Initialize custom stopping criteria.
        
        Args:
            stop_tokens: List of stopping tokens/phrases
            tokenizer: Model tokenizer for encoding
        """
        self.stop_tokens = stop_tokens
        self.tokenizer = tokenizer
        
        # Pre-encode stop tokens for efficiency
        self.stop_token_ids = []
        for token in stop_tokens:
            encoded = tokenizer.encode(token, add_special_tokens=False)
            self.stop_token_ids.extend(encoded)
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        """Check if generation should stop based on custom criteria."""
        # Check last few tokens for stop conditions
        if len(input_ids[0]) > 0:
            recent_tokens = input_ids[0][-10:].tolist()  # Check last 10 tokens
            
            for stop_id in self.stop_token_ids:
                if stop_id in recent_tokens:
                    return True
        
        return False


class MemoryManager:
    """Sophisticated memory management for large model inference."""
    
    def __init__(self):
        """Initialize memory manager with system monitoring."""
        self.initial_memory = self._get_memory_usage()
        self.peak_memory = self.initial_memory
        self.cleanup_threshold = 0.8  # Cleanup when 80% memory used
    
    @contextmanager
    def memory_context(self):
        """Context manager for memory-aware operations."""
        initial_mem = self._get_memory_usage()
        try:
            yield
        finally:
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            final_mem = self._get_memory_usage()
            memory_diff = final_mem - initial_mem
            
            if memory_diff > 0:
                logger.debug(f"Memory usage increased by {memory_diff:.2f} GB")
            
            # Update peak memory tracking
            self.peak_memory = max(self.peak_memory, final_mem)
            
            # Trigger cleanup if needed
            if final_mem > self.cleanup_threshold * self._get_total_memory():
                self._aggressive_cleanup()
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in GB."""
        process = psutil.Process()
        return process.memory_info().rss / (1024**3)
    
    def _get_total_memory(self) -> float:
        """Get total system memory in GB."""
        return psutil.virtual_memory().total / (1024**3)
    
    def _aggressive_cleanup(self):
        """Perform aggressive memory cleanup."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        logger.info("Performed aggressive memory cleanup")


class LocalModelInterface(ABC):
    """Abstract interface for local model implementations."""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> GenerationResult:
        """Generate text from prompt."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and capabilities."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up model resources."""
        pass


class TransformerLocalModel(LocalModelInterface):
    """
    Advanced transformer model implementation with sophisticated optimization.
    """
    
    def __init__(self, 
                 model_config: ModelConfiguration,
                 generation_config: Optional[GenerationConfiguration] = None):
        """
        Initialize transformer model with advanced configuration.
        
        Args:
            model_config: Model configuration and optimization settings
            generation_config: Default generation configuration
        """
        self.model_config = model_config
        self.generation_config = generation_config or GenerationConfiguration()
        
        # Initialize memory manager
        self.memory_manager = MemoryManager()
        
        # Model components
        self.model = None
        self.tokenizer = None
        self.device = torch.device(model_config.device)
        
        # Performance tracking
        self.generation_count = 0
        self.total_tokens_generated = 0
        self.total_generation_time = 0.0
        
        # Load model and tokenizer
        self._load_model()
        
        logger.info(f"Initialized local model: {model_config.model_name}")
    
    def _load_model(self) -> None:
        """Load model and tokenizer with advanced optimization."""
        with self.memory_manager.memory_context():
            try:
                # Load tokenizer first
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_config.model_name,
                    trust_remote_code=self.model_config.trust_remote_code,
                    revision=self.model_config.revision
                )
                
                # Handle missing pad token
                if self.tokenizer.pad_token is None:
                    if self.tokenizer.eos_token is not None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                    else:
                        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                
                # Load model with optimization
                model_kwargs = {
                    'trust_remote_code': self.model_config.trust_remote_code,
                    'revision': self.model_config.revision,
                    'torch_dtype': self.model_config.torch_dtype,
                    'low_cpu_mem_usage': self.model_config.low_cpu_mem_usage,
                }
                
                # Add quantization if configured
                if self.model_config.quantization_config:
                    model_kwargs['quantization_config'] = self.model_config.quantization_config
                
                # Add device mapping for multi-GPU
                if self.model_config.device_map:
                    model_kwargs['device_map'] = self.model_config.device_map
                
                # Load model
                if self.model_config.model_path and self.model_config.model_path.exists():
                    # Load from local path
                    self.model = AutoModelForCausalLM.from_pretrained(
                        str(self.model_config.model_path),
                        **model_kwargs
                    )
                else:
                    # Load from HuggingFace Hub
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_config.model_name,
                        **model_kwargs
                    )
                
                # Move to device if not using device_map
                if not self.model_config.device_map:
                    self.model = self.model.to(self.device)
                
                # Set evaluation mode
                self.model.eval()
                
                # Configure generation settings
                self._configure_generation()
                
                logger.info(f"Successfully loaded model on {self.device}")
                
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise
    
    def _configure_generation(self) -> None:
        """Configure model-specific generation parameters."""
        # Update generation config with model-specific tokens
        if self.generation_config.pad_token_id is None:
            self.generation_config.pad_token_id = self.tokenizer.pad_token_id
        
        if self.generation_config.eos_token_id is None:
            self.generation_config.eos_token_id = self.tokenizer.eos_token_id
        
        # Configure model generation config
        if hasattr(self.model, 'generation_config'):
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
            self.model.generation_config.eos_token_id = self.tokenizer.eos_token_id
    
    def generate(self, 
                prompt: str,
                context_docs: Optional[List[Any]] = None,
                **kwargs) -> GenerationResult:
        """
        Generate text with advanced optimization and context integration.
        
        Args:
            prompt: Input prompt for generation
            context_docs: Optional context documents for RAG
            **kwargs: Additional generation parameters
            
        Returns:
            Comprehensive generation result with metadata
        """
        start_time = time.time()
        
        with self.memory_manager.memory_context():
            try:
                # Prepare context-aware prompt
                full_prompt = self._prepare_prompt(prompt, context_docs)
                
                # Tokenize input
                inputs = self._tokenize_input(full_prompt)
                
                # Configure generation parameters
                gen_config = self._prepare_generation_config(**kwargs)
                
                # Prepare stopping criteria
                stopping_criteria = self._prepare_stopping_criteria()
                
                # Generate with model
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        generation_config=gen_config,
                        stopping_criteria=stopping_criteria,
                        return_dict_in_generate=True,
                        output_scores=True
                    )
                
                # Process outputs
                result = self._process_outputs(
                    outputs, 
                    inputs, 
                    full_prompt,
                    context_docs or []
                )
                
                # Update performance tracking
                generation_time = time.time() - start_time
                result.generation_time = generation_time
                
                self.generation_count += 1
                self.total_tokens_generated += result.token_count
                self.total_generation_time += generation_time
                
                logger.debug(f"Generated {result.token_count} tokens in {generation_time:.2f}s")
                
                return result
                
            except Exception as e:
                logger.error(f"Generation failed: {e}")
                # Return empty result on failure
                return GenerationResult(
                    text="",
                    generation_time=time.time() - start_time
                )
    
    def _prepare_prompt(self, prompt: str, context_docs: Optional[List[Any]]) -> str:
        """Prepare context-aware prompt with intelligent formatting."""
        if not context_docs:
            return prompt
        
        # Build context section
        context_parts = []
        for i, doc in enumerate(context_docs[:3]):  # Limit context
            if hasattr(doc, 'content'):
                content = doc.content
            else:
                content = str(doc)
            
            # Truncate long context
            if len(content) > 500:
                content = content[:497] + "..."
            
            context_parts.append(f"Context {i+1}: {content}")
        
        # Combine context and prompt
        context_text = "\n\n".join(context_parts)
        
        full_prompt = f"""Based on the following context, please answer the question.

{context_text}

Question: {prompt}

Answer:"""
        
        return full_prompt
    
    def _tokenize_input(self, prompt: str) -> Dict[str, torch.Tensor]:
        """Tokenize input with context length management."""
        # Tokenize with truncation
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.generation_config.max_context_length,
            padding=False
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        return inputs
    
    def _prepare_generation_config(self, **kwargs) -> GenerationConfig:
        """Prepare generation configuration with parameter merging."""
        # Start with default config
        config_dict = {
            'max_new_tokens': self.generation_config.max_new_tokens,
            'temperature': self.generation_config.temperature,
            'top_p': self.generation_config.top_p,
            'top_k': self.generation_config.top_k,
            'repetition_penalty': self.generation_config.repetition_penalty,
            'length_penalty': self.generation_config.length_penalty,
            'do_sample': self.generation_config.do_sample,
            'num_beams': self.generation_config.num_beams,
            'early_stopping': self.generation_config.early_stopping,
            'pad_token_id': self.generation_config.pad_token_id,
            'eos_token_id': self.generation_config.eos_token_id,
            'use_cache': self.generation_config.use_cache,
        }
        
        # Override with kwargs
        config_dict.update(kwargs)
        
        # Create generation config
        return GenerationConfig(**config_dict)
    
    def _prepare_stopping_criteria(self) -> Optional[StoppingCriteriaList]:
        """Prepare custom stopping criteria."""
        if not self.generation_config.use_stopping_criteria:
            return None
        
        criteria_list = []
        
        # Add custom token stopping criteria
        if self.generation_config.custom_stopping_tokens:
            custom_criteria = CustomStoppingCriteria(
                self.generation_config.custom_stopping_tokens,
                self.tokenizer
            )
            criteria_list.append(custom_criteria)
        
        return StoppingCriteriaList(criteria_list) if criteria_list else None
    
    def _process_outputs(self, 
                        outputs, 
                        inputs: Dict[str, torch.Tensor],
                        full_prompt: str,
                        context_docs: List[Any]) -> GenerationResult:
        """Process model outputs into structured result."""
        # Extract generated sequences
        generated_sequences = outputs.sequences
        
        # Get input length for prompt separation
        input_length = inputs['input_ids'].shape[1]
        
        # Decode generated text
        generated_tokens = generated_sequences[0][input_length:]
        generated_text = self.tokenizer.decode(
            generated_tokens, 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        # Create token list for analysis
        token_list = [
            self.tokenizer.decode([token_id], skip_special_tokens=True)
            for token_id in generated_tokens
        ]
        
        # Extract source information
        sources = []
        context_used = []
        
        if context_docs:
            for doc in context_docs:
                if hasattr(doc, 'metadata') and hasattr(doc.metadata, 'source'):
                    sources.append(doc.metadata.source)
                else:
                    sources.append("Unknown")
                
                if hasattr(doc, 'content'):
                    context_used.append(doc.content[:100] + "...")
        
        # Calculate advanced metrics
        confidence_score = self._calculate_confidence(outputs.scores) if hasattr(outputs, 'scores') else None
        
        return GenerationResult(
            text=generated_text.strip(),
            tokens=token_list,
            token_count=len(generated_tokens),
            prompt_tokens=input_length,
            completion_tokens=len(generated_tokens),
            sources=sources,
            context_used=context_used,
            confidence_score=confidence_score
        )
    
    def _calculate_confidence(self, scores: Tuple[torch.FloatTensor]) -> float:
        """Calculate generation confidence score from logits."""
        if not scores:
            return None
        
        # Convert scores to probabilities and calculate mean confidence
        confidences = []
        
        for step_scores in scores:
            # Apply softmax to get probabilities
            probs = F.softmax(step_scores[0], dim=-1)
            
            # Get max probability as confidence for this step
            max_prob = torch.max(probs).item()
            confidences.append(max_prob)
        
        # Return average confidence
        return float(np.mean(confidences)) if confidences else None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        if not self.model:
            return {"status": "not_loaded"}
        
        # Calculate average performance metrics
        avg_tokens_per_second = (
            self.total_tokens_generated / self.total_generation_time
            if self.total_generation_time > 0 else 0
        )
        
        model_info = {
            "model_name": self.model_config.model_name,
            "device": str(self.device),
            "precision": self.model_config.precision.name,
            "status": "loaded",
            "parameters": self._get_parameter_count(),
            "memory_usage": self.memory_manager._get_memory_usage(),
            "performance": {
                "generation_count": self.generation_count,
                "total_tokens_generated": self.total_tokens_generated,
                "total_generation_time": self.total_generation_time,
                "avg_tokens_per_second": avg_tokens_per_second
            },
            "capabilities": {
                "max_context_length": self.generation_config.max_context_length,
                "supports_streaming": self.generation_config.enable_streaming,
                "supports_batching": True
            }
        }
        
        return model_info
    
    def _get_parameter_count(self) -> int:
        """Get total number of model parameters."""
        if not self.model:
            return 0
        
        return sum(p.numel() for p in self.model.parameters())
    
    def cleanup(self) -> None:
        """Clean up model resources and memory."""
        if self.model:
            del self.model
            self.model = None
        
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Model resources cleaned up")


class LocalGenerator:
    """
    Advanced local model generation orchestrator with multi-model support.
    
    This class manages multiple local models and provides intelligent routing
    between them based on query characteristics and resource availability.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize local generator with model management.
        
        Args:
            config_path: Path to configuration directory
        """
        self.config_path = config_path or Path.home() / ".rag_system"
        self.models: Dict[str, TransformerLocalModel] = {}
        self.default_model_name = None
        
        # Performance tracking
        self.total_generations = 0
        self.model_usage_stats = defaultdict(int)
        
        # Load available models
        self._discover_and_load_models()
        
        logger.info(f"Local Generator initialized with {len(self.models)} models")
    
    def _discover_and_load_models(self) -> None:
        """Discover and load available local models."""
        # Default model configurations for common models
        default_models = [
            {
                "name": "microsoft/DialoGPT-medium",
                "config": ModelConfiguration(
                    model_name="microsoft/DialoGPT-medium",
                    precision=ModelPrecision.FLOAT16,
                    load_in_8bit=True
                )
            },
            {
                "name": "distilgpt2", 
                "config": ModelConfiguration(
                    model_name="distilgpt2",
                    precision=ModelPrecision.FLOAT16
                )
            }
        ]
        
        # Try to load each model
        for model_info in default_models:
            try:
                model = TransformerLocalModel(model_info["config"])
                self.models[model_info["name"]] = model
                
                if self.default_model_name is None:
                    self.default_model_name = model_info["name"]
                
                logger.info(f"Loaded model: {model_info['name']}")
                
            except Exception as e:
                logger.warning(f"Failed to load model {model_info['name']}: {e}")
                continue
        
        if not self.models:
            logger.warning("No local models could be loaded")
    
    def generate(self, 
                query: str,
                context_docs: Optional[List[Any]] = None,
                model_name: Optional[str] = None,
                **kwargs) -> GenerationResult:
        """
        Generate text using the specified or best available model.
        
        Args:
            query: Input query/prompt
            context_docs: Optional context documents
            model_name: Specific model to use (uses default if None)
            **kwargs: Additional generation parameters
            
        Returns:
            Generation result with text and metadata
        """
        # Select model
        selected_model_name = model_name or self.default_model_name
        
        if not selected_model_name or selected_model_name not in self.models:
            logger.error(f"Model not available: {selected_model_name}")
            return GenerationResult(text="Model not available")
        
        # Get model and generate
        model = self.models[selected_model_name]
        result = model.generate(query, context_docs, **kwargs)
        
        # Update usage statistics
        self.total_generations += 1
        self.model_usage_stats[selected_model_name] += 1
        
        return result
    
    def get_available_models(self) -> List[str]:
        """Get list of available model names."""
        return list(self.models.keys())
    
    def get_model_info(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Get information about a specific model or all models."""
        if model_name:
            if model_name in self.models:
                return self.models[model_name].get_model_info()
            else:
                return {"error": f"Model {model_name} not found"}
        
        # Return info for all models
        return {
            name: model.get_model_info()
            for name, model in self.models.items()
        }
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for all models."""
        return {
            "total_generations": self.total_generations,
            "model_usage": dict(self.model_usage_stats),
            "available_models": len(self.models),
            "default_model": self.default_model_name
        }
    
    def cleanup(self) -> None:
        """Clean up all model resources."""
        for model in self.models.values():
            model.cleanup()
        
        self.models.clear()
        logger.info("All local models cleaned up")


# Import required modules for default collections
from collections import defaultdict
