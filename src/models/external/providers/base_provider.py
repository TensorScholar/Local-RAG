"""
Abstract Base Provider for External API Integration.

This module defines the foundational interface for all external API providers,
establishing a uniform contract for model interaction while allowing provider-specific
implementations to handle the unique characteristics of each API.

The architecture employs the Strategy pattern to enable polymorphic provider selection
at runtime while maintaining consistent behavior expectations across implementations.

Design Principles:
1. Interface Segregation - Minimal, cohesive interface requirements
2. Dependency Inversion - High-level components depend on abstractions
3. Open/Closed - Extensible for new providers without modification
4. Liskov Substitution - Provider implementations maintain contract guarantees
5. Single Responsibility - Each provider focused on one API integration

Performance Characteristics:
- Initialization: O(1) with credential validation
- Model Listing: O(1) with cached metadata
- Response Generation: Dependent on external API latency
- Error Handling: Comprehensive with exponential backoff

Author: Advanced RAG System Team
Version: 1.0.0
"""

import abc
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Any

# Type aliases for clarity
QueryType = str
ResponseType = Dict[str, Any]


class ModelCapability(Enum):
    """Enumeration of model capabilities for capability-aware routing."""
    BASIC_COMPLETION = auto()
    CODE_GENERATION = auto()
    SCIENTIFIC_REASONING = auto()
    MATHEMATICAL_COMPUTATION = auto()
    MULTIMODAL_UNDERSTANDING = auto()
    LONG_CONTEXT = auto()
    ADVANCED_REASONING = auto()


@dataclass
class ModelMetadata:
    """Metadata about external models to facilitate intelligent routing."""
    provider_name: str
    model_name: str
    max_tokens: int
    context_window: int
    capabilities: List[ModelCapability]
    token_cost_input: float  # Cost per 1000 tokens for input
    token_cost_output: float  # Cost per 1000 tokens for output
    latency_estimate_ms: int  # Estimated average latency


class APIProvider(abc.ABC):
    """
    Abstract base class defining the interface for all external API providers.
    
    This class establishes the contract that all provider implementations must fulfill,
    ensuring consistent behavior and interchangeability within the system.
    
    The interface is designed to be:
    - Minimal: Only essential operations required
    - Complete: Covers all necessary interactions
    - Robust: Includes error handling and resource management
    - Flexible: Accommodates different provider capabilities
    """
    
    @abc.abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the provider with necessary credentials and configuration.
        
        Returns:
            bool: True if initialization was successful, False otherwise
            
        This method performs:
        - Credential validation
        - API client setup
        - Connection testing (optional)
        - Resource preparation
        
        Implementation must be idempotent and thread-safe.
        """
        pass
    
    @abc.abstractmethod
    def get_available_models(self) -> List[ModelMetadata]:
        """
        Return a list of available models from this provider with their metadata.
        
        Returns:
            List[ModelMetadata]: Available models with capabilities and specifications
            
        This method should:
        - Return detailed metadata for intelligent routing
        - Include capability information for each model
        - Provide performance characteristics for optimization
        - Cache results when appropriate for efficiency
        """
        pass
    
    @abc.abstractmethod
    async def generate_response(
        self, 
        query: QueryType, 
        model_name: str, 
        context: Optional[List[str]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> ResponseType:
        """
        Generate a response from the specified model.
        
        Args:
            query: The user's query text
            model_name: Name of the model to use
            context: Optional context information to include
            temperature: Generation temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            
        Returns:
            ResponseType: Standardized response dictionary containing:
                - provider: Provider name
                - model: Model name used
                - content: Generated text
                - latency_ms: Request latency
                - input_tokens: Number of input tokens
                - output_tokens: Number of output tokens
                - cost: Estimated cost in USD
                
        Raises:
            ValueError: If model_name is invalid
            RuntimeError: If provider not initialized
            Exception: For provider-specific errors
            
        Implementation must handle:
        - API communication errors with appropriate retry logic
        - Authentication errors with clear diagnostics
        - Rate limiting with backoff strategies
        - Response formatting for consistency
        """
        pass
    
    @abc.abstractmethod
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in the provided text.
        
        Args:
            text: The input text to estimate
            
        Returns:
            int: Estimated token count
            
        This method enables:
        - Cost estimation before requests
        - Context window management
        - Token budget allocation
        
        Implementation should balance accuracy with performance.
        """
        pass
    
    @abc.abstractmethod
    def calculate_cost(self, input_tokens: int, output_tokens: int, model_name: str) -> float:
        """
        Calculate the cost of a request based on token counts.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model_name: Name of the model used
            
        Returns:
            float: Estimated cost in USD
            
        This method enables:
        - Cost monitoring and budgeting
        - Usage optimization
        - Billing reconciliation
        
        Implementation should follow the provider's current pricing model.
        """
        pass
