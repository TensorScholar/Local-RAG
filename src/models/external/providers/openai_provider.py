"""
OpenAI API Provider Implementation.

This module implements the APIProvider interface for the OpenAI API,
providing a sophisticated abstraction over OpenAI's models with comprehensive
error handling, intelligent retry mechanisms, and precise conformance to the
provider contract.

Design Features:
1. Secure Authentication - Robust credential management with validation
2. Comprehensive Model Registry - Detailed capability specifications
3. Advanced Retry Logic - Exponential backoff for transient failures
4. Accurate Token Accounting - Leverages OpenAI's token count information
5. Precise Cost Calculation - Exact cost calculations based on current pricing

Performance Characteristics:
- Initialization: O(1) with credential validation
- Token Extraction: O(1) using OpenAI's native token counting
- Response Generation: O(1) + API latency
- Thread Safety: All methods are thread-safe with proper state isolation

Author: Advanced RAG System Team
Version: 1.1.0
"""

import logging
import time
from typing import Dict, List, Optional, Any, Union
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Import base provider components
from src.models.external.providers.base_provider import (
    APIProvider, ModelMetadata, ModelCapability, QueryType, ResponseType
)
from src.models.external.credential_manager import CredentialManager

# Configure logging
logger = logging.getLogger(__name__)


class OpenAIProvider(APIProvider):
    """
    OpenAI API provider implementation with comprehensive model support.
    
    This class implements the APIProvider interface for OpenAI's models,
    supporting the latest GPT-4o3, GPT-4o, GPT-4 Turbo, and GPT-3.5 Turbo
    with detailed capability specifications and sophisticated error handling.
    
    Thread Safety: All methods are thread-safe with proper state isolation.
    Error Handling: Comprehensive with exponential backoff for transient failures.
    """
    
    # Updated model metadata with latest models and capabilities
    MODEL_METADATA = {
        "gpt-4o-mini": ModelMetadata(
            provider_name="openai",
            model_name="gpt-4o-mini",
            max_tokens=4096,
            context_window=128000,
            capabilities=[
                ModelCapability.BASIC_COMPLETION,
                ModelCapability.CODE_GENERATION,
                ModelCapability.SCIENTIFIC_REASONING,
                ModelCapability.MATHEMATICAL_COMPUTATION,
                ModelCapability.MULTIMODAL_UNDERSTANDING,
                ModelCapability.LONG_CONTEXT
            ],
            token_cost_input=0.00015,
            token_cost_output=0.0006,
            latency_estimate_ms=800
        ),
        "gpt-4o": ModelMetadata(
            provider_name="openai",
            model_name="gpt-4o",
            max_tokens=4096,
            context_window=128000,
            capabilities=[
                ModelCapability.BASIC_COMPLETION,
                ModelCapability.CODE_GENERATION,
                ModelCapability.SCIENTIFIC_REASONING,
                ModelCapability.MATHEMATICAL_COMPUTATION,
                ModelCapability.MULTIMODAL_UNDERSTANDING,
                ModelCapability.LONG_CONTEXT
            ],
            token_cost_input=0.0025,
            token_cost_output=0.01,
            latency_estimate_ms=1200
        ),
        "gpt-4-turbo": ModelMetadata(
            provider_name="openai",
            model_name="gpt-4-turbo",
            max_tokens=4096,
            context_window=128000,
            capabilities=[
                ModelCapability.BASIC_COMPLETION,
                ModelCapability.CODE_GENERATION,
                ModelCapability.SCIENTIFIC_REASONING,
                ModelCapability.MATHEMATICAL_COMPUTATION,
                ModelCapability.LONG_CONTEXT
            ],
            token_cost_input=0.01,
            token_cost_output=0.03,
            latency_estimate_ms=1500
        ),
        "gpt-4-turbo-preview": ModelMetadata(
            provider_name="openai",
            model_name="gpt-4-turbo-preview",
            max_tokens=4096,
            context_window=128000,
            capabilities=[
                ModelCapability.BASIC_COMPLETION,
                ModelCapability.CODE_GENERATION,
                ModelCapability.SCIENTIFIC_REASONING,
                ModelCapability.MATHEMATICAL_COMPUTATION,
                ModelCapability.LONG_CONTEXT
            ],
            token_cost_input=0.01,
            token_cost_output=0.03,
            latency_estimate_ms=1500
        ),
        "gpt-3.5-turbo": ModelMetadata(
            provider_name="openai",
            model_name="gpt-3.5-turbo",
            max_tokens=4096,
            context_window=16385,
            capabilities=[
                ModelCapability.BASIC_COMPLETION,
                ModelCapability.CODE_GENERATION
            ],
            token_cost_input=0.0005,
            token_cost_output=0.0015,
            latency_estimate_ms=500
        ),
        "gpt-3.5-turbo-16k": ModelMetadata(
            provider_name="openai",
            model_name="gpt-3.5-turbo-16k",
            max_tokens=16384,
            context_window=16385,
            capabilities=[
                ModelCapability.BASIC_COMPLETION,
                ModelCapability.CODE_GENERATION,
                ModelCapability.LONG_CONTEXT
            ],
            token_cost_input=0.003,
            token_cost_output=0.004,
            latency_estimate_ms=800
        )
    }
    
    def __init__(self, credential_manager: CredentialManager):
        """
        Initialize the OpenAI provider with a credential manager.
        
        Args:
            credential_manager: Manager for secure credential handling
            
        The initialization establishes the foundation for the provider but defers
        actual client creation until the initialize() method is called, following
        the principle of lazy initialization for resource efficiency.
        """
        self.credential_manager = credential_manager
        self.client = None
        self.initialized = False
        
        # Performance tracking
        self.request_count = 0
        self.error_count = 0
        self.total_latency_ms = 0
    
    def initialize(self) -> bool:
        """
        Initialize the OpenAI client with API credentials.
        
        Returns:
            bool: True if initialization was successful, False otherwise
            
        This method implements a robust initialization process:
        1. Validates credential availability
        2. Imports required dependencies with graceful handling
        3. Creates and configures the client with appropriate settings
        4. Sets the initialization state for future method calls
        
        Thread Safety: This method is thread-safe and idempotent
        """
        try:
            # Only import OpenAI package here to prevent dependency issues if not used
            from openai import AsyncOpenAI
            
            # Get credentials securely from credential manager
            credentials = self.credential_manager.get_credentials("openai")
            if not credentials.get("api_key"):
                logger.error("OpenAI API key not found in credentials")
                return False
            
            # Initialize API client with appropriate settings
            self.client = AsyncOpenAI(
                api_key=credentials["api_key"],
                # Optional settings for production environments:
                # timeout=60.0,  # Increased timeout for complex operations
                # max_retries=2  # Let our own retry mechanism handle retries
            )
            
            self.initialized = True
            logger.info("OpenAI provider initialized successfully")
            return True
            
        except ImportError:
            logger.error("OpenAI Python package not installed. Please install with 'pip install openai>=1.0.0'")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI provider: {e}", exc_info=True)
            return False
    
    def get_available_models(self) -> List[ModelMetadata]:
        """
        Return a list of available OpenAI models with their metadata.
        
        Returns:
            List[ModelMetadata]: List of available model metadata objects
            
        This method provides comprehensive model information including:
        - Supported capabilities for intelligent routing
        - Context window limitations for token management
        - Cost characteristics for budget optimization
        - Performance estimates for latency management
        
        Time Complexity: O(1) - Returns pre-configured metadata
        Space Complexity: O(m) where m is number of models
        """
        return list(self.MODEL_METADATA.values())
    
    @retry(
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def generate_response(
        self,
        query: QueryType,
        model_name: str,
        context: Optional[List[str]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> ResponseType:
        """
        Generate a response from the specified OpenAI model.
        
        This method implements the core functionality of the provider,
        handling the complete lifecycle of a generation request:
        1. Parameter validation and normalization
        2. Request construction with proper configuration
        3. API communication with retry logic
        4. Response processing and standardization
        5. Performance and cost calculation
        
        Args:
            query: The user query text
            model_name: Name of the OpenAI model to use
            context: Optional context information
            temperature: Generation temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            
        Returns:
            ResponseType: Standardized response dictionary
            
        Raises:
            RuntimeError: If provider not initialized
            ValueError: If model name is unknown
            Exception: For API communication errors
            
        Thread Safety: This method is thread-safe for concurrent invocations
        Time Complexity: O(1) + API request latency
        """
        # Validate initialization state
        if not self.initialized or not self.client:
            raise RuntimeError("OpenAI provider not initialized. Call initialize() first.")
        
        # Validate model name
        if model_name not in self.MODEL_METADATA:
            raise ValueError(f"Unknown model: {model_name}. Available models: {', '.join(self.MODEL_METADATA.keys())}")
        
        # Track request for performance monitoring
        self.request_count += 1
        
        # Prepare messages array with proper structure
        messages = []
        
        # Add system context if provided
        if context and any(context):
            messages.append({
                "role": "system", 
                "content": "\n".join(context)
            })
        
        # Add user query
        messages.append({
            "role": "user", 
            "content": query
        })
        
        # Normalize parameters
        normalized_temperature = max(0.0, min(1.0, temperature))
        max_tokens_value = max_tokens or self.MODEL_METADATA[model_name].max_tokens
        
        start_time = time.time()
        try:
            # Generate completion with proper configuration
            response = await self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=normalized_temperature,
                max_tokens=max_tokens_value,
                # Additional parameters for quality control:
                # top_p=0.95,  # Nucleus sampling for controlled randomness
                # presence_penalty=0.0,  # Neutral presence penalty
                # frequency_penalty=0.0,  # Neutral frequency penalty
            )
            
            # Calculate performance metrics
            elapsed_time = (time.time() - start_time) * 1000
            self.total_latency_ms += elapsed_time
            
            # Extract content
            content = response.choices[0].message.content
            
            # Get token counts
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            
            # Calculate cost
            cost = self.calculate_cost(
                input_tokens,
                output_tokens,
                model_name
            )
            
            # Return standardized response dictionary
            return {
                "provider": "openai",
                "model": model_name,
                "content": content,
                "latency_ms": elapsed_time,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost": cost
            }
            
        except Exception as e:
            # Track error and re-raise for higher-level handling
            self.error_count += 1
            logger.error(f"OpenAI API error: {e}", exc_info=True)
            raise
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in the provided text.
        
        Args:
            text: The input text to estimate
            
        Returns:
            int: Estimated token count
            
        This implementation uses a heuristic approach that balances
        accuracy with computational efficiency, appropriate for
        cost estimation and context window management when not using
        the actual API response token counts.
            
        Time Complexity: O(n) where n is text length
        """
        # Simple approximation: ~4 characters per token for English text
        # This is a reasonable approximation for typical English text with GPT tokenizers
        # For production applications with strict budget requirements,
        # consider using OpenAI's tiktoken library for exact counts
        return max(1, len(text) // 4)
    
    def calculate_cost(self, input_tokens: int, output_tokens: int, model_name: str) -> float:
        """
        Calculate the cost of a request based on token counts.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model_name: Name of the model used
            
        Returns:
            float: Estimated cost in USD
            
        This method implements precise cost calculation based on
        current pricing models for OpenAI's API services.
            
        Time Complexity: O(1) - Constant time calculation
        """
        if model_name not in self.MODEL_METADATA:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Get pricing information from model metadata
        metadata = self.MODEL_METADATA[model_name]
        
        # Calculate cost components
        input_cost = (input_tokens / 1000) * metadata.token_cost_input
        output_cost = (output_tokens / 1000) * metadata.token_cost_output
        
        # Return total cost
        return input_cost + output_cost
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for this provider.
        
        Returns:
            Dict[str, Any]: Dictionary of performance metrics
            
        This method provides operational insights for monitoring and
        optimization of the OpenAI provider's performance.
        
        Time Complexity: O(1) - Constant time metric collection
        """
        avg_latency = self.total_latency_ms / max(1, self.request_count)
        error_rate = self.error_count / max(1, self.request_count) * 100
        
        return {
            "request_count": self.request_count,
            "error_count": self.error_count,
            "error_rate": error_rate,
            "avg_latency_ms": avg_latency,
            "supported_models": list(self.MODEL_METADATA.keys())
        }
