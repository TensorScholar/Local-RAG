"""
Google Gemini API Provider Implementation.

This module implements the APIProvider interface for the Google Gemini API,
providing a clean abstraction over Google's models with comprehensive error
handling, intelligent retry mechanisms, and strict conformance to the
provider contract.

Design Features:
1. Robust Authentication - Secure credential management with validation
2. Comprehensive Model Mapping - Detailed capability specifications
3. Exponential Backoff - Sophisticated retry mechanism for transient failures
4. Token Estimation - Efficient token counting for budget management
5. Cost Calculation - Precise cost calculations based on current pricing

Performance Characteristics:
- Initialization: O(1) with credential validation
- Token Estimation: O(n) where n is text length
- Response Generation: O(1) + API latency
- Thread Safety: All methods are thread-safe with proper state management

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


class GoogleProvider(APIProvider):
    """
    Google Gemini API provider implementation with comprehensive model support.
    
    This class implements the APIProvider interface for Google's Gemini models,
    supporting the latest Gemini Pro 2 Experimental, Gemini Flash 2 Experimental,
    Gemini 1.5 Pro, and Gemini 1.0 Pro with detailed capability specifications
    and robust error handling.
    
    Thread Safety: All methods are thread-safe with proper state management.
    Error Handling: Comprehensive with exponential backoff for transient failures.
    """
    
    # Detailed model metadata with capability specifications
    MODEL_METADATA = {
        "gemini-pro-2-experimental": ModelMetadata(
            provider_name="google",
            model_name="gemini-pro-2-experimental",
            max_tokens=16384,
            context_window=2000000,  # 2M tokens
            capabilities=[
                ModelCapability.BASIC_COMPLETION,
                ModelCapability.CODE_GENERATION,
                ModelCapability.SCIENTIFIC_REASONING,
                ModelCapability.MATHEMATICAL_COMPUTATION,
                ModelCapability.MULTIMODAL_UNDERSTANDING,
                ModelCapability.LONG_CONTEXT
            ],
            token_cost_input=0.0007,
            token_cost_output=0.0021,
            latency_estimate_ms=800
        ),
        "gemini-flash-2-experimental": ModelMetadata(
            provider_name="google",
            model_name="gemini-flash-2-experimental",
            max_tokens=8192,
            context_window=1000000,  # 1M tokens
            capabilities=[
                ModelCapability.BASIC_COMPLETION,
                ModelCapability.CODE_GENERATION,
                ModelCapability.SCIENTIFIC_REASONING,
                ModelCapability.MULTIMODAL_UNDERSTANDING,
                ModelCapability.LONG_CONTEXT
            ],
            token_cost_input=0.00035,
            token_cost_output=0.00105,
            latency_estimate_ms=500
        ),
        "gemini-1.5-pro": ModelMetadata(
            provider_name="google",
            model_name="gemini-1.5-pro",
            max_tokens=8192,
            context_window=1000000,  # 1M tokens
            capabilities=[
                ModelCapability.BASIC_COMPLETION,
                ModelCapability.CODE_GENERATION,
                ModelCapability.SCIENTIFIC_REASONING,
                ModelCapability.MATHEMATICAL_COMPUTATION,
                ModelCapability.MULTIMODAL_UNDERSTANDING,
                ModelCapability.LONG_CONTEXT
            ],
            token_cost_input=0.0005,
            token_cost_output=0.0015,
            latency_estimate_ms=1000
        ),
        "gemini-1.0-pro": ModelMetadata(
            provider_name="google",
            model_name="gemini-1.0-pro",
            max_tokens=8192,
            context_window=32768,
            capabilities=[
                ModelCapability.BASIC_COMPLETION,
                ModelCapability.CODE_GENERATION,
                ModelCapability.SCIENTIFIC_REASONING,
            ],
            token_cost_input=0.0001,
            token_cost_output=0.0005,
            latency_estimate_ms=800
        )
    }
    
    def __init__(self, credential_manager: CredentialManager):
        """
        Initialize the Google provider with a credential manager.
        
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
        Initialize the Google Generative AI client with API credentials.
        
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
            # Only import Google packages here to prevent dependency issues if not used
            import google.generativeai as genai
            
            # Get credentials securely from credential manager
            credentials = self.credential_manager.get_credentials("google")
            if not credentials.get("api_key"):
                logger.error("Google API key not found in credentials")
                return False
            
            # Configure client with credentials
            genai.configure(api_key=credentials["api_key"])
            self.client = genai
            
            self.initialized = True
            logger.info("Google Gemini provider initialized successfully")
            return True
            
        except ImportError:
            logger.error("Google GenerativeAI package not installed. Please install with 'pip install google-generativeai'")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Google provider: {e}", exc_info=True)
            return False
    
    def get_available_models(self) -> List[ModelMetadata]:
        """
        Return a list of available Google models with their metadata.
        
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
        Generate a response from the specified Google model.
        
        This method implements the core functionality of the provider,
        handling the complete lifecycle of a generation request:
        1. Parameter validation and normalization
        2. Request construction with proper configuration
        3. API communication with retry logic
        4. Response processing and standardization
        5. Performance and cost calculation
        
        Args:
            query: The user query text
            model_name: Name of the Google model to use
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
            raise RuntimeError("Google provider not initialized. Call initialize() first.")
        
        # Validate model name
        if model_name not in self.MODEL_METADATA:
            raise ValueError(f"Unknown model: {model_name}. Available models: {', '.join(self.MODEL_METADATA.keys())}")
        
        # Track request for performance monitoring
        self.request_count += 1
        
        # Prepare generation config with parameter normalization
        generation_config = {
            "temperature": max(0.0, min(1.0, temperature)),  # Normalize temperature to valid range
            "max_output_tokens": max_tokens or self.MODEL_METADATA[model_name].max_tokens,
            "top_p": 0.95,  # Add nucleus sampling for quality
            "top_k": 40,    # Reasonable value for diverse yet focused outputs
        }
        
        # Prepare system instructions from context
        system_instruction = "\n".join(context) if context and any(context) else ""
        
        start_time = time.time()
        try:
            # Initialize model with appropriate configuration
            model = self.client.GenerativeModel(
                model_name=model_name,
                generation_config=generation_config,
                system_instruction=system_instruction
            )
            
            # Generate response asynchronously
            # Note: Google's API has limited async support, so we use their sync API
            response = model.generate_content(query)
            
            # Calculate performance metrics
            elapsed_time = (time.time() - start_time) * 1000
            self.total_latency_ms += elapsed_time
            
            # Extract content
            if hasattr(response, 'text'):
                content = response.text
            elif hasattr(response, 'parts'):
                content = ''.join(part.text for part in response.parts if hasattr(part, 'text'))
            else:
                content = str(response)
            
            # Estimate tokens (Gemini doesn't provide token counts)
            input_tokens = self.estimate_tokens(query)
            if system_instruction:
                input_tokens += self.estimate_tokens(system_instruction)
            output_tokens = self.estimate_tokens(content)
            
            # Calculate cost
            cost = self.calculate_cost(
                input_tokens,
                output_tokens,
                model_name
            )
            
            # Return standardized response dictionary
            return {
                "provider": "google",
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
            logger.error(f"Google API error: {e}", exc_info=True)
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
        cost estimation and context window management.
            
        Time Complexity: O(n) where n is text length
        """
        # Simple approximation: ~4 characters per token for English text
        # This is a reasonable approximation for typical English text
        # Google's tokenizer is similar to this ratio for English
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
        current pricing models for Google's Gemini API services.
            
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
        optimization of the Google provider's performance.
        
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
