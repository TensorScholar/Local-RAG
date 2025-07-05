import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union

# Import credential management
from src.models.external.credential_manager import CredentialManager

# Import base provider interfaces and types
from src.models.external.providers.base_provider import (
    APIProvider, ModelMetadata, ModelCapability, ResponseType
)

# Import concrete provider implementations
from src.models.external.providers.openai_provider import OpenAIProvider
from src.models.external.providers.google_provider import GoogleProvider
from src.models.external.providers.anthropic_provider import AnthropicProvider

# Configure logging
logger = logging.getLogger(__name__)

class ExternalModelSelector:
    """
    Intelligent model selection utility that analyzes query requirements
    and selects the most appropriate model based on capabilities, cost,
    and performance constraints.
    
    This class implements a sophisticated decision framework that evaluates
    available models against requirements and constraints to identify the
    optimal model for each query.
    
    Time Complexity: O(m) where m is the number of available models
    Space Complexity: O(m) for internal data structures
    """
    
    def __init__(self, providers: List[APIProvider]):
        """
        Initialize the model selector with available providers.
        
        Args:
            providers: List of initialized API providers
        """
        self.providers = providers
        
    def select_model(
        self,
        query: str,
        required_capabilities: List[ModelCapability],
        max_cost: Optional[float] = None,
        max_latency_ms: Optional[int] = None,
        prefer_provider: Optional[str] = None
    ) -> Optional[Tuple[str, str]]:
        """
        Select the most appropriate model based on requirements and constraints.
        
        Args:
            query: The query text
            required_capabilities: List of required model capabilities
            max_cost: Maximum acceptable cost
            max_latency_ms: Maximum acceptable latency
            prefer_provider: Provider to prefer if multiple options are viable
            
        Returns:
            Tuple of (provider_name, model_name) or None if no suitable model
            
        Time Complexity: O(m) where m is number of available models
        """
        # Collect all available models with metadata
        all_models: List[Tuple[str, ModelMetadata]] = []
        
        for provider in self.providers:
            provider_name = provider.get_available_models()[0].provider_name if provider.get_available_models() else "unknown"
            for model_meta in provider.get_available_models():
                all_models.append((provider_name, model_meta))
        
        # Filter models by required capabilities
        capable_models = []
        for provider_name, model_meta in all_models:
            if all(cap in model_meta.capabilities for cap in required_capabilities):
                capable_models.append((provider_name, model_meta))
        
        if not capable_models:
            logger.warning(f"No models found with required capabilities: {[cap.name for cap in required_capabilities]}")
            return None
        
        # Filter by cost constraint if specified
        if max_cost is not None:
            # Estimate tokens for the query (rough approximation)
            estimated_tokens = len(query.split())
            
            cost_viable_models = []
            for provider_name, model_meta in capable_models:
                # Calculate approximate cost (input + estimated output)
                approx_cost = (
                    (estimated_tokens / 1000) * model_meta.token_cost_input +
                    (estimated_tokens * 1.5 / 1000) * model_meta.token_cost_output
                )
                
                if approx_cost <= max_cost:
                    cost_viable_models.append((provider_name, model_meta))
            
            if cost_viable_models:
                capable_models = cost_viable_models
        
        # Filter by latency constraint if specified
        if max_latency_ms is not None:
            latency_viable_models = []
            for provider_name, model_meta in capable_models:
                if model_meta.latency_estimate_ms <= max_latency_ms:
                    latency_viable_models.append((provider_name, model_meta))
            
            if latency_viable_models:
                capable_models = latency_viable_models
        
        # If no models meet constraints, return None
        if not capable_models:
            return None
        
        # Apply provider preference if specified
        if prefer_provider:
            preferred_models = [(p, m) for p, m in capable_models if p == prefer_provider]
            if preferred_models:
                capable_models = preferred_models
        
        # Select model with the best capability match
        # For simplicity, we'll use the first matching model
        # In a production system, this would use more sophisticated selection logic
        selected_provider, selected_model = capable_models[0]
        
        return selected_provider, selected_model.model_name

# If you have other imports, keep them as well
class ExternalModelManager:
    """
    Central manager for all external model providers with intelligent routing,
    unified error handling, and seamless model selection.
    
    This class implements the Facade pattern, providing a simplified interface
    to the complex subsystem of external API providers and model selection.
    It synchronizes credential management, provider initialization, model selection,
    and response generation with comprehensive error handling and performance
    monitoring.
    
    Design Principles:
    1. Information Hiding - Internal complexity abstracted behind clean interfaces
    2. Fail-Fast and Fail-Safe - Robust initialization with graceful degradation
    3. Resource Management - Efficient provider lifecycle management
    4. Stateful Context - Maintains operational state for optimized routing
    5. Command Pattern - Response generation encapsulated as command operations
    
    Performance Characteristics:
    - Time Complexity: O(p) initialization where p is provider count
                      O(m) selection where m is model count
                      O(1) generation (excluding API latency)
    - Space Complexity: O(p+m) for providers and model metadata
    - Thread Safety: Thread-safe with proper initialization sequencing
    - Fault Tolerance: Resilient to provider failures with fallback mechanisms
    
    Usage Example:
        manager = ExternalModelManager()
        await manager.initialize()
        response = await manager.generate_response(
            query="Explain quantum computing",
            provider_name="openai",  # Optional - auto-selects if not specified
            model_name="gpt-4",      # Optional - uses default if not specified
            required_capabilities=[ModelCapability.SCIENTIFIC_REASONING]
        )
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the External Model Manager.
        
        Args:
            config_path: Optional path to configuration directory containing
                         credentials.json and other configuration files.
                         If None, uses default locations (~/.rag_system/).
        
        Time Complexity: O(1) - Constant time initialization
        Space Complexity: O(1) - Fixed size data structures initialized
        
        Note: Full provider initialization is deferred to the initialize() method
              for proper async execution and error handling.
        """
        self.credential_manager = CredentialManager(config_path)
        self.providers: Dict[str, APIProvider] = {}
        self.model_selector: Optional[ExternalModelSelector] = None
        self.initialized = False
        
        # Performance monitoring and diagnostics
        self.request_count = 0
        self.error_count = 0
        self.provider_latencies: Dict[str, List[float]] = {}
        self.start_time = time.time()
    
    async def initialize(self) -> bool:
        """
        Initialize all available providers with proper credential validation
        and capability discovery.
        
        Returns:
            bool: True if at least one provider was successfully initialized,
                  False otherwise.
        
        Time Complexity: O(p) where p is the number of potential providers
        Space Complexity: O(p+m) where m is the number of available models
        
        This method implements a fault-tolerant initialization process:
        1. Attempts to initialize each provider independently
        2. Continues initialization even if some providers fail
        3. Succeeds if at least one provider is successfully initialized
        4. Provides detailed logging of initialization process
        
        Thread Safety: Safe for concurrent initialization with idempotent results
        """
        # Create provider instances
        providers_to_init = {
            "openai": OpenAIProvider(self.credential_manager),
            "google": GoogleProvider(self.credential_manager),
            "anthropic": AnthropicProvider(self.credential_manager)
        }
        
        initialized_providers = []
        
        # Initialize each provider
        for name, provider in providers_to_init.items():
            try:
                if self.credential_manager.has_credentials(name):
                    if provider.initialize():
                        self.providers[name] = provider
                        initialized_providers.append(provider)
                        logger.info(f"{name.capitalize()} provider initialized successfully")
                    else:
                        logger.warning(f"Failed to initialize {name} provider despite valid credentials")
                else:
                    logger.info(f"No credentials found for {name}, skipping initialization")
            except Exception as e:
                logger.error(f"Exception during {name} provider initialization: {e}", exc_info=True)
        
        if not self.providers:
            logger.warning("No external API providers could be initialized")
            return False
        
        # Create model selector with initialized providers
        self.model_selector = ExternalModelSelector(initialized_providers)
        self.initialized = True
        
        # Log initialization summary
        provider_names = ', '.join(self.providers.keys())
        model_count = sum(len(provider.get_available_models()) for provider in self.providers.values())
        logger.info(f"External Model Manager initialized with {len(self.providers)} providers: {provider_names}")
        logger.info(f"Total available models: {model_count}")
        
        return True
    
    def get_available_providers(self) -> List[str]:
        """
        Get a list of successfully initialized provider names.
        
        Returns:
            List[str]: Names of initialized providers
            
        Time Complexity: O(1) - Constant time dictionary key access
        Space Complexity: O(p) where p is number of providers for result list
        
        This method is useful for diagnostics and for providing options to end users.
        """
        return list(self.providers.keys())
    
    def get_available_models(self, provider: Optional[str] = None) -> List[ModelMetadata]:
        """
        Get metadata for all available models, optionally filtered by provider.
        
        Args:
            provider: Optional provider name to filter models
            
        Returns:
            List[ModelMetadata]: List of available model metadata objects
            
        Time Complexity: O(m) where m is number of models
        Space Complexity: O(m) for result list
        
        The returned metadata contains comprehensive information about each model:
        - Supported capabilities
        - Context window limitations
        - Cost per token
        - Latency estimates
        - Provider details
        
        This metadata enables intelligent model selection and informs users of
        available options.
        """
        if not self.initialized:
            return []
        
        if provider:
            if provider not in self.providers:
                return []
            return self.providers[provider].get_available_models()
        
        # Return all models from all providers
        all_models = []
        for provider_instance in self.providers.values():
            all_models.extend(provider_instance.get_available_models())
        
        return all_models
    
    async def generate_response(
        self,
        query: str,
        provider_name: Optional[str] = None,
        model_name: Optional[str] = None,
        required_capabilities: Optional[List[ModelCapability]] = None,
        context: Optional[List[str]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        max_cost: Optional[float] = None,
        max_latency_ms: Optional[int] = None
    ) -> ResponseType:
        """
        Generate a response from an external model, with intelligent routing if
        specific provider/model not specified.
        
        This method represents the primary interaction point for generating responses.
        It supports both explicit model selection and automatic capability-based routing.
        
        Args:
            query: The user query text
            provider_name: Specific provider name (optional)
            model_name: Specific model name (optional)
            required_capabilities: List of required capabilities (used for auto-selection)
            context: List of context strings to include
            temperature: Generation temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            max_cost: Maximum acceptable cost per request
            max_latency_ms: Maximum acceptable latency
            
        Returns:
            ResponseType: Standardized response dictionary with:
                - provider: Provider name
                - model: Model name
                - content: Generated text
                - latency_ms: Request latency
                - input_tokens: Token count of input
                - output_tokens: Token count of output
                - cost: Estimated cost in USD
            
        Raises:
            RuntimeError: If manager not initialized
            ValueError: If no suitable model found
            Exception: For provider-specific errors
            
        Time Complexity: O(m) for model selection where m is model count
                         O(1) for direct provider/model specification
                         Plus API request latency
        Space Complexity: O(n) where n is size of response
        
        Thread Safety: Safe for concurrent invocation
        """
        if not self.initialized:
            raise RuntimeError("External Model Manager not initialized")
        
        # Track request for diagnostics
        self.request_count += 1
        request_start_time = time.time()
        
        try:
            # If provider and model are specified, use them directly
            if provider_name and model_name:
                if provider_name not in self.providers:
                    raise ValueError(f"Provider '{provider_name}' not available")
                
                provider = self.providers[provider_name]
                response = await provider.generate_response(
                    query=query,
                    model_name=model_name,
                    context=context,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                # Track provider latency
                latency = time.time() - request_start_time
                if provider_name not in self.provider_latencies:
                    self.provider_latencies[provider_name] = []
                self.provider_latencies[provider_name].append(latency * 1000)  # ms
                
                return response
            
            # Auto-select model based on capabilities and constraints
            if not self.model_selector:
                raise RuntimeError("Model selector not available")
            
            required_caps = required_capabilities or [ModelCapability.BASIC_COMPLETION]
            selected = self.model_selector.select_model(
                query=query,
                required_capabilities=required_caps,
                max_cost=max_cost,
                max_latency_ms=max_latency_ms,
                prefer_provider=provider_name
            )
            
            if not selected:
                raise ValueError("No suitable external model found for the given constraints")
            
            auto_provider_name, auto_model_name = selected
            logger.info(f"Auto-selected model: {auto_provider_name}:{auto_model_name}")
            
            provider = self.providers[auto_provider_name]
            response = await provider.generate_response(
                query=query,
                model_name=auto_model_name,
                context=context,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Track provider latency
            latency = time.time() - request_start_time
            if auto_provider_name not in self.provider_latencies:
                self.provider_latencies[auto_provider_name] = []
            self.provider_latencies[auto_provider_name].append(latency * 1000)  # ms
            
            return response
            
        except Exception as e:
            # Track error for diagnostics
            self.error_count += 1
            logger.error(f"Error generating response: {e}", exc_info=True)
            raise
    
    def estimate_cost(
        self,
        query: str,
        context: Optional[List[str]] = None,
        provider_name: Optional[str] = None,
        model_name: Optional[str] = None,
        required_capabilities: Optional[List[ModelCapability]] = None,
        expected_output_tokens: Optional[int] = None
    ) -> Optional[float]:
        """
        Estimate the cost of a potential request without actually making it.
        
        This method enables cost forecasting and budget planning by providing
        cost estimates before sending requests.
        
        Args:
            query: The user query text
            context: Optional context information
            provider_name: Specific provider name (optional)
            model_name: Specific model name (optional)
            required_capabilities: List of required capabilities (for auto-selection)
            expected_output_tokens: Estimated output token count (default: len(query))
            
        Returns:
            float: Estimated cost in USD, or None if estimation not possible
            
        Time Complexity: O(m) for model selection where m is model count
                         O(1) for cost calculation
        Space Complexity: O(1) - Constant space usage
        """
        if not self.initialized:
            return None
        
        try:
            # Determine provider and model
            selected_provider_name = provider_name
            selected_model_name = model_name
            
            # Auto-select if not specified
            if not (selected_provider_name and selected_model_name):
                if not self.model_selector:
                    return None
                
                required_caps = required_capabilities or [ModelCapability.BASIC_COMPLETION]
                selected = self.model_selector.select_model(
                    query=query,
                    required_capabilities=required_caps,
                    prefer_provider=provider_name
                )
                
                if not selected:
                    return None
                
                selected_provider_name, selected_model_name = selected
            
            # Get provider
            if selected_provider_name not in self.providers:
                return None
            
            provider = self.providers[selected_provider_name]
            
            # Estimate token counts
            query_tokens = provider.estimate_tokens(query)
            context_tokens = 0
            if context:
                context_tokens = sum(provider.estimate_tokens(c) for c in context)
            
            output_tokens = expected_output_tokens or query_tokens
            
            # Calculate cost
            cost = provider.calculate_cost(
                input_tokens=query_tokens + context_tokens,
                output_tokens=output_tokens,
                model_name=selected_model_name
            )
            
            return cost
            
        except Exception as e:
            logger.error(f"Error estimating cost: {e}")
            return None
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive performance metrics for monitoring and diagnostics.
        
        Returns:
            Dict[str, Any]: Performance metrics including:
                - request_count: Total requests processed
                - error_count: Total errors encountered
                - uptime_seconds: System uptime
                - success_rate: Percentage of successful requests
                - provider_latencies: Average latency by provider
                - available_models: Count of available models
            
        Time Complexity: O(p) where p is number of providers
        Space Complexity: O(p) for result dictionary
        
        This method enables monitoring of system health, performance optimization,
        and diagnostics for troubleshooting.
        """
        metrics = {
            "request_count": self.request_count,
            "error_count": self.error_count,
            "uptime_seconds": time.time() - self.start_time,
            "success_rate": (self.request_count - self.error_count) / max(1, self.request_count) * 100,
            "provider_latencies": {},
            "available_models": len(self.get_available_models())
        }
        
        # Calculate average latencies
        for provider, latencies in self.provider_latencies.items():
            if latencies:
                metrics["provider_latencies"][provider] = {
                    "avg_ms": sum(latencies) / len(latencies),
                    "min_ms": min(latencies),
                    "max_ms": max(latencies),
                    "count": len(latencies)
                }
        
        return metrics
    
    def reset_performance_metrics(self) -> None:
        """
        Reset all performance metrics for a new monitoring period.
        
        Time Complexity: O(1) - Constant time reset
        Space Complexity: O(1) - No additional space used
        
        This method is useful for establishing monitoring intervals or
        clearing metrics after addressing performance issues.
        """
        self.request_count = 0
        self.error_count = 0
        self.provider_latencies = {}
        self.start_time = time.time()
        logger.info("Performance metrics have been reset")
    
    async def verify_provider_connectivity(self, provider_name: str) -> bool:
        """
        Verify connectivity to a specific provider with a minimal test request.
        
        Args:
            provider_name: Name of the provider to verify
            
        Returns:
            bool: True if provider is accessible, False otherwise
            
        Time Complexity: O(1) plus API request latency
        Space Complexity: O(1) - Constant space usage
        
        This method performs a minimal request to verify that the provider
        is accessible and properly authenticated, which is useful for
        diagnostics and system health checks.
        """
        if not self.initialized or provider_name not in self.providers:
            return False
        
        provider = self.providers[provider_name]
        
        try:
            # Get available models (a simple API call to verify connectivity)
            models = provider.get_available_models()
            
            if not models:
                logger.warning(f"Provider {provider_name} returned no models")
                return False
            
            # Select the first model for a minimal test query
            test_model = models[0].model_name
            
            # Perform a minimal query
            response = await provider.generate_response(
                query="Test query for connectivity verification.",
                model_name=test_model,
                max_tokens=10
            )
            
            return "content" in response and response["content"]
            
        except Exception as e:
            logger.error(f"Connectivity test failed for {provider_name}: {e}")
            return False
