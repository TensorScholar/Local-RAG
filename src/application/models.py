"""
APEX Universal Model Interface - Intelligent Model Abstraction

This module implements a universal model interface that provides intelligent
abstraction over multiple model providers with advanced error handling,
circuit breaker patterns, and automatic fallback mechanisms.

Author: APEX Development Team
Version: 1.0.0
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    TypeVar, Generic, Protocol, Union, Optional, 
    List, Dict, Any, Callable, Awaitable, Tuple
)
import asyncio
from datetime import datetime, timedelta
import aiohttp
import json
import hashlib
from contextlib import asynccontextmanager

from ..domain.models import (
    ModelProvider, ModelProviderProtocol, Result, Success, Failure,
    QueryContext, PerformanceMetrics, ModelDecision
)
from ..domain.reactive import Observable, Event

# Type variables
T = TypeVar('T')
U = TypeVar('U')

@dataclass
class ModelConfig:
    """Immutable model configuration"""
    provider: ModelProvider
    model_name: str
    api_key: str
    base_url: str
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0

@dataclass
class ModelResponse:
    """Immutable model response"""
    content: str
    model_used: str
    tokens_used: int
    latency_ms: float
    cost: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance"""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    async def call(self, func: Callable[..., Awaitable[T]], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection"""
        
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _on_success(self) -> None:
        """Handle successful call"""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def _on_failure(self) -> None:
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset"""
        if not self.last_failure_time:
            return True
        
        return datetime.utcnow() - self.last_failure_time > timedelta(seconds=self.timeout)

class RateLimiter:
    """Token bucket rate limiter"""
    
    def __init__(self, tokens_per_second: float, bucket_size: int):
        self.tokens_per_second = tokens_per_second
        self.bucket_size = bucket_size
        self.tokens = bucket_size
        self.last_refill = datetime.utcnow()
        self._lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1) -> None:
        """Acquire tokens from rate limiter"""
        async with self._lock:
            await self._refill_tokens()
            
            if self.tokens < tokens:
                # Calculate wait time
                wait_time = (tokens - self.tokens) / self.tokens_per_second
                await asyncio.sleep(wait_time)
                await self._refill_tokens()
            
            self.tokens -= tokens
    
    async def _refill_tokens(self) -> None:
        """Refill tokens based on time elapsed"""
        now = datetime.utcnow()
        time_elapsed = (now - self.last_refill).total_seconds()
        tokens_to_add = time_elapsed * self.tokens_per_second
        
        self.tokens = min(self.bucket_size, self.tokens + tokens_to_add)
        self.last_refill = now

class BaseModelProvider(ABC):
    """Abstract base class for model providers"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.circuit_breaker = CircuitBreaker()
        self.rate_limiter = RateLimiter(tokens_per_second=10, bucket_size=100)
        self.session: Optional[aiohttp.ClientSession] = None
        self.performance_metrics = Observable[PerformanceMetrics](name=f"{config.provider}_metrics")
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    @abstractmethod
    async def generate_response(
        self, 
        query: str, 
        context: List[str] = None,
        **kwargs
    ) -> Result[ModelResponse]:
        """Generate response from model"""
        ...
    
    @abstractmethod
    async def estimate_cost(self, query: str, context: List[str] = None) -> float:
        """Estimate cost for query"""
        ...
    
    async def check_availability(self) -> bool:
        """Check provider availability"""
        try:
            # Simple health check
            await self._make_request("GET", "/health", timeout=5)
            return True
        except Exception:
            return False
    
    async def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        data: Dict[str, Any] = None,
        timeout: int = None
    ) -> Dict[str, Any]:
        """Make HTTP request with retry logic"""
        
        async def _request() -> Dict[str, Any]:
            if not self.session:
                raise Exception("Session not initialized")
            
            url = f"{self.config.base_url}{endpoint}"
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json"
            }
            
            timeout_obj = aiohttp.ClientTimeout(total=timeout or self.config.timeout)
            
            async with self.session.request(
                method, url, json=data, headers=headers, timeout=timeout_obj
            ) as response:
                if response.status >= 400:
                    raise Exception(f"HTTP {response.status}: {await response.text()}")
                
                return await response.json()
        
        # Apply circuit breaker and rate limiting
        await self.rate_limiter.acquire()
        return await self.circuit_breaker.call(_request)

class OpenAIProvider(BaseModelProvider):
    """OpenAI model provider implementation"""
    
    async def generate_response(
        self, 
        query: str, 
        context: List[str] = None,
        **kwargs
    ) -> Result[ModelResponse]:
        """Generate response using OpenAI API"""
        
        try:
            start_time = datetime.utcnow()
            
            # Prepare request
            messages = []
            if context:
                messages.append({
                    "role": "system",
                    "content": "You are a helpful assistant. Use the following context to answer the question."
                })
                messages.append({
                    "role": "user",
                    "content": f"Context: {' '.join(context)}\n\nQuestion: {query}"
                })
            else:
                messages.append({
                    "role": "user",
                    "content": query
                })
            
            data = {
                "model": self.config.model_name,
                "messages": messages,
                "max_tokens": kwargs.get('max_tokens', self.config.max_tokens),
                "temperature": kwargs.get('temperature', self.config.temperature)
            }
            
            # Make request
            response = await self._make_request("POST", "/v1/chat/completions", data=data)
            
            # Calculate metrics
            end_time = datetime.utcnow()
            latency_ms = (end_time - start_time).total_seconds() * 1000
            
            content = response["choices"][0]["message"]["content"]
            tokens_used = response["usage"]["total_tokens"]
            cost = self._calculate_cost(tokens_used)
            
            model_response = ModelResponse(
                content=content,
                model_used=self.config.model_name,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                cost=cost,
                metadata=response
            )
            
            # Emit performance metrics
            metrics = PerformanceMetrics(
                latency_p50=latency_ms,
                latency_p95=latency_ms,
                latency_p99=latency_ms,
                throughput=1.0,
                error_rate=0.0,
                cost_per_query=cost
            )
            self.performance_metrics.emit_sync(Event(value=metrics))
            
            return Success(model_response)
            
        except Exception as e:
            return Failure(
                error=str(e),
                error_code="OPENAI_ERROR",
                context={"query": query, "model": self.config.model_name}
            )
    
    async def estimate_cost(self, query: str, context: List[str] = None) -> float:
        """Estimate cost for OpenAI query"""
        # Rough estimation based on token count
        total_tokens = len(query.split()) * 1.3  # Approximate token ratio
        
        if context:
            context_tokens = sum(len(c.split()) for c in context) * 1.3
            total_tokens += context_tokens
        
        # OpenAI pricing (approximate)
        input_cost_per_1k = 0.002  # $0.002 per 1K input tokens
        output_cost_per_1k = 0.002  # $0.002 per 1K output tokens
        
        estimated_output_tokens = total_tokens * 0.5  # Assume 50% output ratio
        
        total_cost = (
            (total_tokens / 1000) * input_cost_per_1k +
            (estimated_output_tokens / 1000) * output_cost_per_1k
        )
        
        return total_cost
    
    def _calculate_cost(self, tokens_used: int) -> float:
        """Calculate actual cost based on tokens used"""
        # OpenAI pricing (approximate)
        input_cost_per_1k = 0.002
        output_cost_per_1k = 0.002
        
        # Assume 70% input, 30% output
        input_tokens = int(tokens_used * 0.7)
        output_tokens = tokens_used - input_tokens
        
        total_cost = (
            (input_tokens / 1000) * input_cost_per_1k +
            (output_tokens / 1000) * output_cost_per_1k
        )
        
        return total_cost

class AnthropicProvider(BaseModelProvider):
    """Anthropic model provider implementation"""
    
    async def generate_response(
        self, 
        query: str, 
        context: List[str] = None,
        **kwargs
    ) -> Result[ModelResponse]:
        """Generate response using Anthropic API"""
        
        try:
            start_time = datetime.utcnow()
            
            # Prepare request
            system_prompt = "You are a helpful assistant."
            if context:
                system_prompt += f" Use the following context to answer questions: {' '.join(context)}"
            
            data = {
                "model": self.config.model_name,
                "max_tokens": kwargs.get('max_tokens', self.config.max_tokens),
                "temperature": kwargs.get('temperature', self.config.temperature),
                "system": system_prompt,
                "messages": [{"role": "user", "content": query}]
            }
            
            # Make request
            response = await self._make_request("POST", "/v1/messages", data=data)
            
            # Calculate metrics
            end_time = datetime.utcnow()
            latency_ms = (end_time - start_time).total_seconds() * 1000
            
            content = response["content"][0]["text"]
            tokens_used = response["usage"]["input_tokens"] + response["usage"]["output_tokens"]
            cost = self._calculate_cost(tokens_used)
            
            model_response = ModelResponse(
                content=content,
                model_used=self.config.model_name,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                cost=cost,
                metadata=response
            )
            
            # Emit performance metrics
            metrics = PerformanceMetrics(
                latency_p50=latency_ms,
                latency_p95=latency_ms,
                latency_p99=latency_ms,
                throughput=1.0,
                error_rate=0.0,
                cost_per_query=cost
            )
            self.performance_metrics.emit_sync(Event(value=metrics))
            
            return Success(model_response)
            
        except Exception as e:
            return Failure(
                error=str(e),
                error_code="ANTHROPIC_ERROR",
                context={"query": query, "model": self.config.model_name}
            )
    
    async def estimate_cost(self, query: str, context: List[str] = None) -> float:
        """Estimate cost for Anthropic query"""
        # Rough estimation based on token count
        total_tokens = len(query.split()) * 1.3
        
        if context:
            context_tokens = sum(len(c.split()) for c in context) * 1.3
            total_tokens += context_tokens
        
        # Anthropic pricing (approximate)
        input_cost_per_1k = 0.003  # $0.003 per 1K input tokens
        output_cost_per_1k = 0.015  # $0.015 per 1K output tokens
        
        estimated_output_tokens = total_tokens * 0.5
        
        total_cost = (
            (total_tokens / 1000) * input_cost_per_1k +
            (estimated_output_tokens / 1000) * output_cost_per_1k
        )
        
        return total_cost
    
    def _calculate_cost(self, tokens_used: int) -> float:
        """Calculate actual cost based on tokens used"""
        # Anthropic pricing (approximate)
        input_cost_per_1k = 0.003
        output_cost_per_1k = 0.015
        
        # Assume 70% input, 30% output
        input_tokens = int(tokens_used * 0.7)
        output_tokens = tokens_used - input_tokens
        
        total_cost = (
            (input_tokens / 1000) * input_cost_per_1k +
            (output_tokens / 1000) * output_cost_per_1k
        )
        
        return total_cost

class GoogleProvider(BaseModelProvider):
    """Google model provider implementation"""
    
    async def generate_response(
        self, 
        query: str, 
        context: List[str] = None,
        **kwargs
    ) -> Result[ModelResponse]:
        """Generate response using Google API"""
        
        try:
            start_time = datetime.utcnow()
            
            # Prepare request
            prompt = query
            if context:
                prompt = f"Context: {' '.join(context)}\n\nQuestion: {query}"
            
            data = {
                "model": self.config.model_name,
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "maxOutputTokens": kwargs.get('max_tokens', self.config.max_tokens),
                    "temperature": kwargs.get('temperature', self.config.temperature)
                }
            }
            
            # Make request
            response = await self._make_request("POST", "/v1/models/generateContent", data=data)
            
            # Calculate metrics
            end_time = datetime.utcnow()
            latency_ms = (end_time - start_time).total_seconds() * 1000
            
            content = response["candidates"][0]["content"]["parts"][0]["text"]
            tokens_used = response.get("usageMetadata", {}).get("totalTokenCount", 0)
            cost = self._calculate_cost(tokens_used)
            
            model_response = ModelResponse(
                content=content,
                model_used=self.config.model_name,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                cost=cost,
                metadata=response
            )
            
            # Emit performance metrics
            metrics = PerformanceMetrics(
                latency_p50=latency_ms,
                latency_p95=latency_ms,
                latency_p99=latency_ms,
                throughput=1.0,
                error_rate=0.0,
                cost_per_query=cost
            )
            self.performance_metrics.emit_sync(Event(value=metrics))
            
            return Success(model_response)
            
        except Exception as e:
            return Failure(
                error=str(e),
                error_code="GOOGLE_ERROR",
                context={"query": query, "model": self.config.model_name}
            )
    
    async def estimate_cost(self, query: str, context: List[str] = None) -> float:
        """Estimate cost for Google query"""
        # Rough estimation based on token count
        total_tokens = len(query.split()) * 1.3
        
        if context:
            context_tokens = sum(len(c.split()) for c in context) * 1.3
            total_tokens += context_tokens
        
        # Google pricing (approximate)
        input_cost_per_1k = 0.001  # $0.001 per 1K input tokens
        output_cost_per_1k = 0.002  # $0.002 per 1K output tokens
        
        estimated_output_tokens = total_tokens * 0.5
        
        total_cost = (
            (total_tokens / 1000) * input_cost_per_1k +
            (estimated_output_tokens / 1000) * output_cost_per_1k
        )
        
        return total_cost
    
    def _calculate_cost(self, tokens_used: int) -> float:
        """Calculate actual cost based on tokens used"""
        # Google pricing (approximate)
        input_cost_per_1k = 0.001
        output_cost_per_1k = 0.002
        
        # Assume 70% input, 30% output
        input_tokens = int(tokens_used * 0.7)
        output_tokens = tokens_used - input_tokens
        
        total_cost = (
            (input_tokens / 1000) * input_cost_per_1k +
            (output_tokens / 1000) * output_cost_per_1k
        )
        
        return total_cost

class UniversalModelInterface:
    """Universal model interface with intelligent abstraction"""
    
    def __init__(self):
        self.providers: Dict[ModelProvider, BaseModelProvider] = {}
        self.model_registry: Dict[str, ModelConfig] = {}
        self.performance_events = Observable[PerformanceMetrics](name="model_performance")
        self._setup_providers()
    
    def _setup_providers(self) -> None:
        """Setup model providers"""
        # This would be loaded from configuration
        pass
    
    async def generate_response(
        self, 
        query: str, 
        context: List[str] = None,
        model_preference: Optional[str] = None,
        auto_fallback: bool = True
    ) -> Result[ModelResponse]:
        """Universal response generation with intelligent model selection"""
        
        # Auto-select model if not specified
        if not model_preference:
            model_preference = await self._select_optimal_model(query, context)
        
        # Attempt primary model
        try:
            provider = self._get_provider_for_model(model_preference)
            response = await provider.generate_response(
                query=query,
                context=context
            )
            
            if isinstance(response, Success):
                # Emit performance event
                metrics = PerformanceMetrics(
                    latency_p50=response.value.latency_ms,
                    latency_p95=response.value.latency_ms,
                    latency_p99=response.value.latency_ms,
                    throughput=1.0,
                    error_rate=0.0,
                    cost_per_query=response.value.cost
                )
                self.performance_events.emit_sync(Event(value=metrics))
            
            return response
            
        except Exception as e:
            if auto_fallback:
                # Intelligent fallback to alternative model
                fallback_model = await self._select_fallback_model(
                    original_model=model_preference,
                    error=e
                )
                return await self.generate_response(
                    query=query,
                    context=context,
                    model_preference=fallback_model,
                    auto_fallback=False  # Prevent infinite fallback
                )
            raise
    
    async def _select_optimal_model(self, query: str, context: List[str] = None) -> str:
        """Select optimal model based on query characteristics"""
        # Placeholder for intelligent model selection
        # In production, this would use ML-based selection
        return "gpt-4"
    
    async def _select_fallback_model(self, original_model: str, error: Exception) -> str:
        """Select fallback model based on error type"""
        # Placeholder for intelligent fallback selection
        # In production, this would use error analysis
        return "claude-3-sonnet"
    
    def _get_provider_for_model(self, model_name: str) -> BaseModelProvider:
        """Get provider for model name"""
        # Placeholder for provider mapping
        # In production, this would use model registry
        raise ValueError(f"Model {model_name} not found")
    
    def register_provider(self, provider: BaseModelProvider) -> None:
        """Register a model provider"""
        self.providers[provider.config.provider] = provider
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        return list(self.model_registry.keys())
    
    def subscribe_to_performance(self, observer: Callable[[PerformanceMetrics], None]) -> 'Subscription':
        """Subscribe to performance events"""
        return self.performance_events.subscribe(lambda event: observer(event.value))
