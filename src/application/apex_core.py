"""
APEX Core Integration Layer - Intelligent Orchestration Engine

This module implements the core APEX integration layer that orchestrates
all components with intelligent request handling, advanced caching,
and comprehensive monitoring.

Author: Mohammad Atashi (mohammadaliatashi@icloud.com)
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
import hashlib
import json
import uuid
from pathlib import Path
from collections import defaultdict

from ..domain.models import (
    QueryContext, ModelDecision, ModelProvider, 
    QueryComplexity, PerformanceTier, Result, Success, Failure,
    PerformanceMetrics, UserId, SessionId, QueryId, APEXConfig
)
from ..domain.reactive import Observable, Event, StateManager, compose, memoize
from .router import IntelligentRouter
from .models import UniversalModelInterface, ModelResponse
from .autoconfig import ConfigurationManager

# Type variables
T = TypeVar('T')
U = TypeVar('U')

@dataclass
class APEXRequest:
    """Immutable APEX request with comprehensive metadata"""
    query_id: QueryId
    user_id: UserId
    session_id: SessionId
    query_text: str
    context: QueryContext
    retrieval_context: List[str] = field(default_factory=list)
    model_preference: Optional[str] = None
    auto_fallback: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class APEXResponse:
    """Immutable APEX response with comprehensive results"""
    query_id: QueryId
    content: str
    model_used: str
    route_decision: ModelDecision
    performance_metrics: PerformanceMetrics
    processing_time_ms: float
    cost: float
    confidence_score: float
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

class APEXCache:
    """Advanced caching system with intelligent invalidation"""
    
    def __init__(self, max_size: int = 1000, ttl_hours: int = 1):
        self.max_size = max_size
        self.ttl = timedelta(hours=ttl_hours)
        self.cache: Dict[str, Tuple[Any, datetime]] = {}
        self.access_count: Dict[str, int] = defaultdict(int)
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache with access tracking"""
        async with self._lock:
            if key in self.cache:
                value, timestamp = self.cache[key]
                
                # Check if expired
                if datetime.utcnow() - timestamp > self.ttl:
                    del self.cache[key]
                    return None
                
                # Update access count
                self.access_count[key] += 1
                return value
            
            return None
    
    async def set(self, key: str, value: Any) -> None:
        """Set value in cache with size management"""
        async with self._lock:
            # Evict least recently used if cache is full
            if len(self.cache) >= self.max_size:
                await self._evict_lru()
            
            self.cache[key] = (value, datetime.utcnow())
    
    async def _evict_lru(self) -> None:
        """Evict least recently used items"""
        if not self.cache:
            return
        
        # Find item with lowest access count
        lru_key = min(self.access_count.keys(), key=lambda k: self.access_count[k])
        del self.cache[lru_key]
        del self.access_count[lru_key]
    
    def generate_key(self, request: APEXRequest) -> str:
        """Generate cache key for request"""
        # Create deterministic key based on request content
        key_data = {
            'query': request.query_text,
            'user_id': request.user_id,
            'expertise_level': request.context.expertise_level.value,
            'performance_target': request.context.performance_target.value,
            'retrieval_context_hash': hashlib.md5(
                json.dumps(request.retrieval_context, sort_keys=True).encode()
            ).hexdigest()[:8]
        }
        
        return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()

class APEXOrchestrator:
    """Intelligent orchestration engine for APEX"""
    
    def __init__(self, config_path: Path = None):
        # Initialize components
        self.config_manager = ConfigurationManager(config_path)
        self.router = IntelligentRouter()
        self.model_interface = UniversalModelInterface()
        self.cache = APEXCache()
        
        # Reactive state management
        self.request_events = Observable[APEXRequest](name="request_events")
        self.response_events = Observable[APEXResponse](name="response_events")
        self.error_events = Observable[Failure](name="error_events")
        self.performance_events = Observable[PerformanceMetrics](name="performance_events")
        
        # Performance tracking
        self.performance_history: List[PerformanceMetrics] = []
        self.request_count = 0
        self.success_count = 0
        self.error_count = 0
        
        # Start monitoring
        asyncio.create_task(self._monitor_performance())
    
    async def process_request(self, request: APEXRequest) -> Result[APEXResponse]:
        """Process APEX request with intelligent orchestration"""
        
        try:
            start_time = datetime.utcnow()
            self.request_count += 1
            
            # Emit request event
            await self.request_events.emit(Event(value=request))
            
            # Check cache first
            cache_key = self.cache.generate_key(request)
            cached_response = await self.cache.get(cache_key)
            
            if cached_response:
                # Return cached response
                return Success(cached_response)
            
            # Route request
            route_result = await self.router.route_request(
                query=request.query_text,
                context=request.context
            )
            
            if isinstance(route_result, Failure):
                self.error_count += 1
                await self.error_events.emit(Event(value=route_result))
                return route_result
            
            route_decision = route_result.value
            
            # Generate response using model interface
            model_result = await self.model_interface.generate_response(
                query=request.query_text,
                context=request.retrieval_context,
                model_preference=request.model_preference,
                auto_fallback=request.auto_fallback
            )
            
            if isinstance(model_result, Failure):
                self.error_count += 1
                await self.error_events.emit(Event(value=model_result))
                return model_result
            
            model_response = model_result.value
            
            # Calculate performance metrics
            end_time = datetime.utcnow()
            processing_time_ms = (end_time - start_time).total_seconds() * 1000
            
            # Create performance metrics
            performance_metrics = PerformanceMetrics(
                latency_p50=processing_time_ms,
                latency_p95=processing_time_ms,
                latency_p99=processing_time_ms,
                throughput=1.0,
                error_rate=0.0,
                cost_per_query=model_response.cost
            )
            
            # Create APEX response
            apex_response = APEXResponse(
                query_id=request.query_id,
                content=model_response.content,
                model_used=model_response.model_used,
                route_decision=route_decision,
                performance_metrics=performance_metrics,
                processing_time_ms=processing_time_ms,
                cost=model_response.cost,
                confidence_score=route_decision.confidence_score,
                metadata={
                    'cache_hit': False,
                    'route_decision': route_decision.__dict__,
                    'model_response': model_response.__dict__
                }
            )
            
            # Cache response
            await self.cache.set(cache_key, apex_response)
            
            # Update performance history
            self.performance_history.append(performance_metrics)
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-1000:]
            
            # Emit events
            await self.response_events.emit(Event(value=apex_response))
            await self.performance_events.emit(Event(value=performance_metrics))
            
            self.success_count += 1
            
            return Success(apex_response)
            
        except Exception as e:
            self.error_count += 1
            error = Failure(
                error=str(e),
                error_code="ORCHESTRATION_ERROR",
                context={"request": request.__dict__}
            )
            await self.error_events.emit(Event(value=error))
            return error
    
    async def _monitor_performance(self) -> None:
        """Continuous performance monitoring"""
        while True:
            try:
                # Auto-configure based on performance
                if self.performance_history:
                    await self.config_manager.auto_configure(self.performance_history)
                
                await asyncio.sleep(60)  # Monitor every minute
                
            except Exception as e:
                print(f"Performance monitoring error: {e}")
                await asyncio.sleep(300)  # Longer delay on error
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive APEX metrics"""
        total_requests = self.request_count
        success_rate = self.success_count / total_requests if total_requests > 0 else 0
        error_rate = self.error_count / total_requests if total_requests > 0 else 0
        
        # Calculate performance statistics
        if self.performance_history:
            latencies = [m.latency_p95 for m in self.performance_history]
            costs = [m.cost_per_query for m in self.performance_history]
            
            avg_latency = sum(latencies) / len(latencies)
            avg_cost = sum(costs) / len(costs)
        else:
            avg_latency = 0
            avg_cost = 0
        
        return {
            'request_metrics': {
                'total_requests': total_requests,
                'success_count': self.success_count,
                'error_count': self.error_count,
                'success_rate': success_rate,
                'error_rate': error_rate
            },
            'performance_metrics': {
                'avg_latency_ms': avg_latency,
                'avg_cost_per_query': avg_cost,
                'performance_history_size': len(self.performance_history)
            },
            'component_metrics': {
                'config_manager': self.config_manager.get_metrics(),
                'router': self.router.get_metrics(),
                'cache': {
                    'size': len(self.cache.cache),
                    'max_size': self.cache.max_size
                }
            },
            'event_metrics': {
                'request_events': self.request_events.get_metrics(),
                'response_events': self.response_events.get_metrics(),
                'error_events': self.error_events.get_metrics(),
                'performance_events': self.performance_events.get_metrics()
            }
        }
    
    def subscribe_to_requests(self, observer: Callable[[APEXRequest], None]) -> 'Subscription':
        """Subscribe to request events"""
        return self.request_events.subscribe(lambda event: observer(event.value))
    
    def subscribe_to_responses(self, observer: Callable[[APEXResponse], None]) -> 'Subscription':
        """Subscribe to response events"""
        return self.response_events.subscribe(lambda event: observer(event.value))
    
    def subscribe_to_errors(self, observer: Callable[[Failure], None]) -> 'Subscription':
        """Subscribe to error events"""
        return self.error_events.subscribe(lambda event: observer(event.value))
    
    def subscribe_to_performance(self, observer: Callable[[PerformanceMetrics], None]) -> 'Subscription':
        """Subscribe to performance events"""
        return self.performance_events.subscribe(lambda event: observer(event.value))

class APEXService:
    """High-level APEX service with simplified interface"""
    
    def __init__(self, config_path: Path = None):
        self.orchestrator = APEXOrchestrator(config_path)
        self.config = self.orchestrator.config_manager.get_current_config()
    
    async def query(
        self,
        query_text: str,
        user_id: str,
        session_id: str,
        expertise_level: QueryComplexity = QueryComplexity.MODERATE,
        performance_target: PerformanceTier = PerformanceTier.STANDARD,
        cost_constraints: Optional[float] = None,
        latency_target: Optional[int] = None,
        retrieval_context: List[str] = None,
        model_preference: Optional[str] = None,
        auto_fallback: bool = True
    ) -> Result[APEXResponse]:
        """Execute query with APEX intelligent processing"""
        
        # Create query context
        context = QueryContext(
            user_id=UserId(user_id),
            session_id=SessionId(session_id),
            expertise_level=expertise_level,
            performance_target=performance_target,
            cost_constraints=cost_constraints,
            latency_target=latency_target
        )
        
        # Create APEX request
        request = APEXRequest(
            query_id=QueryId(str(uuid.uuid4())),
            user_id=UserId(user_id),
            session_id=SessionId(session_id),
            query_text=query_text,
            context=context,
            retrieval_context=retrieval_context or [],
            model_preference=model_preference,
            auto_fallback=auto_fallback
        )
        
        # Process request
        return await self.orchestrator.process_request(request)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get service metrics"""
        return {
            'service_metrics': {
                'config': self.config.__dict__,
                'uptime': 'running'  # Placeholder for actual uptime
            },
            'orchestrator_metrics': self.orchestrator.get_metrics()
        }
    
    def subscribe_to_events(self, event_type: str, observer: Callable) -> 'Subscription':
        """Subscribe to specific event types"""
        if event_type == 'requests':
            return self.orchestrator.subscribe_to_requests(observer)
        elif event_type == 'responses':
            return self.orchestrator.subscribe_to_responses(observer)
        elif event_type == 'errors':
            return self.orchestrator.subscribe_to_errors(observer)
        elif event_type == 'performance':
            return self.orchestrator.subscribe_to_performance(observer)
        else:
            raise ValueError(f"Unknown event type: {event_type}")

# Factory functions for easy instantiation
def create_apex_service(config_path: Path = None) -> APEXService:
    """Create APEX service with default configuration"""
    return APEXService(config_path)

def create_apex_orchestrator(config_path: Path = None) -> APEXOrchestrator:
    """Create APEX orchestrator with default configuration"""
    return APEXOrchestrator(config_path)

# Utility functions
@memoize
def generate_query_id() -> QueryId:
    """Generate unique query ID"""
    return QueryId(str(uuid.uuid4()))

def create_simple_context(user_id: str, session_id: str) -> QueryContext:
    """Create simple query context"""
    return QueryContext(
        user_id=UserId(user_id),
        session_id=SessionId(session_id),
        expertise_level=QueryComplexity.SIMPLE,
        performance_target=PerformanceTier.STANDARD
    )

def create_expert_context(user_id: str, session_id: str) -> QueryContext:
    """Create expert query context"""
    return QueryContext(
        user_id=UserId(user_id),
        session_id=SessionId(session_id),
        expertise_level=QueryComplexity.EXPERT,
        performance_target=PerformanceTier.ULTRA
    )
