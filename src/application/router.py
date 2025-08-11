"""
APEX Intelligent Router - Advanced Routing with Context-Aware Decision Making

This module implements a sophisticated intelligent router that provides
context-aware request routing, multi-factor model selection, and advanced
performance optimization for the APEX platform.

Author: APEX Development Team
Version: 1.0.0
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    TypeVar, Generic, Protocol, Union, Optional, 
    List, Dict, Any, Callable, Awaitable, Tuple,
    AsyncIterator, AsyncGenerator
)
import asyncio
from datetime import datetime, timedelta
from collections import defaultdict
import weakref
import time
import statistics
import hashlib
import json

from ..domain.models import (
    QueryContext, ModelDecision, ModelProvider, 
    QueryComplexity, PerformanceTier, Result, Success, Failure,
    PerformanceMetrics, UserId, SessionId, QueryId
)
from ..domain.reactive import Observable, Event, compose, curry, memoize

# Type variables
T = TypeVar('T')
U = TypeVar('U')

@dataclass
class RouteDecision:
    """Immutable routing decision with comprehensive metadata"""
    model_decision: ModelDecision
    retrieval_strategy: str
    optimization_level: str
    estimated_total_cost: float
    estimated_total_latency: int
    confidence_score: float
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_viable(self, max_cost: Optional[float] = None, max_latency: Optional[int] = None) -> bool:
        """Check if route decision is viable given constraints"""
        if max_cost and self.estimated_total_cost > max_cost:
            return False
        if max_latency and self.estimated_total_latency > max_latency:
            return False
        return True

@dataclass
class LoadMetrics:
    """Comprehensive load metrics for intelligent routing"""
    cpu_usage: float
    memory_usage: float
    active_connections: int
    queue_length: int
    error_rate: float
    average_response_time: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_healthy(self, thresholds: Dict[str, float] = None) -> bool:
        """Health check based on load thresholds"""
        default_thresholds = {
            'cpu_usage': 80.0,
            'memory_usage': 85.0,
            'error_rate': 0.05,
            'average_response_time': 500.0
        }
        thresholds = thresholds or default_thresholds
        
        return (
            self.cpu_usage < thresholds.get('cpu_usage', 80.0) and
            self.memory_usage < thresholds.get('memory_usage', 85.0) and
            self.error_rate < thresholds.get('error_rate', 0.05) and
            self.average_response_time < thresholds.get('average_response_time', 500.0)
        )

class QueryAnalyzer:
    """Advanced query analysis with ML-driven insights"""
    
    def __init__(self):
        self._complexity_models = self._load_complexity_models()
        self._intent_classifiers = self._load_intent_classifiers()
        self._domain_classifiers = self._load_domain_classifiers()
        self._cache = {}
    
    @memoize
    async def analyze_query_complexity(self, query: str) -> QueryComplexity:
        """Analyze query complexity using multiple heuristics"""
        # Token count analysis
        token_count = len(query.split())
        
        # Semantic complexity analysis
        semantic_score = await self._analyze_semantic_complexity(query)
        
        # Intent complexity analysis
        intent_score = await self._analyze_intent_complexity(query)
        
        # Domain complexity analysis
        domain_score = await self._analyze_domain_complexity(query)
        
        # Combined scoring with weights
        total_score = (
            token_count * 0.2 + 
            semantic_score * 0.3 + 
            intent_score * 0.3 + 
            domain_score * 0.2
        )
        
        # Complexity classification
        if total_score < 10:
            return QueryComplexity.SIMPLE
        elif total_score < 25:
            return QueryComplexity.MODERATE
        elif total_score < 50:
            return QueryComplexity.COMPLEX
        else:
            return QueryComplexity.EXPERT
    
    async def extract_requirements(self, query: str, context: QueryContext) -> Dict[str, Any]:
        """Extract comprehensive requirements from query and context"""
        requirements = {
            'complexity': await self.analyze_query_complexity(query),
            'intent': await self._classify_intent(query),
            'domain': await self._classify_domain(query),
            'performance_target': context.performance_target,
            'cost_constraints': context.cost_constraints,
            'latency_target': context.latency_target,
            'expertise_level': context.expertise_level,
            'user_preferences': await self._extract_user_preferences(context),
            'query_features': await self._extract_query_features(query)
        }
        
        return requirements
    
    async def _analyze_semantic_complexity(self, query: str) -> float:
        """Analyze semantic complexity using NLP techniques"""
        # Placeholder for advanced NLP analysis
        # In production, this would use transformer models
        
        # Simple heuristics for now
        complexity_indicators = [
            'explain', 'analyze', 'compare', 'contrast', 'evaluate',
            'synthesize', 'design', 'implement', 'optimize', 'debug'
        ]
        
        score = 0.0
        query_lower = query.lower()
        
        for indicator in complexity_indicators:
            if indicator in query_lower:
                score += 5.0
        
        # Length factor
        score += len(query) / 100.0
        
        return min(score, 50.0)  # Cap at 50
    
    async def _analyze_intent_complexity(self, query: str) -> float:
        """Analyze intent complexity"""
        # Placeholder for intent analysis
        # In production, this would use intent classification models
        
        intent_complexity_map = {
            'information_retrieval': 1.0,
            'question_answering': 3.0,
            'analysis': 5.0,
            'synthesis': 7.0,
            'creation': 10.0,
            'optimization': 8.0
        }
        
        # Simple keyword-based intent detection
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['create', 'generate', 'write']):
            return intent_complexity_map['creation']
        elif any(word in query_lower for word in ['optimize', 'improve', 'enhance']):
            return intent_complexity_map['optimization']
        elif any(word in query_lower for word in ['analyze', 'examine', 'study']):
            return intent_complexity_map['analysis']
        elif any(word in query_lower for word in ['compare', 'contrast', 'synthesize']):
            return intent_complexity_map['synthesis']
        elif any(word in query_lower for word in ['what', 'how', 'why', 'when']):
            return intent_complexity_map['question_answering']
        else:
            return intent_complexity_map['information_retrieval']
    
    async def _analyze_domain_complexity(self, query: str) -> float:
        """Analyze domain complexity"""
        # Placeholder for domain analysis
        # In production, this would use domain classification models
        
        domain_complexity_map = {
            'general': 1.0,
            'technical': 3.0,
            'scientific': 5.0,
            'mathematical': 7.0,
            'programming': 6.0,
            'research': 8.0
        }
        
        # Simple keyword-based domain detection
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['algorithm', 'code', 'program', 'function']):
            return domain_complexity_map['programming']
        elif any(word in query_lower for word in ['theorem', 'proof', 'equation', 'mathematical']):
            return domain_complexity_map['mathematical']
        elif any(word in query_lower for word in ['research', 'study', 'experiment']):
            return domain_complexity_map['research']
        elif any(word in query_lower for word in ['scientific', 'physics', 'chemistry', 'biology']):
            return domain_complexity_map['scientific']
        elif any(word in query_lower for word in ['technical', 'system', 'architecture']):
            return domain_complexity_map['technical']
        else:
            return domain_complexity_map['general']
    
    async def _classify_intent(self, query: str) -> str:
        """Classify query intent"""
        # Placeholder for intent classification
        # In production, this would use ML models
        return "information_retrieval"
    
    async def _classify_domain(self, query: str) -> str:
        """Classify query domain"""
        # Placeholder for domain classification
        # In production, this would use ML models
        return "general"
    
    async def _extract_user_preferences(self, context: QueryContext) -> Dict[str, Any]:
        """Extract user preferences from context"""
        return {
            'expertise_level': context.expertise_level,
            'performance_target': context.performance_target,
            'cost_sensitivity': context.cost_constraints is not None,
            'latency_sensitivity': context.latency_target is not None
        }
    
    async def _extract_query_features(self, query: str) -> Dict[str, Any]:
        """Extract query features for analysis"""
        return {
            'length': len(query),
            'word_count': len(query.split()),
            'has_technical_terms': any(word in query.lower() for word in ['algorithm', 'function', 'system']),
            'has_questions': any(word in query.lower() for word in ['what', 'how', 'why', 'when', 'where']),
            'has_commands': any(word in query.lower() for word in ['create', 'generate', 'analyze', 'compare'])
        }
    
    def _load_complexity_models(self) -> Dict[str, Any]:
        """Load complexity analysis models"""
        # Placeholder for model loading
        return {}
    
    def _load_intent_classifiers(self) -> Dict[str, Any]:
        """Load intent classification models"""
        # Placeholder for classifier loading
        return {}
    
    def _load_domain_classifiers(self) -> Dict[str, Any]:
        """Load domain classification models"""
        # Placeholder for classifier loading
        return {}

class ModelSelector:
    """Intelligent model selection with multi-factor decision making"""
    
    def __init__(self):
        self._performance_history = defaultdict(list)
        self._cost_history = defaultdict(list)
        self._availability_cache = {}
        self._cache_ttl = timedelta(minutes=5)
        self._model_capabilities = self._load_model_capabilities()
        self._selection_weights = {
            'performance': 0.3,
            'cost': 0.25,
            'availability': 0.2,
            'capability': 0.25
        }
    
    async def select_optimal_model(
        self, 
        complexity: QueryComplexity,
        requirements: Dict[str, Any],
        load_metrics: LoadMetrics
    ) -> ModelDecision:
        """Select optimal model using multi-factor analysis"""
        
        # Get available models
        available_models = await self._get_available_models()
        
        # Score each model
        model_scores = []
        for model in available_models:
            score = await self._calculate_model_score(
                model=model,
                complexity=complexity,
                requirements=requirements,
                load_metrics=load_metrics
            )
            model_scores.append((model, score))
        
        # Sort by score and select best model
        model_scores.sort(key=lambda x: x[1].confidence_score, reverse=True)
        
        if not model_scores:
            raise ValueError("No suitable models available")
        
        best_model, best_score = model_scores[0]
        
        # Generate fallback models
        fallback_models = []
        for model, score in model_scores[1:4]:  # Top 3 fallbacks
            fallback_decision = ModelDecision(
                provider=model.provider,
                model_name=model.model_name,
                confidence_score=score.confidence_score * 0.8,
                reasoning=f"Fallback option with {score.confidence_score * 0.8:.2f} confidence",
                estimated_cost=score.estimated_cost,
                estimated_latency=score.estimated_latency,
                capabilities=score.capabilities
            )
            fallback_models.append(fallback_decision)
        
        return ModelDecision(
            provider=best_model.provider,
            model_name=best_model.model_name,
            confidence_score=best_score.confidence_score,
            reasoning=best_score.reasoning,
            estimated_cost=best_score.estimated_cost,
            estimated_latency=best_score.estimated_latency,
            capabilities=best_score.capabilities,
            fallback_models=fallback_models
        )
    
    async def _calculate_model_score(
        self,
        model: Any,
        complexity: QueryComplexity,
        requirements: Dict[str, Any],
        load_metrics: LoadMetrics
    ) -> ModelDecision:
        """Calculate comprehensive model score"""
        
        # Performance score
        performance_score = await self._calculate_performance_score(
            model, complexity, load_metrics
        )
        
        # Cost score
        cost_score = await self._calculate_cost_score(
            model, requirements
        )
        
        # Availability score
        availability_score = await self._calculate_availability_score(model)
        
        # Capability score
        capability_score = await self._calculate_capability_score(
            model, complexity, requirements
        )
        
        # Weighted combination
        total_score = (
            performance_score * self._selection_weights['performance'] +
            cost_score * self._selection_weights['cost'] +
            availability_score * self._selection_weights['availability'] +
            capability_score * self._selection_weights['capability']
        )
        
        # Generate reasoning
        reasoning = (
            f"Performance: {performance_score:.2f}, "
            f"Cost: {cost_score:.2f}, "
            f"Availability: {availability_score:.2f}, "
            f"Capability: {capability_score:.2f}"
        )
        
        return ModelDecision(
            provider=model.provider,
            model_name=model.model_name,
            confidence_score=total_score,
            reasoning=reasoning,
            estimated_cost=await self._estimate_cost(model, requirements),
            estimated_latency=await self._estimate_latency(model, complexity),
            capabilities=await self._get_model_capabilities(model)
        )
    
    async def _calculate_performance_score(
        self, 
        model: Any, 
        complexity: QueryComplexity,
        load_metrics: LoadMetrics
    ) -> float:
        """Calculate performance score based on historical data and current load"""
        # Get historical performance
        history = self._performance_history.get(model.model_name, [])
        
        if not history:
            return 0.5  # Default score for new models
        
        # Calculate average performance
        avg_latency = statistics.mean([h.latency_p95 for h in history])
        avg_throughput = statistics.mean([h.throughput for h in history])
        avg_error_rate = statistics.mean([h.error_rate for h in history])
        
        # Normalize scores
        latency_score = max(0, 1 - (avg_latency / 1000))  # Normalize to 1 second
        throughput_score = min(1, avg_throughput / 1000)  # Normalize to 1000 req/s
        error_score = max(0, 1 - avg_error_rate)  # Lower error rate is better
        
        # Consider current load
        load_factor = 1 - (load_metrics.cpu_usage / 100)
        
        # Weighted performance score
        performance_score = (
            latency_score * 0.4 + 
            throughput_score * 0.3 + 
            error_score * 0.3
        ) * load_factor
        
        return max(0, min(1, performance_score))  # Clamp to [0, 1]
    
    async def _calculate_cost_score(self, model: Any, requirements: Dict[str, Any]) -> float:
        """Calculate cost efficiency score"""
        estimated_cost = await self._estimate_cost(model, requirements)
        
        if requirements.get('cost_constraints'):
            max_cost = requirements['cost_constraints']
            return max(0, 1 - (estimated_cost / max_cost))
        
        # Normalize to typical cost range
        return max(0, 1 - (estimated_cost / 0.1))  # Normalize to $0.10
    
    async def _calculate_availability_score(self, model: Any) -> float:
        """Calculate availability score with caching"""
        cache_key = f"{model.provider}_{model.model_name}"
        
        # Check cache
        if cache_key in self._availability_cache:
            cached_time, cached_score = self._availability_cache[cache_key]
            if datetime.utcnow() - cached_time < self._cache_ttl:
                return cached_score
        
        # Check availability
        try:
            is_available = await self._check_model_availability(model)
            score = 1.0 if is_available else 0.0
            
            # Cache result
            self._availability_cache[cache_key] = (datetime.utcnow(), score)
            
            return score
        except Exception:
            return 0.0
    
    async def _calculate_capability_score(
        self, 
        model: Any, 
        complexity: QueryComplexity,
        requirements: Dict[str, Any]
    ) -> float:
        """Calculate capability score based on model capabilities"""
        model_capabilities = await self._get_model_capabilities(model)
        required_capabilities = requirements.get('required_capabilities', [])
        
        if not required_capabilities:
            # Use complexity-based scoring
            complexity_capability_map = {
                QueryComplexity.SIMPLE: 0.8,
                QueryComplexity.MODERATE: 0.9,
                QueryComplexity.COMPLEX: 0.7,
                QueryComplexity.EXPERT: 0.6
            }
            return complexity_capability_map.get(complexity, 0.5)
        
        # Calculate capability match
        matched_capabilities = sum(1 for cap in required_capabilities if cap in model_capabilities)
        capability_score = matched_capabilities / len(required_capabilities)
        
        return capability_score
    
    async def _get_available_models(self) -> List[Any]:
        """Get list of available models"""
        # Placeholder - would return actual model configurations
        return []
    
    async def _estimate_cost(self, model: Any, requirements: Dict[str, Any]) -> float:
        """Estimate cost for model and requirements"""
        # Placeholder - would use actual cost estimation
        return 0.01
    
    async def _estimate_latency(self, model: Any, complexity: QueryComplexity) -> int:
        """Estimate latency for model and complexity"""
        # Placeholder - would use actual latency estimation
        return 200
    
    async def _check_model_availability(self, model: Any) -> bool:
        """Check if model is available"""
        # Placeholder - would check actual availability
        return True
    
    async def _get_model_capabilities(self, model: Any) -> List[str]:
        """Get model capabilities"""
        # Placeholder - would return actual capabilities
        return ["text_generation", "question_answering"]
    
    def _load_model_capabilities(self) -> Dict[str, List[str]]:
        """Load model capabilities database"""
        # Placeholder for capabilities database
        return {}

class PerformanceMonitor:
    """Real-time performance monitoring with statistical analysis"""
    
    def __init__(self):
        self._metrics_history = deque(maxlen=1000)
        self._alerts = Observable[Dict[str, Any]](name="performance_alerts")
        self._thresholds = {
            'cpu_usage': 80.0,
            'memory_usage': 85.0,
            'error_rate': 0.05,
            'response_time': 500.0
        }
    
    async def collect_metrics(self) -> LoadMetrics:
        """Collect comprehensive performance metrics"""
        # Placeholder - would collect actual metrics
        # In production, this would use system monitoring tools
        
        import psutil
        
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_usage = psutil.virtual_memory().percent
        
        # Simulate other metrics
        active_connections = 100
        queue_length = 5
        error_rate = 0.01
        average_response_time = 150.0
        
        metrics = LoadMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            active_connections=active_connections,
            queue_length=queue_length,
            error_rate=error_rate,
            average_response_time=average_response_time
        )
        
        # Store in history
        self._metrics_history.append(metrics)
        
        # Check for alerts
        await self._check_alerts(metrics)
        
        return metrics
    
    async def _check_alerts(self, metrics: LoadMetrics) -> None:
        """Check for performance alerts"""
        alerts = []
        
        if metrics.cpu_usage > self._thresholds['cpu_usage']:
            alerts.append({
                'type': 'high_cpu_usage',
                'value': metrics.cpu_usage,
                'threshold': self._thresholds['cpu_usage'],
                'severity': 'warning'
            })
        
        if metrics.memory_usage > self._thresholds['memory_usage']:
            alerts.append({
                'type': 'high_memory_usage',
                'value': metrics.memory_usage,
                'threshold': self._thresholds['memory_usage'],
                'severity': 'warning'
            })
        
        if metrics.error_rate > self._thresholds['error_rate']:
            alerts.append({
                'type': 'high_error_rate',
                'value': metrics.error_rate,
                'threshold': self._thresholds['error_rate'],
                'severity': 'critical'
            })
        
        if metrics.average_response_time > self._thresholds['response_time']:
            alerts.append({
                'type': 'high_response_time',
                'value': metrics.average_response_time,
                'threshold': self._thresholds['response_time'],
                'severity': 'warning'
            })
        
        # Emit alerts
        for alert in alerts:
            await self._alerts.emit(Event(value=alert))
    
    def get_metrics_history(self) -> List[LoadMetrics]:
        """Get metrics history"""
        return list(self._metrics_history)
    
    def subscribe_to_alerts(self, observer: Callable[[Dict[str, Any]], None]) -> 'Subscription':
        """Subscribe to performance alerts"""
        return self._alerts.subscribe(lambda event: observer(event.value))

class IntelligentRouter:
    """Advanced intelligent router with reactive architecture"""
    
    def __init__(self):
        self.query_analyzer = QueryAnalyzer()
        self.model_selector = ModelSelector()
        self.performance_monitor = PerformanceMonitor()
        self.state_manager = StateManager(LoadMetrics(0, 0, 0, 0, 0, 0), name="router_state")
        
        # Reactive streams
        self.route_events = Observable[RouteDecision](name="route_events")
        self.performance_events = Observable[LoadMetrics](name="performance_events")
        self.error_events = Observable[Failure](name="error_events")
        
        # Start monitoring
        asyncio.create_task(self._monitor_performance())
    
    async def route_request(
        self, 
        query: str, 
        context: QueryContext
    ) -> Result[RouteDecision]:
        """Route request with intelligent decision making"""
        
        try:
            # Analyze query
            complexity = await self.query_analyzer.analyze_query_complexity(query)
            requirements = await self.query_analyzer.extract_requirements(query, context)
            
            # Get current load metrics
            load_metrics = self.state_manager.get_state().value
            
            # Select optimal model
            model_decision = await self.model_selector.select_optimal_model(
                complexity=complexity,
                requirements=requirements,
                load_metrics=load_metrics
            )
            
            # Create route decision
            route_decision = RouteDecision(
                model_decision=model_decision,
                retrieval_strategy=self._select_retrieval_strategy(complexity),
                optimization_level=self._determine_optimization_level(requirements),
                estimated_total_cost=model_decision.estimated_cost,
                estimated_total_latency=model_decision.estimated_latency,
                confidence_score=model_decision.confidence_score,
                reasoning=model_decision.reasoning,
                metadata={
                    'complexity': complexity,
                    'requirements': requirements,
                    'load_metrics': load_metrics.__dict__
                }
            )
            
            # Emit route event
            await self.route_events.emit(Event(value=route_decision))
            
            return Success(route_decision)
            
        except Exception as e:
            error = Failure(
                error=str(e),
                error_code="ROUTING_ERROR",
                context={"query": query, "context": context.__dict__}
            )
            await self.error_events.emit(Event(value=error))
            return error
    
    def _select_retrieval_strategy(self, complexity: QueryComplexity) -> str:
        """Select retrieval strategy based on complexity"""
        strategy_map = {
            QueryComplexity.SIMPLE: "basic_retrieval",
            QueryComplexity.MODERATE: "enhanced_retrieval",
            QueryComplexity.COMPLEX: "advanced_retrieval",
            QueryComplexity.EXPERT: "expert_retrieval"
        }
        return strategy_map.get(complexity, "basic_retrieval")
    
    def _determine_optimization_level(self, requirements: Dict[str, Any]) -> str:
        """Determine optimization level based on requirements"""
        performance_target = requirements.get('performance_target', PerformanceTier.STANDARD)
        
        optimization_map = {
            PerformanceTier.ECONOMY: "minimal",
            PerformanceTier.STANDARD: "standard",
            PerformanceTier.PERFORMANCE: "aggressive",
            PerformanceTier.ULTRA: "maximum"
        }
        return optimization_map.get(performance_target, "standard")
    
    async def _monitor_performance(self) -> None:
        """Continuous performance monitoring"""
        while True:
            try:
                # Collect performance metrics
                metrics = await self.performance_monitor.collect_metrics()
                
                # Update state
                self.state_manager.update_state_sync(lambda _: metrics)
                
                # Emit performance event
                await self.performance_events.emit(Event(value=metrics))
                
                await asyncio.sleep(10)  # Monitor every 10 seconds
                
            except Exception as e:
                # Log error and continue monitoring
                print(f"Performance monitoring error: {e}")
                await asyncio.sleep(30)  # Longer delay on error
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get router metrics"""
        return {
            'state_metrics': self.state_manager.get_metrics(),
            'route_events_metrics': self.route_events.get_metrics(),
            'performance_events_metrics': self.performance_events.get_metrics(),
            'error_events_metrics': self.error_events.get_metrics()
        }
