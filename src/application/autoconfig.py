"""
APEX Auto-Configuration System - Intelligent Parameter Optimization

This module implements an intelligent auto-configuration system that provides
ML-driven parameter optimization, configuration validation, and safe default
fallbacks for the APEX platform.

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
import json
import yaml
from pathlib import Path
import hashlib
import statistics
from collections import defaultdict

from ..domain.models import (
    APEXConfig, PerformanceTier, QueryComplexity,
    PerformanceMetrics, Result, Success, Failure
)
from ..domain.reactive import Observable, Event, StateManager

# Type variables
T = TypeVar('T')
U = TypeVar('U')

@dataclass
class ConfigurationParameter:
    """Immutable configuration parameter with validation"""
    name: str
    value: Any
    type: str
    description: str
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    allowed_values: Optional[List[Any]] = None
    required: bool = True
    default_value: Optional[Any] = None
    validation_rules: List[Callable[[Any], bool]] = field(default_factory=list)
    
    def validate(self) -> Result[Any]:
        """Validate parameter value"""
        try:
            # Type validation
            if not self._validate_type():
                return Failure(
                    error=f"Invalid type for {self.name}. Expected {self.type}, got {type(self.value).__name__}",
                    error_code="TYPE_VALIDATION_ERROR"
                )
            
            # Range validation
            if not self._validate_range():
                return Failure(
                    error=f"Value {self.value} for {self.name} is outside allowed range [{self.min_value}, {self.max_value}]",
                    error_code="RANGE_VALIDATION_ERROR"
                )
            
            # Allowed values validation
            if not self._validate_allowed_values():
                return Failure(
                    error=f"Value {self.value} for {self.name} is not in allowed values {self.allowed_values}",
                    error_code="VALUE_VALIDATION_ERROR"
                )
            
            # Custom validation rules
            for rule in self.validation_rules:
                if not rule(self.value):
                    return Failure(
                        error=f"Custom validation failed for {self.name}",
                        error_code="CUSTOM_VALIDATION_ERROR"
                    )
            
            return Success(self.value)
            
        except Exception as e:
            return Failure(
                error=f"Validation error for {self.name}: {str(e)}",
                error_code="VALIDATION_ERROR"
            )
    
    def _validate_type(self) -> bool:
        """Validate parameter type"""
        type_map = {
            'int': int,
            'float': float,
            'str': str,
            'bool': bool,
            'list': list,
            'dict': dict
        }
        
        expected_type = type_map.get(self.type)
        if not expected_type:
            return True  # Unknown type, skip validation
        
        return isinstance(self.value, expected_type)
    
    def _validate_range(self) -> bool:
        """Validate parameter range"""
        if self.min_value is None and self.max_value is None:
            return True
        
        if self.min_value is not None and self.value < self.min_value:
            return False
        
        if self.max_value is not None and self.value > self.max_value:
            return False
        
        return True
    
    def _validate_allowed_values(self) -> bool:
        """Validate allowed values"""
        if self.allowed_values is None:
            return True
        
        return self.value in self.allowed_values

class ConfigurationOptimizer:
    """ML-driven configuration optimizer with performance analysis"""
    
    def __init__(self):
        self.performance_history = defaultdict(list)
        self.optimization_rules = self._load_optimization_rules()
        self.ml_model = self._load_ml_model()
        self.optimization_events = Observable[Dict[str, Any]](name="optimization_events")
    
    async def optimize_configuration(
        self, 
        current_config: APEXConfig,
        performance_metrics: List[PerformanceMetrics]
    ) -> Result[APEXConfig]:
        """Optimize configuration based on performance metrics"""
        
        try:
            # Analyze performance patterns
            analysis = await self._analyze_performance_patterns(performance_metrics)
            
            # Generate optimization suggestions
            suggestions = await self._generate_optimization_suggestions(
                current_config, analysis
            )
            
            # Apply optimizations
            optimized_config = await self._apply_optimizations(
                current_config, suggestions
            )
            
            # Validate optimized configuration
            validation_result = await self._validate_optimized_config(optimized_config)
            
            if isinstance(validation_result, Failure):
                return validation_result
            
            # Emit optimization event
            await self.optimization_events.emit(Event(value={
                'type': 'configuration_optimized',
                'original_config': current_config.__dict__,
                'optimized_config': optimized_config.__dict__,
                'suggestions': suggestions,
                'performance_improvement': analysis.get('expected_improvement', 0.0)
            }))
            
            return Success(optimized_config)
            
        except Exception as e:
            return Failure(
                error=f"Configuration optimization failed: {str(e)}",
                error_code="OPTIMIZATION_ERROR"
            )
    
    async def _analyze_performance_patterns(
        self, 
        metrics: List[PerformanceMetrics]
    ) -> Dict[str, Any]:
        """Analyze performance patterns for optimization insights"""
        
        if not metrics:
            return {'expected_improvement': 0.0}
        
        # Calculate statistical measures
        latencies = [m.latency_p95 for m in metrics]
        error_rates = [m.error_rate for m in metrics]
        costs = [m.cost_per_query for m in metrics]
        
        analysis = {
            'avg_latency': statistics.mean(latencies),
            'latency_variance': statistics.variance(latencies),
            'avg_error_rate': statistics.mean(error_rates),
            'avg_cost': statistics.mean(costs),
            'performance_trend': self._calculate_trend(latencies),
            'bottlenecks': self._identify_bottlenecks(metrics),
            'optimization_opportunities': self._identify_opportunities(metrics)
        }
        
        # Calculate expected improvement
        analysis['expected_improvement'] = self._calculate_expected_improvement(analysis)
        
        return analysis
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend in performance values"""
        if len(values) < 2:
            return "stable"
        
        # Simple linear trend calculation
        recent_avg = statistics.mean(values[-5:]) if len(values) >= 5 else values[-1]
        older_avg = statistics.mean(values[:5]) if len(values) >= 5 else values[0]
        
        if recent_avg < older_avg * 0.9:
            return "improving"
        elif recent_avg > older_avg * 1.1:
            return "degrading"
        else:
            return "stable"
    
    def _identify_bottlenecks(self, metrics: List[PerformanceMetrics]) -> List[str]:
        """Identify performance bottlenecks"""
        bottlenecks = []
        
        # Analyze recent metrics
        recent_metrics = metrics[-10:] if len(metrics) >= 10 else metrics
        
        avg_latency = statistics.mean([m.latency_p95 for m in recent_metrics])
        avg_error_rate = statistics.mean([m.error_rate for m in recent_metrics])
        
        if avg_latency > 500:  # High latency threshold
            bottlenecks.append("high_latency")
        
        if avg_error_rate > 0.05:  # High error rate threshold
            bottlenecks.append("high_error_rate")
        
        return bottlenecks
    
    def _identify_opportunities(self, metrics: List[PerformanceMetrics]) -> List[str]:
        """Identify optimization opportunities"""
        opportunities = []
        
        # Analyze cost efficiency
        recent_metrics = metrics[-10:] if len(metrics) >= 10 else metrics
        avg_cost = statistics.mean([m.cost_per_query for m in recent_metrics])
        
        if avg_cost > 0.01:  # High cost threshold
            opportunities.append("cost_optimization")
        
        # Analyze throughput
        avg_throughput = statistics.mean([m.throughput for m in recent_metrics])
        if avg_throughput < 50:  # Low throughput threshold
            opportunities.append("throughput_optimization")
        
        return opportunities
    
    def _calculate_expected_improvement(self, analysis: Dict[str, Any]) -> float:
        """Calculate expected performance improvement"""
        improvement = 0.0
        
        # Latency improvement potential
        if analysis['avg_latency'] > 300:
            improvement += 0.2  # 20% improvement potential
        
        # Error rate improvement potential
        if analysis['avg_error_rate'] > 0.02:
            improvement += 0.15  # 15% improvement potential
        
        # Cost optimization potential
        if analysis['avg_cost'] > 0.005:
            improvement += 0.1  # 10% improvement potential
        
        return min(improvement, 0.5)  # Cap at 50% improvement
    
    async def _generate_optimization_suggestions(
        self, 
        config: APEXConfig, 
        analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate optimization suggestions based on analysis"""
        
        suggestions = []
        
        # Latency optimization
        if 'high_latency' in analysis.get('bottlenecks', []):
            suggestions.append({
                'type': 'latency_optimization',
                'parameter': 'default_timeout_ms',
                'current_value': config.default_timeout_ms,
                'suggested_value': min(config.default_timeout_ms * 0.8, 20000),
                'reasoning': 'Reduce timeout to improve responsiveness'
            })
        
        # Error rate optimization
        if 'high_error_rate' in analysis.get('bottlenecks', []):
            suggestions.append({
                'type': 'reliability_optimization',
                'parameter': 'max_concurrent_queries',
                'current_value': config.max_concurrent_queries,
                'suggested_value': max(config.max_concurrent_queries * 0.7, 50),
                'reasoning': 'Reduce concurrency to improve reliability'
            })
        
        # Cost optimization
        if 'cost_optimization' in analysis.get('optimization_opportunities', []):
            suggestions.append({
                'type': 'cost_optimization',
                'parameter': 'performance_target',
                'current_value': config.performance_target,
                'suggested_value': PerformanceTier.STANDARD,
                'reasoning': 'Use standard performance tier for cost efficiency'
            })
        
        return suggestions
    
    async def _apply_optimizations(
        self, 
        config: APEXConfig, 
        suggestions: List[Dict[str, Any]]
    ) -> APEXConfig:
        """Apply optimization suggestions to configuration"""
        
        optimized_config = config
        
        for suggestion in suggestions:
            param_name = suggestion['parameter']
            suggested_value = suggestion['suggested_value']
            
            if param_name == 'default_timeout_ms':
                optimized_config = optimized_config.with_timeout(suggested_value)
            elif param_name == 'performance_target':
                optimized_config = optimized_config.with_performance_target(suggested_value)
            elif param_name == 'max_concurrent_queries':
                # Create new config with updated concurrency
                optimized_config = APEXConfig(
                    auto_configure=optimized_config.auto_configure,
                    performance_target=optimized_config.performance_target,
                    max_concurrent_queries=suggested_value,
                    default_timeout_ms=optimized_config.default_timeout_ms,
                    cache_ttl_hours=optimized_config.cache_ttl_hours,
                    enable_monitoring=optimized_config.enable_monitoring,
                    enable_auto_optimization=optimized_config.enable_auto_optimization,
                    log_level=optimized_config.log_level,
                    metadata=optimized_config.metadata
                )
        
        return optimized_config
    
    async def _validate_optimized_config(self, config: APEXConfig) -> Result[APEXConfig]:
        """Validate optimized configuration"""
        
        # Basic validation rules
        if config.default_timeout_ms < 5000:
            return Failure(
                error="Timeout too low for reliable operation",
                error_code="VALIDATION_ERROR"
            )
        
        if config.max_concurrent_queries < 10:
            return Failure(
                error="Concurrency too low for practical use",
                error_code="VALIDATION_ERROR"
            )
        
        return Success(config)
    
    def _load_optimization_rules(self) -> Dict[str, Any]:
        """Load optimization rules"""
        # Placeholder for optimization rules
        return {}
    
    def _load_ml_model(self) -> Any:
        """Load ML model for optimization"""
        # Placeholder for ML model
        return None

class ConfigurationManager:
    """Intelligent configuration manager with auto-configuration"""
    
    def __init__(self, config_path: Path = None):
        self.config_path = config_path or Path("config/apex_config.yaml")
        self.optimizer = ConfigurationOptimizer()
        self.state_manager = StateManager(APEXConfig(), name="config_manager")
        self.config_events = Observable[Dict[str, Any]](name="config_events")
        
        # Load initial configuration
        self._load_initial_config()
    
    def _load_initial_config(self) -> None:
        """Load initial configuration from file or defaults"""
        try:
            if self.config_path.exists():
                config = self._load_from_file()
            else:
                config = self._create_default_config()
            
            self.state_manager.update_state_sync(lambda _: config)
            
        except Exception as e:
            print(f"Error loading configuration: {e}")
            # Use safe defaults
            config = self._create_safe_default_config()
            self.state_manager.update_state_sync(lambda _: config)
    
    def _load_from_file(self) -> APEXConfig:
        """Load configuration from file"""
        with open(self.config_path, 'r') as f:
            data = yaml.safe_load(f)
        
        return APEXConfig(
            auto_configure=data.get('auto_configure', True),
            performance_target=PerformanceTier(data.get('performance_target', 'standard')),
            max_concurrent_queries=data.get('max_concurrent_queries', 100),
            default_timeout_ms=data.get('default_timeout_ms', 30000),
            cache_ttl_hours=data.get('cache_ttl_hours', 1),
            enable_monitoring=data.get('enable_monitoring', True),
            enable_auto_optimization=data.get('enable_auto_optimization', True),
            log_level=data.get('log_level', 'INFO'),
            metadata=data.get('metadata', {})
        )
    
    def _create_default_config(self) -> APEXConfig:
        """Create default configuration"""
        return APEXConfig()
    
    def _create_safe_default_config(self) -> APEXConfig:
        """Create safe default configuration for error recovery"""
        return APEXConfig(
            auto_configure=False,  # Disable auto-configuration for safety
            performance_target=PerformanceTier.STANDARD,
            max_concurrent_queries=50,  # Conservative concurrency
            default_timeout_ms=60000,  # Longer timeout for reliability
            cache_ttl_hours=1,
            enable_monitoring=True,
            enable_auto_optimization=False,  # Disable for safety
            log_level='WARNING',
            metadata={'safe_mode': True}
        )
    
    async def auto_configure(self, performance_metrics: List[PerformanceMetrics] = None) -> Result[APEXConfig]:
        """Perform intelligent auto-configuration"""
        
        current_config = self.state_manager.get_state().value
        
        if not current_config.auto_configure:
            return Success(current_config)
        
        if not performance_metrics:
            performance_metrics = []
        
        # Perform optimization
        optimization_result = await self.optimizer.optimize_configuration(
            current_config, performance_metrics
        )
        
        if isinstance(optimization_result, Success):
            optimized_config = optimization_result.value
            
            # Update state
            self.state_manager.update_state_sync(lambda _: optimized_config)
            
            # Save to file
            await self._save_configuration(optimized_config)
            
            # Emit configuration event
            await self.config_events.emit(Event(value={
                'type': 'configuration_updated',
                'config': optimized_config.__dict__,
                'timestamp': datetime.utcnow().isoformat()
            }))
        
        return optimization_result
    
    async def _save_configuration(self, config: APEXConfig) -> None:
        """Save configuration to file"""
        try:
            # Ensure config directory exists
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert to dictionary
            config_dict = {
                'auto_configure': config.auto_configure,
                'performance_target': config.performance_target.value,
                'max_concurrent_queries': config.max_concurrent_queries,
                'default_timeout_ms': config.default_timeout_ms,
                'cache_ttl_hours': config.cache_ttl_hours,
                'enable_monitoring': config.enable_monitoring,
                'enable_auto_optimization': config.enable_auto_optimization,
                'log_level': config.log_level,
                'metadata': config.metadata
            }
            
            # Save to file
            with open(self.config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
                
        except Exception as e:
            print(f"Error saving configuration: {e}")
    
    def get_current_config(self) -> APEXConfig:
        """Get current configuration"""
        return self.state_manager.get_state().value
    
    def update_config(self, updates: Dict[str, Any]) -> APEXConfig:
        """Update configuration with new values"""
        current_config = self.state_manager.get_state().value
        
        # Apply updates
        if 'performance_target' in updates:
            current_config = current_config.with_performance_target(
                PerformanceTier(updates['performance_target'])
            )
        
        if 'default_timeout_ms' in updates:
            current_config = current_config.with_timeout(updates['default_timeout_ms'])
        
        # Update state
        self.state_manager.update_state_sync(lambda _: current_config)
        
        return current_config
    
    def subscribe_to_config_changes(self, observer: Callable[[APEXConfig], None]) -> 'Subscription':
        """Subscribe to configuration changes"""
        return self.state_manager.subscribe(observer)
    
    def subscribe_to_config_events(self, observer: Callable[[Dict[str, Any]], None]) -> 'Subscription':
        """Subscribe to configuration events"""
        return self.config_events.subscribe(lambda event: observer(event.value))
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get configuration manager metrics"""
        return {
            'state_metrics': self.state_manager.get_metrics(),
            'config_events_metrics': self.config_events.get_metrics(),
            'optimizer_metrics': self.optimizer.optimization_events.get_metrics()
        }
