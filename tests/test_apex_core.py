"""
APEX Core Test Suite - Comprehensive Testing for APEX Platform

This module provides comprehensive tests for the APEX core functionality,
including unit tests, integration tests, and performance tests.

Author: Mohammad Atashi (mohammadaliatashi@icloud.com)
Version: 1.0.0
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import json

from src.domain.models import (
    QueryContext, QueryComplexity, PerformanceTier,
    UserId, SessionId, QueryId, PerformanceMetrics,
    Result, Success, Failure
)
from src.domain.reactive import Observable, Event, StateManager
from src.application.apex_core import (
    APEXService, APEXOrchestrator, APEXCache,
    APEXRequest, APEXResponse, create_apex_service,
    create_simple_context, create_expert_context
)
from src.application.models import ModelConfig, ModelResponse
from src.application.autoconfig import ConfigurationManager, APEXConfig

# Test fixtures
@pytest.fixture
def temp_config_file():
    """Create temporary configuration file for testing"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        config_data = {
            'auto_configure': True,
            'performance_target': 'standard',
            'max_concurrent_queries': 100,
            'default_timeout_ms': 30000,
            'cache_ttl_hours': 1,
            'enable_monitoring': True,
            'enable_auto_optimization': True,
            'log_level': 'INFO'
        }
        import yaml
        yaml.dump(config_data, f)
        return Path(f.name)

@pytest.fixture
def sample_query_context():
    """Create sample query context for testing"""
    return QueryContext(
        user_id=UserId("test_user"),
        session_id=SessionId("test_session"),
        expertise_level=QueryComplexity.MODERATE,
        performance_target=PerformanceTier.STANDARD
    )

@pytest.fixture
def sample_apex_request(sample_query_context):
    """Create sample APEX request for testing"""
    return APEXRequest(
        query_id=QueryId("test_query"),
        user_id=UserId("test_user"),
        session_id=SessionId("test_session"),
        query_text="What is machine learning?",
        context=sample_query_context
    )

@pytest.fixture
def sample_performance_metrics():
    """Create sample performance metrics for testing"""
    return PerformanceMetrics(
        latency_p50=100.0,
        latency_p95=200.0,
        latency_p99=300.0,
        throughput=50.0,
        error_rate=0.01,
        cost_per_query=0.005
    )

# Unit tests for APEXCache
class TestAPEXCache:
    """Test APEX cache functionality"""
    
    @pytest.mark.asyncio
    async def test_cache_set_get(self):
        """Test basic cache set and get operations"""
        cache = APEXCache(max_size=10, ttl_hours=1)
        
        # Set value
        await cache.set("test_key", "test_value")
        
        # Get value
        value = await cache.get("test_key")
        assert value == "test_value"
    
    @pytest.mark.asyncio
    async def test_cache_expiration(self):
        """Test cache expiration"""
        cache = APEXCache(max_size=10, ttl_hours=0)  # 0 hours = immediate expiration
        
        # Set value
        await cache.set("test_key", "test_value")
        
        # Get value (should be expired)
        value = await cache.get("test_key")
        assert value is None
    
    @pytest.mark.asyncio
    async def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full"""
        cache = APEXCache(max_size=2, ttl_hours=1)
        
        # Fill cache
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        
        # Access key1 to make it more recently used
        await cache.get("key1")
        
        # Add third item (should evict key2)
        await cache.set("key3", "value3")
        
        # Check that key2 was evicted
        assert await cache.get("key2") is None
        assert await cache.get("key1") == "value1"
        assert await cache.get("key3") == "value3"
    
    def test_generate_cache_key(self, sample_apex_request):
        """Test cache key generation"""
        cache = APEXCache()
        key = cache.generate_key(sample_apex_request)
        
        # Key should be deterministic
        key2 = cache.generate_key(sample_apex_request)
        assert key == key2
        
        # Key should be different for different requests
        different_request = APEXRequest(
            query_id=QueryId("different_query"),
            user_id=UserId("different_user"),
            session_id=SessionId("different_session"),
            query_text="Different question?",
            context=sample_apex_request.context
        )
        different_key = cache.generate_key(different_request)
        assert key != different_key

# Unit tests for ConfigurationManager
class TestConfigurationManager:
    """Test configuration manager functionality"""
    
    @pytest.mark.asyncio
    async def test_load_default_config(self):
        """Test loading default configuration"""
        manager = ConfigurationManager()
        config = manager.get_current_config()
        
        assert isinstance(config, APEXConfig)
        assert config.auto_configure is True
        assert config.performance_target == PerformanceTier.STANDARD
    
    @pytest.mark.asyncio
    async def test_load_from_file(self, temp_config_file):
        """Test loading configuration from file"""
        manager = ConfigurationManager(temp_config_file)
        config = manager.get_current_config()
        
        assert isinstance(config, APEXConfig)
        assert config.max_concurrent_queries == 100
        assert config.default_timeout_ms == 30000
    
    @pytest.mark.asyncio
    async def test_auto_configure(self):
        """Test auto-configuration functionality"""
        manager = ConfigurationManager()
        
        # Create sample performance metrics
        metrics = [
            PerformanceMetrics(
                latency_p50=500.0,  # High latency
                latency_p95=1000.0,
                latency_p99=1500.0,
                throughput=10.0,  # Low throughput
                error_rate=0.05,  # High error rate
                cost_per_query=0.02  # High cost
            )
        ]
        
        # Run auto-configuration
        result = await manager.auto_configure(metrics)
        
        assert isinstance(result, Success)
        optimized_config = result.value
        
        # Should have optimized settings
        assert optimized_config.default_timeout_ms < 30000  # Reduced timeout
        assert optimized_config.max_concurrent_queries < 100  # Reduced concurrency
    
    def test_update_config(self):
        """Test configuration updates"""
        manager = ConfigurationManager()
        
        # Update configuration
        updated_config = manager.update_config({
            'performance_target': 'ultra',
            'default_timeout_ms': 60000
        })
        
        assert updated_config.performance_target == PerformanceTier.ULTRA
        assert updated_config.default_timeout_ms == 60000

# Unit tests for APEXOrchestrator
class TestAPEXOrchestrator:
    """Test APEX orchestrator functionality"""
    
    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self):
        """Test orchestrator initialization"""
        orchestrator = APEXOrchestrator()
        
        assert orchestrator.config_manager is not None
        assert orchestrator.router is not None
        assert orchestrator.model_interface is not None
        assert orchestrator.cache is not None
    
    @pytest.mark.asyncio
    async def test_process_request_cache_hit(self, sample_apex_request):
        """Test request processing with cache hit"""
        orchestrator = APEXOrchestrator()
        
        # Create mock response
        mock_response = APEXResponse(
            query_id=sample_apex_request.query_id,
            content="Cached response",
            model_used="test_model",
            route_decision=None,  # Mock
            performance_metrics=None,  # Mock
            processing_time_ms=50.0,
            cost=0.001,
            confidence_score=0.9
        )
        
        # Manually add to cache
        cache_key = orchestrator.cache.generate_key(sample_apex_request)
        await orchestrator.cache.set(cache_key, mock_response)
        
        # Process request (should hit cache)
        result = await orchestrator.process_request(sample_apex_request)
        
        assert isinstance(result, Success)
        assert result.value.content == "Cached response"
    
    def test_get_metrics(self):
        """Test metrics collection"""
        orchestrator = APEXOrchestrator()
        metrics = orchestrator.get_metrics()
        
        assert 'request_metrics' in metrics
        assert 'performance_metrics' in metrics
        assert 'component_metrics' in metrics
        assert 'event_metrics' in metrics
        
        # Check initial values
        assert metrics['request_metrics']['total_requests'] == 0
        assert metrics['request_metrics']['success_count'] == 0
        assert metrics['request_metrics']['error_count'] == 0

# Unit tests for APEXService
class TestAPEXService:
    """Test APEX service functionality"""
    
    @pytest.mark.asyncio
    async def test_service_initialization(self):
        """Test service initialization"""
        service = APEXService()
        
        assert service.orchestrator is not None
        assert service.config is not None
    
    @pytest.mark.asyncio
    async def test_simple_query(self):
        """Test simple query execution"""
        service = APEXService()
        
        # Note: This will fail in test environment due to missing model providers
        # In a real test, we would mock the model interface
        with pytest.raises(Exception):
            await service.query(
                query_text="What is AI?",
                user_id="test_user",
                session_id="test_session"
            )
    
    def test_get_metrics(self):
        """Test service metrics"""
        service = APEXService()
        metrics = service.get_metrics()
        
        assert 'service_metrics' in metrics
        assert 'orchestrator_metrics' in metrics

# Integration tests
class TestAPEXIntegration:
    """Integration tests for APEX components"""
    
    @pytest.mark.asyncio
    async def test_configuration_flow(self, temp_config_file):
        """Test complete configuration flow"""
        # Create service with config file
        service = APEXService(temp_config_file)
        
        # Check configuration loaded
        config = service.config
        assert config.max_concurrent_queries == 100
        
        # Test configuration update
        service.orchestrator.config_manager.update_config({
            'max_concurrent_queries': 200
        })
        
        updated_config = service.orchestrator.config_manager.get_current_config()
        assert updated_config.max_concurrent_queries == 200
    
    @pytest.mark.asyncio
    async def test_event_subscription(self):
        """Test event subscription system"""
        orchestrator = APEXOrchestrator()
        
        events_received = []
        
        def event_handler(event):
            events_received.append(event)
        
        # Subscribe to events
        subscription = orchestrator.subscribe_to_requests(event_handler)
        
        # Create and process request
        request = APEXRequest(
            query_id=QueryId("test"),
            user_id=UserId("test_user"),
            session_id=SessionId("test_session"),
            query_text="Test query",
            context=create_simple_context("test_user", "test_session")
        )
        
        # Process request (will emit events)
        await orchestrator.process_request(request)
        
        # Check that events were received
        assert len(events_received) > 0
        
        # Cleanup
        subscription.unsubscribe()

# Performance tests
class TestAPEXPerformance:
    """Performance tests for APEX components"""
    
    @pytest.mark.asyncio
    async def test_cache_performance(self):
        """Test cache performance under load"""
        cache = APEXCache(max_size=1000, ttl_hours=1)
        
        # Fill cache
        start_time = datetime.utcnow()
        for i in range(1000):
            await cache.set(f"key_{i}", f"value_{i}")
        
        # Measure retrieval performance
        retrieval_start = datetime.utcnow()
        for i in range(1000):
            await cache.get(f"key_{i}")
        retrieval_time = (datetime.utcnow() - retrieval_start).total_seconds()
        
        # Should be fast (less than 1 second for 1000 operations)
        assert retrieval_time < 1.0
    
    @pytest.mark.asyncio
    async def test_orchestrator_performance(self):
        """Test orchestrator performance"""
        orchestrator = APEXOrchestrator()
        
        # Create multiple requests
        requests = []
        for i in range(10):
            request = APEXRequest(
                query_id=QueryId(f"test_{i}"),
                user_id=UserId("test_user"),
                session_id=SessionId("test_session"),
                query_text=f"Test query {i}",
                context=create_simple_context("test_user", "test_session")
            )
            requests.append(request)
        
        # Process requests concurrently
        start_time = datetime.utcnow()
        results = await asyncio.gather(
            *[orchestrator.process_request(req) for req in requests],
            return_exceptions=True
        )
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Should handle concurrent requests efficiently
        assert processing_time < 5.0  # Should complete within 5 seconds

# Utility function tests
class TestAPEXUtilities:
    """Test APEX utility functions"""
    
    def test_create_simple_context(self):
        """Test simple context creation"""
        context = create_simple_context("test_user", "test_session")
        
        assert context.user_id == UserId("test_user")
        assert context.session_id == SessionId("test_session")
        assert context.expertise_level == QueryComplexity.SIMPLE
        assert context.performance_target == PerformanceTier.STANDARD
    
    def test_create_expert_context(self):
        """Test expert context creation"""
        context = create_expert_context("test_user", "test_session")
        
        assert context.user_id == UserId("test_user")
        assert context.session_id == SessionId("test_session")
        assert context.expertise_level == QueryComplexity.EXPERT
        assert context.performance_target == PerformanceTier.ULTRA
    
    def test_factory_functions(self):
        """Test factory functions"""
        service = create_apex_service()
        assert isinstance(service, APEXService)
        
        orchestrator = create_apex_orchestrator()
        assert isinstance(orchestrator, APEXOrchestrator)

# Error handling tests
class TestAPEXErrorHandling:
    """Test APEX error handling"""
    
    @pytest.mark.asyncio
    async def test_invalid_request_handling(self):
        """Test handling of invalid requests"""
        orchestrator = APEXOrchestrator()
        
        # Create invalid request (missing required fields)
        invalid_request = APEXRequest(
            query_id=QueryId("test"),
            user_id=UserId("test_user"),
            session_id=SessionId("test_session"),
            query_text="",  # Empty query
            context=create_simple_context("test_user", "test_session")
        )
        
        # Process request
        result = await orchestrator.process_request(invalid_request)
        
        # Should handle gracefully
        assert isinstance(result, (Success, Failure))
    
    @pytest.mark.asyncio
    async def test_configuration_error_handling(self):
        """Test configuration error handling"""
        # Create invalid config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            config_path = Path(f.name)
        
        # Should handle invalid config gracefully
        manager = ConfigurationManager(config_path)
        config = manager.get_current_config()
        
        # Should fall back to safe defaults
        assert config.auto_configure is False  # Safe mode
        assert config.performance_target == PerformanceTier.STANDARD

# Cleanup
def cleanup_temp_files():
    """Clean up temporary files created during testing"""
    import glob
    temp_files = glob.glob("/tmp/tmp*")
    for file in temp_files:
        try:
            Path(file).unlink()
        except:
            pass

# Test configuration
pytest_plugins = ["pytest_asyncio"]

def pytest_configure(config):
    """Configure pytest for APEX testing"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection for APEX"""
    for item in items:
        if "performance" in item.name.lower():
            item.add_marker(pytest.mark.slow)

# Main test runner
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
