"""
Integration Tests for External API Components.

This module provides comprehensive tests for validating the integration of external
API providers with the Advanced RAG System. It ensures correct behavior, robustness,
and performance of the integration layer through systematic test cases.

Testing Strategy:
1. Component Isolation - Testing individual components in isolation
2. Integration Validation - Testing interaction between components
3. End-to-End Scenarios - Testing complete query processing pipelines
4. Error Handling - Validating robustness under error conditions
5. Performance Characteristics - Validating expected performance

Each test class focuses on a specific component or interaction point, with
individual test methods addressing specific requirements and behaviors.

Author: Advanced RAG System Team
Version: 1.0.0
"""

import unittest
import asyncio
import os
import tempfile
import json
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

# Configure logging for tests
logging.basicConfig(level=logging.ERROR)

# Import components to test
from src.models.external.credential_manager import CredentialManager
from src.models.external.external_model_manager import (
    ExternalModelManager, ModelCapability, ModelMetadata
)
from src.models.external.model_integration_manager import (
    ModelIntegrationManager, QueryComplexity, Document, QueryResult
)


class TestCredentialManager(unittest.TestCase):
    """
    Test suite for the CredentialManager component.
    
    This class validates the secure storage, retrieval, and management of API credentials
    across different storage mechanisms and environment configurations.
    """
    
    def setUp(self):
        """Set up test environment with isolated credential storage."""
        # Create a temporary directory for test config
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_path = Path(self.temp_dir.name) / "credentials.json"
        
        # Sample test credentials
        self.test_credentials = {
            "openai": {"api_key": "test-openai-key"},
            "google": {"api_key": "test-google-key"},
            "anthropic": {"api_key": "test-anthropic-key"}
        }
        
        # Write test credentials to file
        with open(self.config_path, "w") as f:
            json.dump(self.test_credentials, f)
        
        # Save original environment variables
        self.original_env = os.environ.copy()
    
    def tearDown(self):
        """Clean up test environment and restore original state."""
        # Clean up temporary directory
        self.temp_dir.cleanup()
        
        # Restore original environment variables
        os.environ.clear()
        os.environ.update(self.original_env)
    
    def test_load_credentials_from_file(self):
        """Test loading credentials from config file."""
        manager = CredentialManager(self.config_path)
        
        # Verify all credentials were loaded
        for provider in ["openai", "google", "anthropic"]:
            self.assertTrue(manager.has_credentials(provider))
            self.assertEqual(
                manager.get_credentials(provider)["api_key"],
                self.test_credentials[provider]["api_key"]
            )
    
    def test_load_credentials_from_env(self):
        """Test loading credentials from environment variables."""
        # Set environment variables
        os.environ["OPENAI_API_KEY"] = "env-openai-key"
        
        # Initialize with non-existent config path
        manager = CredentialManager(Path(self.temp_dir.name) / "nonexistent.json")
        
        # Verify environment credentials were loaded
        self.assertTrue(manager.has_credentials("openai"))
        self.assertEqual(
            manager.get_credentials("openai")["api_key"],
            "env-openai-key"
        )
        
        # Verify missing credentials
        self.assertFalse(manager.has_credentials("google"))
        self.assertFalse(manager.has_credentials("anthropic"))
    
    def test_environment_variables_override_file(self):
        """Test that environment variables override file credentials."""
        # Set environment variable
        os.environ["OPENAI_API_KEY"] = "override-key"
        
        # Initialize with existing config
        manager = CredentialManager(self.config_path)
        
        # Verify environment variable takes precedence
        self.assertEqual(
            manager.get_credentials("openai")["api_key"],
            "override-key"
        )
        
        # Verify other credentials still loaded from file
        self.assertEqual(
            manager.get_credentials("google")["api_key"],
            self.test_credentials["google"]["api_key"]
        )
    
    def test_runtime_credential_updates(self):
        """Test adding and updating credentials at runtime."""
        manager = CredentialManager(self.config_path)
        
        # Add new provider
        manager.set_credentials("cohere", {"api_key": "test-cohere-key"})
        self.assertTrue(manager.has_credentials("cohere"))
        
        # Update existing provider
        manager.set_credentials("openai", {"api_key": "updated-openai-key"})
        self.assertEqual(
            manager.get_credentials("openai")["api_key"],
            "updated-openai-key"
        )
        
        # Save and reload to verify persistence
        manager.save()
        new_manager = CredentialManager(self.config_path)
        self.assertEqual(
            new_manager.get_credentials("cohere")["api_key"],
            "test-cohere-key"
        )
        self.assertEqual(
            new_manager.get_credentials("openai")["api_key"],
            "updated-openai-key"
        )
    
    def test_credential_validation(self):
        """Test validation of credential format."""
        # Test with invalid API key format
        with self.assertLogs(level='WARNING') as cm:
            # Write invalid credentials to file
            invalid_creds = {
                "openai": {"api_key": "invalid"}  # Too short
            }
            with open(self.config_path, "w") as f:
                json.dump(invalid_creds, f)
                
            # Initialize manager, which should trigger validation warning
            manager = CredentialManager(self.config_path)
            
            # Verify warning was logged
            self.assertTrue(any("short api_key" in msg for msg in cm.output))


class TestExternalModelManager(unittest.TestCase):
    """
    Test suite for the ExternalModelManager component.
    
    This class validates the initialization, model selection, and response generation
    capabilities of the external model manager across different providers and conditions.
    """
    
    def setUp(self):
        """Set up test environment with mock providers and credentials."""
        # Create a temporary directory for test config
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_path = Path(self.temp_dir.name) / "credentials.json"
        
        # Write test credentials to file
        test_credentials = {
            "openai": {"api_key": "test-openai-key"},
            "google": {"api_key": "test-google-key"},
            "anthropic": {"api_key": "test-anthropic-key"}
        }
        with open(self.config_path, "w") as f:
            json.dump(test_credentials, f)
        
        # Create the manager instance
        self.manager = ExternalModelManager(self.config_path)
    
    def tearDown(self):
        """Clean up test environment."""
        self.temp_dir.cleanup()
    
    @patch("src.models.external.external_model_manager.OpenAIProvider")
    @patch("src.models.external.external_model_manager.GoogleProvider")
    @patch("src.models.external.external_model_manager.AnthropicProvider")
    async def test_initialize(self, mock_anthropic, mock_google, mock_openai):
        """Test manager initialization with multiple providers."""
        # Configure provider mocks
        for mock_provider in [mock_openai, mock_google, mock_anthropic]:
            instance = mock_provider.return_value
            instance.initialize.return_value = True
            instance.get_available_models.return_value = [
                ModelMetadata(
                    provider_name=mock_provider.__name__.lower().replace("provider", ""),
                    model_name="test-model",
                    max_tokens=100,
                    context_window=1000,
                    capabilities=[ModelCapability.BASIC_COMPLETION],
                    token_cost_input=0.001,
                    token_cost_output=0.002,
                    latency_estimate_ms=100
                )
            ]
        
        # Initialize manager
        result = await self.manager.initialize()
        
        # Verify initialization
        self.assertTrue(result)
        self.assertTrue(self.manager.initialized)
        
        # Verify providers were initialized
        mock_openai.return_value.initialize.assert_called_once()
        mock_google.return_value.initialize.assert_called_once()
        mock_anthropic.return_value.initialize.assert_called_once()
    
    @patch("src.models.external.external_model_manager.OpenAIProvider")
    async def test_partial_initialization(self, mock_openai):
        """Test manager initialization with only some providers succeeding."""
        # Configure OpenAI provider to succeed
        openai_instance = mock_openai.return_value
        openai_instance.initialize.return_value = True
        openai_instance.get_available_models.return_value = [
            ModelMetadata(
                provider_name="openai",
                model_name="test-model",
                max_tokens=100,
                context_window=1000,
                capabilities=[ModelCapability.BASIC_COMPLETION],
                token_cost_input=0.001,
                token_cost_output=0.002,
                latency_estimate_ms=100
            )
        ]
        
        # Patch the other providers to fail
        with patch("src.models.external.external_model_manager.GoogleProvider") as mock_google:
            google_instance = mock_google.return_value
            google_instance.initialize.return_value = False
            
            with patch("src.models.external.external_model_manager.AnthropicProvider") as mock_anthropic:
                anthropic_instance = mock_anthropic.return_value
                anthropic_instance.initialize.side_effect = Exception("Test exception")
                
                # Initialize manager
                result = await self.manager.initialize()
                
                # Verify initialization succeeded with only OpenAI
                self.assertTrue(result)
                self.assertTrue(self.manager.initialized)
                self.assertEqual(len(self.manager.get_available_providers()), 1)
                self.assertEqual(self.manager.get_available_providers()[0], "openai")
    
    @patch("src.models.external.external_model_manager.OpenAIProvider")
    async def test_generate_response_direct(self, mock_openai):
        """Test response generation with specific provider and model."""
        # Configure OpenAI provider
        openai_instance = mock_openai.return_value
        openai_instance.initialize.return_value = True
        openai_instance.get_available_models.return_value = [
            ModelMetadata(
                provider_name="openai",
                model_name="test-model",
                max_tokens=100,
                context_window=1000,
                capabilities=[ModelCapability.BASIC_COMPLETION],
                token_cost_input=0.001,
                token_cost_output=0.002,
                latency_estimate_ms=100
            )
        ]
        
        # Mock response generation
        mock_response = {
            "provider": "openai",
            "model": "test-model",
            "content": "Test response",
            "latency_ms": 150,
            "input_tokens": 10,
            "output_tokens": 5,
            "cost": 0.00015
        }
        openai_instance.generate_response = AsyncMock(return_value=mock_response)
        
        # Initialize manager
        await self.manager.initialize()
        
        # Generate response with specific provider and model
        response = await self.manager.generate_response(
            query="Test query",
            provider_name="openai",
            model_name="test-model"
        )
        
        # Verify response
        self.assertEqual(response, mock_response)
        openai_instance.generate_response.assert_called_once_with(
            query="Test query",
            model_name="test-model",
            context=None,
            temperature=0.7,
            max_tokens=None
        )
    
    
@patch("src.models.external.external_model_manager.OpenAIProvider")
@patch("src.models.external.external_model_manager.ExternalModelSelector")
async def test_generate_response_auto_select(self, mock_selector, mock_openai):
    """Test response generation with automatic model selection."""
    # Configure OpenAI provider
    openai_instance = mock_openai.return_value
    openai_instance.initialize.return_value = True
    openai_instance.get_available_models.return_value = [
        ModelMetadata(
            provider_name="openai",
            model_name="test-model",
            max_tokens=100,
            context_window=1000,
            capabilities=[ModelCapability.BASIC_COMPLETION],
            token_cost_input=0.001,
            token_cost_output=0.002,
            latency_estimate_ms=100
        )
    ]
    
    # Mock response generation
    mock_response = {
        "provider": "openai",
        "model": "test-model",
        "content": "Test response",
        "latency_ms": 150,
        "input_tokens": 10,
        "output_tokens": 5,
        "cost": 0.00015
    }
    openai_instance.generate_response = AsyncMock(return_value=mock_response)
    
    # Configure model selector
    selector_instance = mock_selector.return_value
    selector_instance.select_model.return_value = ("openai", "test-model")
    
    # Initialize manager
    self.manager.model_selector = selector_instance
    self.manager.providers = {"openai": openai_instance}
    self.manager.initialized = True
    
    # Generate response with auto-selection
    response = await self.manager.generate_response(
        query="Test query",
        required_capabilities=[ModelCapability.BASIC_COMPLETION]
    )
    
    # Verify response
    self.assertEqual(response, mock_response)
    selector_instance.select_model.assert_called_once()
    openai_instance.generate_response.assert_called_once()
    
@patch("src.models.external.external_model_manager.OpenAIProvider")
async def test_error_handling(self, mock_openai):
    """Test error handling during response generation."""
    # Configure OpenAI provider
    openai_instance = mock_openai.return_value
    openai_instance.initialize.return_value = True
    openai_instance.get_available_models.return_value = [
        ModelMetadata(
            provider_name="openai",
            model_name="test-model",
            max_tokens=100,
            context_window=1000,
            capabilities=[ModelCapability.BASIC_COMPLETION],
            token_cost_input=0.001,
            token_cost_output=0.002,
            latency_estimate_ms=100
        )
    ]
    
    # Mock response generation to raise exception
    openai_instance.generate_response = AsyncMock(side_effect=Exception("API error"))
    
    # Initialize manager
    await self.manager.initialize()
    
    # Attempt to generate response
    with self.assertRaises(Exception):
        await self.manager.generate_response(
            query="Test query",
            provider_name="openai",
            model_name="test-model"
        )
    
    # Verify error was tracked
    self.assertEqual(self.manager.error_count, 1)
    self.assertEqual(self.manager.request_count, 1)

@patch("src.models.external.external_model_manager.OpenAIProvider")
async def test_performance_metrics(self, mock_openai):
    """Test performance metric tracking during operation."""
    # Configure OpenAI provider
    openai_instance = mock_openai.return_value
    openai_instance.initialize.return_value = True
    openai_instance.get_available_models.return_value = [
        ModelMetadata(
            provider_name="openai",
            model_name="test-model",
            max_tokens=100,
            context_window=1000,
            capabilities=[ModelCapability.BASIC_COMPLETION],
            token_cost_input=0.001,
            token_cost_output=0.002,
            latency_estimate_ms=100
        )
    ]
    
    # Mock response generation
    mock_response = {
        "provider": "openai",
        "model": "test-model",
        "content": "Test response",
        "latency_ms": 150,
        "input_tokens": 10,
        "output_tokens": 5,
        "cost": 0.00015
    }
    openai_instance.generate_response = AsyncMock(return_value=mock_response)
    
    # Initialize manager
    await self.manager.initialize()
    
    # Generate multiple responses
    for _ in range(3):
        await self.manager.generate_response(
            query="Test query",
            provider_name="openai",
            model_name="test-model"
        )
    
    # Get performance metrics
    metrics = self.manager.get_performance_metrics()
    
    # Verify metric tracking
    self.assertEqual(metrics["request_count"], 3)
    self.assertEqual(metrics["error_count"], 0)
    self.assertGreaterEqual(metrics["uptime_seconds"], 0)
    self.assertEqual(metrics["success_rate"], 100.0)
    self.assertIn("openai", metrics["provider_latencies"])
    self.assertEqual(metrics["provider_latencies"]["openai"]["count"], 3)

class TestModelIntegrationManager(unittest.TestCase):
    """
    Test suite for the ModelIntegrationManager component.
    
    This class validates the intelligent routing, model selection, and query processing
    capabilities of the model integration manager under various conditions and scenarios.
    """
    
    def setUp(self):
        """Set up test environment with mock models and services."""
        # Create mock local models
        self.local_models = {
            "llama-3-8b": MagicMock(),
            "mistral-7b": MagicMock(),
            "phi-3": MagicMock()
        }
        
        # Configure mock responses for local models
        for model_name, model in self.local_models.items():
            mock_response = MagicMock()
            mock_response.text = f"Response from {model_name}"
            mock_response.token_count = 20
            mock_response.sources = ["doc1", "doc2"]
            model.generate.return_value = mock_response
        
        # Create temporary directory for configuration
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_path = Path(self.temp_dir.name)
        
        # Mock external model manager
        self.mock_external_manager = MagicMock()
        self.mock_external_manager.initialize = AsyncMock(return_value=True)
        
        # Initialize the integration manager with mocks
        # (This setup enables isolation of the manager from actual providers)
        with patch("src.models.external.model_integration_manager.ExternalModelManager", 
                  return_value=self.mock_external_manager):
            self.manager = ModelIntegrationManager(
                config_path=self.config_path,
                local_models=self.local_models,
                external_api_preference="openai",
                cost_limit_per_query=0.05,
                max_latency_ms=2000
            )
            # Patch complexity analyzer
            self.manager.complexity_analyzer = MagicMock()
            # Mark as initialized
            self.manager.initialized = True
    
    def tearDown(self):
        """Clean up test environment."""
        self.temp_dir.cleanup()
    
    def _create_test_documents(self, count=3, length=100):
        """Helper method to create test documents."""
        return [
            Document(
                id=f"doc{i}",
                content=f"Document {i} content " * (length // 10),
                metadata={"source": f"source{i}", "title": f"Document {i}"}
            )
            for i in range(1, count + 1)
        ]
    
    async def test_local_model_routing(self):
        """Test routing to local model when appropriate."""
        # Configure complexity analyzer to return SIMPLE complexity
        self.manager.complexity_analyzer.analyze_query.return_value = (
            QueryComplexity.SIMPLE, 
            [ModelCapability.BASIC_COMPLETION]
        )
        
        # Mock should_use_external_model to return False
        with patch.object(
            self.manager, 
            "should_use_external_model", 
            return_value=(False, {"reason": "Local model sufficient"})
        ):
            # Process a simple query
            documents = self._create_test_documents()
            result = await self.manager.process_query(
                query="What is the capital of France?",
                context_documents=documents
            )
            
            # Verify local model was used
            self.assertFalse(result.is_external)
            self.assertIn(result.model, self.local_models)
            self.mock_external_manager.generate_response.assert_not_called()
    
    async def test_external_model_routing(self):
        """Test routing to external model when appropriate."""
        # Configure complexity analyzer to return COMPLEX complexity
        self.manager.complexity_analyzer.analyze_query.return_value = (
            QueryComplexity.COMPLEX, 
            [ModelCapability.SCIENTIFIC_REASONING]
        )
        
        # Mock should_use_external_model to return True
        with patch.object(
            self.manager, 
            "should_use_external_model", 
            return_value=(True, {"reason": "Complex query requires external model"})
        ):
            # Configure external model response
            mock_response = {
                "provider": "openai",
                "model": "gpt-4",
                "content": "Paris is the capital of France",
                "latency_ms": 500,
                "input_tokens": 50,
                "output_tokens": 10,
                "cost": 0.002
            }
            self.mock_external_manager.generate_response = AsyncMock(return_value=mock_response)
            
            # Process a complex query
            documents = self._create_test_documents()
            result = await self.manager.process_query(
                query="Explain quantum mechanics and its implications for cosmology",
                context_documents=documents
            )
            
            # Verify external model was used
            self.assertTrue(result.is_external)
            self.assertEqual(result.model, "openai:gpt-4")
            self.mock_external_manager.generate_response.assert_called_once()
    
    async def test_force_local_model(self):
        """Test forcing the use of local model regardless of complexity."""
        # Configure complexity analyzer to return COMPLEX complexity
        self.manager.complexity_analyzer.analyze_query.return_value = (
            QueryComplexity.COMPLEX, 
            [ModelCapability.SCIENTIFIC_REASONING]
        )
        
        # should_use_external_model should not be called when forcing local
        with patch.object(
            self.manager, 
            "should_use_external_model"
        ) as mock_should_use:
            # Process a complex query but force local model
            documents = self._create_test_documents()
            result = await self.manager.process_query(
                query="Explain quantum mechanics and its implications for cosmology",
                context_documents=documents,
                force_local=True
            )
            
            # Verify local model was used despite complexity
            self.assertFalse(result.is_external)
            self.assertIn(result.model, self.local_models)
            mock_should_use.assert_not_called()
            self.mock_external_manager.generate_response.assert_not_called()
    
    async def test_force_external_model(self):
        """Test forcing the use of external model regardless of complexity."""
        # Configure complexity analyzer to return SIMPLE complexity
        self.manager.complexity_analyzer.analyze_query.return_value = (
            QueryComplexity.SIMPLE, 
            [ModelCapability.BASIC_COMPLETION]
        )
        
        # should_use_external_model should not be called when forcing external
        with patch.object(
            self.manager, 
            "should_use_external_model"
        ) as mock_should_use:
            # Configure external model response
            mock_response = {
                "provider": "openai",
                "model": "gpt-3.5-turbo",
                "content": "Paris is the capital of France",
                "latency_ms": 300,
                "input_tokens": 30,
                "output_tokens": 5,
                "cost": 0.0005
            }
            self.mock_external_manager.generate_response = AsyncMock(return_value=mock_response)
            
            # Process a simple query but force external model
            documents = self._create_test_documents()
            result = await self.manager.process_query(
                query="What is the capital of France?",
                context_documents=documents,
                force_external=True
            )
            
            # Verify external model was used despite simplicity
            self.assertTrue(result.is_external)
            self.assertEqual(result.model, "openai:gpt-3.5-turbo")
            mock_should_use.assert_not_called()
            self.mock_external_manager.generate_response.assert_called_once()
    
    async def test_specific_model_selection(self):
        """Test using a specific model by name."""
        # Configure complexity analyzer
        self.manager.complexity_analyzer.analyze_query.return_value = (
            QueryComplexity.MODERATE, 
            [ModelCapability.BASIC_COMPLETION]
        )
        
        # Process query with specific local model
        documents = self._create_test_documents()
        result = await self.manager.process_query(
            query="What is the capital of France?",
            context_documents=documents,
            specific_model="phi-3"
        )
        
        # Verify specific local model was used
        self.assertFalse(result.is_external)
        self.assertEqual(result.model, "phi-3")
        self.local_models["phi-3"].generate.assert_called_once()
        
        # Test specific external model
        # Configure external model response
        mock_response = {
            "provider": "openai",
            "model": "gpt-4",
            "content": "Paris is the capital of France",
            "latency_ms": 500,
            "input_tokens": 50,
            "output_tokens": 10,
            "cost": 0.002
        }
        self.mock_external_manager.generate_response = AsyncMock(return_value=mock_response)
        
        # Process query with specific external model
        result = await self.manager.process_query(
            query="What is the capital of France?",
            context_documents=documents,
            specific_model="openai:gpt-4"
        )
        
        # Verify specific external model was used
        self.assertTrue(result.is_external)
        self.assertEqual(result.model, "openai:gpt-4")
    
    async def test_fallback_to_local(self):
        """Test fallback to local model when external API fails."""
        # Configure complexity analyzer to return COMPLEX complexity
        self.manager.complexity_analyzer.analyze_query.return_value = (
            QueryComplexity.COMPLEX, 
            [ModelCapability.SCIENTIFIC_REASONING]
        )
        
        # Mock should_use_external_model to return True
        with patch.object(
            self.manager, 
            "should_use_external_model", 
            return_value=(True, {"reason": "Complex query requires external model"})
        ):
            # Configure external model to fail
            self.mock_external_manager.generate_response = AsyncMock(
                side_effect=Exception("API Error")
            )
            
            # Process a complex query
            documents = self._create_test_documents()
            result = await self.manager.process_query(
                query="Explain quantum mechanics and its implications for cosmology",
                context_documents=documents
            )
            
            # Verify fallback to local model occurred
            self.assertFalse(result.is_external)
            self.assertIn(result.model, self.local_models)
            self.mock_external_manager.generate_response.assert_called_once()
    
    def test_complexity_to_local_model_mapping(self):
        """Test mapping of query complexity to appropriate local model."""
        # Test SIMPLE complexity
        model = self.manager.map_complexity_to_local_model(
            QueryComplexity.SIMPLE,
            [ModelCapability.BASIC_COMPLETION]
        )
        self.assertEqual(model, "phi-3")
        
        # Test MODERATE complexity
        model = self.manager.map_complexity_to_local_model(
            QueryComplexity.MODERATE,
            [ModelCapability.BASIC_COMPLETION, ModelCapability.CODE_GENERATION]
        )
        self.assertEqual(model, "mistral-7b")
        
        # Test COMPLEX complexity
        model = self.manager.map_complexity_to_local_model(
            QueryComplexity.COMPLEX,
            [ModelCapability.BASIC_COMPLETION, ModelCapability.CODE_GENERATION]
        )
        self.assertEqual(model, "llama-3-8b")
        
        # Test SPECIALIZED complexity (should return None)
        model = self.manager.map_complexity_to_local_model(
            QueryComplexity.SPECIALIZED,
            [ModelCapability.SCIENTIFIC_REASONING]
        )
        self.assertIsNone(model)
        
        # Test capability mismatch
        model = self.manager.map_complexity_to_local_model(
            QueryComplexity.SIMPLE,
            [ModelCapability.MULTIMODAL_UNDERSTANDING]
        )
        self.assertIsNone(model)
    
    def test_external_model_decision_factors(self):
        """Test decision factors for external model routing."""
        # Test case: No suitable local model
        with patch.object(
            self.manager,
            "map_complexity_to_local_model",
            return_value=None
        ):
            use_external, factors = self.manager.should_use_external_model(
                "What is quantum computing?",
                self._create_test_documents(),
                QueryComplexity.MODERATE,
                [ModelCapability.SCIENTIFIC_REASONING]
            )
            self.assertTrue(use_external)
            self.assertIn("No suitable local model", factors["reason"])
        
        # Test case: Complex query
        use_external, factors = self.manager.should_use_external_model(
            "Explain in detail the implications of quantum mechanics for our understanding of reality",
            self._create_test_documents(),
            QueryComplexity.COMPLEX,
            [ModelCapability.BASIC_COMPLETION]
        )
        self.assertTrue(use_external)
        self.assertIn("Complex query", factors["reason"])
        
        # Test case: Specialized capability
        use_external, factors = self.manager.should_use_external_model(
            "Analyze this equation: E=mc²",
            self._create_test_documents(),
            QueryComplexity.MODERATE,
            [ModelCapability.MATHEMATICAL_COMPUTATION]
        )
        self.assertTrue(use_external)
        self.assertIn("Specialized capabilities", factors["reason"])
        
        # Test case: Large context length
        large_docs = self._create_test_documents(count=5, length=5000)
        use_external, factors = self.manager.should_use_external_model(
            "Summarize these documents",
            large_docs,
            QueryComplexity.MODERATE,
            [ModelCapability.BASIC_COMPLETION]
        )
        self.assertTrue(use_external)
        self.assertIn("Large context length", factors["reason"])
        
        # Test case: Simple query suitable for local model
        use_external, factors = self.manager.should_use_external_model(
            "What is the capital of France?",
            self._create_test_documents(),
            QueryComplexity.SIMPLE,
            [ModelCapability.BASIC_COMPLETION]
        )
        self.assertFalse(use_external)
        self.assertIn("Local model sufficient", factors["reason"])


class TestQueryComplexityAnalyzer(unittest.TestCase):
    """
    Test suite for the QueryComplexityAnalyzer component.
    
    This class validates the complexity analysis and capability detection
    functionality that drives intelligent model routing decisions.
    """
    
    def setUp(self):
        """Set up the complexity analyzer for testing."""
        self.analyzer = QueryComplexityAnalyzer()
    
    def test_simple_query_classification(self):
        """Test detection of simple queries."""
        simple_queries = [
            "What is Python?",
            "Who is Albert Einstein?",
            "Define artificial intelligence",
            "List the 5 largest cities in the world",
            "When was the Declaration of Independence signed?",
            "Where is the Eiffel Tower located?",
            "Give me a brief overview of climate change",
            "What's the capital of Japan?"
        ]
        
        for query in simple_queries:
            complexity, capabilities = self.analyzer.analyze_query(query)
            self.assertEqual(complexity, QueryComplexity.SIMPLE)
            self.assertIn(ModelCapability.BASIC_COMPLETION, capabilities)
    
    def test_moderate_query_classification(self):
        """Test detection of moderate complexity queries."""
        moderate_queries = [
            "Explain how neural networks learn from data",
            "Compare and contrast democracy and autocracy",
            "Describe the process of photosynthesis",
            "What causes climate change and what are its effects?",
            "Analyze the main themes in Shakespeare's Hamlet",
            "How does the immune system defend against pathogens?",
            "Evaluate the pros and cons of renewable energy sources",
            "What factors contributed to the fall of the Roman Empire?"
        ]
        
        for query in moderate_queries:
            complexity, capabilities = self.analyzer.analyze_query(query)
            self.assertEqual(complexity, QueryComplexity.MODERATE)
            self.assertIn(ModelCapability.BASIC_COMPLETION, capabilities)
    
    def test_complex_query_classification(self):
        """Test detection of complex queries."""
        complex_queries = [
            "Explain in detail how quantum computing differs from classical computing and its implications for cryptography",
            "Analyze comprehensively the economic, social, and political factors that led to the Great Depression",
            "Provide an intricate analysis of how climate change affects global ecosystems and biodiversity",
            "Derive from first principles the equations governing general relativity",
            "Synthesize the current understanding of consciousness from neuroscientific, philosophical, and psychological perspectives",
            "Elaborate on the interconnections between monetary policy, inflation, and unemployment in modern economies"
        ]
        
        for query in complex_queries:
            complexity, capabilities = self.analyzer.analyze_query(query)
            self.assertEqual(complexity, QueryComplexity.COMPLEX)
            self.assertIn(ModelCapability.BASIC_COMPLETION, capabilities)
    
    def test_capability_detection(self):
        """Test detection of specific capabilities from query text."""
        # Test code generation capability
        code_queries = [
            "Write a Python function to sort a list using quicksort",
            "Create a JavaScript function to fetch data from an API",
            "Implement a binary search tree in C++",
            "Debug this code snippet: for i in range(10): print(i"
        ]
        
        for query in code_queries:
            _, capabilities = self.analyzer.analyze_query(query)
            self.assertIn(ModelCapability.CODE_GENERATION, capabilities)
        
        # Test scientific reasoning capability
        science_queries = [
            "Explain Einstein's theory of relativity",
            "Calculate the molarity of a solution with 5g of NaCl in 100ml of water",
            "Describe the process of nuclear fusion in stars",
            "What is the relationship between force, mass, and acceleration?"
        ]
        
        for query in science_queries:
            _, capabilities = self.analyzer.analyze_query(query)
            self.assertIn(ModelCapability.SCIENTIFIC_REASONING, capabilities)
        
        # Test mathematical computation capability
        math_queries = [
            "Solve the equation 3x² + 5x - 2 = 0",
            "Calculate the integral of sin(x) from 0 to π",
            "What is the derivative of f(x) = x³ + 2x² - 5x + 3?",
            "Compute the dot product of vectors [1,2,3] and [4,5,6]"
        ]
        
        for query in math_queries:
            _, capabilities = self.analyzer.analyze_query(query)
            self.assertIn(ModelCapability.MATHEMATICAL_COMPUTATION, capabilities)
        
        # Test multimodal understanding capability
        multimodal_queries = [
            "Analyze this image and describe what you see",
            "Extract text from this screenshot",
            "Interpret the data in this chart",
            "What does this diagram represent?"
        ]
        
        for query in multimodal_queries:
            _, capabilities = self.analyzer.analyze_query(query)
            self.assertIn(ModelCapability.MULTIMODAL_UNDERSTANDING, capabilities)
        
        # Test long context capability
        long_context_queries = [
            "Summarize this research paper",
            "Analyze this 50-page document and extract key insights",
            "Compare and contrast these three articles",
            "Review this book chapter and highlight important concepts"
        ]
        
        for query in long_context_queries:
            _, capabilities = self.analyzer.analyze_query(query)
            self.assertIn(ModelCapability.LONG_CONTEXT, capabilities)
    
    def test_structural_complexity_factors(self):
        """Test detection of complexity based on query structure."""
        # Test nested parentheses
        query = "Explain how the mechanism of action for aspirin (acetylsalicylic acid) differs from that of ibuprofen (which belongs to the class of non-steroidal anti-inflammatory drugs (NSAIDs))."
        complexity, _ = self.analyzer.analyze_query(query)
        self.assertEqual(complexity, QueryComplexity.COMPLEX)
        
        # Test mathematical symbols
        query = "Calculate 5 + 3 * 4 / 2 - √(16) = ?"
        complexity, _ = self.analyzer.analyze_query(query)
        self.assertEqual(complexity, QueryComplexity.MODERATE)
        
        # Test multiple mathematical symbols
        query = "Solve for x: 3x² + 4x - 7 = 0 using the quadratic formula x = (-b ± √(b² - 4ac)) / 2a"
        complexity, _ = self.analyzer.analyze_query(query)
        self.assertEqual(complexity, QueryComplexity.COMPLEX)
        
        # Test length-based complexity
        long_query = "What is " + "very " * 50 + "interesting about this topic?"
        complexity, _ = self.analyzer.analyze_query(long_query)
        self.assertEqual(complexity, QueryComplexity.MODERATE)
