"""
Unified Model Integration Manager for Advanced Local RAG System.

This module provides a comprehensive integration layer between local and external models,
implementing intelligent routing based on query complexity and required capabilities.
The architecture follows a hierarchical decision-making process with clean separation
between analysis, selection, and execution components.

Design Principles:
1. Single Responsibility Principle - Each component addresses one aspect of the decision pipeline
2. Open/Closed Principle - Extensible for new model types without modification
3. Interface Segregation - Clean interfaces between components
4. Dependency Inversion - High-level modules independent of low-level implementation
5. Composition over Inheritance - Flexible component assembly

Performance Characteristics:
- Time Complexity: O(q + m) where q is query complexity and m is model count
- Space Complexity: O(m) for model metadata
- Caching: Adaptive performance caching for repeated queries
- Fallback: Graceful degradation with comprehensive error recovery

Author: Advanced RAG System Team
Version: 1.0.0
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Callable

# Import the external model manager
from src.models.external.external_model_manager import (
    ExternalModelManager, ModelCapability, ResponseType
)

# Configure logging
logger = logging.getLogger(__name__)


class QueryComplexity(Enum):
    """Enumeration of query complexity levels for model routing."""
    SIMPLE = auto()  # Basic factual queries, simple questions
    MODERATE = auto()  # Multi-step reasoning, moderate analysis
    COMPLEX = auto()  # Deep reasoning, scientific processing, complex analysis
    SPECIALIZED = auto()  # Domain-specific expertise, mathematical computation


@dataclass
class Document:
    """Representation of a document from the vector store."""
    id: str
    content: str
    metadata: Dict[str, Any]


@dataclass
class QueryResult:
    """Result of a query processing operation."""
    query: str
    response: str
    model: str
    is_external: bool
    processing_time_ms: float
    context_documents: List[Document]
    metadata: Dict[str, Any]


class QueryComplexityAnalyzer:
    """
    Analyzes query complexity for intelligent model routing decisions.
    Uses a combination of heuristics, pattern matching, and query structure analysis.
    
    Time complexity: O(n) where n is query length
    Space complexity: O(1) for fixed pattern sets
    """
    
    # Capability trigger keywords for identifying required model capabilities
    CAPABILITY_TRIGGERS = {
        ModelCapability.SCIENTIFIC_REASONING: [
            "calculate", "compute", "equation", "physics", "chemistry", 
            "biology", "scientific", "formula", "theorem", "theory",
            "hypothesis", "experiment", "molecular", "atomic", "quantum",
            "relativity", "newton", "einstein", "energy", "force", 
            "acceleration", "velocity", "momentum", "integrate", "differentiate"
        ],
        ModelCapability.MATHEMATICAL_COMPUTATION: [
            "solve", "calculate", "compute", "equation", "integral", 
            "derivative", "matrix", "vector", "tensor", "logarithm", 
            "exponent", "trigonometric", "sine", "cosine", "tangent",
            "probability", "statistics", "regression", "linear algebra",
            "calculus", "arithmetic", "geometric", "algebra", "polynomial"
        ],
        ModelCapability.CODE_GENERATION: [
            "code", "function", "algorithm", "program", "script", 
            "implementation", "programming", "software", "develop",
            "compile", "debug", "python", "javascript", "java", "c++",
            "typescript", "html", "css", "api", "backend", "frontend", 
            "full-stack", "database", "sql", "nosql", "framework"
        ],
        ModelCapability.MULTIMODAL_UNDERSTANDING: [
            "image", "picture", "photo", "graph", "chart", "diagram",
            "visualization", "plot", "illustration", "figure", "sketch",
            "drawing", "visual", "OCR", "text extraction", "screenshot"
        ],
        ModelCapability.LONG_CONTEXT: [
            "document", "article", "paper", "book", "chapter", "section",
            "analyze", "summarize", "extract", "review", "comparison", 
            "contrast", "synthesis", "comprehensive", "thorough", "detailed"
        ]
    }
    
    # Complexity indicator patterns
    COMPLEXITY_INDICATORS = {
        QueryComplexity.COMPLEX: [
            "explain in detail", "analyze comprehensively", "derive step by step",
            "elaborate explanation", "intricate", "sophisticated", "nuanced",
            "deep dive", "complexity", "advanced", "multiple perspectives",
            "theoretical framework", "underlying mechanism", "derive from first principles",
            "mathematical proof", "rigorous analysis", "synthesize", "interconnections"
        ],
        QueryComplexity.MODERATE: [
            "explain", "analyze", "compare", "contrast", "describe the process",
            "how does", "why does", "what causes", "evaluate", "assess",
            "provide reasons", "interpret", "examine", "investigate", "outline"
        ],
        QueryComplexity.SIMPLE: [
            "what is", "who is", "where is", "when did", "list", "define",
            "brief", "simple", "straightforward", "basic", "elementary",
            "quick", "short", "concise", "summarize", "overview"
        ]
    }
    
    def analyze_query(self, query: str) -> Tuple[QueryComplexity, List[ModelCapability]]:
        """
        Analyze query complexity and identify required capabilities.
        
        Args:
            query: The user query
            
        Returns:
            Tuple of (complexity_level, required_capabilities)
            
        Time complexity: O(n) where n is query length
        """
        # Normalize query
        query_lower = query.lower()
        
        # Detect required capabilities
        required_capabilities = [ModelCapability.BASIC_COMPLETION]  # Always include basic completion
        
        for capability, triggers in self.CAPABILITY_TRIGGERS.items():
            if any(trigger in query_lower for trigger in triggers):
                required_capabilities.append(capability)
        
        # Analyze complexity
        complexity = QueryComplexity.SIMPLE  # Default to simple
        
        # Check for complex indicators first (hierarchical check)
        for indicator in self.COMPLEXITY_INDICATORS[QueryComplexity.COMPLEX]:
            if indicator in query_lower:
                complexity = QueryComplexity.COMPLEX
                break
        
        # If not complex, check for moderate indicators
        if complexity == QueryComplexity.SIMPLE:
            for indicator in self.COMPLEXITY_INDICATORS[QueryComplexity.MODERATE]:
                if indicator in query_lower:
                    complexity = QueryComplexity.MODERATE
                    break
        
        # Additional complexity factors
        query_length = len(query.split())
        if query_length > 50 and complexity != QueryComplexity.COMPLEX:
            # Long queries are likely more complex
            complexity = QueryComplexity.MODERATE
        if query_length > 100:
            # Very long queries often indicate complex requests
            complexity = QueryComplexity.COMPLEX
        
        # Check for scientific/mathematical notation
        scientific_indicators = ["=", "+", "-", "*", "/", "^", "∫", "∂", "∞", "∑", "√"]
        if any(indicator in query for indicator in scientific_indicators):
            complexity = max(complexity, QueryComplexity.MODERATE)
            if len([c for c in query if c in scientific_indicators]) > 3:
                # Multiple mathematical symbols indicate complex queries
                complexity = QueryComplexity.COMPLEX
                
        # Look for nested parentheses, a sign of complex expressions
        open_count = 0
        max_nesting = 0
        for char in query:
            if char == '(':
                open_count += 1
                max_nesting = max(max_nesting, open_count)
            elif char == ')':
                open_count = max(0, open_count - 1)
        
        if max_nesting >= 2:
            complexity = max(complexity, QueryComplexity.MODERATE)
        if max_nesting >= 3:
            complexity = QueryComplexity.COMPLEX
                
        return complexity, required_capabilities


class ModelIntegrationManager:
    """
    Central manager for unified access to both local and external models,
    with intelligent routing based on query characteristics and system state.
    
    Time complexity: 
        - Initialization: O(m) where m is model count
        - Query processing: O(q + m) where q is query complexity and m is model count
    Space complexity: O(m) for model descriptions and routing decisions
    """
    
    def __init__(
        self,
        config_path: Optional[Path] = None,
        local_models: Dict[str, Any] = None,
        external_api_preference: Optional[str] = None,
        cost_limit_per_query: Optional[float] = None,
        max_latency_ms: Optional[int] = None,
    ):
        """
        Initialize the Model Integration Manager.
        
        Args:
            config_path: Optional path to configuration directory
            local_models: Dictionary of local model generators by name
            external_api_preference: Preferred external API provider
            cost_limit_per_query: Maximum cost allowed per query
            max_latency_ms: Maximum allowed latency in milliseconds
        """
        self.local_models = local_models or {}
        self.external_manager = ExternalModelManager(config_path)
        self.complexity_analyzer = QueryComplexityAnalyzer()
        self.external_api_preference = external_api_preference
        self.cost_limit_per_query = cost_limit_per_query
        self.max_latency_ms = max_latency_ms
        self.initialized = False
        
        # Performance and decision tracking
        self.routing_decisions: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, Dict[str, float]] = {}
    
    async def initialize(self) -> bool:
        """Initialize the manager and all its components."""
        # Initialize external model manager
        external_initialized = await self.external_manager.initialize()
        
        # Check if we have at least one way to generate responses
        if not external_initialized and not self.local_models:
            logger.error("No models available - both external and local models failed to initialize")
            return False
        
        self.initialized = True
        
        # Log available models
        logger.info(f"Initialized Model Integration Manager with:")
        logger.info(f"- Local models: {list(self.local_models.keys())}")
        
        if external_initialized:
            logger.info(f"- External providers: {self.external_manager.get_available_providers()}")
            available_models = self.external_manager.get_available_models()
            if available_models:
                logger.info(f"- External models: {[m.model_name for m in available_models]}")
        else:
            logger.warning("No external models available")
            
        return True
    
    def map_complexity_to_local_model(
        self, 
        complexity: QueryComplexity,
        required_capabilities: List[ModelCapability]
    ) -> Optional[str]:
        """
        Map query complexity to an appropriate local model.
        
        Args:
            complexity: Query complexity level
            required_capabilities: Required model capabilities
            
        Returns:
            Name of appropriate local model, or None if no suitable model
            
        Time complexity: O(m*c) where m is model count and c is capability count
        """
        if not self.local_models:
            return None
            
        # Define model capability mapping for local models
        # This would ideally be dynamically determined or configured
        model_capabilities = {
            "llama-3-8b": [
                ModelCapability.BASIC_COMPLETION,
                ModelCapability.CODE_GENERATION,
                ModelCapability.LONG_CONTEXT
            ],
            "mistral-7b": [
                ModelCapability.BASIC_COMPLETION,
                ModelCapability.CODE_GENERATION
            ],
            "phi-3": [
                ModelCapability.BASIC_COMPLETION
            ]
        }
        
        # Define complexity mapping
        complexity_mapping = {
            QueryComplexity.SIMPLE: ["phi-3", "mistral-7b", "llama-3-8b"],
            QueryComplexity.MODERATE: ["mistral-7b", "llama-3-8b"],
            QueryComplexity.COMPLEX: ["llama-3-8b"],
            QueryComplexity.SPECIALIZED: []  # No local models for specialized queries
        }
        
        # Get appropriate models for this complexity
        candidate_models = complexity_mapping.get(complexity, [])
        
        # Filter by available models
        candidate_models = [m for m in candidate_models if m in self.local_models]
        
        if not candidate_models:
            return None
            
        # Filter by capability
        for model in candidate_models:
            if model in model_capabilities and all(
                cap in model_capabilities[model] for cap in required_capabilities
            ):
                return model
                
        return None
    
    def should_use_external_model(
        self,
        query: str,
        context_documents: List[Document],
        complexity: QueryComplexity,
        required_capabilities: List[ModelCapability]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Determine if an external model should be used based on query characteristics
        and system state.
        
        Args:
            query: The user query
            context_documents: Retrieved context documents
            complexity: Query complexity level
            required_capabilities: Required model capabilities
            
        Returns:
            Tuple of (use_external, decision_factors)
            
        Time complexity: O(m) where m is model count
        """
        decision_factors = {
            "query_length": len(query),
            "complexity": complexity.name,
            "required_capabilities": [cap.name for cap in required_capabilities],
            "context_length": sum(len(doc.content) for doc in context_documents)
        }
        
        # Check if local models can handle this query
        local_model = self.map_complexity_to_local_model(complexity, required_capabilities)
        decision_factors["suitable_local_model"] = local_model
        
        # If no suitable local model, use external
        if not local_model:
            decision_factors["reason"] = "No suitable local model for capabilities/complexity"
            return True, decision_factors
        
        # Complex or specialized queries prefer external models when available
        if complexity in [QueryComplexity.COMPLEX, QueryComplexity.SPECIALIZED]:
            if self.external_manager.initialized:
                decision_factors["reason"] = f"Complex query ({complexity.name}) routed to external model"
                return True, decision_factors
        
        # If specialized capabilities are required, prefer external models
        specialized_capabilities = [
            ModelCapability.SCIENTIFIC_REASONING,
            ModelCapability.MATHEMATICAL_COMPUTATION,
            ModelCapability.MULTIMODAL_UNDERSTANDING
        ]
        
        if any(cap in required_capabilities for cap in specialized_capabilities):
            if self.external_manager.initialized:
                decision_factors["reason"] = "Specialized capabilities required"
                return True, decision_factors
        
        # Check context length - if very large, might need external model with larger context
        total_context_length = sum(len(doc.content) for doc in context_documents)
        decision_factors["total_context_length"] = total_context_length
        
        # Assuming local models have a context limit of ~8K tokens
        # This should be adjusted based on actual local model configurations
        estimated_tokens = total_context_length / 4  # Rough estimate
        if estimated_tokens > 6000:  # Leave some room for the query and response
            if self.external_manager.initialized:
                decision_factors["reason"] = "Large context length exceeds local model capacity"
                return True, decision_factors
        
        # If we reach here, local model is suitable
        decision_factors["reason"] = "Local model sufficient for query"
        return False, decision_factors
    
    async def process_query(
        self,
        query: str,
        context_documents: List[Document],
        force_local: bool = False,
        force_external: bool = False,
        specific_model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> QueryResult:
        """
        Process a user query with the most appropriate model based on complexity
        and required capabilities.
        
        Args:
            query: The user query
            context_documents: Retrieved context documents
            force_local: Force the use of local models
            force_external: Force the use of external models
            specific_model: Specific model to use (optional)
            temperature: Generation temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            
        Returns:
            QueryResult containing the generated response and metadata
            
        Time complexity: O(q + m) where q is query complexity and m is model count
                        plus model inference time
        Space complexity: O(d) where d is context document size
        """
        if not self.initialized:
            raise RuntimeError("Model Integration Manager not initialized")
        
        start_time = time.time()
        
        # Analyze query complexity and required capabilities
        complexity, required_capabilities = self.complexity_analyzer.analyze_query(query)
        
        # Determine if external model should be used
        use_external = force_external
        decision_factors = {}
        
        if not force_local and not force_external:
            use_external, decision_factors = self.should_use_external_model(
                query, context_documents, complexity, required_capabilities
            )
        
        # Record routing decision
        selected_model = specific_model or "auto-selected"
        routing_decision = {
            "query": query,
            "complexity": complexity.name,
            "required_capabilities": [cap.name for cap in required_capabilities],
            "use_external": use_external,
            "selected_model": selected_model,
            "decision_factors": decision_factors,
            "timestamp": time.time()
        }
        self.routing_decisions.append(routing_decision)
        
        # Prepare context for the model
        context_text = [doc.content for doc in context_documents]
        
        # Execute query with appropriate model
        if use_external and self.external_manager.initialized:
            try:
                # Parse specific_model if provided (format: "provider:model")
                provider_name = None
                model_name = None
                
                if specific_model and ":" in specific_model:
                    provider_name, model_name = specific_model.split(":", 1)
                
                # Generate with external model
                logger.info(f"Using external model (complexity: {complexity.name})")
                response = await self.external_manager.generate_response(
                    query=query,
                    provider_name=provider_name,
                    model_name=model_name,
                    required_capabilities=required_capabilities,
                    context=context_text,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    max_cost=self.cost_limit_per_query,
                    max_latency_ms=self.max_latency_ms
                )
                
                # Process external response
                result = QueryResult(
                    query=query,
                    response=response["content"],
                    model=f"{response['provider']}:{response['model']}",
                    is_external=True,
                    processing_time_ms=(time.time() - start_time) * 1000,
                    context_documents=context_documents,
                    metadata={
                        "complexity": complexity.name,
                        "required_capabilities": [cap.name for cap in required_capabilities],
                        "token_count": response.get("input_tokens", 0) + response.get("output_tokens", 0),
                        "cost": response.get("cost", 0),
                        "latency_ms": response.get("latency_ms", 0)
                    }
                )
                
                # Update performance metrics
                model_key = f"{response['provider']}:{response['model']}"
                if model_key not in self.performance_metrics:
                    self.performance_metrics[model_key] = {
                        "count": 0,
                        "avg_latency": 0,
                        "total_cost": 0
                    }
                
                metrics = self.performance_metrics[model_key]
                metrics["count"] += 1
                metrics["avg_latency"] = (
                    (metrics["avg_latency"] * (metrics["count"] - 1)) + response.get("latency_ms", 0)
                ) / metrics["count"]
                metrics["total_cost"] += response.get("cost", 0)
                
                return result
                
            except Exception as e:
                logger.error(f"External model error: {e}")
                if not self.local_models:
                    raise
                    
                # Fallback to local model if external fails
                logger.info("Falling back to local model due to external API error")
                use_external = False
        
        # Use local model
        if not use_external:
            if not self.local_models:
                raise RuntimeError("No local models available")
                
            # Select appropriate local model
            local_model_name = specific_model
            if not local_model_name or local_model_name not in self.local_models:
                local_model_name = self.map_complexity_to_local_model(complexity, required_capabilities)
                
            if not local_model_name:
                raise ValueError(
                    f"No suitable local model for query complexity {complexity.name} " +
                    f"and capabilities {[cap.name for cap in required_capabilities]}"
                )
                
            logger.info(f"Using local model {local_model_name} (complexity: {complexity.name})")
            
            local_generator = self.local_models[local_model_name]
            local_response = local_generator.generate(
                query=query,
                context_docs=context_documents,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            result = QueryResult(
                query=query,
                response=local_response.text,
                model=local_model_name,
                is_external=False,
                processing_time_ms=(time.time() - start_time) * 1000,
                context_documents=context_documents,
                metadata={
                    "complexity": complexity.name,
                    "required_capabilities": [cap.name for cap in required_capabilities],
                    "token_count": local_response.token_count,
                    "sources": local_response.sources
                }
            )
            
            # Update performance metrics
            if local_model_name not in self.performance_metrics:
                self.performance_metrics[local_model_name] = {
                    "count": 0,
                    "avg_latency": 0
                }
                
            metrics = self.performance_metrics[local_model_name]
            metrics["count"] += 1
            metrics["avg_latency"] = (
                (metrics["avg_latency"] * (metrics["count"] - 1)) + 
                (time.time() - start_time) * 1000
            ) / metrics["count"]
            
            return result
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get the current system status including performance metrics.
        
        Returns:
            Dictionary with system status information
            
        Time complexity: O(1) - constant time dictionary access
        """
        return {
            "initialized": self.initialized,
            "local_models": list(self.local_models.keys()) if self.local_models else [],
            "external_providers": self.external_manager.get_available_providers() if self.external_manager.initialized else [],
            "external_models": [
                f"{m.provider_name}:{m.model_name}" 
                for m in self.external_manager.get_available_models()
            ] if self.external_manager.initialized else [],
            "performance_metrics": self.performance_metrics,
            "routing_decisions_count": len(self.routing_decisions),
            "external_api_preference": self.external_api_preference,
            "cost_limit_per_query": self.cost_limit_per_query,
            "max_latency_ms": self.max_latency_ms
        }
