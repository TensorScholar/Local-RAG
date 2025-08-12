"""
Distributed Processing module for Local-RAG system.

This module provides advanced distributed processing capabilities including:
- Horizontal scaling and load balancing across multiple nodes
- Microservices orchestration and service discovery
- Distributed vector stores with sharding and replication
- Real-time stream processing pipelines
- Fault tolerance and high availability mechanisms
- Distributed caching and state management
- Cross-node communication and synchronization

Author: Elite Technical Implementation Team
Version: 2.3.0
License: MIT
"""

from .load_balancer import AdvancedLoadBalancer
from .microservices_orchestrator import MicroservicesOrchestrator
from .distributed_vector_store import DistributedVectorStore
from .stream_processor import AdvancedStreamProcessor

__all__ = [
    'AdvancedLoadBalancer',
    'MicroservicesOrchestrator',
    'DistributedVectorStore',
    'AdvancedStreamProcessor'
]
