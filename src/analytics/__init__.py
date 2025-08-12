"""
Advanced Analytics & Insights module for Local-RAG system.

This module provides sophisticated analytics capabilities including:
- Real-time performance metrics and monitoring
- Query complexity and behavior analysis
- Document quality assessment and content analysis
- Predictive engines for usage forecasting and optimization
- Advanced statistical analysis and machine learning insights

Author: Elite Technical Implementation Team
Version: 2.2.0
License: MIT
"""

from .performance_analyzer import AdvancedPerformanceAnalyzer
from .query_analyzer import AdvancedQueryAnalyzer
from .content_analyzer import AdvancedContentAnalyzer
from .predictive_engine import AdvancedPredictiveEngine

__all__ = [
    'AdvancedPerformanceAnalyzer',
    'AdvancedQueryAnalyzer',
    'AdvancedContentAnalyzer',
    'AdvancedPredictiveEngine'
]
