"""
Advanced ML Models module for Local-RAG system.

This module provides sophisticated machine learning capabilities including:
- Custom model training and fine-tuning with domain-specific optimization
- Model ensemble methods with advanced combination strategies
- Active learning systems for continuous model improvement
- Domain adaptation for industry-specific model customization
- Transfer learning with pre-trained model optimization
- Hyperparameter optimization and automated model selection
- Model interpretability and explainability frameworks
- Performance monitoring and model lifecycle management

Author: Elite Technical Implementation Team
Version: 2.4.0
License: MIT
"""

from .custom_model_trainer import AdvancedCustomModelTrainer
from .model_ensemble import AdvancedModelEnsemble
from .active_learning import AdvancedActiveLearning
from .domain_adaptation import AdvancedDomainAdaptation

__all__ = [
    'AdvancedCustomModelTrainer',
    'AdvancedModelEnsemble',
    'AdvancedActiveLearning',
    'AdvancedDomainAdaptation'
]
