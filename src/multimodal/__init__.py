"""
Multi-modal document processing module for Local-RAG system.

This module provides advanced multi-modal capabilities including:
- Image processing with OCR and visual content analysis
- Audio processing with speech-to-text and audio analysis
- Video processing with frame extraction and temporal analysis
- Cross-modal retrieval with unified search capabilities
"""

from .image_processor import AdvancedImageProcessor
from .audio_processor import AdvancedAudioProcessor
from .video_processor import AdvancedVideoProcessor
from .cross_modal_retriever import CrossModalRetriever

__all__ = [
    'AdvancedImageProcessor',
    'AdvancedAudioProcessor', 
    'AdvancedVideoProcessor',
    'CrossModalRetriever'
]
