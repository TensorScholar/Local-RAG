#!/usr/bin/env python3
"""
Advanced RAG System - Computational Excellence Entry Point

A state-of-the-art implementation showcasing modern software engineering
principles and cutting-edge computational architecture.
"""

import asyncio
import sys
from pathlib import Path

# Add src to Python path for modular imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def main():
    """Main application entry point with sophisticated error handling."""
    try:
        from src.core.config import get_config
        from src.core.logging import setup_logging
        
        # Initialize advanced configuration system
        config = get_config()
        loggers = setup_logging(config.log_level)
        
        print("🧬 Advanced RAG System - Computational Excellence Engine")
        print("=========================================================")
        print(f"🔬 Environment: {config.environment.value}")
        print(f"📊 Log Level: {config.log_level}")
        print("✅ System initialization complete!")
        print("")
        print("🌐 Available Interfaces:")
        print("   • Web UI: http://localhost:8000")
        print("   • API Documentation: http://localhost:8000/api/docs")
        print("   • Development Server: python scripts/dev-server.py")
        
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
