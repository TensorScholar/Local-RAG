#!/usr/bin/env python3
"""
Local-RAG Web Server Startup Script

This script starts the FastAPI web server for the Local-RAG system.
"""

import sys
import os
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.web.server import run_server

if __name__ == "__main__":
    # Get configuration from environment variables
    host = os.getenv("RAG_HOST", "0.0.0.0")
    port = int(os.getenv("RAG_PORT", "8000"))
    reload = os.getenv("RAG_RELOAD", "false").lower() == "true"
    
    print(f"🚀 Starting Local-RAG Web Server...")
    print(f"📍 Host: {host}")
    print(f"🔌 Port: {port}")
    print(f"🔄 Reload: {reload}")
    print(f"🌐 Web Interface: http://{host}:{port}")
    print(f"📚 API Documentation: http://{host}:{port}/docs")
    print(f"❤️  Health Check: http://{host}:{port}/health")
    print("=" * 60)
    
    try:
        run_server(host=host, port=port, reload=reload)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)
