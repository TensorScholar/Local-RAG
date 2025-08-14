#!/usr/bin/env python3
"""
System Validation Test Script
"""

import asyncio
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_system():
    """Test the system functionality."""
    try:
        from src.integration_interface import AdvancedRAGSystem
        
        print("🧪 Testing Local-RAG System...")
        print("=" * 50)
        
        # Test initialization
        print("1. Testing system initialization...")
        rag = AdvancedRAGSystem()
        print("✅ System initialized successfully")
        
        # Test system status
        print("\n2. Testing system status...")
        status = await rag.get_system_status()
        print("✅ System status retrieved successfully")
        
        # Display key metrics
        print(f"📊 Document count: {status.get('document_count', 'N/A')}")
        print(f"📄 Supported formats: {status.get('supported_formats', 'N/A')}")
        print(f"🔍 Vector store status: {status.get('vector_store', {}).get('status', 'N/A')}")
        print(f"🤖 Model integration: {status.get('model_integration', {}).get('status', 'N/A')}")
        
        # Test model providers
        print("\n3. Testing model providers...")
        providers = status.get('model_integration', {}).get('providers', {})
        for provider, info in providers.items():
            print(f"   ✅ {provider}: {info.get('status', 'N/A')}")
        
        print("\n🎉 System validation completed successfully!")
        
    except Exception as e:
        print(f"❌ System validation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_system())
