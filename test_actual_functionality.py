#!/usr/bin/env python3
"""
Actual Functionality Validation Test Script
"""

import asyncio
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_actual_functionality():
    """Test the actual functionality available in the system."""
    try:
        from src.integration_interface import AdvancedRAGSystem
        
        print("🔍 Testing ACTUAL System Functionality...")
        print("=" * 60)
        
        # Initialize system
        print("1. Testing system initialization...")
        rag = AdvancedRAGSystem()
        init_result = await rag.initialize()
        print(f"✅ System initialization: {init_result}")
        
        # Test system status
        print("\n2. Testing system status...")
        status = await rag.get_system_status()
        print("✅ System status retrieved successfully")
        
        # Display actual status
        print(f"📊 Document count: {status.get('document_count', 'N/A')}")
        print(f"📄 Supported formats: {status.get('supported_formats', 'N/A')}")
        print(f"🔍 Vector store status: {status.get('vector_store', {}).get('status', 'N/A')}")
        print(f"🤖 Model integration: {status.get('model_integration', {}).get('status', 'N/A')}")
        
        # Test document processing
        print("\n3. Testing document processing...")
        try:
            # Create a test document file
            test_file = Path("test_document.txt")
            test_content = """
            This is a test document for the Local-RAG system.
            It contains multiple paragraphs to test document processing capabilities.
            
            The system should be able to:
            - Parse text documents
            - Extract meaningful content
            - Process multiple formats
            - Handle different document structures
            """
            
            with open(test_file, 'w') as f:
                f.write(test_content)
            
            # Test document processing
            result = await rag.process_document(test_file)
            print(f"✅ Document processing: {result}")
            
            # Clean up
            test_file.unlink()
            
        except Exception as e:
            print(f"❌ Document processing failed: {e}")
        
        # Test query functionality
        print("\n4. Testing query functionality...")
        try:
            query_result = await rag.query("What can the system do?")
            print(f"✅ Query executed: {query_result}")
        except Exception as e:
            print(f"❌ Query failed: {e}")
        
        # Test available methods
        print("\n5. Available methods in AdvancedRAGSystem:")
        methods = [method for method in dir(rag) if not method.startswith('_')]
        for method in methods:
            print(f"   • {method}")
        
        print("\n🎉 Actual functionality validation completed!")
        
    except Exception as e:
        print(f"❌ Functionality validation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_actual_functionality())
