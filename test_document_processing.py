#!/usr/bin/env python3
"""
Document Processing Validation Test Script
"""

import asyncio
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_document_processing():
    """Test document processing capabilities."""
    try:
        from src.integration_interface import AdvancedRAGSystem
        
        print("📄 Testing Document Processing...")
        print("=" * 50)
        
        # Initialize system
        rag = AdvancedRAGSystem()
        
        # Test document processing
        print("1. Testing document processing...")
        
        # Create a test document
        test_content = """
        This is a test document for the Local-RAG system.
        It contains multiple paragraphs to test document processing capabilities.
        
        The system should be able to:
        - Parse text documents
        - Extract meaningful content
        - Process multiple formats
        - Handle different document structures
        """
        
        # Test document addition
        print("2. Testing document addition...")
        try:
            result = await rag.add_document(
                content=test_content,
                metadata={"title": "Test Document", "type": "text"}
            )
            print(f"✅ Document added successfully: {result}")
        except Exception as e:
            print(f"❌ Document addition failed: {e}")
        
        # Test document retrieval
        print("\n3. Testing document retrieval...")
        try:
            query = "What can the system do?"
            results = await rag.query_documents(query, max_results=5)
            print(f"✅ Query executed successfully")
            print(f"📊 Found {len(results) if results else 0} results")
        except Exception as e:
            print(f"❌ Document retrieval failed: {e}")
        
        print("\n🎉 Document processing validation completed!")
        
    except Exception as e:
        print(f"❌ Document processing validation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_document_processing())
