#!/usr/bin/env python3
"""
Comprehensive Validation Test for Local-RAG System

This test validates all the critical fixes implemented to address the validation issues.
"""

import asyncio
import sys
import time
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def comprehensive_validation():
    """Run comprehensive validation of all fixes."""
    print("🧪 COMPREHENSIVE VALIDATION TEST")
    print("=" * 80)
    print("Testing all critical fixes implemented")
    print("")
    
    try:
        from src.integration_interface import AdvancedRAGSystem
        
        # Test 1: System Initialization
        print("1️⃣ Testing System Initialization...")
        start_time = time.time()
        rag = AdvancedRAGSystem()
        init_result = await rag.initialize()
        init_time = time.time() - start_time
        
        if init_result:
            print(f"✅ System initialized successfully in {init_time:.2f}s")
            if init_time < 30:
                print("✅ Performance target met (<30s)")
            else:
                print(f"⚠️  Performance target missed ({init_time:.2f}s > 30s)")
        else:
            print("❌ System initialization failed")
            return
        
        # Test 2: Document Processing
        print("\n2️⃣ Testing Document Processing...")
        
        # Create test document
        test_file = Path("validation_test_doc.txt")
        test_content = """
        Local-RAG System Validation Document
        
        This is a test document to validate the Local-RAG system functionality.
        The system should be able to process this document and extract meaningful content.
        
        Key Features:
        - Document processing and chunking
        - Vector storage and retrieval
        - Query processing with LLM integration
        - Web interface and API endpoints
        
        This document contains multiple paragraphs to test the chunking functionality
        and ensure that the system can handle different types of content effectively.
        """
        
        with open(test_file, 'w') as f:
            f.write(test_content)
        
        # Process document
        doc_start = time.time()
        doc_result = await rag.process_document(test_file)
        doc_time = time.time() - doc_start
        
        if doc_result:
            print(f"✅ Document processed successfully in {doc_time:.2f}s")
        else:
            print("❌ Document processing failed")
        
        # Clean up
        test_file.unlink(missing_ok=True)
        
        # Test 3: Query Processing
        print("\n3️⃣ Testing Query Processing...")
        
        query_start = time.time()
        query_result = await rag.query("What are the key features of the Local-RAG system?")
        query_time = time.time() - query_start
        
        if query_result.get('success', False):
            print(f"✅ Query processed successfully in {query_time:.2f}s")
            print(f"📝 Response: {query_result.get('response', '')[:200]}...")
            print(f"🤖 Model used: {query_result.get('model', 'unknown')}")
            print(f"📊 Retrieved documents: {len(query_result.get('sources', []))}")
        else:
            print(f"❌ Query processing failed: {query_result.get('error', 'Unknown error')}")
        
        # Test 4: System Status
        print("\n4️⃣ Testing System Status...")
        status = await rag.get_system_status()
        
        print(f"📊 Document count: {status.get('document_count', 'N/A')}")
        print(f"📄 Supported formats: {status.get('supported_formats', 'N/A')}")
        print(f"🔍 Vector store status: {status.get('vector_store', {}).get('status', 'N/A')}")
        print(f"🤖 Model integration: {status.get('model_integration', {}).get('status', 'N/A')}")
        
        # Test 5: Web Server (if available)
        print("\n5️⃣ Testing Web Server Availability...")
        try:
            from src.web.server import app
            print("✅ FastAPI web server is available")
            print("🌐 API endpoints ready for deployment")
        except ImportError as e:
            print(f"❌ Web server not available: {e}")
        
        # Test 6: Docker Support
        print("\n6️⃣ Testing Docker Support...")
        dockerfile_exists = Path("Dockerfile").exists()
        docker_compose_exists = Path("docker-compose.yml").exists()
        
        if dockerfile_exists and docker_compose_exists:
            print("✅ Docker support is available")
            print("🐳 Containerization ready for deployment")
        else:
            print("❌ Docker support missing")
            if not dockerfile_exists:
                print("   - Dockerfile missing")
            if not docker_compose_exists:
                print("   - docker-compose.yml missing")
        
        # Test 7: Performance Metrics
        print("\n7️⃣ Performance Metrics...")
        print(f"⏱️  System initialization: {init_time:.2f}s")
        print(f"📄 Document processing: {doc_time:.2f}s")
        print(f"🔍 Query processing: {query_time:.2f}s")
        print(f"📊 Total queries: {rag.query_count}")
        print(f"✅ Successful queries: {rag.successful_queries}")
        print(f"❌ Error count: {rag.error_count}")
        
        # Overall Assessment
        print("\n" + "=" * 80)
        print("📊 OVERALL ASSESSMENT")
        print("=" * 80)
        
        # Calculate scores
        init_score = 100 if init_result else 0
        doc_score = 100 if doc_result else 0
        query_score = 100 if query_result.get('success', False) else 0
        web_score = 100 if Path("src/web/server.py").exists() else 0
        docker_score = 100 if dockerfile_exists and docker_compose_exists else 0
        perf_score = 100 if init_time < 30 else max(0, 100 - (init_time - 30) * 10)
        
        total_score = (init_score + doc_score + query_score + web_score + docker_score + perf_score) / 6
        
        print(f"🎯 Overall Score: {total_score:.1f}%")
        print("")
        
        if total_score >= 80:
            print("🎉 EXCELLENT - System is ready for production!")
        elif total_score >= 60:
            print("✅ GOOD - System is functional with minor issues")
        elif total_score >= 40:
            print("⚠️  FAIR - System needs improvements")
        else:
            print("❌ POOR - System needs major fixes")
        
        print("")
        print("📋 Detailed Scores:")
        print(f"   • System Initialization: {init_score}%")
        print(f"   • Document Processing: {doc_score}%")
        print(f"   • Query Processing: {query_score}%")
        print(f"   • Web Server: {web_score}%")
        print(f"   • Docker Support: {docker_score}%")
        print(f"   • Performance: {perf_score}%")
        
        print("\n🎉 Comprehensive validation completed!")
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(comprehensive_validation())
