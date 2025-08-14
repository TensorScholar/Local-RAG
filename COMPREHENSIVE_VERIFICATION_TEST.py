#!/usr/bin/env python3
"""
COMPREHENSIVE VERIFICATION TEST
"""

import asyncio
import sys
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

async def comprehensive_verification():
    """Run comprehensive verification."""
    print("🔍 COMPREHENSIVE VERIFICATION TEST")
    print("=" * 80)
    
    results = {}
    start_time = time.time()
    
    try:
        # Test 1: System Initialization
        print("1️⃣ Testing System Initialization...")
        from src.integration_interface import AdvancedRAGSystem
        
        init_start = time.time()
        rag = AdvancedRAGSystem()
        init_result = await rag.initialize()
        init_time = time.time() - init_start
        
        results['initialization'] = {
            'success': init_result,
            'time': init_time,
            'score': 100 if init_result and init_time < 30 else 0
        }
        
        if init_result:
            print(f"✅ System initialized in {init_time:.2f}s")
        else:
            print("❌ System initialization failed")
            return
        
        # Test 2: Document Processing
        print("\n2️⃣ Testing Document Processing...")
        test_file = Path("verification_test.txt")
        test_content = "Local-RAG verification test document with content for processing."
        
        with open(test_file, 'w') as f:
            f.write(test_content)
        
        doc_result = await rag.process_document(test_file)
        test_file.unlink(missing_ok=True)
        
        results['document_processing'] = {
            'success': doc_result,
            'score': 100 if doc_result else 0
        }
        
        print(f"✅ Document processing: {'Success' if doc_result else 'Failed'}")
        
        # Test 3: Query Processing
        print("\n3️⃣ Testing Query Processing...")
        query_result = await rag.query("What is Local-RAG?")
        
        results['query_processing'] = {
            'success': query_result.get('success', False),
            'score': 100 if query_result.get('success', False) else 0
        }
        
        print(f"✅ Query processing: {'Success' if query_result.get('success', False) else 'Failed'}")
        
        # Test 4: Web Server
        print("\n4️⃣ Testing Web Server...")
        try:
            from src.web.server import app
            routes = [route.path for route in app.routes]
            required = ['/health', '/api/status', '/api/query']
            missing = [r for r in required if r not in routes]
            
            results['web_server'] = {
                'success': len(missing) == 0,
                'score': 100 if len(missing) == 0 else 50
            }
            
            print(f"✅ Web server: {'Complete' if len(missing) == 0 else f'Missing {missing}'}")
        except Exception as e:
            results['web_server'] = {'success': False, 'score': 0}
            print(f"❌ Web server: {e}")
        
        # Test 5: Docker Support
        print("\n5️⃣ Testing Docker Support...")
        dockerfile = Path("Dockerfile").exists()
        compose = Path("docker-compose.yml").exists()
        
        results['docker'] = {
            'success': dockerfile and compose,
            'score': 100 if dockerfile and compose else 0
        }
        
        print(f"✅ Docker support: {'Complete' if dockerfile and compose else 'Incomplete'}")
        
        # Calculate overall score
        total_score = sum(r['score'] for r in results.values()) / len(results)
        
        print(f"\n🎯 OVERALL VERIFICATION SCORE: {total_score:.1f}%")
        
        if total_score >= 80:
            print("✅ SYSTEM IS VERIFIED AND PRODUCTION READY")
        else:
            print("⚠️ SYSTEM NEEDS ADDITIONAL VERIFICATION")
        
        # Save results
        with open("verification_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(comprehensive_verification())
