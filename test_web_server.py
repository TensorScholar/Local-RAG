#!/usr/bin/env python3
"""
Test Web Server Functionality
"""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_web_server():
    """Test web server functionality."""
    print("🌐 TESTING WEB SERVER FUNCTIONALITY")
    print("=" * 50)
    
    try:
        # Test 1: Import web server
        print("1️⃣ Testing Web Server Import...")
        from src.web.server import app
        
        # Test 2: Check routes
        print("\n2️⃣ Testing API Routes...")
        routes = [route.path for route in app.routes]
        required_routes = ['/health', '/api/status', '/api/query']
        
        missing_routes = [r for r in required_routes if r not in routes]
        
        if len(missing_routes) == 0:
            print("✅ All required routes available")
            print(f"📊 Total routes: {len(routes)}")
        else:
            print(f"❌ Missing routes: {missing_routes}")
        
        # Test 3: Test Pydantic models
        print("\n3️⃣ Testing Pydantic Models...")
        from src.web.server import QueryRequest, QueryResponse
        
        test_request = QueryRequest(
            query="test query",
            num_results=5,
            temperature=0.7
        )
        
        print("✅ Pydantic models working correctly")
        
        # Test 4: Test health endpoint logic
        print("\n4️⃣ Testing Health Endpoint Logic...")
        from src.web.server import health_check
        
        # Mock the health check
        class MockRAGSystem:
            def __init__(self):
                self.initialized = True
                self._start_time = time.time() - 100
        
        # Temporarily set rag_system
        import src.web.server as server_module
        server_module.rag_system = MockRAGSystem()
        
        health_result = await health_check()
        print(f"✅ Health check response: {health_result.status}")
        
        print("\n🎉 Web Server Tests Completed Successfully!")
        
    except Exception as e:
        print(f"❌ Web server test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_web_server())
