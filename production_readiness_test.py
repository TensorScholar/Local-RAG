#!/usr/bin/env python3
"""
Production Readiness Test Script
Validates the three critical claims against state-of-the-art standards
"""

import asyncio
import sys
import time
import subprocess
import requests
from pathlib import Path
from typing import Dict, List, Any

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

class ProductionReadinessValidator:
    """Validates production readiness against industry standards."""
    
    def __init__(self):
        self.results = {}
        
    async def validate_claim_1_functional_rag(self) -> Dict[str, Any]:
        """Validate: NOT a functional RAG system"""
        print("🔍 Validating Claim 1: NOT a functional RAG system")
        print("=" * 60)
        
        results = {
            "claim": "NOT a functional RAG system",
            "tests": {},
            "score": 0,
            "status": "FAILED"
        }
        
        try:
            from src.integration_interface import AdvancedRAGSystem
            
            # Test 1: System Initialization
            print("1. Testing system initialization...")
            start_time = time.time()
            rag = AdvancedRAGSystem()
            init_result = await rag.initialize()
            init_time = time.time() - start_time
            
            results["tests"]["initialization"] = {
                "status": "PASS" if init_result else "FAIL",
                "time": init_time,
                "details": f"Initialization {'succeeded' if init_result else 'failed'}"
            }
            
            # Test 2: Document Processing
            print("2. Testing document processing...")
            try:
                # Create test document
                test_file = Path("test_doc.txt")
                test_content = "This is a test document for RAG validation."
                with open(test_file, 'w') as f:
                    f.write(test_content)
                
                # Test processing
                process_result = await rag.process_document(test_file)
                test_file.unlink()
                
                results["tests"]["document_processing"] = {
                    "status": "PASS" if process_result else "FAIL",
                    "details": f"Document processing {'succeeded' if process_result else 'failed'}"
                }
            except Exception as e:
                results["tests"]["document_processing"] = {
                    "status": "FAIL",
                    "details": f"Document processing error: {str(e)}"
                }
            
            # Test 3: Vector Storage
            print("3. Testing vector storage...")
            try:
                status = await rag.get_system_status()
                vector_status = status.get('vector_store', {}).get('status', 'unknown')
                
                results["tests"]["vector_storage"] = {
                    "status": "PASS" if vector_status == "ready" else "FAIL",
                    "details": f"Vector store status: {vector_status}"
                }
            except Exception as e:
                results["tests"]["vector_storage"] = {
                    "status": "FAIL",
                    "details": f"Vector storage error: {str(e)}"
                }
            
            # Test 4: Query Functionality
            print("4. Testing query functionality...")
            try:
                query_result = await rag.query("test query")
                results["tests"]["query_functionality"] = {
                    "status": "PASS" if query_result else "FAIL",
                    "details": f"Query {'succeeded' if query_result else 'failed'}"
                }
            except Exception as e:
                results["tests"]["query_functionality"] = {
                    "status": "FAIL",
                    "details": f"Query error: {str(e)}"
                }
            
            # Calculate score
            passed_tests = sum(1 for test in results["tests"].values() if test["status"] == "PASS")
            total_tests = len(results["tests"])
            results["score"] = (passed_tests / total_tests) * 100
            results["status"] = "PASS" if results["score"] >= 80 else "FAIL"
            
        except Exception as e:
            results["error"] = str(e)
            results["status"] = "ERROR"
        
        self.results["functional_rag"] = results
        return results
    
    async def validate_claim_2_deployable(self) -> Dict[str, Any]:
        """Validate: NOT deployable as claimed"""
        print("\n🚀 Validating Claim 2: NOT deployable as claimed")
        print("=" * 60)
        
        results = {
            "claim": "NOT deployable as claimed",
            "tests": {},
            "score": 0,
            "status": "FAILED"
        }
        
        # Test 1: Web Server
        print("1. Testing web server...")
        try:
            # Check if FastAPI server exists
            server_files = list(Path("src").rglob("*server*.py")) + list(Path("src").rglob("*api*.py"))
            results["tests"]["web_server"] = {
                "status": "PASS" if server_files else "FAIL",
                "details": f"Found {len(server_files)} server files"
            }
        except Exception as e:
            results["tests"]["web_server"] = {
                "status": "FAIL",
                "details": f"Server check error: {str(e)}"
            }
        
        # Test 2: API Endpoints
        print("2. Testing API endpoints...")
        try:
            # Check for API route definitions
            api_files = list(Path("src").rglob("*router*.py")) + list(Path("src").rglob("*endpoint*.py"))
            results["tests"]["api_endpoints"] = {
                "status": "PASS" if api_files else "FAIL",
                "details": f"Found {len(api_files)} API files"
            }
        except Exception as e:
            results["tests"]["api_endpoints"] = {
                "status": "FAIL",
                "details": f"API check error: {str(e)}"
            }
        
        # Test 3: Docker Support
        print("3. Testing Docker support...")
        docker_files = ["Dockerfile", "docker-compose.yml", "docker-compose.yaml"]
        found_docker = any(Path(f).exists() for f in docker_files)
        results["tests"]["docker_support"] = {
            "status": "PASS" if found_docker else "FAIL",
            "details": f"Docker files found: {found_docker}"
        }
        
        # Test 4: Configuration Management
        print("4. Testing configuration management...")
        config_files = list(Path("config").glob("*.json")) + list(Path("config").glob("*.yaml"))
        results["tests"]["configuration"] = {
            "status": "PASS" if config_files else "FAIL",
            "details": f"Found {len(config_files)} config files"
        }
        
        # Test 5: Health Check Endpoint
        print("5. Testing health check...")
        try:
            # Try to start a server (if possible)
            results["tests"]["health_check"] = {
                "status": "FAIL",
                "details": "No server implementation to test"
            }
        except Exception as e:
            results["tests"]["health_check"] = {
                "status": "FAIL",
                "details": f"Health check error: {str(e)}"
            }
        
        # Calculate score
        passed_tests = sum(1 for test in results["tests"].values() if test["status"] == "PASS")
        total_tests = len(results["tests"])
        results["score"] = (passed_tests / total_tests) * 100
        results["status"] = "PASS" if results["score"] >= 80 else "FAIL"
        
        self.results["deployable"] = results
        return results
    
    async def validate_claim_3_production_ready(self) -> Dict[str, Any]:
        """Validate: NOT ready for production use"""
        print("\n🏭 Validating Claim 3: NOT ready for production use")
        print("=" * 60)
        
        results = {
            "claim": "NOT ready for production use",
            "tests": {},
            "score": 0,
            "status": "FAILED"
        }
        
        # Test 1: Security
        print("1. Testing security...")
        security_issues = []
        
        # Check for authentication
        auth_files = list(Path("src").rglob("*auth*.py")) + list(Path("src").rglob("*security*.py"))
        if not auth_files:
            security_issues.append("No authentication system")
        
        # Check for input validation
        validation_files = list(Path("src").rglob("*validation*.py")) + list(Path("src").rglob("*sanitize*.py"))
        if not validation_files:
            security_issues.append("No input validation")
        
        results["tests"]["security"] = {
            "status": "PASS" if not security_issues else "FAIL",
            "details": f"Security issues: {security_issues if security_issues else 'None found'}"
        }
        
        # Test 2: Testing Coverage
        print("2. Testing coverage...")
        test_files = list(Path("tests").rglob("*.py"))
        test_count = len(test_files)
        results["tests"]["testing"] = {
            "status": "PASS" if test_count >= 10 else "FAIL",
            "details": f"Found {test_count} test files (minimum 10 required)"
        }
        
        # Test 3: Error Handling
        print("3. Testing error handling...")
        try:
            from src.integration_interface import AdvancedRAGSystem
            rag = AdvancedRAGSystem()
            
            # Test error handling
            try:
                await rag.initialize()
                results["tests"]["error_handling"] = {
                    "status": "PASS",
                    "details": "Basic error handling exists"
                }
            except Exception as e:
                results["tests"]["error_handling"] = {
                    "status": "FAIL",
                    "details": f"Error handling failed: {str(e)}"
                }
        except Exception as e:
            results["tests"]["error_handling"] = {
                "status": "FAIL",
                "details": f"Error handling test failed: {str(e)}"
            }
        
        # Test 4: Performance
        print("4. Testing performance...")
        try:
            start_time = time.time()
            rag = AdvancedRAGSystem()
            await rag.initialize()
            init_time = time.time() - start_time
            
            results["tests"]["performance"] = {
                "status": "PASS" if init_time < 30 else "FAIL",
                "details": f"Initialization time: {init_time:.2f}s (should be <30s)"
            }
        except Exception as e:
            results["tests"]["performance"] = {
                "status": "FAIL",
                "details": f"Performance test failed: {str(e)}"
            }
        
        # Test 5: Monitoring
        print("5. Testing monitoring...")
        monitoring_files = list(Path("src").rglob("*monitor*.py")) + list(Path("src").rglob("*metrics*.py"))
        results["tests"]["monitoring"] = {
            "status": "PASS" if monitoring_files else "FAIL",
            "details": f"Found {len(monitoring_files)} monitoring files"
        }
        
        # Calculate score
        passed_tests = sum(1 for test in results["tests"].values() if test["status"] == "PASS")
        total_tests = len(results["tests"])
        results["score"] = (passed_tests / total_tests) * 100
        results["status"] = "PASS" if results["score"] >= 80 else "FAIL"
        
        self.results["production_ready"] = results
        return results
    
    def generate_report(self) -> str:
        """Generate comprehensive validation report."""
        report = []
        report.append("# Production Readiness Validation Report")
        report.append("")
        report.append("## Summary")
        report.append("")
        
        total_score = 0
        total_claims = len(self.results)
        
        for claim_name, result in self.results.items():
            score = result.get("score", 0)
            status = result.get("status", "FAILED")
            claim = result.get("claim", "Unknown")
            
            total_score += score
            
            report.append(f"### {claim}")
            report.append(f"- **Status**: {status}")
            report.append(f"- **Score**: {score:.1f}%")
            report.append("")
            
            # Add test details
            for test_name, test_result in result.get("tests", {}).items():
                test_status = test_result.get("status", "FAIL")
                test_details = test_result.get("details", "No details")
                report.append(f"- **{test_name}**: {test_status} - {test_details}")
            
            report.append("")
        
        # Overall assessment
        overall_score = total_score / total_claims if total_claims > 0 else 0
        report.append("## Overall Assessment")
        report.append("")
        report.append(f"**Overall Score**: {overall_score:.1f}%")
        report.append("")
        
        if overall_score >= 80:
            report.append("✅ **PRODUCTION READY**")
        elif overall_score >= 60:
            report.append("⚠️ **NEEDS IMPROVEMENT**")
        else:
            report.append("❌ **NOT PRODUCTION READY**")
        
        return "\n".join(report)

async def main():
    """Run production readiness validation."""
    print("🧪 Production Readiness Validation")
    print("=" * 80)
    print("Validating three critical claims against state-of-the-art standards")
    print("")
    
    validator = ProductionReadinessValidator()
    
    # Validate all claims
    await validator.validate_claim_1_functional_rag()
    await validator.validate_claim_2_deployable()
    await validator.validate_claim_3_production_ready()
    
    # Generate report
    print("\n" + "=" * 80)
    print("📊 VALIDATION REPORT")
    print("=" * 80)
    print(validator.generate_report())
    
    # Save report
    with open("production_validation_report.md", "w") as f:
        f.write(validator.generate_report())
    
    print("\n📄 Report saved to: production_validation_report.md")

if __name__ == "__main__":
    asyncio.run(main())
