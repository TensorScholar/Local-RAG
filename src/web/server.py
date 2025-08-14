#!/usr/bin/env python3
"""
FastAPI Web Server for Local-RAG System

This module implements a comprehensive RESTful API server for the Local-RAG system,
providing endpoints for document processing, querying, and system management.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import uvicorn
from contextlib import asynccontextmanager

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.integration_interface import AdvancedRAGSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

# Global RAG system instance
rag_system: Optional[AdvancedRAGSystem] = None

# Pydantic models for API
class QueryRequest(BaseModel):
    query: str = Field(..., description="The query text")
    num_results: int = Field(5, description="Number of results to retrieve")
    temperature: float = Field(0.7, description="Generation temperature")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate")

class QueryResponse(BaseModel):
    success: bool
    response: str
    query: str
    processing_time_ms: float
    model_used: str
    context_documents: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class DocumentResponse(BaseModel):
    success: bool
    document_id: str
    file_name: str
    processing_time_ms: float
    chunks_created: int
    message: str

class SystemStatus(BaseModel):
    initialized: bool
    document_count: int
    supported_formats: List[str]
    vector_store_status: str
    model_integration_status: str
    uptime_seconds: float

class HealthResponse(BaseModel):
    status: str
    timestamp: float
    version: str
    uptime_seconds: float

# Startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    global rag_system
    
    # Startup
    logger.info("Starting Local-RAG Web Server...")
    try:
        rag_system = AdvancedRAGSystem()
        await rag_system.initialize()
        logger.info("Local-RAG system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Local-RAG system: {e}")
        rag_system = None
    
    yield
    
    # Shutdown
    logger.info("Shutting down Local-RAG Web Server...")
    if rag_system:
        # Cleanup if needed
        pass

# Create FastAPI app
app = FastAPI(
    title="Local-RAG API",
    description="Advanced Retrieval-Augmented Generation System API",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency for authentication (placeholder for now)
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current user from JWT token."""
    # TODO: Implement proper JWT validation
    return {"user_id": "default_user"}

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if rag_system and rag_system.initialized else "unhealthy",
        timestamp=time.time(),
        version="1.0.0",
        uptime_seconds=time.time() - (getattr(rag_system, '_start_time', time.time()) if rag_system else time.time())
    )

# System status endpoint
@app.get("/api/status", response_model=SystemStatus)
async def get_system_status():
    """Get system status and statistics."""
    if not rag_system or not rag_system.initialized:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        status_data = await rag_system.get_system_status()
        return SystemStatus(
            initialized=rag_system.initialized,
            document_count=status_data.get('document_count', 0),
            supported_formats=status_data.get('supported_formats', []),
            vector_store_status=status_data.get('vector_store', {}).get('status', 'unknown'),
            model_integration_status=status_data.get('model_integration', {}).get('status', 'unknown'),
            uptime_seconds=time.time() - (getattr(rag_system, '_start_time', time.time()) if rag_system else time.time())
        )
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving system status")

# Document upload endpoint
@app.post("/api/documents/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    document_id: Optional[str] = Form(None)
):
    """Upload and process a document."""
    if not rag_system or not rag_system.initialized:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        # Save uploaded file temporarily
        temp_path = Path(f"/tmp/{file.filename}")
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process document
        start_time = time.time()
        success = await rag_system.process_document(temp_path, document_id)
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Clean up temp file
        temp_path.unlink(missing_ok=True)
        
        if success:
            return DocumentResponse(
                success=True,
                document_id=document_id or file.filename,
                file_name=file.filename,
                processing_time_ms=processing_time_ms,
                chunks_created=0,  # TODO: Get actual chunk count
                message="Document processed successfully"
            )
        else:
            raise HTTPException(status_code=400, detail="Document processing failed")
            
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

# Query endpoint
@app.post("/api/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the RAG system."""
    if not rag_system or not rag_system.initialized:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        start_time = time.time()
        
        # Process query
        result = await rag_system.query(
            query_text=request.query,
            num_results=request.num_results,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        if result.get('success', False):
            return QueryResponse(
                success=True,
                response=result.get('response', ''),
                query=request.query,
                processing_time_ms=processing_time_ms,
                model_used=result.get('model_used', 'unknown'),
                context_documents=result.get('context_documents', []),
                metadata=result.get('metadata', {})
            )
        else:
            raise HTTPException(status_code=400, detail=result.get('error', 'Query failed'))
            
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# Clear index endpoint
@app.delete("/api/documents/clear")
async def clear_documents():
    """Clear all documents from the index."""
    if not rag_system or not rag_system.initialized:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        success = await rag_system.clear_index()
        if success:
            return {"message": "Index cleared successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to clear index")
    except Exception as e:
        logger.error(f"Error clearing index: {e}")
        raise HTTPException(status_code=500, detail=f"Error clearing index: {str(e)}")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

# Main function to run the server
def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run the FastAPI server."""
    uvicorn.run(
        "src.web.server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )

if __name__ == "__main__":
    run_server()
