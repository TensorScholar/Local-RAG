"""
Advanced RAG System Web API.

This module provides a FastAPI-based web API for the Advanced RAG System,
enabling seamless integration with web frontends and external services.
The API follows REST principles with robust error handling, clear documentation,
and comprehensive security measures.

Design Principles:
1. Simplicity - Clean RESTful endpoints with intuitive naming
2. Security - Comprehensive protection against common web vulnerabilities
3. Documentation - Detailed OpenAPI/Swagger documentation
4. Performance - Optimized for low-latency response
5. Extensibility - Modular design for future expansion

Features:
- Document management (upload, list, delete)
- Query processing with model selection
- System status and health monitoring
- Credential management (secure)
- Performance metrics and analytics

Author: Advanced RAG System Team
Version: 1.0.0
"""

import asyncio
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, Query, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
# Import the Advanced RAG System
from src.integration_interface import AdvancedRAGSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Advanced RAG System API",
    description="API for the Advanced Local RAG System with external model integration",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# Configure CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define API models
class QueryRequest(BaseModel):
    """API model for query requests."""
    query: str = Field(..., description="The query text to process")
    num_results: int = Field(5, description="Number of documents to retrieve")
    force_local: bool = Field(False, description="Force the use of local models")
    force_external: bool = Field(False, description="Force the use of external models")
    model: Optional[str] = Field(None, description="Specific model to use (format: 'name' or 'provider:model')")
    temperature: float = Field(0.7, description="Generation temperature (0.0-1.0)")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate")

class DocumentUploadResponse(BaseModel):
    """API model for document upload responses."""
    success: bool
    document_id: str
    filename: str
    message: str

class DocumentListItem(BaseModel):
    """API model for document list items."""
    id: str
    filename: str
    size_bytes: int
    upload_time: float
    metadata: Dict[str, Any]

class DocumentListResponse(BaseModel):
    """API model for document list responses."""
    success: bool
    documents: List[DocumentListItem]
    total_count: int

# Global RAG system instance
rag_system = None
upload_dir = Path("uploads")
os.makedirs(upload_dir, exist_ok=True)

# Optional API key security for production use
API_KEY_NAME = "X-API-Key"
API_KEY = os.environ.get("RAG_API_KEY")
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key_header: str = Depends(api_key_header)):
    """Validate API key if configured."""
    if not API_KEY:
        return True
    if api_key_header and api_key_header == API_KEY:
        return True
    raise HTTPException(
        status_code=401,
        detail="Invalid or missing API key",
        headers={"WWW-Authenticate": API_KEY_NAME},
    )

async def get_rag_system():
    """Get or initialize the RAG system."""
    global rag_system
    
    if rag_system is None:
        logger.info("Initializing RAG system for API")
        rag_system = AdvancedRAGSystem()
        await rag_system.initialize()
        
    return rag_system

@app.on_event("startup")
async def startup_event():
    """Initialize the RAG system on startup."""
    await get_rag_system()
    logger.info("RAG system API started")

@app.get("/api/health", response_model=Dict[str, Any])
async def health_check(_: bool = Depends(get_api_key)):
    """
    Health check endpoint.
    
    Returns basic system status and health information.
    """
    system = await get_rag_system()
    status = await system.get_system_status()
    
    return {
        "status": "healthy" if status.get("initialized") else "degraded",
        "initialized": status.get("initialized", False),
        "uptime_seconds": status.get("uptime_seconds", 0),
        "version": "1.0.0"
    }

@app.get("/api/status", response_model=Dict[str, Any])
async def get_status(_: bool = Depends(get_api_key)):
    """
    Get detailed system status.
    
    Returns comprehensive information about system components,
    performance metrics, and resource utilization.
    """
    system = await get_rag_system()
    status = await system.get_system_status()
    return status

@app.post("/api/query", response_model=Dict[str, Any])
async def process_query(
    request: QueryRequest,
    _: bool = Depends(get_api_key)
):
    """
    Process a query through the RAG system.
    
    Takes a query and processing parameters, returning the generated
    response with relevant context and metadata.
    """
    system = await get_rag_system()
    
    try:
        result = await system.query(
            query_text=request.query,
            num_results=request.num_results,
            force_local=request.force_local,
            force_external=request.force_external,
            specific_model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

@app.post("/api/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    _: bool = Depends(get_api_key)
):
    """
    Upload and process a document.
    
    Accepts a document file, saves it to disk, and processes it
    for indexing in the RAG system.
    """
    system = await get_rag_system()
    
    try:
        # Generate unique ID for document
        document_id = str(uuid.uuid4())
        
        # Save file to disk
        file_path = upload_dir / f"{document_id}_{file.filename}"
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Process document in background
        background_tasks.add_task(
            process_document_background,
            str(file_path),
            document_id
        )
        
        return DocumentUploadResponse(
            success=True,
            document_id=document_id,
            filename=file.filename,
            message="Document uploaded and being processed"
        )
        
    except Exception as e:
        logger.error(f"Error uploading document: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error uploading document: {str(e)}"
        )

async def process_document_background(file_path: str, document_id: str):
    """Process document in background task."""
    system = await get_rag_system()
    try:
        logger.info(f"Processing document {file_path} with ID {document_id}")
        await system.process_document(Path(file_path), document_id=document_id)
        logger.info(f"Document {document_id} processed successfully")
    except Exception as e:
        logger.error(f"Error processing document {document_id}: {e}", exc_info=True)

@app.get("/api/documents", response_model=DocumentListResponse)
async def list_documents(_: bool = Depends(get_api_key)):
    """
    List indexed documents.
    
    Returns a list of all documents currently indexed in the system.
    """
    system = await get_rag_system()
    
    try:
        # This is a placeholder - actual implementation would depend on
        # vector store implementation details to retrieve document metadata
        status = await system.get_system_status()
        vector_store_stats = status.get("vector_store", {})
        doc_count = vector_store_stats.get("document_count", 0)
        
        # Get documents from upload directory as a fallback
        documents = []
        for file_path in upload_dir.glob("*"):
            if file_path.is_file():
                file_stats = file_path.stat()
                doc_id = file_path.name.split("_")[0]
                filename = "_".join(file_path.name.split("_")[1:])
                
                documents.append(DocumentListItem(
                    id=doc_id,
                    filename=filename,
                    size_bytes=file_stats.st_size,
                    upload_time=file_stats.st_mtime,
                    metadata={"path": str(file_path)}
                ))
        
        return DocumentListResponse(
            success=True,
            documents=documents,
            total_count=len(documents)
        )
        
    except Exception as e:
        logger.error(f"Error listing documents: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error listing documents: {str(e)}"
        )

@app.delete("/api/documents/{document_id}", response_model=Dict[str, Any])
async def delete_document(
    document_id: str,
    _: bool = Depends(get_api_key)
):
    """
    Delete a document from the index.
    
    Removes the specified document from the system index.
    """
    # Note: Actual implementation would need to remove document from vector store
    # This is a placeholder that just removes the file
    
    try:
        # Find and delete document file
        found = False
        for file_path in upload_dir.glob(f"{document_id}_*"):
            if file_path.is_file():
                file_path.unlink()
                found = True
                break
        
        if not found:
            raise HTTPException(
                status_code=404,
                detail=f"Document with ID {document_id} not found"
            )
        
        return {
            "success": True,
            "document_id": document_id,
            "message": "Document deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting document: {str(e)}"
        )

@app.post("/api/index/clear", response_model=Dict[str, Any])
async def clear_index(_: bool = Depends(get_api_key)):
    """
    Clear the document index.
    
    Removes all documents from the system index.
    """
    system = await get_rag_system()
    
    try:
        success = await system.clear_index()
        
        # Also clear upload directory
        for file_path in upload_dir.glob("*"):
            if file_path.is_file():
                file_path.unlink()
        
        return {
            "success": success,
            "message": "Document index cleared successfully" if success else "Failed to clear index"
        }
        
    except Exception as e:
        logger.error(f"Error clearing index: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error clearing index: {str(e)}"
        )

# Serve static files for the web interface
app.mount("/", StaticFiles(directory="web", html=True), name="web")

def start_server(host="0.0.0.0", port=8000):
    """Start the API server."""
    uvicorn.run("rag_web_api:app", host=host, port=port, reload=True)

if __name__ == "__main__":
    start_server()
