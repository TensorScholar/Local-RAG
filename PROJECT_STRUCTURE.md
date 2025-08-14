# Local-RAG Project Structure

## Overview

This document describes the clean and well-organized structure of the Local-RAG system.

## Directory Structure

```
Local-RAG/
├── README.md                           # Main project documentation
├── requirements.txt                    # Python dependencies
├── Dockerfile                          # Docker containerization
├── docker-compose.yml                  # Multi-service deployment
├── run_server.py                       # Web server startup script
├── .gitignore                          # Git ignore rules
├── PROJECT_STRUCTURE.md                # This file
│
├── src/                                # Main source code
│   ├── __init__.py
│   ├── main.py                         # Main application entry point
│   ├── integration_interface.py        # Core RAG system interface
│   │
│   ├── application/                    # Application layer
│   │   ├── __init__.py
│   │   ├── apex_core.py
│   │   ├── autoconfig.py
│   │   ├── models.py
│   │   └── router.py
│   │
│   ├── core/                           # Core system components
│   │   ├── __init__.py
│   │   ├── rag_cli.py
│   │   └── rag_web_api.py
│   │
│   ├── domain/                         # Domain models
│   │   ├── __init__.py
│   │   ├── models.py
│   │   └── reactive.py
│   │
│   ├── embeddings/                     # Embedding generation
│   │   ├── __init__.py
│   │   └── embedder.py
│   │
│   ├── exploration/                    # Knowledge exploration
│   │   ├── __init__.py
│   │   ├── contradiction_detector.py
│   │   └── knowledge_graph.py
│   │
│   ├── generation/                     # Text generation
│   │   ├── __init__.py
│   │   └── local_generator.py
│   │
│   ├── indexing/                       # Vector indexing
│   │   ├── __init__.py
│   │   ├── test_imports.py
│   │   └── vector_store.py
│   │
│   ├── models/                         # Model management
│   │   ├── __init__.py
│   │   └── external/                   # External model providers
│   │       ├── __init__.py
│   │       ├── credential_manager.py
│   │       ├── external_model_manager.py
│   │       ├── model_integration_manager.py
│   │       └── providers/              # API providers
│   │           ├── __init__.py
│   │           ├── base_provider.py
│   │           ├── openai_provider.py
│   │           ├── google_provider.py
│   │           └── anthropic_provider.py
│   │
│   ├── processing/                     # Document processing
│   │   ├── __init__.py
│   │   ├── setup.py
│   │   ├── start_rag.py
│   │   ├── update_references.py
│   │   ├── document/                   # Document processing
│   │   │   ├── __init__.py
│   │   │   ├── Document.py
│   │   │   └── processor.py
│   │   ├── image/                      # Image processing
│   │   │   └── __init__.py
│   │   └── scientific/                 # Scientific content processing
│   │       ├── __init__.py
│   │       └── processor.py
│   │
│   ├── retrieval/                      # Document retrieval
│   │   ├── __init__.py
│   │   └── retriever.py
│   │
│   ├── utils/                          # Utility functions
│   │   ├── __init__.py
│   │   ├── reriever.py
│   │   └── test_path.py
│   │
│   └── web/                            # Web interface
│       ├── __init__.py
│       └── server.py                   # FastAPI web server
│
├── web/                                # Web assets
│   ├── index.html                      # Main web interface
│   └── assets/                         # Web assets
│       ├── css/
│       ├── js/
│       └── img/
│
├── config/                             # Configuration files
│   ├── config-manager.py
│   ├── credentials.json.template
│   └── docker-compose.yml
│
├── docs/                               # Documentation
│   └── technical/                      # Technical documentation
│       ├── architectural_decisions.md
│       ├── mathematical_specifications.md
│       └── performance_benchmarks.md
│
├── tests/                              # Test files
│   ├── __init__.py
│   ├── test_apex_core.py
│   └── integration/                    # Integration tests
│       ├── __init__.py
│       └── test_external_api.py
│
└── verification/                       # Verification and validation
    ├── COMPREHENSIVE_VERIFICATION_TEST.py
    ├── comprehensive_validation_test.py
    ├── test_web_server.py
    ├── verification_results.json
    ├── FINAL_COMPREHENSIVE_VERIFICATION_REPORT.md
    ├── FINAL_VALIDATION_REPORT.md
    ├── CRITICAL_ISSUES_ROADMAP.md
    └── VERIFICATION_SUMMARY.md
```

## Key Components

### Core RAG System
- **integration_interface.py**: Main system interface
- **embeddings/**: Text embedding generation
- **indexing/**: Vector storage and retrieval
- **retrieval/**: Document retrieval system
- **generation/**: Text generation capabilities

### Model Management
- **models/external/**: External API integration
- **providers/**: Support for OpenAI, Google, Anthropic

### Document Processing
- **processing/**: Multi-format document processing
- **document/**: Core document handling
- **scientific/**: Scientific content processing

### Web Interface
- **web/**: FastAPI web server
- **web/assets/**: Frontend assets

### Deployment
- **Dockerfile**: Containerization
- **docker-compose.yml**: Multi-service deployment
- **run_server.py**: Server startup

### Verification
- **verification/**: All verification and validation files
- **tests/**: Unit and integration tests

## Clean Architecture Principles

1. **Separation of Concerns**: Each module has a specific responsibility
2. **Dependency Inversion**: High-level modules don't depend on low-level modules
3. **Single Responsibility**: Each class has one reason to change
4. **Open/Closed**: Open for extension, closed for modification

## File Organization

- **Source Code**: All Python code in `src/`
- **Configuration**: All config files in `config/`
- **Documentation**: All docs in `docs/`
- **Tests**: All tests in `tests/`
- **Verification**: All verification files in `verification/`
- **Web Assets**: All web files in `web/`

This structure ensures a clean, maintainable, and professional codebase.
