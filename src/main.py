"""
Advanced Local RAG System - Main Integration Module.

This module integrates all components of the advanced local RAG system into a
cohesive application with a command-line interface. It implements the core RAG
workflow with sophisticated document processing, vector storage, and local
language model inference.

Architecture features:
- Modular component design with clean interfaces
- Adaptive processing pipeline for different document types
- Resource-efficient operation with hardware-specific optimizations
- Comprehensive logging and performance monitoring
- Exception handling with graceful degradation
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('rag_system.log')
    ]
)

logger = logging.getLogger(__name__)

# Import system components
try:
    from src.core.config.settings import *
    from src.embeddings.embeddings_creator import EmbeddingsCreator
    from src.indexing.vector_store import VectorStore
    from src.processing.document.processor import DocumentProcessor
    from src.generation.local_generator import LocalResponseGenerator
    from src.processing.scientific.processor import ScientificProcessor
except ImportError as e:
    logger.error(f"Error importing system components: {e}")
    logger.error("Make sure the src directory is in your Python path")
    sys.exit(1)

class RAGSystem:
    """
    Integrated retrieval-augmented generation system with local operation.
    
    This class orchestrates the end-to-end RAG pipeline, from document processing
    and indexing to retrieval and response generation using local language models.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the RAG system with all components and configuration.
        
        Args:
            config_path: Optional path to a custom configuration file
        """
        self.startup_time = time.time()
        
        # Load custom configuration if provided
        if config_path:
            # Custom configuration loading would go here
            logger.info(f"Loading custom configuration from {config_path}")
        
        # Initialize components
        logger.info("Initializing RAG system components...")
        
        # Document processor
        self.document_processor = DocumentProcessor(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            enable_ocr=True,
            extract_tables=True
        )
        
        # Embeddings creator
        self.embeddings_creator = EmbeddingsCreator(
            model_name=EMBEDDING_MODEL,
            device="auto",
            batch_size=32
        )
        
        # Vector store
        self.vector_store = VectorStore(
            persist_directory=str(VECTOR_STORE_DIR),
            collection_name="documents",
            embedding_function=self.embeddings_creator.model,
            distance_function=DISTANCE_METRIC
        )
        
        # Response generator
        self.response_generator = LocalResponseGenerator(
            model_path=str(get_default_model_path()),
            model_type="gguf",
            device="auto",
            context_length=get_model_config()['context_length'],
            max_tokens=1000,
            temperature=0.1
        )
        
        # Scientific processor
        self.scientific_processor = ScientificProcessor()
        
        logger.info("RAG system initialization complete")
        
        # System stats
        self.total_queries = 0
        self.total_documents_indexed = 0
        self.startup_duration = time.time() - self.startup_time
        logger.info(f"System startup completed in {self.startup_duration:.2f} seconds")
    
    def index_document(self, document_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Process and index a single document.
        
        Args:
            document_path: Path to the document file
            
        Returns:
            Dictionary with indexing results and metadata
        """
        document_path = Path(document_path)
        result = {
            'success': False,
            'document_path': str(document_path),
            'chunks_indexed': 0,
            'processing_time': 0,
            'embedding_time': 0,
            'indexing_time': 0,
            'total_time': 0,
            'error': None
        }
        
        total_start_time = time.time()
        
        try:
            # Check if file exists
            if not document_path.exists():
                result['error'] = f"Document not found: {document_path}"
                return result
            
            logger.info(f"Processing document: {document_path}")
            
            # Process the document
            processing_start = time.time()
            processed_doc = self.document_processor.process_document(document_path)
            processing_time = time.time() - processing_start
            
            result['processing_time'] = processing_time
            
            if not processed_doc['success']:
                result['error'] = processed_doc.get('error', 'Document processing failed')
                return result
            
            # Get document chunks
            chunks = processed_doc.get('chunks', [])
            
            if not chunks:
                result['error'] = "No content chunks created from document"
                return result
            
            logger.info(f"Created {len(chunks)} chunks from document")
            
            # Create embeddings for chunks
            embedding_start = time.time()
            documents_with_embeddings = self.embeddings_creator.process_documents(chunks)
            embedding_time = time.time() - embedding_start
            
            result['embedding_time'] = embedding_time
            
            # Add to vector store
            indexing_start = time.time()
            doc_ids = self.vector_store.add_documents(documents_with_embeddings)
            indexing_time = time.time() - indexing_start
            
            result['indexing_time'] = indexing_time
            result['chunks_indexed'] = len(doc_ids)
            
            # Update stats
            self.total_documents_indexed += 1
            
            # Set success
            result['success'] = True
            
        except Exception as e:
            logger.error(f"Error indexing document: {e}")
            result['error'] = str(e)
        
        # Calculate total time
        result['total_time'] = time.time() - total_start_time
        
        return result
    
    def index_directory(self, directory_path: Union[str, Path], recursive: bool = True) -> Dict[str, Any]:
        """
        Process and index all documents in a directory.
        
        Args:
            directory_path: Path to the directory containing documents
            recursive: Whether to process subdirectories
            
        Returns:
            Dictionary with indexing results and metadata
        """
        directory_path = Path(directory_path)
        result = {
            'success': False,
            'directory_path': str(directory_path),
            'files_processed': 0,
            'files_failed': 0,
            'total_chunks_indexed': 0,
            'total_time': 0,
            'errors': [],
            'file_results': {}
        }
        
        total_start_time = time.time()
        
        try:
            # Check if directory exists
            if not directory_path.exists() or not directory_path.is_dir():
                result['error'] = f"Directory not found: {directory_path}"
                return result
            
            logger.info(f"Indexing directory: {directory_path}")
            
            # Process all documents in the directory
            directory_results = self.document_processor.process_directory(directory_path, recursive)
            
            # Create embeddings and index each processed document
            for doc_result in directory_results:
                doc_path = doc_result['file_path']
                
                if doc_result['success']:
                    # Create embeddings for chunks
                    chunks = doc_result.get('chunks', [])
                    
                    if chunks:
                        documents_with_embeddings = self.embeddings_creator.process_documents(chunks)
                        doc_ids = self.vector_store.add_documents(documents_with_embeddings)
                        
                        # Update results
                        result['files_processed'] += 1
                        result['total_chunks_indexed'] += len(doc_ids)
                        result['file_results'][doc_path] = {
                            'success': True,
                            'chunks_indexed': len(doc_ids)
                        }
                    else:
                        result['files_failed'] += 1
                        result['errors'].append(f"No chunks created for {doc_path}")
                        result['file_results'][doc_path] = {
                            'success': False,
                            'error': "No chunks created"
                        }
                else:
                    result['files_failed'] += 1
                    result['errors'].append(f"Failed to process {doc_path}: {doc_result.get('error', 'Unknown error')}")
                    result['file_results'][doc_path] = {
                        'success': False,
                        'error': doc_result.get('error', 'Unknown error')
                    }
            
            # Update stats
            self.total_documents_indexed += result['files_processed']
            
            # Set success
            result['success'] = True
            
        except Exception as e:
            logger.error(f"Error indexing directory: {e}")
            result['errors'].append(str(e))
        
        # Calculate total time
        result['total_time'] = time.time() - total_start_time
        
        logger.info(f"Directory indexing complete. Processed {result['files_processed']} files " + 
                    f"with {result['total_chunks_indexed']} chunks in {result['total_time']:.2f} seconds")
        
        return result
    
    def query(self, query_text: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Process a query through the RAG pipeline and generate a response.
        
        This method implements the core RAG workflow, retrieving relevant documents
        based on the query and generating a response using the local language model.
        
        Args:
            query_text: The user's query text
            n_results: Number of relevant documents to retrieve
            
        Returns:
            Dictionary with query results, retrieved documents, and generated response
        """
        result = {
            'query': query_text,
            'success': False,
            'response': None,
            'retrieved_documents': [],
            'scientific_processing': None,
            'retrieval_time': 0,
            'generation_time': 0,
            'total_time': 0,
            'error': None
        }
        
        total_start_time = time.time()
        
        try:
            # Check for scientific queries first
            scientific_result = self.scientific_processor.process_query(query_text)
            
            if scientific_result['processed']:
                # This is a scientific query, use the scientific processor's result
                result['scientific_processing'] = scientific_result
                result['response'] = {
                    'content': self._format_scientific_response(scientific_result),
                    'source': "scientific_processor"
                }
                result['success'] = True
            else:
                # Regular RAG query
                # Create query embedding
                query_embedding = self.embeddings_creator.create_embeddings([query_text])[0]
                
                # Retrieve relevant documents
                retrieval_start = time.time()
                retrieved_docs = self.vector_store.query(
                    query_embedding=query_embedding,
                    n_results=n_results
                )
                retrieval_time = time.time() - retrieval_start
                
                result['retrieval_time'] = retrieval_time
                result['retrieved_documents'] = retrieved_docs
                
                # Generate response
                if retrieved_docs:
                    generation_start = time.time()
                    response = self.response_generator.generate_response(
                        query=query_text,
                        retrieved_documents=retrieved_docs
                    )
                    generation_time = time.time() - generation_start
                    
                    result['generation_time'] = generation_time
                    result['response'] = response
                else:
                    result['response'] = {
                        'content': "I don't have enough information to answer that question.",
                        'source': "no_documents"
                    }
                
                result['success'] = True
            
            # Update stats
            self.total_queries += 1
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            result['error'] = str(e)
        
        # Calculate total time
        result['total_time'] = time.time() - total_start_time
        
        return result
    
    def _format_scientific_response(self, scientific_result: Dict[str, Any]) -> str:
        """
        Format a scientific processing result into a user-friendly response.
        
        Args:
            scientific_result: Result from scientific processor
            
        Returns:
            Formatted response string
        """
        query_type = scientific_result.get('query_type', 'unknown')
        result_data = scientific_result.get('result', {})
        
        if query_type == 'equation':
            # Format equation solution
            equation = result_data.get('equation', '')
            solution = result_data.get('solution', '')
            
            response = f"Equation: {equation}\n"
            if isinstance(solution, list):
                response += "Solutions:\n"
                for i, sol in enumerate(solution):
                    response += f"  {i+1}. {sol}\n"
            else:
                response += f"Solution: {solution}\n"
            
            return response
            
        elif query_type == 'unit_conversion':
            # Format unit conversion
            from_value = result_data.get('from_value', '')
            from_unit = result_data.get('from_unit', '')
            to_unit = result_data.get('to_unit', '')
            converted_value = result_data.get('converted_value', '')
            
            response = f"Converting {from_value} {from_unit} to {to_unit}:\n"
            response += f"Result: {converted_value} {to_unit}"
            
            return response
            
        elif query_type == 'symbolic':
            # Format symbolic manipulation
            expression = result_data.get('expression', '')
            operation = result_data.get('operation', '')
            result_value = result_data.get('result', '')
            
            response = f"{operation.capitalize()} expression: {expression}\n"
            response += f"Result: {result_value}"
            
            return response
            
        elif query_type == 'chemical':
            # Format chemical formula analysis
            formula = result_data.get('formula', '')
            elements = result_data.get('elements', {})
            molecular_weight = result_data.get('molecular_weight', '')
            
            response = f"Chemical formula: {formula}\n"
            response += "Elements:\n"
            for element, count in elements.items():
                response += f"  {element}: {count}\n"
            response += f"Molecular weight: {molecular_weight} g/mol"
            
            return response
            
        elif query_type == 'math_expression':
            # Format math expression evaluation
            expression = result_data.get('expression', '')
            value = result_data.get('value', '')
            
            response = f"Expression: {expression}\n"
            if isinstance(value, dict) and 'symbolic' in value:
                response += f"Symbolic result: {value.get('symbolic', '')}"
            else:
                response += f"Value: {value}"
            
            return response
            
        else:
            # Generic response for other types
            return f"Scientific processing result: {result_data}"
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the RAG system's performance.
        
        Returns:
            Dictionary with system statistics and component performance metrics
        """
        stats = {
            'system': {
                'total_queries': self.total_queries,
                'total_documents_indexed': self.total_documents_indexed,
                'startup_time': self.startup_duration,
                'uptime': time.time() - self.startup_time
            },
            'components': {
                'document_processor': self.document_processor.get_performance_stats(),
                'embeddings_creator': self.embeddings_creator.get_performance_stats(),
                'vector_store': self.vector_store.get_collection_stats(),
                'response_generator': self.response_generator.get_performance_stats()
            },
            'models': {
                'embedding_model': EMBEDDING_MODEL,
                'llm_model': Path(get_default_model_path()).name
            }
        }
        
        return stats


def parse_arguments():
    """Parse command-line arguments for the RAG system."""
    parser = argparse.ArgumentParser(description="Advanced Local RAG System")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Index command
    index_parser = subparsers.add_parser("index", help="Index documents")
    index_parser.add_argument("path", help="Path to document or directory to index")
    index_parser.add_argument("--recursive", "-r", action="store_true", help="Recursively process directories")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the system")
    query_parser.add_argument("query", help="Query text")
    query_parser.add_argument("--results", "-n", type=int, default=5, help="Number of results to retrieve")
    
    # Interactive command
    interactive_parser = subparsers.add_parser("interactive", help="Start interactive query mode")
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show system statistics")
    
    return parser.parse_args()


def main():
    """Main entry point for the RAG system CLI."""
    # Parse arguments
    args = parse_arguments()
    
    # Initialize the RAG system
    rag_system = RAGSystem()
    
    if args.command == "index":
        # Index documents
        path = Path(args.path)
        if path.is_dir():
            result = rag_system.index_directory(path, args.recursive)
            print(f"Indexed {result['files_processed']} documents with {result['total_chunks_indexed']} chunks")
            if result['files_failed'] > 0:
                print(f"Failed to index {result['files_failed']} documents")
        else:
            result = rag_system.index_document(path)
            if result['success']:
                print(f"Successfully indexed document with {result['chunks_indexed']} chunks")
            else:
                print(f"Failed to index document: {result['error']}")
    
    elif args.command == "query":
        # Process a single query
        result = rag_system.query(args.query, args.results)
        
        if result['success']:
            if 'scientific_processing' in result and result['scientific_processing']:
                print("\n=== Scientific Processing Result ===")
                print(result['response']['content'])
            else:
                print("\n=== Response ===")
                print(result['response']['response'])
                
                print("\n=== Retrieved Documents ===")
                for i, doc in enumerate(result['retrieved_documents']):
                    print(f"Document {i+1} (Similarity: {doc['similarity']:.2f}):")
                    print(f"  {doc['text'][:100]}...")
        else:
            print(f"Query failed: {result['error']}")
    
    elif args.command == "interactive":
        # Start interactive mode
        print("=== Advanced Local RAG System - Interactive Mode ===")
        print("Type 'exit' or 'quit' to end the session")
        
        while True:
            try:
                query = input("\nEnter your query: ")
                
                if query.lower() in ['exit', 'quit', 'q']:
                    break
                
                # Process query
                result = rag_system.query(query)
                
                if result['success']:
                    if 'scientific_processing' in result and result['scientific_processing']:
                        print("\n=== Scientific Processing Result ===")
                        print(result['response']['content'])
                    else:
                        print("\n=== Response ===")
                        print(result['response']['response'])
                        
                        print("\n=== Retrieved Documents ===")
                        for i, doc in enumerate(result['retrieved_documents']):
                            print(f"Document {i+1} (Similarity: {doc['similarity']:.2f}):")
                            source = doc['metadata'].get('source', 'Unknown')
                            print(f"  Source: {source}")
                            print(f"  {doc['text'][:100]}...")
                    
                    print(f"\nResponse generated in {result['total_time']:.2f} seconds")
                else:
                    print(f"Query failed: {result['error']}")
            
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
    
    elif args.command == "stats":
        # Show system statistics
        stats = rag_system.get_system_stats()
        
        print("=== Advanced Local RAG System - Statistics ===")
        print(f"Total queries processed: {stats['system']['total_queries']}")
        print(f"Total documents indexed: {stats['system']['total_documents_indexed']}")
        print(f"System uptime: {stats['system']['uptime']:.2f} seconds")
        
        print("\n=== Model Information ===")
        print(f"Embedding model: {stats['models']['embedding_model']}")
        print(f"Language model: {stats['models']['llm_model']}")
        
        print("\n=== Vector Store Statistics ===")
        vs_stats = stats['components']['vector_store']
        print(f"Document chunks: {vs_stats.get('document_count', 0)}")
        print(f"Distance function: {vs_stats.get('distance_function', '')}")
        
        if vs_stats.get('performance', {}).get('avg_query_time'):
            print(f"Average query time: {vs_stats['performance']['avg_query_time']:.4f} seconds")
            
    else:
        # No command or invalid command
        print("Please specify a command. Use --help for available commands.")
        
if __name__ == "__main__":
    main()
