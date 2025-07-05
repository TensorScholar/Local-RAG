#!/usr/bin/env python3
"""
Advanced RAG System Command Line Interface.

This module provides a sophisticated command-line interface for interacting with the
Advanced Local RAG System. It offers comprehensive access to all system functionality
through a well-structured command hierarchy with intuitive argument parsing and
intelligent default behaviors.

Design Principles:
1. Usability - Intuitive commands and helpful documentation
2. Completeness - Access to all system functionality
3. Robustness - Comprehensive error handling and input validation
4. Performance - Efficient execution with progress reporting
5. Extensibility - Modular command structure for future expansion

Command Structure:
- init: Initialize the system
- index: Manage document indexing
  - add: Add documents to the index
  - clear: Clear the document index
  - stats: Show index statistics
- query: Process queries with flexible model selection
  - text: Process a text query (default)
  - file: Process queries from a file
- status: Show system status and performance metrics
- config: Manage system configuration
  - show: Show current configuration
  - update: Update configuration settings

Author: Advanced RAG System Team
Version: 1.0.0
"""

import asyncio
import argparse
import logging
import json
import sys
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple

# Import the main integration interface
from src.integration_interface import AdvancedRAGSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("rag_cli.log")
    ]
)
logger = logging.getLogger(__name__)


class ProgressReporter:
    """
    Progress reporting utility for long-running operations.
    
    This class provides visual feedback for operations like document indexing
    and query processing, enhancing the user experience with real-time progress
    information.
    
    Example:
        with ProgressReporter("Indexing documents", total=10) as progress:
            for i in range(10):
                # Do work
                progress.update()
    """
    
    def __init__(self, description: str, total: int = 0, enabled: bool = True):
        """
        Initialize progress reporter.
        
        Args:
            description: Operation description
            total: Total number of steps (0 for indeterminate)
            enabled: Whether progress reporting is enabled
        """
        self.description = description
        self.total = total
        self.current = 0
        self.start_time = 0
        self.enabled = enabled and sys.stdout.isatty()  # Only enable for interactive terminals
    
    def __enter__(self):
        """Start progress reporting."""
        self.start_time = time.time()
        if self.enabled:
            if self.total > 0:
                sys.stdout.write(f"{self.description}: 0/{self.total} [0%]")
            else:
                sys.stdout.write(f"{self.description}...")
            sys.stdout.flush()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        """Complete progress reporting."""
        if self.enabled:
            elapsed = time.time() - self.start_time
            if self.total > 0:
                sys.stdout.write(f"\r{self.description}: {self.current}/{self.total} [100%] - {elapsed:.2f}s\n")
            else:
                sys.stdout.write(f"\r{self.description}... done in {elapsed:.2f}s\n")
            sys.stdout.flush()
    
    def update(self, increment: int = 1):
        """
        Update progress.
        
        Args:
            increment: Number of steps to increment by
        """
        self.current += increment
        if self.enabled and self.total > 0:
            percent = min(100, int(self.current / self.total * 100))
            sys.stdout.write(f"\r{self.description}: {self.current}/{self.total} [{percent}%]")
            sys.stdout.flush()


class RAGCommandHandler:
    """
    Command handler for the Advanced RAG System CLI.
    
    This class implements the command pattern to process CLI commands,
    delegating to appropriate handlers for each command category.
    It maintains system state across commands and ensures proper
    initialization and error handling.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize command handler.
        
        Args:
            config_path: Optional path to configuration directory
        """
        self.config_path = config_path
        self.system = None
        self.initialized = False
    
    async def initialize_system(self) -> bool:
        """
        Initialize the RAG system if not already initialized.
        
        Returns:
            bool: True if initialization was successful or system was already initialized
        """
        if self.initialized and self.system:
            return True
        
        with ProgressReporter("Initializing system", enabled=True):
            self.system = AdvancedRAGSystem(self.config_path)
            self.initialized = await self.system.initialize()
        
        if not self.initialized:
            print("System initialization failed. See log for details.")
            return False
        
        return True
    
    async def handle_init(self, args: argparse.Namespace) -> int:
        """
        Handle 'init' command to initialize the system.
        
        Args:
            args: Command-line arguments
            
        Returns:
            int: Exit code (0 for success, non-zero for error)
        """
        if await self.initialize_system():
            print("System initialized successfully.")
            return 0
        else:
            return 1
    
    async def handle_index(self, args: argparse.Namespace) -> int:
        """
        Handle 'index' command and subcommands for document indexing.
        
        Args:
            args: Command-line arguments
            
        Returns:
            int: Exit code (0 for success, non-zero for error)
        """
        # Ensure system is initialized
        if not await self.initialize_system():
            return 1
        
        if args.index_cmd == "add":
            # Get all document paths with glob expansion
            all_documents = []
            for path_expr in args.docs:
                path = Path(path_expr)
                if path.is_dir():
                    # Add all files in directory if it's a directory
                    all_documents.extend([p for p in path.glob("**/*") if p.is_file()])
                elif "*" in path_expr:
                    # Handle glob patterns
                    parent = path.parent or Path(".")
                    pattern = path.name
                    all_documents.extend(list(parent.glob(pattern)))
                else:
                    # Add single file
                    all_documents.append(path)
            
            # Remove duplicates and sort for consistency
            all_documents = sorted(set(all_documents))
            
            if not all_documents:
                print("No documents found matching the specified paths.")
                return 1
            
            print(f"Adding {len(all_documents)} documents to index...")
            
            with ProgressReporter(f"Indexing documents", total=len(all_documents)) as progress:
                successful = 0
                for doc_path in all_documents:
                    if await self.system.process_document(doc_path):
                        successful += 1
                    progress.update()
            
            print(f"Successfully indexed {successful} out of {len(all_documents)} documents.")
            return 0 if successful > 0 else 1
            
        elif args.index_cmd == "clear":
            if not args.confirm and not args.yes:
                confirm = input("Are you sure you want to clear the entire document index? [y/N] ")
                if confirm.lower() not in ["y", "yes"]:
                    print("Operation canceled.")
                    return 0
            
            with ProgressReporter("Clearing document index"):
                success = await self.system.clear_index()
            
            if success:
                print("Document index successfully cleared.")
                return 0
            else:
                print("Failed to clear document index. See log for details.")
                return 1
                
        elif args.index_cmd == "stats":
            status = await self.system.get_system_status()
            
            print("\n=== Document Index Statistics ===")
            if "vector_store" in status:
                vs_stats = status["vector_store"]
                print(f"Document count: {vs_stats.get('document_count', 0)}")
                print(f"Embedding dimensions: {vs_stats.get('embedding_dimensions', 0)}")
                size_mb = vs_stats.get('index_size_bytes', 0) / (1024 * 1024)
                print(f"Index size: {size_mb:.2f} MB")
                
                if "performance_metrics" in status:
                    perf = status["performance_metrics"]
                    print(f"\nIndexed documents: {perf.get('document_count', 0)}")
                    print(f"Average processing time: {perf.get('avg_document_processing_ms', 0):.2f} ms")
            else:
                print("No index statistics available.")
            
            return 0
        
        # Default case should not be reached due to argparse subparsers
        print(f"Unknown index command: {args.index_cmd}")
        return 1
    
    async def handle_query(self, args: argparse.Namespace) -> int:
        """
        Handle 'query' command for processing queries through the RAG system.
        
        This handler implements the core query processing workflow, with comprehensive
        options for model selection, output formatting, and result handling.
        
        Args:
            args: Command-line arguments
            
        Returns:
            int: Exit code (0 for success, non-zero for error)
        """
        # Ensure system is initialized
        if not await self.initialize_system():
            return 1
        
        # Process based on query source
        if args.query_cmd == "text" or not hasattr(args, "query_cmd"):
            # Single query from command line
            query_text = args.text
            
            with ProgressReporter(f"Processing query"):
                result = await self.system.query(
                    query_text=query_text,
                    num_results=args.num_results,
                    force_local=args.local,
                    force_external=args.external,
                    specific_model=args.model,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens
                )
            
            # Display result based on output format
            if args.json:
                # Output as formatted JSON
                print(json.dumps(result, indent=2))
            else:
                # Format as human-readable output
                print("\n" + "=" * 80)
                print(f"Query: {result['query']}")
                print("-" * 80)
                print(f"Response:\n\n{result['response']}")
                print("-" * 80)
                
                # Show metadata if requested
                if args.verbose:
                    print("Metadata:")
                    model_info = f"{result['model']} (External: {result['is_external']})"
                    print(f"  Model: {model_info}")
                    print(f"  Processing time: {result['processing_time_ms']:.2f} ms")
                    
                    # Show sources if available
                    if "sources" in result and result["sources"]:
                        print("\nSources:")
                        for i, source in enumerate(result["sources"], 1):
                            title = source.get("title", source.get("id", "Unknown"))
                            src = source.get("source", "Unknown")
                            similarity = source.get("similarity", 0) * 100  # Convert to percentage
                            print(f"  {i}. {title} ({src}) - Relevance: {similarity:.1f}%")
                
                print("=" * 80 + "\n")
            
            # Return success if query was processed
            return 0 if result.get("success", False) else 1
            
        elif args.query_cmd == "file":
            # Process multiple queries from file
            query_file = Path(args.file)
            if not query_file.exists():
                print(f"Query file not found: {query_file}")
                return 1
            
            try:
                # Read queries from file (one per line)
                with open(query_file, "r") as f:
                    queries = [line.strip() for line in f if line.strip()]
                
                if not queries:
                    print("No queries found in file.")
                    return 1
                
                print(f"Processing {len(queries)} queries from {query_file}...")
                
                # Process each query
                results = []
                with ProgressReporter(f"Processing queries", total=len(queries)) as progress:
                    for query_text in queries:
                        result = await self.system.query(
                            query_text=query_text,
                            num_results=args.num_results,
                            force_local=args.local,
                            force_external=args.external,
                            specific_model=args.model,
                            temperature=args.temperature,
                            max_tokens=args.max_tokens
                        )
                        results.append(result)
                        progress.update()
                
                # Output results
                if args.output:
                    # Save results to file
                    output_path = Path(args.output)
                    with open(output_path, "w") as f:
                        json.dump(results, f, indent=2)
                    print(f"Results saved to {output_path}")
                else:
                    # Print summary
                    successful = sum(1 for r in results if r.get("success", False))
                    print(f"Successfully processed {successful} out of {len(queries)} queries.")
                    
                    # Print brief results if not too many
                    if len(queries) <= 5 and not args.json:
                        for i, result in enumerate(results, 1):
                            print(f"\nQuery {i}: {result['query']}")
                            print(f"Response: {result['response'][:100]}...")
                
                return 0
                
            except Exception as e:
                print(f"Error processing queries from file: {e}")
                logger.error(f"Error processing queries from file: {e}", exc_info=True)
                return 1
        
        # Default case should not be reached due to argparse subparsers
        print(f"Unknown query command: {args.query_cmd}")
        return 1
    
    async def handle_status(self, args: argparse.Namespace) -> int:
        """
        Handle 'status' command to display system status and metrics.
        
        This command provides comprehensive information about the system state,
        including component status, performance metrics, and resource utilization.
        
        Args:
            args: Command-line arguments
            
        Returns:
            int: Exit code (0 for success, non-zero for error)
        """
        # Ensure system is initialized
        if not await self.initialize_system():
            return 1
        
        with ProgressReporter("Retrieving system status"):
            status = await self.system.get_system_status()
        
        # Output as JSON if requested
        if args.json:
            print(json.dumps(status, indent=2, default=str))
            return 0
        
        # Format as human-readable output
        print("\n=== Advanced RAG System Status ===")
        print(f"Initialized: {status['initialized']}")
        print(f"Uptime: {status.get('uptime_seconds', 0):.1f} seconds")
        
        # Vector store information
        if "vector_store" in status:
            print("\nVector Store:")
            vs = status["vector_store"]
            print(f"  Document count: {vs.get('document_count', 0)}")
            print(f"  Embedding dimensions: {vs.get('embedding_dimensions', 0)}")
            size_mb = vs.get('index_size_bytes', 0) / (1024 * 1024)
            print(f"  Index size: {size_mb:.2f} MB")
        
        # Model information
        if "models" in status:
            models = status["models"]
            print("\nModels:")
            
            # Local models
            local_models = models.get("local_models", [])
            print(f"  Local models: {', '.join(local_models) if local_models else 'None'}")
            
            # External providers
            ext_providers = models.get("external_providers", [])
            print(f"  External providers: {', '.join(ext_providers) if ext_providers else 'None'}")
            
            # External models if available and verbose mode
            if args.verbose and "external_models" in models and models["external_models"]:
                print("  Available external models:")
                for model in models["external_models"]:
                    print(f"    - {model}")
            
            # Model preferences if available
            if "external_api_preference" in models and models["external_api_preference"]:
                print(f"  Preferred provider: {models['external_api_preference']}")
        
        # Performance metrics
        if "performance_metrics" in status:
            print("\nPerformance:")
            perf = status["performance_metrics"]
            print(f"  Document count: {perf.get('document_count', 0)}")
            print(f"  Query count: {perf.get('query_count', 0)}")
            print(f"  Successful queries: {perf.get('successful_queries', 0)}")
            print(f"  Error count: {perf.get('error_count', 0)}")
            print(f"  Avg. document processing: {perf.get('avg_document_processing_ms', 0):.2f} ms")
            print(f"  Avg. query time: {perf.get('avg_query_time_ms', 0):.2f} ms")
            print(f"  Query success rate: {perf.get('query_success_rate', 0):.1f}%")
        
        # Routing decisions if available and verbose mode
        if args.verbose and "models" in status and "routing_decisions_count" in status["models"]:
            print(f"\nRouting decisions: {status['models']['routing_decisions_count']}")
        
        # Document processor information
        if "document_processor" in status:
            if args.verbose and "supported_formats" in status["document_processor"]:
                formats = status["document_processor"]["supported_formats"]
                print("\nSupported document formats:")
                for fmt in formats:
                    print(f"  - {fmt}")
        
        return 0
    
    async def handle_config(self, args: argparse.Namespace) -> int:
        """
        Handle 'config' command for configuration management.
        
        This command provides facilities for viewing and updating system configuration,
        supporting both interactive and non-interactive usage patterns.
        
        Args:
            args: Command-line arguments
            
        Returns:
            int: Exit code (0 for success, non-zero for error)
        """
        if args.config_cmd == "show":
            # Display current configuration
            config_path = self.config_path or Path.home() / ".rag_system"
            
            # Check for configuration files
            files = []
            if (config_path / "config.yaml").exists():
                files.append("config.yaml")
            if (config_path / "config.json").exists():
                files.append("config.json")
            if (config_path / "external_api.yaml").exists():
                files.append("external_api.yaml")
            if (config_path / "credentials.json").exists():
                files.append("credentials.json")
            
            if not files:
                print(f"No configuration files found in {config_path}")
                return 1
            
            print(f"Configuration directory: {config_path}")
            print(f"Available configuration files: {', '.join(files)}")
            
            # Display specific file content if requested
            if args.file and args.file in files:
                file_path = config_path / args.file
                print(f"\nContents of {args.file}:")
                with open(file_path, "r") as f:
                    content = f.read()
                print(content)
            
            return 0
            
        elif args.config_cmd == "create":
            # Create default configuration
            config_path = self.config_path or Path.home() / ".rag_system"
            
            # Create directory if it doesn't exist
            os.makedirs(config_path, exist_ok=True)
            
            # Create external API configuration
            external_api_path = config_path / "external_api.yaml"
            if not external_api_path.exists() or args.force:
                # Default configuration content
                external_api_config = """# External API Configuration for Advanced Local RAG System

external_models:
  enabled: true
  preferred_provider: null  # Auto-select based on capabilities
  cost_limit_per_query: 0.02  # USD
  max_latency_ms: 3000
  performance_logging: true
  fallback_to_local: true
  
  providers:
    openai:
      enabled: true
      default_model: "gpt-3.5-turbo"
      temperature: 0.7
      preferred_models: ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
    
    google:
      enabled: true
      default_model: "gemini-1.0-pro"
      temperature: 0.7
      preferred_models: ["gemini-1.5-pro", "gemini-1.0-pro"]
    
    anthropic:
      enabled: true
      default_model: "claude-3-haiku"
      temperature: 0.7
      preferred_models: ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"]
  
  # Capability-based routing preferences
  capability_preferences:
    scientific_reasoning: ["anthropic", "openai", "google"]
    mathematical_computation: ["openai", "anthropic", "google"]
    code_generation: ["openai", "anthropic", "google"]
    multimodal_understanding: ["openai", "google"]
    long_context: ["anthropic", "google", "openai"]
  
  # Complexity-based routing preferences
  complexity_preferences:
    simple: ["local", "external"]  # Prefer local for simple queries
    moderate: ["local", "external"]  # Prefer local for moderate queries
    complex: ["external", "local"]  # Prefer external for complex queries
    specialized: ["external"]  # External only for specialized queries
"""
                # Write configuration
                with open(external_api_path, "w") as f:
                    f.write(external_api_config)
                print(f"Created external API configuration: {external_api_path}")
            else:
                print(f"External API configuration already exists: {external_api_path}")
                print("Use --force to overwrite")
            
            # Create credentials template
            credentials_path = config_path / "credentials.json"
            if not credentials_path.exists() or args.force:
                # Default credentials template
                credentials_template = """{
  "openai": {
    "api_key": "your_openai_key_here"
  },
  "google": {
    "api_key": "your_google_key_here"
  },
  "anthropic": {
    "api_key": "your_anthropic_key_here"
  }
}
"""
                # Write credentials template
                with open(credentials_path, "w") as f:
                    f.write(credentials_template)
                
                # Set secure permissions (owner read/write only)
                os.chmod(credentials_path, 0o600)
                
                print(f"Created credentials template: {credentials_path}")
                print("Please edit this file to add your API keys")
            else:
                print(f"Credentials file already exists: {credentials_path}")
                print("Use --force to overwrite")
            
            return 0
            
        # Default case should not be reached due to argparse subparsers
        print(f"Unknown config command: {args.config_cmd}")
        return 1


async def main():
    """
    Main entry point for the Advanced RAG System CLI.
    
    This function defines the command-line interface structure, parses arguments,
    and delegates to appropriate command handlers. It implements a sophisticated
    subcommand hierarchy with comprehensive help documentation and intelligent
    default behaviors.
    """
    # Create main parser
    parser = argparse.ArgumentParser(
        description="Advanced RAG System Command Line Interface",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Global options
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to configuration directory"
    )
    parser.add_argument(
        "--verbose", 
        "-v", 
        action="store_true", 
        help="Enable verbose output"
    )
    parser.add_argument(
        "--quiet", 
        "-q", 
        action="store_true", 
        help="Suppress non-essential output"
    )
    
    # Create subcommand parsers
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Initialize command
    init_parser = subparsers.add_parser(
        "init", 
        help="Initialize the system"
    )
    
    # Index commands
    index_parser = subparsers.add_parser(
        "index", 
        help="Manage document indexing"
    )
    index_subparsers = index_parser.add_subparsers(
        dest="index_cmd", 
        help="Indexing subcommand"
    )
    
    # Index add subcommand
    add_parser = index_subparsers.add_parser(
        "add", 
        help="Add documents to the index"
    )
    add_parser.add_argument(
        "docs", 
        nargs="+", 
        help="Document paths to index (files, directories, or glob patterns)"
    )
    
    # Index clear subcommand
    clear_parser = index_subparsers.add_parser(
        "clear", 
        help="Clear the document index"
    )
    clear_parser.add_argument(
        "--yes", 
        "-y", 
        action="store_true", 
        help="Skip confirmation prompt"
    )
    clear_parser.add_argument(
        "--confirm", 
        action="store_true", 
        help="Confirm index clearing"
    )
    
    # Index stats subcommand
    stats_parser = index_subparsers.add_parser(
        "stats", 
        help="Show index statistics"
    )
    
    # Query commands
    query_parser = subparsers.add_parser(
        "query", 
        help="Process queries"
    )
    query_subparsers = query_parser.add_subparsers(
        dest="query_cmd", 
        help="Query subcommand"
    )
    
    # Common query options
    def add_query_options(parser):
        """Add common query options to parser."""
        parser.add_argument(
            "--local", 
            action="store_true", 
            help="Force use of local models"
        )
        parser.add_argument(
            "--external", 
            action="store_true", 
            help="Force use of external API models"
        )
        parser.add_argument(
            "--model", 
            type=str, 
            help="Specific model to use (format: 'name' or 'provider:model')"
        )
        parser.add_argument(
            "--temperature", 
            type=float, 
            default=0.7, 
            help="Generation temperature (0.0-1.0)"
        )
        parser.add_argument(
            "--max-tokens", 
            type=int, 
            help="Maximum tokens to generate"
        )
        parser.add_argument(
            "--num-results", 
            type=int, 
            default=5, 
            help="Number of documents to retrieve"
        )
        parser.add_argument(
            "--json", 
            action="store_true", 
            help="Output results as JSON"
        )
    
    # Query text subcommand
    text_parser = query_subparsers.add_parser(
        "text", 
        help="Process a text query"
    )
    text_parser.add_argument(
        "text", 
        help="Query text"
    )
    add_query_options(text_parser)
    
    # Query file subcommand
    file_parser = query_subparsers.add_parser(
        "file", 
        help="Process queries from a file"
    )
    file_parser.add_argument(
        "file", 
        help="File containing queries (one per line)"
    )
    file_parser.add_argument(
        "--output", 
        "-o", 
        help="Output file for results (JSON format)"
    )
    add_query_options(file_parser)
    
    # Status command
    status_parser = subparsers.add_parser(
        "status", 
        help="Show system status"
    )
    status_parser.add_argument(
        "--json", 
        action="store_true", 
        help="Output status as JSON"
    )
    
    # Config commands
    config_parser = subparsers.add_parser(
        "config", 
        help="Manage system configuration"
    )
    config_subparsers = config_parser.add_subparsers(
        dest="config_cmd", 
        help="Configuration subcommand"
    )
    
    # Config show subcommand
    show_parser = config_subparsers.add_parser(
        "show", 
        help="Show current configuration"
    )
    show_parser.add_argument(
        "--file", 
        help="Specific configuration file to show"
    )
    
    # Config create subcommand
    create_parser = config_subparsers.add_parser(
        "create", 
        help="Create default configuration files"
    )
    create_parser.add_argument(
        "--force", 
        "-f", 
        action="store_true", 
        help="Overwrite existing files"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Convert config path to Path object if provided
    config_path = Path(args.config) if args.config else None
    
    # Create command handler
    handler = RAGCommandHandler(config_path)
    
    # Determine logging level based on verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Handle command
    if args.command == "init":
        return await handler.handle_init(args)
    elif args.command == "index":
        return await handler.handle_index(args)
    elif args.command == "query":
        return await handler.handle_query(args)
    elif args.command == "status":
        return await handler.handle_status(args)
    elif args.command == "config":
        return await handler.handle_config(args)
    else:
        # No command specified, show help
        parser.print_help()
        return 0


if __name__ == "__main__":
    # Run the CLI with proper asyncio handling
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nOperation canceled by user.")
        sys.exit(130)  # Standard exit code for SIGINT
    except Exception as e:
        print(f"Error: {e}")
        logger.error(f"Unhandled exception: {e}", exc_info=True)
        sys.exit(1)
                
