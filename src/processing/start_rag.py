#!/usr/bin/env python3
"""
Advanced RAG System Launcher

This script provides a unified entry point for launching the Advanced RAG System
with its various components and services. It implements a sophisticated startup
sequence with dependency management, health checks, and graceful error handling.

Design Philosophy:
1. Elegant Orchestration - Coordination of multiple system components
2. Progressive Initialization - Staged startup with dependency resolution
3. Fault Tolerance - Graceful handling of initialization failures
4. Resource Management - Proper allocation and cleanup of resources
5. Operational Transparency - Clear feedback on system status

Usage:
  python start_rag.py [options]

Options:
  --headless       Start without web UI (API only)
  --port PORT      Port for web server (default: 8000)
  --host HOST      Host address (default: 0.0.0.0)
  --config PATH    Path to configuration directory
  --debug          Enable debug logging
  --no-browser     Don't open browser window automatically
  --init-only      Initialize system without starting server

Author: Advanced RAG System Team
Version: 1.0.0
"""

import argparse
import asyncio
import logging
import os
import platform
import shutil
import signal
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("rag_launcher")


class SystemComponent:
    """
    Abstract representation of a system component with lifecycle management.
    
    This class encapsulates the common behavior of various system components,
    providing a unified interface for initialization, status monitoring, and
    shutdown operations.
    
    Attributes:
        name (str): Component identifier
        enabled (bool): Whether component is active
        initialized (bool): Whether component is initialized
        depends_on (List[str]): Component dependencies
        process (Optional[subprocess.Popen]): Process handle if applicable
    """
    
    def __init__(self, name: str, enabled: bool = True, depends_on: List[str] = None):
        """Initialize component with name and dependency information."""
        self.name = name
        self.enabled = enabled
        self.initialized = False
        self.depends_on = depends_on or []
        self.process = None
        self._stop_event = threading.Event()
    
    async def initialize(self, components: Dict[str, 'SystemComponent'], config_path: Optional[Path]) -> bool:
        """
        Initialize this component, ensuring all dependencies are initialized first.
        
        Args:
            components: Dictionary of all available components
            config_path: Path to configuration directory
            
        Returns:
            bool: True if initialization successful, False otherwise
        """
        if not self.enabled:
            logger.info(f"Component {self.name} is disabled, skipping initialization")
            return False
        
        # Check dependencies
        for dep_name in self.depends_on:
            if dep_name not in components:
                logger.error(f"Component {self.name} depends on unknown component {dep_name}")
                return False
                
            dep = components[dep_name]
            if not dep.enabled or not dep.initialized:
                logger.error(f"Component {self.name} depends on {dep_name} which is not initialized")
                return False
        
        # Perform component-specific initialization
        try:
            result = await self._initialize(config_path)
            self.initialized = result
            return result
        except Exception as e:
            logger.error(f"Error initializing component {self.name}: {e}", exc_info=True)
            return False
    
    async def _initialize(self, config_path: Optional[Path]) -> bool:
        """
        Component-specific initialization logic.
        
        Args:
            config_path: Path to configuration directory
            
        Returns:
            bool: True if initialization successful, False otherwise
            
        This method should be overridden by subclasses.
        """
        logger.warning(f"Component {self.name} using default initialization")
        return True
    
    async def start(self) -> bool:
        """
        Start component operation.
        
        Returns:
            bool: True if started successfully, False otherwise
        """
        if not self.initialized:
            logger.error(f"Cannot start uninitialized component {self.name}")
            return False
            
        try:
            return await self._start()
        except Exception as e:
            logger.error(f"Error starting component {self.name}: {e}", exc_info=True)
            return False
    
    async def _start(self) -> bool:
        """
        Component-specific start logic.
        
        Returns:
            bool: True if started successfully, False otherwise
            
        This method should be overridden by subclasses.
        """
        logger.info(f"Component {self.name} started")
        return True
    
    def stop(self) -> bool:
        """
        Stop component operation.
        
        Returns:
            bool: True if stopped successfully, False otherwise
        """
        self._stop_event.set()
        
        try:
            if self.process:
                logger.info(f"Stopping process for component {self.name}")
                
                # Try graceful shutdown first
                if platform.system() != "Windows":
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=5)
                        return True
                    except subprocess.TimeoutExpired:
                        logger.warning(f"Timeout waiting for {self.name} to terminate, forcing...")
                        self.process.kill()
                else:
                    # On Windows, we need a different approach
                    self.process.terminate()
                
                self.process = None
                
            return self._stop()
        except Exception as e:
            logger.error(f"Error stopping component {self.name}: {e}", exc_info=True)
            return False
    
    def _stop(self) -> bool:
        """
        Component-specific stop logic.
        
        Returns:
            bool: True if stopped successfully, False otherwise
            
        This method should be overridden by subclasses.
        """
        logger.info(f"Component {self.name} stopped")
        return True


class RAGSystemComponent(SystemComponent):
    """
    Core RAG System component that manages the initialization of the
    integrated system.
    
    This component ensures the RAG system is properly initialized with correct
    configuration before any dependent services are started.
    """
    
    async def _initialize(self, config_path: Optional[Path]) -> bool:
        """
        Initialize the RAG system by running the CLI init command.
        
        Args:
            config_path: Path to configuration directory
            
        Returns:
            bool: True if initialization successful, False otherwise
        """
        logger.info("Initializing RAG System...")
        
        cmd = [sys.executable, "rag_cli.py", "init"]
        if config_path:
            cmd.extend(["--config", str(config_path)])
        
        try:
            # Run the initialization command
            process = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            
            if "System initialized successfully" in process.stdout:
                logger.info("RAG System initialized successfully")
                return True
            else:
                logger.warning(f"RAG System initialization may have issues: {process.stdout}")
                return True  # Continue anyway, as we have partial initialization
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Error initializing RAG System: {e}")
            logger.error(f"Error output: {e.stderr}")
            return False
    
    async def _start(self) -> bool:
        """
        Start any background processes for the core RAG system.
        
        Returns:
            bool: True if started successfully, False otherwise
        """
        # The core system doesn't have a specific 'start' process
        # It's more of a library that other components use
        return True


class WebServerComponent(SystemComponent):
    """
    Web Server component that manages the FastAPI web server process.
    
    This component handles the web server lifecycle, including startup,
    monitoring, and graceful shutdown.
    """
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8000, enabled: bool = True):
        """
        Initialize web server component with network parameters.
        
        Args:
            host: Host address to bind to
            port: Port to listen on
            enabled: Whether this component is enabled
        """
        super().__init__("web_server", enabled, ["rag_system"])
        self.host = host
        self.port = port
        self.ready = threading.Event()
    
    async def _initialize(self, config_path: Optional[Path]) -> bool:
        """
        Initialize web server component.
        
        Args:
            config_path: Path to configuration directory
            
        Returns:
            bool: True if initialization successful, False otherwise
        """
        # Ensure web directory exists
        web_dir = Path("web")
        if not web_dir.exists():
            logger.warning("Web directory not found, creating empty directory")
            web_dir.mkdir(exist_ok=True)
        
        # Ensure assets directory exists
        assets_dir = web_dir / "assets"
        if not assets_dir.exists():
            assets_dir.mkdir(exist_ok=True)
            (assets_dir / "css").mkdir(exist_ok=True)
            (assets_dir / "js").mkdir(exist_ok=True)
            (assets_dir / "js" / "components").mkdir(exist_ok=True)
            (assets_dir / "js" / "utils").mkdir(exist_ok=True)
            (assets_dir / "js" / "state").mkdir(exist_ok=True)
            (assets_dir / "img").mkdir(exist_ok=True)
        
        # Check for web server script
        if not Path("rag_web_api.py").exists():
            logger.error("Web server script (rag_web_api.py) not found")
            return False
        
        logger.info("Web server component initialized successfully")
        return True
    
    async def _start(self) -> bool:
        """
        Start the web server process.
        
        Returns:
            bool: True if started successfully, False otherwise
        """
        logger.info(f"Starting web server on {self.host}:{self.port}...")
        
        # Construct the command to start the web server
        cmd = [
            sys.executable,
            "rag_web_api.py",
        ]
        
        # Set environment variables for the server
        env = os.environ.copy()
        env["RAG_WEB_HOST"] = self.host
        env["RAG_WEB_PORT"] = str(self.port)
        
        # Start the web server process
        try:
            self.process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True
            )
            
            # Start threads to monitor the output
            threading.Thread(target=self._monitor_output, daemon=True).start()
            
            # Wait for server to be ready
            if not self.ready.wait(timeout=30):
                logger.warning("Timeout waiting for web server to start, continuing anyway")
                
            logger.info(f"Web server running at http://{self.host}:{self.port}/")
            return True
            
        except Exception as e:
            logger.error(f"Error starting web server: {e}")
            return False
    
    def _monitor_output(self) -> None:
        """
        Monitor web server process output for startup and errors.
        
        This method runs in a background thread, continuously reading from the
        process stdout/stderr and logging relevant information.
        """
        if not self.process:
            return
            
        for line in self.process.stdout:
            line = line.strip()
            if line:
                logger.debug(f"Web server: {line}")
                
                # Check for server startup message
                if "Application startup complete" in line or "Uvicorn running on" in line:
                    self.ready.set()
        
        # Process stderr
        for line in self.process.stderr:
            line = line.strip()
            if line:
                logger.error(f"Web server error: {line}")


class SystemOrchestrator:
    """
    Orchestrator for managing the complete RAG system lifecycle.
    
    This class coordinates the initialization, startup, and shutdown of all 
    system components with proper dependency resolution and error handling.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize orchestrator with system configuration.
        
        Args:
            config: Configuration options for the system
        """
        self.config = config
        self.config_path = Path(config["config"]) if config["config"] else None
        self.components: Dict[str, SystemComponent] = {}
        self.running = False
        self.stop_event = asyncio.Event()
        
        # Create and register components
        self._register_components()
    
    def _register_components(self) -> None:
        """
        Create and register all system components.
        
        This method sets up the component hierarchy with proper dependencies.
        """
        # Core RAG System
        rag_system = RAGSystemComponent("rag_system")
        self.components["rag_system"] = rag_system
        
        # Web Server (if not in headless mode)
        if not self.config["headless"]:
            web_server = WebServerComponent(
                host=self.config["host"],
                port=self.config["port"]
            )
            self.components["web_server"] = web_server
    
    async def initialize(self) -> bool:
        """
        Initialize all system components in dependency order.
        
        Returns:
            bool: True if all components initialized successfully, False otherwise
        """
        logger.info("Initializing Advanced RAG System...")
        
        # Track initialized components for dependency resolution
        initialized = set()
        to_initialize = set(self.components.keys())
        
        # Continue until all components are initialized or we can't make progress
        while to_initialize:
            progress_made = False
            
            for name in list(to_initialize):
                component = self.components[name]
                
                # Check if all dependencies are satisfied
                if all(dep in initialized for dep in component.depends_on):
                    logger.info(f"Initializing component: {name}")
                    
                    # Initialize the component
                    if await component.initialize(self.components, self.config_path):
                        initialized.add(name)
                        to_initialize.remove(name)
                        progress_made = True
                    else:
                        logger.error(f"Failed to initialize component: {name}")
                        
                        # If this is a critical component, fail the initialization
                        if name == "rag_system":
                            return False
                        
                        # Otherwise, remove it from initialization list
                        to_initialize.remove(name)
                        progress_made = True
            
            # If we couldn't initialize any components in this iteration, we have a dependency problem
            if not progress_made and to_initialize:
                remaining = ", ".join(to_initialize)
                logger.error(f"Unable to resolve dependencies for components: {remaining}")
                return False
        
        # Report initialization status
        total_components = len(self.components)
        initialized_count = sum(1 for c in self.components.values() if c.initialized)
        
        logger.info(f"System initialization complete: {initialized_count}/{total_components} components initialized")
        return initialized_count > 0
    
    async def start(self) -> bool:
        """
        Start all initialized components.
        
        Returns:
            bool: True if all components started successfully, False otherwise
        """
        if self.running:
            logger.warning("System is already running")
            return True
        
        logger.info("Starting Advanced RAG System...")
        
        # Start components in dependency order
        for name, component in self.components.items():
            if component.initialized:
                logger.info(f"Starting component: {name}")
                if not await component.start():
                    logger.error(f"Failed to start component: {name}")
                    
                    # If this is a critical component, fail the startup
                    if name == "rag_system":
                        return False
        
        self.running = True
        
        # Open browser if needed
        if (not self.config["headless"] and not self.config["no_browser"] 
                and self.components["web_server"].initialized):
            self._open_browser()
        
        return True
    
    def stop(self) -> None:
        """
        Stop all running components in reverse dependency order.
        
        This ensures dependent components are stopped before their dependencies.
        """
        if not self.running:
            return
            
        logger.info("Stopping Advanced RAG System...")
        
        # Stop in reverse dependency order
        for name, component in reversed(list(self.components.items())):
            if component.initialized:
                logger.info(f"Stopping component: {name}")
                component.stop()
        
        self.running = False
        self.stop_event.set()
    
    def _open_browser(self) -> None:
        """
        Open a web browser pointing to the RAG system interface.
        
        This method delays the browser opening slightly to ensure the server is ready.
        """
        def _delayed_open():
            time.sleep(2)  # Give the server a moment to start
            url = f"http://{self.config['host']}:{self.config['port']}/"
            
            # Use localhost instead of 0.0.0.0 for browser
            if self.config['host'] == '0.0.0.0':
                url = f"http://localhost:{self.config['port']}/"
                
            logger.info(f"Opening browser at {url}")
            
            try:
                webbrowser.open(url)
            except Exception as e:
                logger.error(f"Error opening browser: {e}")
        
        threading.Thread(target=_delayed_open, daemon=True).start()
    
    async def run(self) -> None:
        """
        Run the system until interrupted.
        
        This method starts the system and then waits for a shutdown signal.
        """
        try:
            # Initialize and start the system
            if not await self.initialize():
                logger.error("System initialization failed")
                return
                
            if self.config["init_only"]:
                logger.info("Initialization complete, exiting (--init-only specified)")
                return
                
            if not await self.start():
                logger.error("System startup failed")
                return
            
            # Wait for stop signal
            await self.stop_event.wait()
            
        except asyncio.CancelledError:
            logger.info("System execution cancelled")
            
        finally:
            # Ensure everything is stopped
            self.stop()


async def main() -> int:
    """
    Main entry point for the Advanced RAG System launcher.
    
    Returns:
        int: Exit code (0 for success, non-zero for error)
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Advanced RAG System Launcher")
    parser.add_argument("--headless", action="store_true", help="Start without web UI (API only)")
    parser.add_argument("--port", type=int, default=8000, help="Port for web server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--config", type=str, help="Path to configuration directory")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser window")
    parser.add_argument("--init-only", action="store_true", help="Initialize system without starting server")
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Display startup banner
    print_startup_banner()
    
    # Set up signal handlers for graceful shutdown
    orchestrator = None
    
    def signal_handler():
        logger.info("Shutdown signal received")
        if orchestrator:
            orchestrator.stop()
    
    # Register signal handlers
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda sig, frame: signal_handler())
    
    try:
        # Create and run system orchestrator
        config = {
            "headless": args.headless,
            "port": args.port,
            "host": args.host,
            "config": args.config,
            "debug": args.debug,
            "no_browser": args.no_browser,
            "init_only": args.init_only
        }
        
        orchestrator = SystemOrchestrator(config)
        await orchestrator.run()
        
        return 0
        
    except Exception as e:
        logger.error(f"Error running RAG system: {e}", exc_info=True)
        return 1


def print_startup_banner() -> None:
    """
    Display an attractive startup banner with system information.
    """
    term_width = shutil.get_terminal_size().columns
    width = min(term_width, 80)
    
    # Adjust banner based on terminal width
    if width >= 60:
        banner = [
            "┌" + "─" * (width - 2) + "┐",
            "│" + " " * (width - 2) + "│",
            "│" + "  Advanced Local RAG System  ".center(width - 2) + "│",
            "│" + "  with External API Integration  ".center(width - 2) + "│",
            "│" + " " * (width - 2) + "│",
            "│" + f"  Version: 1.1.0 | Python {sys.version.split()[0]}  ".center(width - 2) + "│",
            "│" + " " * (width - 2) + "│",
            "└" + "─" * (width - 2) + "┘"
        ]
    else:
        # Simplified banner for narrow terminals
        banner = [
            "┌" + "─" * (width - 2) + "┐",
            "│" + "Advanced RAG System".center(width - 2) + "│",
            "│" + f"v1.1.0".center(width - 2) + "│",
            "└" + "─" * (width - 2) + "┘"
        ]
    
    for line in banner:
        print(line)


if __name__ == "__main__":
    # Run with proper asyncio handling
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nOperation canceled by user.")
        sys.exit(130)  # Standard exit code for SIGINT
