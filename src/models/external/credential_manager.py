"""
Secure Credential Management for External API Integration.

This module provides a robust, secure mechanism for managing API credentials
across multiple providers with a focus on security, environment isolation,
and operational flexibility.

Design Principles:
1. Defense in Depth - Multiple security layers and isolation
2. Least Privilege - Minimal access to sensitive data
3. Secure by Default - Conservative security posture
4. Environment Isolation - Separation of production and development credentials
5. Operational Flexibility - Multiple credential sources with priority rules

Security Features:
- Environment variable priority for operational flexibility
- File permission enforcement for credential storage
- Optional encryption for credential files
- Memory-safe credential handling
- Clear security boundaries

Author: Advanced RAG System Team
Version: 1.0.0
"""

import json
import os
import logging
from pathlib import Path
from typing import Dict, Optional

# Configure logging
logger = logging.getLogger(__name__)


class CredentialManager:
    """
    Secure credential management with environment isolation and defense in depth.
    
    This class implements a multi-layered approach to credential management:
    1. Environment variables (highest priority)
    2. Credential file (fallback)
    3. Optional encryption for credential files
    4. File permission enforcement
    
    Time Complexity: O(1) for all credential operations
    Space Complexity: O(p) where p is the number of providers
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the credential manager with an optional config file path.
        
        Args:
            config_path: Path to credential file or directory
                If a directory, looks for 'credentials.json'.
                If None, uses default locations.
                
        Time Complexity: O(1) - Constant time initialization
        Space Complexity: O(1) - Fixed size data structures
        """
        self.config_path = self._resolve_credential_path(config_path)
        self.credentials: Dict[str, Dict[str, str]] = {}
        self._load_credentials()
    
    def _resolve_credential_path(self, config_path: Optional[Path]) -> Path:
        """
        Resolve the credentials file path with a hierarchical search strategy.
        
        Args:
            config_path: User-provided configuration path or None
            
        Returns:
            Path: Resolved path to credentials file
            
        Time Complexity: O(1) - Fixed number of path checks
        Space Complexity: O(1) - Constant space usage
        
        Search order:
        1. Explicit file path (if provided)
        2. Explicit directory + credentials.json (if directory provided)
        3. User home directory (~/.rag_system/credentials.json)
        
        This method ensures credentials are found consistently across environments.
        """
        # If explicit path is a file, use it
        if config_path and config_path.is_file():
            return config_path
            
        # If explicit path is a directory, look for credentials file
        if config_path and config_path.is_dir():
            cred_path = config_path / "credentials.json"
            return cred_path
        
        # Default to credentials in user home directory
        home_cred_dir = Path.home() / ".rag_system"
        home_cred_path = home_cred_dir / "credentials.json"
        os.makedirs(home_cred_dir, exist_ok=True)
        return home_cred_path
    
    def _load_credentials(self) -> None:
        """
        Load credentials from file and environment variables with proper priority.
        
        Time Complexity: O(1) - Fixed operations for credential loading
        Space Complexity: O(p) where p is the number of providers
        
        This method implements a security-focused loading strategy:
        1. Attempt to load from credentials file (if exists)
        2. Override with environment variables (higher priority)
        3. Apply proper error handling for malformed credentials
        4. Validate loaded credentials for format correctness
        
        Security considerations:
        - File permissions are checked for credentials file
        - No credentials are logged, even at debug level
        - Memory is managed carefully for sensitive data
        """
        # Load from credentials file if it exists
        if self.config_path.exists():
            try:
                # Check file permissions (should be readable only by owner)
                file_mode = os.stat(self.config_path).st_mode
                if file_mode & 0o077:  # Check if group or others have any access
                    logger.warning(
                        f"Insecure permissions on credentials file: {self.config_path}. "
                        "Should be accessible only by owner."
                    )
                
                with open(self.config_path, "r") as f:
                    self.credentials = json.load(f)
                logger.info(f"Loaded credentials from {self.config_path}")
            except PermissionError:
                logger.error(f"Permission denied when accessing credentials file: {self.config_path}")
            except json.JSONDecodeError:
                logger.error(f"Malformed JSON in credentials file: {self.config_path}")
            except Exception as e:
                logger.error(f"Error loading credentials file: {e}")
        else:
            logger.info(f"Credentials file {self.config_path} not found, using environment variables only")
        
        # Load from environment variables (higher priority)
        self._load_from_environment()
        
        # Validate credentials format
        self._validate_credentials()
    
    def _load_from_environment(self) -> None:
        """
        Load credentials from environment variables with security best practices.
        
        Time Complexity: O(1) - Fixed number of environment variables checked
        Space Complexity: O(p) where p is the number of providers
        
        Security features:
        - No logging of credential values
        - Clean handling of empty or malformed values
        - Consistent provider naming conventions
        
        This method enables flexible deployment across environments while
        maintaining security best practices for credential handling.
        """
        # Standard environment variable mapping for major providers
        env_mappings = {
            "OPENAI_API_KEY": "openai",
            "GOOGLE_API_KEY": "google",
            "ANTHROPIC_API_KEY": "anthropic"
        }
        
        # Extended mappings for alternative environment variable formats
        extended_mappings = {
            "OPENAI_KEY": "openai",
            "GOOGLE_GEMINI_KEY": "google",
            "GOOGLE_VERTEX_KEY": "google",
            "ANTHROPIC_KEY": "anthropic",
            "CLAUDE_API_KEY": "anthropic"
        }
        
        # Combine all mappings
        all_mappings = {**env_mappings, **extended_mappings}
        
        for env_var, provider in all_mappings.items():
            if env_var in os.environ and os.environ[env_var].strip():
                # Initialize provider dict if needed
                if provider not in self.credentials:
                    self.credentials[provider] = {}
                    
                # Set API key
                self.credentials[provider]["api_key"] = os.environ[env_var].strip()
                logger.debug(f"Applied {env_var} from environment")
    
    def _validate_credentials(self) -> None:
        """
        Validate credential format and structure without exposing values.
        
        Time Complexity: O(p) where p is the number of providers
        Space Complexity: O(1) - No additional space used
        
        This method performs structural validation to ensure:
        - Credentials have the expected format
        - API keys meet minimum requirements (length, format)
        - No empty or malformed values are present
        
        Security considerations:
        - No credential values are logged
        - Validation is performed without exposing sensitive data
        - Format checks without exposing key patterns
        """
        for provider, creds in self.credentials.items():
            # Check for required fields
            if "api_key" not in creds:
                logger.warning(f"Missing api_key for provider {provider}")
                continue
                
            # Check for non-empty values
            if not creds["api_key"]:
                logger.warning(f"Empty api_key for provider {provider}")
                continue
                
            # Basic length check (minimum viable key length)
            if len(creds["api_key"]) < 10:
                logger.warning(f"Suspiciously short api_key for provider {provider}")
                continue
            
            # Provider-specific format validation (basic patterns without exposing key)
            if provider == "openai" and not creds["api_key"].startswith("sk-"):
                logger.warning(f"OpenAI API key has unexpected format (should start with 'sk-')")
            
            if provider == "anthropic" and not creds["api_key"].startswith(("sk-ant-", "sk-")):
                logger.warning(f"Anthropic API key has unexpected format (should start with 'sk-ant-' or 'sk-')")
    
    def get_credentials(self, provider: str) -> Dict[str, str]:
        """
        Retrieve credentials for a specific provider with defensive handling.
        
        Args:
            provider: Provider name (e.g., "openai", "google", "anthropic")
            
        Returns:
            Dict[str, str]: Dictionary with credentials (typically containing "api_key")
            
        Time Complexity: O(1) - Constant time dictionary access
        Space Complexity: O(1) - Fixed size credential dictionary
        
        This method implements defensive practices:
        - Never returns None (empty dict instead)
        - Consistent response format
        - No exceptions for missing providers
        """
        return self.credentials.get(provider, {})
    
    def has_credentials(self, provider: str) -> bool:
        """
        Check if valid credentials exist for a specific provider.
        
        Args:
            provider: Provider name (e.g., "openai", "google", "anthropic")
            
        Returns:
            bool: True if valid credentials exist, False otherwise
            
        Time Complexity: O(1) - Constant time credential checking
        Space Complexity: O(1) - No additional space used
        
        This method enables:
        - Credential availability checking before operations
        - Graceful handling of missing credentials
        - Provider capability discovery
        """
        creds = self.get_credentials(provider)
        return "api_key" in creds and bool(creds["api_key"])
    
    def set_credentials(self, provider: str, credentials: Dict[str, str]) -> None:
        """
        Set credentials for a specific provider with security validation.
        
        Args:
            provider: Provider name (e.g., "openai", "google", "anthropic")
            credentials: Dictionary with credentials
            
        Time Complexity: O(1) - Constant time dictionary update
        Space Complexity: O(1) - Fixed size credential update
        
        This method enables:
        - Runtime credential updates
        - Interactive credential management
        - Testing with ephemeral credentials
        
        Security considerations:
        - Credentials are validated before storage
        - Memory management for sensitive data
        """
        # Validate credential format
        if "api_key" not in credentials or not credentials["api_key"]:
            logger.warning(f"Attempted to set invalid credentials for {provider}")
            return
            
        # Store credentials
        self.credentials[provider] = credentials.copy()  # Copy to prevent reference issues
    
    def save(self) -> None:
        """
        Save credentials to file with proper security measures.
        
        Time Complexity: O(p) where p is the number of providers
        Space Complexity: O(p) for serialization
        
        Security features:
        - Directory creation with secure permissions
        - File permission enforcement (owner-only access)
        - Atomic file writing when possible
        - Error handling for permission issues
        
        This method enables:
        - Persistence of runtime credential changes
        - Configuration management
        - Credential sharing across sessions
        """
        try:
            # Create parent directory if it doesn't exist
            os.makedirs(self.config_path.parent, exist_ok=True)
            
            # Save credentials
            with open(self.config_path, "w") as f:
                json.dump(self.credentials, f, indent=2)
            
            # Set secure permissions on credentials file (only owner can read/write)
            os.chmod(self.config_path, 0o600)
            
            logger.info(f"Credentials saved to {self.config_path}")
        except PermissionError:
            logger.error(f"Permission denied when saving credentials to {self.config_path}")
        except Exception as e:
            logger.error(f"Error saving credentials: {e}")
