"""Abstract base class for log providers."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any


class LogProvider(ABC):
    """Abstract base class for all log providers.
    
    This class defines the interface that all log providers must implement
    to integrate with the OnCall AI Agent system.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the log provider with configuration.
        
        Args:
            config: Provider-specific configuration dictionary
        """
        self.config = config
        self._validate_config()

    @abstractmethod
    def _validate_config(self) -> None:
        """Validate the provider configuration.
        
        Raises:
            ValueError: If configuration is invalid or missing required fields
        """
        pass

    @abstractmethod
    async def fetch_logs(
        self,
        query: str,
        start_time: datetime,
        end_time: datetime,
        limit: Optional[int] = 1000,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Fetch logs based on query and time range.
        
        Args:
            query: Search query or filter expression
            start_time: Start of time range to search
            end_time: End of time range to search
            limit: Maximum number of log entries to return
            **kwargs: Provider-specific additional parameters
            
        Returns:
            List of log entries, each as a dictionary containing:
            - timestamp: Log entry timestamp
            - message: Log message content
            - level: Log level (INFO, ERROR, etc.)
            - source: Log source identifier
            - metadata: Additional log metadata
            
        Raises:
            ConnectionError: If unable to connect to log service
            ValueError: If query parameters are invalid
        """
        pass

    @abstractmethod
    async def search_logs_by_pattern(
        self,
        pattern: str,
        start_time: datetime,
        end_time: datetime,
        log_groups: Optional[List[str]] = None,
        limit: Optional[int] = 1000
    ) -> List[Dict[str, Any]]:
        """Search logs using pattern matching.
        
        Args:
            pattern: Regular expression or search pattern
            start_time: Start of time range to search
            end_time: End of time range to search
            log_groups: Specific log groups to search (if supported)
            limit: Maximum number of matches to return
            
        Returns:
            List of matching log entries with same structure as fetch_logs
            
        Raises:
            ConnectionError: If unable to connect to log service
            ValueError: If pattern or parameters are invalid
        """
        pass

    @abstractmethod
    async def get_log_groups(self) -> List[str]:
        """Get available log groups/streams.
        
        Returns:
            List of available log group names/identifiers
            
        Raises:
            ConnectionError: If unable to connect to log service
        """
        pass

    async def health_check(self) -> bool:
        """Check if the log provider is accessible.
        
        Returns:
            True if provider is healthy, False otherwise
        """
        try:
            await self.get_log_groups()
            return True
        except Exception:
            return False
