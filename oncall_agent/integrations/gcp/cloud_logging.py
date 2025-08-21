"""GCP Cloud Logging integration."""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from google.cloud import logging
from google.auth.exceptions import DefaultCredentialsError
from google.api_core.exceptions import GoogleAPIError

from ..base.log_provider import LogProvider
from .client import GCPClientManager


class GCPCloudLoggingProvider(LogProvider):
    """GCP Cloud Logging provider implementation.
    
    Provides integration with Google Cloud Logging service for fetching
    and searching log data using Cloud Logging filters.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize GCP Cloud Logging provider.
        
        Args:
            config: Configuration dictionary containing:
                - project_id: GCP project ID (required)
                - credentials_path: Path to service account JSON file (optional)
                - log_names: List of log names to search (optional)
        """
        super().__init__(config)
        self.client_manager = GCPClientManager(config)
        self.client = self.client_manager.create_logging_client()

    def _validate_config(self) -> None:
        """Validate GCP Cloud Logging configuration.
        
        Raises:
            ValueError: If required configuration is missing
        """
        if "project_id" not in self.config:
            raise ValueError("GCP project_id is required for Cloud Logging provider")

    async def fetch_logs(
        self,
        query: str,
        start_time: datetime,
        end_time: datetime,
        limit: Optional[int] = 1000,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Fetch logs using Cloud Logging filter query.
        
        Args:
            query: Cloud Logging filter expression
            start_time: Start of time range to search
            end_time: End of time range to search
            limit: Maximum number of log entries to return
            **kwargs: Additional parameters (ignored for GCP)
                
        Returns:
            List of log entries, each containing:
            - timestamp: Log entry timestamp as datetime
            - message: Log message content
            - level: Log level (severity)
            - source: Log name or resource
            - metadata: Additional fields from the log entry
            
        Raises:
            ConnectionError: If unable to connect to Cloud Logging
            ValueError: If query parameters are invalid
        """
        try:
            # Build time filter
            time_filter = (
                f'timestamp >= "{start_time.isoformat()}Z" AND '
                f'timestamp <= "{end_time.isoformat()}Z"'
            )
            
            # Combine query with time filter
            if query:
                full_filter = f"({query}) AND ({time_filter})"
            else:
                full_filter = time_filter

            entries = await asyncio.to_thread(
                lambda: list(self.client.list_entries(
                    filter_=full_filter,
                    order_by=logging.DESCENDING,
                    max_results=limit or 1000
                ))
            )

            # Process entries
            logs = []
            for entry in entries:
                log_entry = self._process_log_entry(entry)
                logs.append(log_entry)

            return logs

        except DefaultCredentialsError as e:
            raise ConnectionError(f"GCP authentication error: {e}")
        except GoogleAPIError as e:
            raise ConnectionError(f"GCP Cloud Logging API error: {e}")
        except Exception as e:
            raise ConnectionError(f"Failed to fetch logs: {e}")

    async def search_logs_by_pattern(
        self,
        pattern: str,
        start_time: datetime,
        end_time: datetime,
        log_groups: Optional[List[str]] = None,
        limit: Optional[int] = 1000
    ) -> List[Dict[str, Any]]:
        """Search logs using pattern matching with Cloud Logging text search.
        
        Args:
            pattern: Text pattern to search for
            start_time: Start of time range to search
            end_time: End of time range to search
            log_groups: Specific log names to search (optional)
            limit: Maximum number of matches to return
            
        Returns:
            List of matching log entries with same structure as fetch_logs
            
        Raises:
            ConnectionError: If unable to connect to Cloud Logging
            ValueError: If pattern or parameters are invalid
        """
        try:
            # Build text search filter
            text_filter = f'textPayload:"{pattern}" OR jsonPayload:"{pattern}"'
            
            # Add log name filter if specified
            if log_groups:
                log_filter = " OR ".join([f'logName="projects/{self.config["project_id"]}/logs/{log_name}"' for log_name in log_groups])
                text_filter = f"({text_filter}) AND ({log_filter})"

            return await self.fetch_logs(
                query=text_filter,
                start_time=start_time,
                end_time=end_time,
                limit=limit
            )

        except Exception as e:
            raise ConnectionError(f"Failed to search logs: {e}")

    async def get_log_groups(self) -> List[str]:
        """Get available Cloud Logging log names.
        
        Returns:
            List of log names available in the project
            
        Raises:
            ConnectionError: If unable to connect to Cloud Logging
        """
        try:
            # Get recent entries to discover log names
            end_time = datetime.now()
            start_time = end_time - timedelta(days=7)  # Look back 7 days
            
            time_filter = (
                f'timestamp >= "{start_time.isoformat()}Z" AND '
                f'timestamp <= "{end_time.isoformat()}Z"'
            )

            entries = await asyncio.to_thread(
                lambda: list(self.client.list_entries(
                    filter_=time_filter,
                    max_results=1000
                ))
            )

            # Extract unique log names
            log_names = set()
            for entry in entries:
                if hasattr(entry, 'log_name') and entry.log_name:
                    # Extract log name from full path
                    # Format: projects/PROJECT_ID/logs/LOG_NAME
                    parts = entry.log_name.split('/')
                    if len(parts) >= 4 and parts[2] == 'logs':
                        log_names.add(parts[3])

            return sorted(list(log_names))

        except Exception as e:
            # Return common log names as fallback
            return [
                "cloudsql.googleapis.com%2Fmysql.err",
                "compute.googleapis.com%2Factivity_log",
                "container.googleapis.com%2Fcluster-autoscaler-visibility",
                "run.googleapis.com%2Frequests",
                "appengine.googleapis.com%2Frequest_log"
            ]

    def _process_log_entry(self, entry) -> Dict[str, Any]:
        """Process a Cloud Logging entry.
        
        Args:
            entry: Raw log entry from Cloud Logging
            
        Returns:
            Processed log entry dictionary
        """
        log_entry = {
            "timestamp": entry.timestamp if hasattr(entry, 'timestamp') else datetime.now(),
            "message": "",
            "level": "INFO",
            "source": "",
            "metadata": {}
        }

        # Extract message
        if hasattr(entry, 'payload'):
            if hasattr(entry.payload, 'get'):
                # Structured payload
                log_entry["message"] = str(entry.payload)
                log_entry["metadata"]["payload_type"] = "structured"
            else:
                # Text payload
                log_entry["message"] = str(entry.payload)
                log_entry["metadata"]["payload_type"] = "text"
        
        # Extract severity/level
        if hasattr(entry, 'severity'):
            log_entry["level"] = entry.severity.name if entry.severity else "INFO"

        # Extract source information
        if hasattr(entry, 'log_name') and entry.log_name:
            # Extract log name from full path
            parts = entry.log_name.split('/')
            if len(parts) >= 4 and parts[2] == 'logs':
                log_entry["source"] = parts[3]
            else:
                log_entry["source"] = entry.log_name

        # Extract resource information
        if hasattr(entry, 'resource') and entry.resource:
            log_entry["metadata"]["resource_type"] = entry.resource.type
            if hasattr(entry.resource, 'labels'):
                log_entry["metadata"]["resource_labels"] = dict(entry.resource.labels)

        # Extract labels
        if hasattr(entry, 'labels') and entry.labels:
            log_entry["metadata"]["labels"] = dict(entry.labels)

        # Extract HTTP request info if available
        if hasattr(entry, 'http_request') and entry.http_request:
            log_entry["metadata"]["http_request"] = {
                "method": entry.http_request.request_method,
                "url": entry.http_request.request_url,
                "status": entry.http_request.status,
                "user_agent": entry.http_request.user_agent
            }

        return log_entry

    async def get_recent_errors(
        self,
        hours: int = 1,
        log_groups: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Get recent error logs from specified time range.
        
        Args:
            hours: Number of hours to look back
            log_groups: Specific log names to search
            
        Returns:
            List of error log entries
            
        Raises:
            ConnectionError: If unable to connect to Cloud Logging
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        # Build error filter
        error_filter = 'severity >= "ERROR"'
        
        # Add log name filter if specified
        if log_groups:
            log_filter = " OR ".join([
                f'logName="projects/{self.config["project_id"]}/logs/{log_name}"' 
                for log_name in log_groups
            ])
            error_filter = f"({error_filter}) AND ({log_filter})"
        
        return await self.fetch_logs(
            query=error_filter,
            start_time=start_time,
            end_time=end_time,
            limit=100
        )

    async def get_app_engine_logs(
        self,
        start_time: datetime,
        end_time: datetime,
        limit: Optional[int] = 1000
    ) -> List[Dict[str, Any]]:
        """Get App Engine application logs.
        
        Args:
            start_time: Start of time range
            end_time: End of time range
            limit: Maximum number of entries to return
            
        Returns:
            List of App Engine log entries
            
        Raises:
            ConnectionError: If unable to connect to Cloud Logging
        """
        app_engine_filter = 'resource.type="gae_app"'
        
        return await self.fetch_logs(
            query=app_engine_filter,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )

    async def get_gke_logs(
        self,
        cluster_name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = 1000
    ) -> List[Dict[str, Any]]:
        """Get Google Kubernetes Engine logs.
        
        Args:
            cluster_name: Specific cluster name to filter by
            start_time: Start of time range (defaults to 1 hour ago)
            end_time: End of time range (defaults to now)
            limit: Maximum number of entries to return
            
        Returns:
            List of GKE log entries
            
        Raises:
            ConnectionError: If unable to connect to Cloud Logging
        """
        if not start_time:
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=1)
        elif not end_time:
            end_time = datetime.now()

        # Build GKE filter
        gke_filter = 'resource.type="k8s_container" OR resource.type="gke_cluster"'
        
        if cluster_name:
            gke_filter += f' AND resource.labels.cluster_name="{cluster_name}"'
        
        return await self.fetch_logs(
            query=gke_filter,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )

    async def get_cloud_function_logs(
        self,
        function_name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = 1000
    ) -> List[Dict[str, Any]]:
        """Get Cloud Functions logs.
        
        Args:
            function_name: Specific function name to filter by
            start_time: Start of time range (defaults to 1 hour ago)
            end_time: End of time range (defaults to now)
            limit: Maximum number of entries to return
            
        Returns:
            List of Cloud Functions log entries
            
        Raises:
            ConnectionError: If unable to connect to Cloud Logging
        """
        if not start_time:
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=1)
        elif not end_time:
            end_time = datetime.now()

        # Build Cloud Functions filter
        cf_filter = 'resource.type="cloud_function"'
        
        if function_name:
            cf_filter += f' AND resource.labels.function_name="{function_name}"'
        
        return await self.fetch_logs(
            query=cf_filter,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )
