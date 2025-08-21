"""GCP client utilities and configuration."""

import os
from typing import Dict, Any, Optional
from google.cloud import logging, monitoring_v3
from google.auth import default
from google.auth.exceptions import DefaultCredentialsError


class GCPClientManager:
    """Manages GCP client creation and configuration.
    
    Provides centralized GCP client creation with proper credential handling
    and error management for Cloud Logging and Cloud Monitoring services.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize GCP client manager.
        
        Args:
            config: GCP configuration containing:
                - project_id: GCP project ID (required)
                - credentials_path: Path to service account JSON file (optional)
        """
        self.config = config
        self._validate_config()
        self._setup_credentials()

    def _validate_config(self) -> None:
        """Validate GCP configuration.
        
        Raises:
            ValueError: If required configuration is missing
        """
        if "project_id" not in self.config:
            raise ValueError("GCP project_id is required")

    def _setup_credentials(self) -> None:
        """Set up GCP credentials.
        
        Uses service account file if provided, otherwise uses default credentials.
        """
        if "credentials_path" in self.config:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.config["credentials_path"]

    def create_logging_client(self) -> logging.Client:
        """Create GCP Cloud Logging client.
        
        Returns:
            Configured Cloud Logging client
            
        Raises:
            DefaultCredentialsError: If credentials are not available
        """
        try:
            return logging.Client(project=self.config["project_id"])
        except Exception as e:
            raise DefaultCredentialsError(f"Failed to create GCP Logging client: {e}")

    def create_monitoring_client(self) -> monitoring_v3.MetricServiceClient:
        """Create GCP Cloud Monitoring client.
        
        Returns:
            Configured Cloud Monitoring client
            
        Raises:
            DefaultCredentialsError: If credentials are not available
        """
        try:
            return monitoring_v3.MetricServiceClient()
        except Exception as e:
            raise DefaultCredentialsError(f"Failed to create GCP Monitoring client: {e}")

    def create_alert_policy_client(self) -> monitoring_v3.AlertPolicyServiceClient:
        """Create GCP Alert Policy client.
        
        Returns:
            Configured Alert Policy client
            
        Raises:
            DefaultCredentialsError: If credentials are not available
        """
        try:
            return monitoring_v3.AlertPolicyServiceClient()
        except Exception as e:
            raise DefaultCredentialsError(f"Failed to create GCP Alert Policy client: {e}")

    async def test_credentials(self) -> Dict[str, Any]:
        """Test GCP credentials by making a simple API call.
        
        Returns:
            Dictionary containing:
            - valid: Whether credentials are valid
            - project_id: GCP project ID
            - credentials_path: Path to credentials file (if used)
            - error: Error message (if invalid)
        """
        try:
            # Try to create a logging client and make a simple call
            logging_client = self.create_logging_client()
            
            # List log entries (limit 1) to test connectivity
            entries = list(logging_client.list_entries(max_results=1))
            
            return {
                "valid": True,
                "project_id": self.config["project_id"],
                "credentials_path": self.config.get("credentials_path"),
                "error": None
            }
            
        except DefaultCredentialsError as e:
            return {
                "valid": False,
                "project_id": self.config["project_id"],
                "credentials_path": self.config.get("credentials_path"),
                "error": f"Credentials error: {e}"
            }
        except Exception as e:
            return {
                "valid": False,
                "project_id": self.config["project_id"],
                "credentials_path": self.config.get("credentials_path"),
                "error": f"Unexpected error: {e}"
            }


def create_gcp_config(
    project_id: str,
    credentials_path: Optional[str] = None
) -> Dict[str, Any]:
    """Create GCP configuration dictionary.
    
    Args:
        project_id: GCP project ID
        credentials_path: Path to service account JSON file (optional)
        
    Returns:
        GCP configuration dictionary
    """
    config = {"project_id": project_id}
    
    if credentials_path:
        config["credentials_path"] = credentials_path
    
    return config
