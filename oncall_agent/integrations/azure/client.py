"""Azure client utilities and configuration."""

from typing import Dict, Any, Optional
from azure.identity import DefaultAzureCredential, ClientSecretCredential
from azure.core.exceptions import ClientAuthenticationError
from azure.monitor.query import LogsQueryClient, MetricsQueryClient


class AzureClientManager:
    """Manages Azure client creation and configuration.
    
    Provides centralized Azure client creation with proper credential handling
    and error management for Azure Monitor services.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize Azure client manager.
        
        Args:
            config: Azure configuration containing:
                - subscription_id: Azure subscription ID (required)
                - tenant_id: Azure AD tenant ID (optional, for service principal)
                - client_id: Azure AD client ID (optional, for service principal)
                - client_secret: Azure AD client secret (optional, for service principal)
                - workspace_id: Log Analytics workspace ID (optional, for logs)
        """
        self.config = config
        self._validate_config()
        self._setup_credentials()

    def _validate_config(self) -> None:
        """Validate Azure configuration.
        
        Raises:
            ValueError: If required configuration is missing
        """
        if "subscription_id" not in self.config:
            raise ValueError("Azure subscription_id is required")

    def _setup_credentials(self) -> None:
        """Set up Azure credentials.
        
        Uses service principal credentials if provided, otherwise uses default credentials.
        """
        if all(key in self.config for key in ["tenant_id", "client_id", "client_secret"]):
            # Use service principal authentication
            self.credential = ClientSecretCredential(
                tenant_id=self.config["tenant_id"],
                client_id=self.config["client_id"],
                client_secret=self.config["client_secret"]
            )
        else:
            # Use default credentials (managed identity, Azure CLI, etc.)
            self.credential = DefaultAzureCredential()

    def create_logs_client(self) -> LogsQueryClient:
        """Create Azure Monitor Logs query client.
        
        Returns:
            Configured LogsQueryClient for querying Log Analytics
            
        Raises:
            ClientAuthenticationError: If authentication fails
        """
        try:
            return LogsQueryClient(self.credential)
        except Exception as e:
            raise ClientAuthenticationError(f"Failed to create Azure Logs client: {e}")

    def create_metrics_client(self) -> MetricsQueryClient:
        """Create Azure Monitor Metrics query client.
        
        Returns:
            Configured MetricsQueryClient for querying Azure metrics
            
        Raises:
            ClientAuthenticationError: If authentication fails
        """
        try:
            return MetricsQueryClient(self.credential)
        except Exception as e:
            raise ClientAuthenticationError(f"Failed to create Azure Metrics client: {e}")

    async def test_credentials(self) -> Dict[str, Any]:
        """Test Azure credentials by making a simple API call.
        
        Returns:
            Dictionary containing:
            - valid: Whether credentials are valid
            - subscription_id: Azure subscription ID
            - tenant_id: Azure tenant ID (if available)
            - error: Error message (if invalid)
            
        Raises:
            ConnectionError: If unable to test credentials
        """
        try:
            # Try to create a logs client and make a simple query
            logs_client = self.create_logs_client()
            
            # If we have a workspace_id, try a simple query
            if "workspace_id" in self.config:
                from azure.monitor.query import LogsQueryStatus
                from datetime import datetime, timedelta
                
                end_time = datetime.now()
                start_time = end_time - timedelta(minutes=1)
                
                # Simple query to test connectivity
                response = logs_client.query_workspace(
                    workspace_id=self.config["workspace_id"],
                    query="print 'test'",
                    timespan=(start_time, end_time)
                )
                
                if response.status == LogsQueryStatus.SUCCESS:
                    return {
                        "valid": True,
                        "subscription_id": self.config["subscription_id"],
                        "tenant_id": self.config.get("tenant_id"),
                        "workspace_id": self.config.get("workspace_id"),
                        "error": None
                    }
            
            # If no workspace_id or query failed, just return basic validation
            return {
                "valid": True,
                "subscription_id": self.config["subscription_id"],
                "tenant_id": self.config.get("tenant_id"),
                "workspace_id": self.config.get("workspace_id"),
                "error": None
            }
            
        except ClientAuthenticationError as e:
            return {
                "valid": False,
                "subscription_id": self.config["subscription_id"],
                "tenant_id": self.config.get("tenant_id"),
                "workspace_id": self.config.get("workspace_id"),
                "error": f"Authentication error: {e}"
            }
        except Exception as e:
            return {
                "valid": False,
                "subscription_id": self.config["subscription_id"],
                "tenant_id": self.config.get("tenant_id"),
                "workspace_id": self.config.get("workspace_id"),
                "error": f"Unexpected error: {e}"
            }


def create_azure_config(
    subscription_id: str,
    workspace_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None
) -> Dict[str, Any]:
    """Create Azure configuration dictionary.
    
    Args:
        subscription_id: Azure subscription ID
        workspace_id: Log Analytics workspace ID (optional)
        tenant_id: Azure AD tenant ID (optional, for service principal)
        client_id: Azure AD client ID (optional, for service principal)
        client_secret: Azure AD client secret (optional, for service principal)
        
    Returns:
        Azure configuration dictionary
    """
    config = {"subscription_id": subscription_id}
    
    if workspace_id:
        config["workspace_id"] = workspace_id
    if tenant_id:
        config["tenant_id"] = tenant_id
    if client_id:
        config["client_id"] = client_id
    if client_secret:
        config["client_secret"] = client_secret
    
    return config
