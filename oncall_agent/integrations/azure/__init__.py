"""Azure integrations for OnCall AI Agent."""

from .monitor_logs import AzureMonitorLogsProvider
from .monitor_metrics import AzureMonitorMetricsProvider
from .client import AzureClientManager, create_azure_config

__all__ = [
    "AzureMonitorLogsProvider",
    "AzureMonitorMetricsProvider",
    "AzureClientManager", 
    "create_azure_config"
]
