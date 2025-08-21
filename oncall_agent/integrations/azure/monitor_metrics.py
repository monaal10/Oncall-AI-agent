"""Azure Monitor Metrics integration."""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from azure.monitor.query import MetricsQueryClient
from azure.core.exceptions import ClientAuthenticationError, HttpResponseError

from ..base.metrics_provider import MetricsProvider
from .client import AzureClientManager


class AzureMonitorMetricsProvider(MetricsProvider):
    """Azure Monitor Metrics provider implementation.
    
    Provides integration with Azure Monitor Metrics service for fetching
    metric data and alert information from Azure resources.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize Azure Monitor Metrics provider.
        
        Args:
            config: Configuration dictionary containing:
                - subscription_id: Azure subscription ID (required)
                - tenant_id: Azure AD tenant ID (optional, for service principal)
                - client_id: Azure AD client ID (optional, for service principal)
                - client_secret: Azure AD client secret (optional, for service principal)
                - resource_groups: List of resource groups to monitor (optional)
        """
        super().__init__(config)
        self.client_manager = AzureClientManager(config)
        self.client = self.client_manager.create_metrics_client()

    def _validate_config(self) -> None:
        """Validate Azure Monitor Metrics configuration.
        
        Raises:
            ValueError: If required configuration is missing
        """
        if "subscription_id" not in self.config:
            raise ValueError("Azure subscription_id is required for Azure Monitor Metrics provider")

    async def get_metric_data(
        self,
        metric_name: str,
        namespace: str,
        start_time: datetime,
        end_time: datetime,
        dimensions: Optional[Dict[str, str]] = None,
        statistic: str = "Average",
        period: int = 300
    ) -> List[Dict[str, Any]]:
        """Get metric data points for a specific metric.
        
        Args:
            metric_name: Name of the metric to retrieve
            namespace: Azure resource ID or resource URI
            start_time: Start of time range
            end_time: End of time range
            dimensions: Metric dimensions as key-value pairs (not used in Azure)
            statistic: Statistic type (Average, Total, Maximum, Minimum, Count)
            period: Period in seconds for data point aggregation (converted to ISO duration)
            
        Returns:
            List of metric data points, each containing:
            - timestamp: Data point timestamp as datetime
            - value: Metric value as float
            - unit: Unit of measurement
            - statistic: Applied statistic
            
        Raises:
            ConnectionError: If unable to connect to Azure Monitor
            ValueError: If metric parameters are invalid
        """
        try:
            # Convert period to ISO 8601 duration format
            iso_interval = f"PT{period}S"
            
            # Map statistic names to Azure aggregation types
            aggregation_map = {
                "Average": "average",
                "Total": "total", 
                "Sum": "total",
                "Maximum": "maximum",
                "Minimum": "minimum",
                "Count": "count",
                "SampleCount": "count"
            }
            
            aggregation = aggregation_map.get(statistic, "average")

            response = await asyncio.to_thread(
                self.client.query_resource,
                resource_uri=namespace,
                metric_names=[metric_name],
                timespan=(start_time, end_time),
                granularity=timedelta(seconds=period),
                aggregations=[aggregation]
            )

            # Process data points
            data_points = []
            for metric in response.metrics:
                for timeseries in metric.timeseries:
                    for data_point in timeseries.data:
                        if hasattr(data_point, aggregation) and getattr(data_point, aggregation) is not None:
                            point = {
                                "timestamp": data_point.timestamp,
                                "value": getattr(data_point, aggregation),
                                "unit": metric.unit.value if metric.unit else "None",
                                "statistic": statistic
                            }
                            data_points.append(point)

            # Sort by timestamp
            data_points.sort(key=lambda x: x["timestamp"])
            return data_points

        except ClientAuthenticationError as e:
            raise ConnectionError(f"Azure authentication error: {e}")
        except HttpResponseError as e:
            raise ConnectionError(f"Azure Monitor API error: {e}")
        except Exception as e:
            raise ConnectionError(f"Failed to get metric data: {e}")

    async def get_alarms(
        self,
        alarm_names: Optional[List[str]] = None,
        state_value: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get Azure Monitor alert rules information.
        
        Note: This is a simplified implementation. Full alert rule querying
        requires Azure Resource Manager API integration.
        
        Args:
            alarm_names: Specific alert rule names to retrieve (None for all)
            state_value: Filter by alert state (not implemented for Azure)
            
        Returns:
            List of alerts (placeholder implementation)
            
        Raises:
            ConnectionError: If unable to connect to Azure Monitor
        """
        # Note: This would require Azure Resource Manager API integration
        # For now, return empty list with explanation
        return []

    async def get_alarm_history(
        self,
        alarm_name: str,
        start_time: datetime,
        end_time: datetime,
        history_item_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get Azure Monitor alert history.
        
        Note: This would require Activity Log integration.
        
        Args:
            alarm_name: Name of the alert rule
            start_time: Start of time range
            end_time: End of time range
            history_item_type: Type of history items (not used in Azure)
            
        Returns:
            List of history items (placeholder implementation)
            
        Raises:
            ConnectionError: If unable to connect to Azure Monitor
        """
        # Note: This would require Activity Log API integration
        # For now, return empty list
        return []

    async def list_metrics(
        self,
        namespace: Optional[str] = None,
        metric_name: Optional[str] = None,
        dimensions: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        """List available Azure Monitor metrics for a resource.
        
        Args:
            namespace: Azure resource URI (required for Azure)
            metric_name: Filter by metric name (not used in listing)
            dimensions: Filter by dimensions (not used in Azure)
            
        Returns:
            List of available metrics, each containing:
            - metric_name: Name of the metric
            - namespace: Resource URI
            - dimensions: Empty list (Azure handles dimensions differently)
            
        Raises:
            ConnectionError: If unable to connect to Azure Monitor
            ValueError: If namespace (resource URI) is not provided
        """
        if not namespace:
            raise ValueError("Resource URI (namespace) is required for listing Azure metrics")

        try:
            response = await asyncio.to_thread(
                self.client.list_metric_definitions,
                resource_uri=namespace
            )

            metrics = []
            for metric_def in response:
                metric_info = {
                    "metric_name": metric_def.name.value,
                    "namespace": namespace,
                    "dimensions": [],  # Azure dimensions are handled differently
                    "unit": metric_def.unit.value if metric_def.unit else "None",
                    "aggregations": [agg.value for agg in metric_def.supported_aggregation_types] if metric_def.supported_aggregation_types else [],
                    "description": metric_def.display_description or ""
                }
                metrics.append(metric_info)

            return metrics

        except ClientAuthenticationError as e:
            raise ConnectionError(f"Azure authentication error: {e}")
        except HttpResponseError as e:
            raise ConnectionError(f"Azure Monitor API error: {e}")
        except Exception as e:
            raise ConnectionError(f"Failed to list metrics: {e}")

    async def get_resource_metrics(
        self,
        resource_uri: str,
        hours: int = 1
    ) -> Dict[str, Any]:
        """Get recent metrics for a specific Azure resource.
        
        Args:
            resource_uri: Azure resource URI
            hours: Number of hours to look back
            
        Returns:
            Dictionary containing:
            - resource_uri: The resource URI
            - available_metrics: List of available metrics
            - sample_data: Sample data for common metrics
            - time_range: Time range analyzed
            
        Raises:
            ConnectionError: If unable to connect to Azure Monitor
        """
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours)
            
            # Get available metrics for this resource
            available_metrics = await self.list_metrics(namespace=resource_uri)
            
            # Try to get sample data for common metrics
            common_metrics = ["Percentage CPU", "Network In", "Network Out", "Disk Read Bytes", "Disk Write Bytes"]
            sample_data = {}
            
            for metric_name in common_metrics:
                # Check if this metric exists for the resource
                if any(m["metric_name"] == metric_name for m in available_metrics):
                    try:
                        data = await self.get_metric_data(
                            metric_name=metric_name,
                            namespace=resource_uri,
                            start_time=start_time,
                            end_time=end_time,
                            statistic="Average",
                            period=300
                        )
                        if data:
                            sample_data[metric_name] = data[-5:]  # Last 5 data points
                    except Exception:
                        # Skip metrics that can't be retrieved
                        continue
            
            return {
                "resource_uri": resource_uri,
                "time_range": {
                    "start": start_time,
                    "end": end_time,
                    "hours": hours
                },
                "total_available_metrics": len(available_metrics),
                "available_metrics": available_metrics[:10],  # First 10 metrics
                "sample_data": sample_data
            }

        except Exception as e:
            raise ConnectionError(f"Failed to get resource metrics: {e}")

    async def get_subscription_resources(self) -> List[Dict[str, Any]]:
        """Get list of resources in the subscription that support metrics.
        
        Note: This would require Azure Resource Manager API integration.
        
        Returns:
            List of resources (placeholder implementation)
            
        Raises:
            ConnectionError: If unable to connect to Azure
        """
        # Note: This would require Azure Resource Manager API integration
        # For now, return common resource URI patterns as examples
        return [
            {
                "name": "Example VM",
                "resource_uri": f"/subscriptions/{self.config['subscription_id']}/resourceGroups/example-rg/providers/Microsoft.Compute/virtualMachines/example-vm",
                "type": "Microsoft.Compute/virtualMachines"
            },
            {
                "name": "Example App Service",
                "resource_uri": f"/subscriptions/{self.config['subscription_id']}/resourceGroups/example-rg/providers/Microsoft.Web/sites/example-app",
                "type": "Microsoft.Web/sites"
            }
        ]

    def _build_resource_uri(
        self,
        resource_group: str,
        resource_type: str,
        resource_name: str
    ) -> str:
        """Build Azure resource URI.
        
        Args:
            resource_group: Resource group name
            resource_type: Resource type (e.g., Microsoft.Compute/virtualMachines)
            resource_name: Resource name
            
        Returns:
            Complete Azure resource URI
        """
        return (
            f"/subscriptions/{self.config['subscription_id']}"
            f"/resourceGroups/{resource_group}"
            f"/providers/{resource_type}"
            f"/{resource_name}"
        )

    async def get_vm_metrics(
        self,
        resource_group: str,
        vm_name: str,
        hours: int = 1
    ) -> Dict[str, Any]:
        """Get metrics for a specific Virtual Machine.
        
        Args:
            resource_group: Resource group containing the VM
            vm_name: Virtual machine name
            hours: Number of hours to look back
            
        Returns:
            Dictionary containing VM metrics data
            
        Raises:
            ConnectionError: If unable to connect to Azure Monitor
        """
        resource_uri = self._build_resource_uri(
            resource_group=resource_group,
            resource_type="Microsoft.Compute/virtualMachines",
            resource_name=vm_name
        )
        
        return await self.get_resource_metrics(resource_uri, hours)

    async def get_app_service_metrics(
        self,
        resource_group: str,
        app_name: str,
        hours: int = 1
    ) -> Dict[str, Any]:
        """Get metrics for a specific App Service.
        
        Args:
            resource_group: Resource group containing the App Service
            app_name: App Service name
            hours: Number of hours to look back
            
        Returns:
            Dictionary containing App Service metrics data
            
        Raises:
            ConnectionError: If unable to connect to Azure Monitor
        """
        resource_uri = self._build_resource_uri(
            resource_group=resource_group,
            resource_type="Microsoft.Web/sites",
            resource_name=app_name
        )
        
        return await self.get_resource_metrics(resource_uri, hours)
