"""GCP Cloud Monitoring integration."""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from google.cloud import monitoring_v3
from google.auth.exceptions import DefaultCredentialsError
from google.api_core.exceptions import GoogleAPIError

from ..base.metrics_provider import MetricsProvider
from .client import GCPClientManager


class GCPCloudMonitoringProvider(MetricsProvider):
    """GCP Cloud Monitoring provider implementation.
    
    Provides integration with Google Cloud Monitoring service for fetching
    metric data and alert policy information.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize GCP Cloud Monitoring provider.
        
        Args:
            config: Configuration dictionary containing:
                - project_id: GCP project ID (required)
                - credentials_path: Path to service account JSON file (optional)
        """
        super().__init__(config)
        self.client_manager = GCPClientManager(config)
        self.client = self.client_manager.create_monitoring_client()
        self.alert_client = self.client_manager.create_alert_policy_client()
        self.project_name = f"projects/{self.config['project_id']}"

    def _validate_config(self) -> None:
        """Validate GCP Cloud Monitoring configuration.
        
        Raises:
            ValueError: If required configuration is missing
        """
        if "project_id" not in self.config:
            raise ValueError("GCP project_id is required for Cloud Monitoring provider")

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
            metric_name: Name of the metric to retrieve (e.g., compute.googleapis.com/instance/cpu/utilization)
            namespace: Resource type filter (e.g., gce_instance)
            start_time: Start of time range
            end_time: End of time range
            dimensions: Resource labels as key-value pairs
            statistic: Statistic type (MEAN, MAX, MIN, SUM, COUNT)
            period: Period in seconds for data point aggregation
            
        Returns:
            List of metric data points, each containing:
            - timestamp: Data point timestamp as datetime
            - value: Metric value as float
            - unit: Unit of measurement
            - statistic: Applied statistic
            
        Raises:
            ConnectionError: If unable to connect to Cloud Monitoring
            ValueError: If metric parameters are invalid
        """
        try:
            # Map statistic names to Cloud Monitoring reducers
            reducer_map = {
                "Average": monitoring_v3.Aggregation.Reducer.REDUCE_MEAN,
                "Mean": monitoring_v3.Aggregation.Reducer.REDUCE_MEAN,
                "Maximum": monitoring_v3.Aggregation.Reducer.REDUCE_MAX,
                "Max": monitoring_v3.Aggregation.Reducer.REDUCE_MAX,
                "Minimum": monitoring_v3.Aggregation.Reducer.REDUCE_MIN,
                "Min": monitoring_v3.Aggregation.Reducer.REDUCE_MIN,
                "Sum": monitoring_v3.Aggregation.Reducer.REDUCE_SUM,
                "Total": monitoring_v3.Aggregation.Reducer.REDUCE_SUM,
                "Count": monitoring_v3.Aggregation.Reducer.REDUCE_COUNT,
                "SampleCount": monitoring_v3.Aggregation.Reducer.REDUCE_COUNT
            }
            
            reducer = reducer_map.get(statistic, monitoring_v3.Aggregation.Reducer.REDUCE_MEAN)

            # Build time interval
            interval = monitoring_v3.TimeInterval({
                "end_time": {"seconds": int(end_time.timestamp())},
                "start_time": {"seconds": int(start_time.timestamp())}
            })

            # Build aggregation
            aggregation = monitoring_v3.Aggregation({
                "alignment_period": {"seconds": period},
                "per_series_aligner": monitoring_v3.Aggregation.Aligner.ALIGN_MEAN,
                "cross_series_reducer": reducer,
                "group_by_fields": []
            })

            # Build filter
            filter_str = f'metric.type="{metric_name}"'
            if namespace:
                filter_str += f' AND resource.type="{namespace}"'
            
            # Add dimension filters
            if dimensions:
                for key, value in dimensions.items():
                    filter_str += f' AND resource.label.{key}="{value}"'

            request = monitoring_v3.ListTimeSeriesRequest({
                "name": self.project_name,
                "filter": filter_str,
                "interval": interval,
                "aggregation": aggregation,
                "view": monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL
            })

            results = await asyncio.to_thread(
                self.client.list_time_series,
                request=request
            )

            # Process data points
            data_points = []
            for result in results:
                for point in result.points:
                    # Extract value based on value type
                    value = None
                    if hasattr(point.value, 'double_value'):
                        value = point.value.double_value
                    elif hasattr(point.value, 'int64_value'):
                        value = float(point.value.int64_value)
                    elif hasattr(point.value, 'bool_value'):
                        value = 1.0 if point.value.bool_value else 0.0
                    
                    if value is not None:
                        data_point = {
                            "timestamp": datetime.fromtimestamp(point.interval.end_time.seconds),
                            "value": value,
                            "unit": result.unit if hasattr(result, 'unit') else "1",
                            "statistic": statistic
                        }
                        data_points.append(data_point)

            # Sort by timestamp
            data_points.sort(key=lambda x: x["timestamp"])
            return data_points

        except DefaultCredentialsError as e:
            raise ConnectionError(f"GCP authentication error: {e}")
        except GoogleAPIError as e:
            raise ConnectionError(f"GCP Cloud Monitoring API error: {e}")
        except Exception as e:
            raise ConnectionError(f"Failed to get metric data: {e}")

    async def get_alarms(
        self,
        alarm_names: Optional[List[str]] = None,
        state_value: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get GCP alert policies information.
        
        Args:
            alarm_names: Specific alert policy names to retrieve (None for all)
            state_value: Filter by alert state (not directly supported in GCP)
            
        Returns:
            List of alert policies, each containing:
            - name: Alert policy name
            - state: Current state (enabled/disabled)
            - display_name: Human-readable name
            - conditions: List of alert conditions
            - notification_channels: List of notification channels
            
        Raises:
            ConnectionError: If unable to connect to Cloud Monitoring
        """
        try:
            request = monitoring_v3.ListAlertPoliciesRequest({
                "name": self.project_name
            })

            policies = await asyncio.to_thread(
                self.alert_client.list_alert_policies,
                request=request
            )

            alerts = []
            for policy in policies:
                # Filter by names if specified
                if alarm_names and policy.display_name not in alarm_names:
                    continue

                alert_info = {
                    "name": policy.name.split('/')[-1],  # Extract policy ID
                    "display_name": policy.display_name,
                    "state": "enabled" if policy.enabled else "disabled",
                    "conditions": [],
                    "notification_channels": list(policy.notification_channels),
                    "documentation": policy.documentation.content if policy.documentation else "",
                    "creation_record": {
                        "mutate_time": policy.creation_record.mutate_time if policy.creation_record else None,
                        "mutated_by": policy.creation_record.mutated_by if policy.creation_record else None
                    }
                }

                # Process conditions
                for condition in policy.conditions:
                    condition_info = {
                        "display_name": condition.display_name,
                        "filter": condition.condition_threshold.filter if hasattr(condition, 'condition_threshold') else "",
                        "comparison": condition.condition_threshold.comparison.name if hasattr(condition, 'condition_threshold') else "",
                        "threshold_value": condition.condition_threshold.threshold_value if hasattr(condition, 'condition_threshold') else None
                    }
                    alert_info["conditions"].append(condition_info)

                alerts.append(alert_info)

            return alerts

        except DefaultCredentialsError as e:
            raise ConnectionError(f"GCP authentication error: {e}")
        except GoogleAPIError as e:
            raise ConnectionError(f"GCP Cloud Monitoring API error: {e}")
        except Exception as e:
            raise ConnectionError(f"Failed to get alert policies: {e}")

    async def get_alarm_history(
        self,
        alarm_name: str,
        start_time: datetime,
        end_time: datetime,
        history_item_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get GCP alert policy incident history.
        
        Note: This requires additional API calls to get incident data.
        
        Args:
            alarm_name: Name of the alert policy
            start_time: Start of time range
            end_time: End of time range
            history_item_type: Type of history items (not used in GCP)
            
        Returns:
            List of history items (placeholder implementation)
            
        Raises:
            ConnectionError: If unable to connect to Cloud Monitoring
        """
        # Note: Getting incident history requires additional API integration
        # For now, return empty list
        return []

    async def list_metrics(
        self,
        namespace: Optional[str] = None,
        metric_name: Optional[str] = None,
        dimensions: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        """List available GCP Cloud Monitoring metrics.
        
        Args:
            namespace: Resource type filter (e.g., gce_instance)
            metric_name: Filter by metric name pattern
            dimensions: Not used in GCP listing
            
        Returns:
            List of available metrics, each containing:
            - metric_name: Name of the metric
            - namespace: Resource type
            - dimensions: Empty list (GCP handles labels differently)
            
        Raises:
            ConnectionError: If unable to connect to Cloud Monitoring
        """
        try:
            request = monitoring_v3.ListMetricDescriptorsRequest({
                "name": self.project_name
            })

            descriptors = await asyncio.to_thread(
                self.client.list_metric_descriptors,
                request=request
            )

            metrics = []
            for descriptor in descriptors:
                # Filter by namespace (resource type)
                if namespace and namespace not in descriptor.type:
                    continue
                
                # Filter by metric name pattern
                if metric_name and metric_name not in descriptor.type:
                    continue

                metric_info = {
                    "metric_name": descriptor.type,
                    "namespace": descriptor.type.split('/')[0] if '/' in descriptor.type else "unknown",
                    "dimensions": [],  # GCP uses labels differently
                    "unit": descriptor.unit,
                    "description": descriptor.description,
                    "display_name": descriptor.display_name,
                    "metric_kind": descriptor.metric_kind.name,
                    "value_type": descriptor.value_type.name
                }
                metrics.append(metric_info)

            return metrics

        except DefaultCredentialsError as e:
            raise ConnectionError(f"GCP authentication error: {e}")
        except GoogleAPIError as e:
            raise ConnectionError(f"GCP Cloud Monitoring API error: {e}")
        except Exception as e:
            raise ConnectionError(f"Failed to list metrics: {e}")

    async def get_compute_instance_metrics(
        self,
        instance_name: str,
        zone: str,
        hours: int = 1
    ) -> Dict[str, Any]:
        """Get metrics for a specific Compute Engine instance.
        
        Args:
            instance_name: Name of the instance
            zone: Zone where the instance is located
            hours: Number of hours to look back
            
        Returns:
            Dictionary containing instance metrics data
            
        Raises:
            ConnectionError: If unable to connect to Cloud Monitoring
        """
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours)
            
            dimensions = {
                "instance_name": instance_name,
                "zone": zone
            }

            # Common Compute Engine metrics
            metrics_to_fetch = [
                "compute.googleapis.com/instance/cpu/utilization",
                "compute.googleapis.com/instance/disk/read_bytes_count",
                "compute.googleapis.com/instance/disk/write_bytes_count",
                "compute.googleapis.com/instance/network/received_bytes_count",
                "compute.googleapis.com/instance/network/sent_bytes_count"
            ]

            instance_data = {
                "instance_name": instance_name,
                "zone": zone,
                "time_range": {
                    "start": start_time,
                    "end": end_time,
                    "hours": hours
                },
                "metrics": {}
            }

            for metric_name in metrics_to_fetch:
                try:
                    data = await self.get_metric_data(
                        metric_name=metric_name,
                        namespace="gce_instance",
                        start_time=start_time,
                        end_time=end_time,
                        dimensions=dimensions,
                        statistic="Average",
                        period=300
                    )
                    if data:
                        instance_data["metrics"][metric_name] = data[-5:]  # Last 5 data points
                except Exception:
                    # Skip metrics that can't be retrieved
                    continue

            return instance_data

        except Exception as e:
            raise ConnectionError(f"Failed to get instance metrics: {e}")

    async def get_gke_cluster_metrics(
        self,
        cluster_name: str,
        location: str,
        hours: int = 1
    ) -> Dict[str, Any]:
        """Get metrics for a specific GKE cluster.
        
        Args:
            cluster_name: Name of the cluster
            location: Location (zone or region) where the cluster is located
            hours: Number of hours to look back
            
        Returns:
            Dictionary containing cluster metrics data
            
        Raises:
            ConnectionError: If unable to connect to Cloud Monitoring
        """
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours)
            
            dimensions = {
                "cluster_name": cluster_name,
                "location": location
            }

            # Common GKE metrics
            metrics_to_fetch = [
                "kubernetes.io/container/cpu/core_usage_time",
                "kubernetes.io/container/memory/used_bytes",
                "kubernetes.io/node/cpu/core_usage_time",
                "kubernetes.io/node/memory/used_bytes"
            ]

            cluster_data = {
                "cluster_name": cluster_name,
                "location": location,
                "time_range": {
                    "start": start_time,
                    "end": end_time,
                    "hours": hours
                },
                "metrics": {}
            }

            for metric_name in metrics_to_fetch:
                try:
                    data = await self.get_metric_data(
                        metric_name=metric_name,
                        namespace="k8s_container" if "container" in metric_name else "k8s_node",
                        start_time=start_time,
                        end_time=end_time,
                        dimensions=dimensions,
                        statistic="Average",
                        period=300
                    )
                    if data:
                        cluster_data["metrics"][metric_name] = data[-5:]  # Last 5 data points
                except Exception:
                    # Skip metrics that can't be retrieved
                    continue

            return cluster_data

        except Exception as e:
            raise ConnectionError(f"Failed to get cluster metrics: {e}")

    async def get_cloud_function_metrics(
        self,
        function_name: str,
        region: str,
        hours: int = 1
    ) -> Dict[str, Any]:
        """Get metrics for a specific Cloud Function.
        
        Args:
            function_name: Name of the function
            region: Region where the function is deployed
            hours: Number of hours to look back
            
        Returns:
            Dictionary containing function metrics data
            
        Raises:
            ConnectionError: If unable to connect to Cloud Monitoring
        """
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours)
            
            dimensions = {
                "function_name": function_name,
                "region": region
            }

            # Common Cloud Functions metrics
            metrics_to_fetch = [
                "cloudfunctions.googleapis.com/function/execution_count",
                "cloudfunctions.googleapis.com/function/execution_times",
                "cloudfunctions.googleapis.com/function/user_memory_bytes",
                "cloudfunctions.googleapis.com/function/network_egress"
            ]

            function_data = {
                "function_name": function_name,
                "region": region,
                "time_range": {
                    "start": start_time,
                    "end": end_time,
                    "hours": hours
                },
                "metrics": {}
            }

            for metric_name in metrics_to_fetch:
                try:
                    data = await self.get_metric_data(
                        metric_name=metric_name,
                        namespace="cloud_function",
                        start_time=start_time,
                        end_time=end_time,
                        dimensions=dimensions,
                        statistic="Average" if "times" in metric_name or "bytes" in metric_name else "Sum",
                        period=300
                    )
                    if data:
                        function_data["metrics"][metric_name] = data[-5:]  # Last 5 data points
                except Exception:
                    # Skip metrics that can't be retrieved
                    continue

            return function_data

        except Exception as e:
            raise ConnectionError(f"Failed to get function metrics: {e}")
