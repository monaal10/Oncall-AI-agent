"""AWS CloudWatch Metrics integration."""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from ..base.metrics_provider import MetricsProvider


class CloudWatchMetricsProvider(MetricsProvider):
    """AWS CloudWatch Metrics provider implementation.
    
    Provides integration with AWS CloudWatch Metrics service for fetching
    metric data and alarm information.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize CloudWatch Metrics provider.
        
        Args:
            config: Configuration dictionary containing:
                - region: AWS region (required)
                - access_key_id: AWS access key (optional, uses default credentials)
                - secret_access_key: AWS secret key (optional, uses default credentials)
                - session_token: AWS session token (optional)
                - namespaces: List of metric namespaces to monitor (optional)
        """
        super().__init__(config)
        self._setup_client()

    def _validate_config(self) -> None:
        """Validate AWS CloudWatch Metrics configuration.
        
        Raises:
            ValueError: If required configuration is missing
        """
        if "region" not in self.config:
            raise ValueError("AWS region is required for CloudWatch Metrics provider")

    def _setup_client(self) -> None:
        """Set up AWS CloudWatch client.
        
        Raises:
            NoCredentialsError: If AWS credentials are not available
        """
        session_kwargs = {"region_name": self.config["region"]}
        
        if "access_key_id" in self.config:
            session_kwargs["aws_access_key_id"] = self.config["access_key_id"]
        if "secret_access_key" in self.config:
            session_kwargs["aws_secret_access_key"] = self.config["secret_access_key"]
        if "session_token" in self.config:
            session_kwargs["aws_session_token"] = self.config["session_token"]

        self.client = boto3.client("cloudwatch", **session_kwargs)

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
            namespace: Metric namespace (e.g., AWS/EC2, AWS/RDS)
            start_time: Start of time range
            end_time: End of time range
            dimensions: Metric dimensions as key-value pairs
            statistic: Statistic type (Average, Sum, Maximum, Minimum, SampleCount)
            period: Period in seconds for data point aggregation
            
        Returns:
            List of metric data points, each containing:
            - timestamp: Data point timestamp as datetime
            - value: Metric value as float
            - unit: Unit of measurement
            - statistic: Applied statistic
            
        Raises:
            ConnectionError: If unable to connect to CloudWatch
            ValueError: If metric parameters are invalid
        """
        try:
            # Build dimensions list for AWS API
            aws_dimensions = []
            if dimensions:
                for key, value in dimensions.items():
                    aws_dimensions.append({"Name": key, "Value": value})

            response = await asyncio.to_thread(
                self.client.get_metric_statistics,
                Namespace=namespace,
                MetricName=metric_name,
                Dimensions=aws_dimensions,
                StartTime=start_time,
                EndTime=end_time,
                Period=period,
                Statistics=[statistic]
            )

            # Process data points
            data_points = []
            for point in response.get("Datapoints", []):
                data_point = {
                    "timestamp": point["Timestamp"],
                    "value": point[statistic],
                    "unit": point.get("Unit", "None"),
                    "statistic": statistic
                }
                data_points.append(data_point)

            # Sort by timestamp
            data_points.sort(key=lambda x: x["timestamp"])
            return data_points

        except ClientError as e:
            raise ConnectionError(f"AWS CloudWatch error: {e}")
        except Exception as e:
            raise ConnectionError(f"Failed to get metric data: {e}")

    async def get_alarms(
        self,
        alarm_names: Optional[List[str]] = None,
        state_value: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get CloudWatch alarm information.
        
        Args:
            alarm_names: Specific alarm names to retrieve (None for all)
            state_value: Filter by alarm state (OK, ALARM, INSUFFICIENT_DATA)
            
        Returns:
            List of alarms, each containing:
            - name: Alarm name
            - state: Current alarm state
            - reason: State change reason
            - timestamp: Last state change time
            - metric_name: Associated metric name
            - namespace: Metric namespace
            - threshold: Alarm threshold value
            - comparison_operator: Comparison operator used
            - dimensions: Metric dimensions
            
        Raises:
            ConnectionError: If unable to connect to CloudWatch
        """
        try:
            kwargs = {}
            if alarm_names:
                kwargs["AlarmNames"] = alarm_names
            if state_value:
                kwargs["StateValue"] = state_value

            response = await asyncio.to_thread(
                self.client.describe_alarms,
                **kwargs
            )

            alarms = []
            for alarm in response.get("MetricAlarms", []):
                alarm_info = {
                    "name": alarm["AlarmName"],
                    "state": alarm["StateValue"],
                    "reason": alarm.get("StateReason", ""),
                    "timestamp": alarm.get("StateUpdatedTimestamp"),
                    "metric_name": alarm.get("MetricName"),
                    "namespace": alarm.get("Namespace"),
                    "threshold": alarm.get("Threshold"),
                    "comparison_operator": alarm.get("ComparisonOperator"),
                    "dimensions": {
                        dim["Name"]: dim["Value"] 
                        for dim in alarm.get("Dimensions", [])
                    },
                    "statistic": alarm.get("Statistic"),
                    "period": alarm.get("Period"),
                    "evaluation_periods": alarm.get("EvaluationPeriods"),
                    "description": alarm.get("AlarmDescription", "")
                }
                alarms.append(alarm_info)

            return alarms

        except ClientError as e:
            raise ConnectionError(f"AWS CloudWatch error: {e}")
        except Exception as e:
            raise ConnectionError(f"Failed to get alarms: {e}")

    async def get_alarm_history(
        self,
        alarm_name: str,
        start_time: datetime,
        end_time: datetime,
        history_item_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get CloudWatch alarm state change history.
        
        Args:
            alarm_name: Name of the alarm
            start_time: Start of time range
            end_time: End of time range
            history_item_type: Type of history items (ConfigurationUpdate, StateUpdate, Action)
            
        Returns:
            List of history items, each containing:
            - timestamp: When the change occurred
            - summary: Summary of the change
            - history_item_type: Type of change
            - history_data: Detailed change data
            
        Raises:
            ConnectionError: If unable to connect to CloudWatch
            ValueError: If alarm name is invalid
        """
        try:
            kwargs = {
                "AlarmName": alarm_name,
                "StartDate": start_time,
                "EndDate": end_time
            }
            if history_item_type:
                kwargs["HistoryItemType"] = history_item_type

            response = await asyncio.to_thread(
                self.client.describe_alarm_history,
                **kwargs
            )

            history_items = []
            for item in response.get("AlarmHistoryItems", []):
                history_item = {
                    "timestamp": item["Timestamp"],
                    "summary": item["HistorySummary"],
                    "history_item_type": item["HistoryItemType"],
                    "history_data": item.get("HistoryData", "")
                }
                history_items.append(history_item)

            # Sort by timestamp (most recent first)
            history_items.sort(key=lambda x: x["timestamp"], reverse=True)
            return history_items

        except ClientError as e:
            raise ConnectionError(f"AWS CloudWatch error: {e}")
        except Exception as e:
            raise ConnectionError(f"Failed to get alarm history: {e}")

    async def list_metrics(
        self,
        namespace: Optional[str] = None,
        metric_name: Optional[str] = None,
        dimensions: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        """List available CloudWatch metrics.
        
        Args:
            namespace: Filter by namespace
            metric_name: Filter by metric name
            dimensions: Filter by dimensions
            
        Returns:
            List of available metrics, each containing:
            - metric_name: Name of the metric
            - namespace: Metric namespace
            - dimensions: List of dimension key-value pairs
            
        Raises:
            ConnectionError: If unable to connect to CloudWatch
        """
        try:
            kwargs = {}
            if namespace:
                kwargs["Namespace"] = namespace
            if metric_name:
                kwargs["MetricName"] = metric_name
            if dimensions:
                aws_dimensions = []
                for key, value in dimensions.items():
                    aws_dimensions.append({"Name": key, "Value": value})
                kwargs["Dimensions"] = aws_dimensions

            metrics = []
            next_token = None

            while True:
                if next_token:
                    kwargs["NextToken"] = next_token

                response = await asyncio.to_thread(
                    self.client.list_metrics,
                    **kwargs
                )

                for metric in response.get("Metrics", []):
                    metric_info = {
                        "metric_name": metric["MetricName"],
                        "namespace": metric["Namespace"],
                        "dimensions": {
                            dim["Name"]: dim["Value"] 
                            for dim in metric.get("Dimensions", [])
                        }
                    }
                    metrics.append(metric_info)

                next_token = response.get("NextToken")
                if not next_token:
                    break

            return metrics

        except ClientError as e:
            raise ConnectionError(f"AWS CloudWatch error: {e}")
        except Exception as e:
            raise ConnectionError(f"Failed to list metrics: {e}")

    async def get_alarming_metrics(self) -> List[Dict[str, Any]]:
        """Get metrics that currently have alarms in ALARM state.
        
        Returns:
            List of metrics with active alarms, each containing metric info and alarm details
            
        Raises:
            ConnectionError: If unable to connect to CloudWatch
        """
        try:
            # Get all alarms in ALARM state
            alarming_alarms = await self.get_alarms(state_value="ALARM")
            
            alarming_metrics = []
            for alarm in alarming_alarms:
                if alarm["metric_name"] and alarm["namespace"]:
                    metric_info = {
                        "metric_name": alarm["metric_name"],
                        "namespace": alarm["namespace"],
                        "dimensions": alarm["dimensions"],
                        "alarm_name": alarm["name"],
                        "alarm_reason": alarm["reason"],
                        "threshold": alarm["threshold"],
                        "comparison_operator": alarm["comparison_operator"],
                        "current_state": alarm["state"],
                        "state_timestamp": alarm["timestamp"]
                    }
                    alarming_metrics.append(metric_info)
            
            return alarming_metrics

        except Exception as e:
            raise ConnectionError(f"Failed to get alarming metrics: {e}")

    async def get_metric_insights(
        self,
        namespace: str,
        hours: int = 24
    ) -> Dict[str, Any]:
        """Get insights about metrics in a namespace over time period.
        
        Args:
            namespace: Metric namespace to analyze
            hours: Number of hours to look back
            
        Returns:
            Dictionary containing:
            - total_metrics: Total number of metrics
            - active_alarms: Number of alarms in ALARM state
            - top_metrics: Most frequently updated metrics
            - namespace: Analyzed namespace
            - time_range: Analysis time range
            
        Raises:
            ConnectionError: If unable to connect to CloudWatch
        """
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours)
            
            # Get all metrics in namespace
            metrics = await self.list_metrics(namespace=namespace)
            
            # Get alarms for this namespace
            all_alarms = await self.get_alarms()
            namespace_alarms = [
                alarm for alarm in all_alarms 
                if alarm["namespace"] == namespace
            ]
            
            active_alarms = [
                alarm for alarm in namespace_alarms 
                if alarm["state"] == "ALARM"
            ]
            
            insights = {
                "namespace": namespace,
                "time_range": {
                    "start": start_time,
                    "end": end_time,
                    "hours": hours
                },
                "total_metrics": len(metrics),
                "total_alarms": len(namespace_alarms),
                "active_alarms": len(active_alarms),
                "alarm_details": active_alarms,
                "sample_metrics": metrics[:10]  # First 10 metrics as sample
            }
            
            return insights

        except Exception as e:
            raise ConnectionError(f"Failed to get metric insights: {e}")
