"""Abstract base class for metrics providers."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any, Union


class MetricsProvider(ABC):
    """Abstract base class for all metrics providers.
    
    This class defines the interface that all metrics providers must implement
    to integrate with the OnCall AI Agent system.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the metrics provider with configuration.
        
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
            namespace: Metric namespace (e.g., AWS/EC2, custom namespace)
            start_time: Start of time range
            end_time: End of time range
            dimensions: Metric dimensions as key-value pairs
            statistic: Statistic type (Average, Sum, Maximum, Minimum, SampleCount)
            period: Period in seconds for data point aggregation
            
        Returns:
            List of metric data points, each containing:
            - timestamp: Data point timestamp
            - value: Metric value
            - unit: Unit of measurement
            - statistic: Applied statistic
            
        Raises:
            ConnectionError: If unable to connect to metrics service
            ValueError: If metric parameters are invalid
        """
        pass

    @abstractmethod
    async def get_alarms(
        self,
        alarm_names: Optional[List[str]] = None,
        state_value: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get alarm information.
        
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
            - threshold: Alarm threshold value
            - comparison_operator: Comparison operator used
            
        Raises:
            ConnectionError: If unable to connect to metrics service
        """
        pass

    @abstractmethod
    async def get_alarm_history(
        self,
        alarm_name: str,
        start_time: datetime,
        end_time: datetime,
        history_item_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get alarm state change history.
        
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
            ConnectionError: If unable to connect to metrics service
            ValueError: If alarm name is invalid
        """
        pass

    @abstractmethod
    async def list_metrics(
        self,
        namespace: Optional[str] = None,
        metric_name: Optional[str] = None,
        dimensions: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        """List available metrics.
        
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
            ConnectionError: If unable to connect to metrics service
        """
        pass

    async def health_check(self) -> bool:
        """Check if the metrics provider is accessible.
        
        Returns:
            True if provider is healthy, False otherwise
        """
        try:
            await self.list_metrics()
            return True
        except Exception:
            return False
