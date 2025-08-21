"""Unit tests for AWS integrations."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from oncall_agent.integrations.aws.cloudwatch_logs import CloudWatchLogsProvider
from oncall_agent.integrations.aws.cloudwatch_metrics import CloudWatchMetricsProvider
from oncall_agent.integrations.aws.client import AWSClientManager, create_aws_config


class TestCloudWatchLogsProvider:
    """Test cases for CloudWatchLogsProvider."""

    @patch('oncall_agent.integrations.aws.cloudwatch_logs.boto3.client')
    def test_init_success(self, mock_boto_client, mock_aws_config):
        """Test CloudWatch Logs provider initialization."""
        mock_client = Mock()
        mock_boto_client.return_value = mock_client
        
        provider = CloudWatchLogsProvider(mock_aws_config)
        
        assert provider.config == mock_aws_config
        assert provider.client == mock_client
        mock_boto_client.assert_called_once_with("logs", region_name="us-west-2")

    def test_validate_config_missing_region(self):
        """Test configuration validation with missing region."""
        config = {"access_key_id": "test", "secret_access_key": "test"}
        
        with pytest.raises(ValueError) as exc_info:
            CloudWatchLogsProvider(config)
        
        assert "AWS region is required" in str(exc_info.value)

    @pytest.mark.asyncio
    @patch('oncall_agent.integrations.aws.cloudwatch_logs.boto3.client')
    @patch('oncall_agent.integrations.aws.cloudwatch_logs.asyncio.to_thread')
    async def test_fetch_logs_success(self, mock_to_thread, mock_boto_client, mock_aws_config):
        """Test successful log fetching."""
        # Mock boto3 client
        mock_client = Mock()
        mock_boto_client.return_value = mock_client
        
        # Mock CloudWatch responses
        start_query_response = {"queryId": "test-query-id"}
        get_query_results_response = {
            "status": "Complete",
            "results": [
                [
                    {"field": "@timestamp", "value": "2024-01-01T10:00:00Z"},
                    {"field": "@message", "value": "Test log message"},
                    {"field": "@logStream", "value": "test-stream"}
                ]
            ]
        }
        
        mock_to_thread.side_effect = [start_query_response, get_query_results_response]
        
        provider = CloudWatchLogsProvider(mock_aws_config)
        
        start_time = datetime.now() - timedelta(hours=1)
        end_time = datetime.now()
        
        logs = await provider.fetch_logs(
            query="fields @timestamp, @message",
            start_time=start_time,
            end_time=end_time,
            limit=100
        )
        
        assert len(logs) == 1
        assert logs[0]["message"] == "Test log message"
        assert logs[0]["source"] == "test-stream"

    @pytest.mark.asyncio
    @patch('oncall_agent.integrations.aws.cloudwatch_logs.boto3.client')
    async def test_get_log_groups(self, mock_boto_client, mock_aws_config):
        """Test getting log groups."""
        mock_client = Mock()
        mock_boto_client.return_value = mock_client
        
        provider = CloudWatchLogsProvider(mock_aws_config)
        
        # Mock the describe_log_groups response
        with patch('oncall_agent.integrations.aws.cloudwatch_logs.asyncio.to_thread') as mock_to_thread:
            mock_to_thread.return_value = {
                "logGroups": [
                    {"logGroupName": "log-group-1"},
                    {"logGroupName": "log-group-2"}
                ]
            }
            
            log_groups = await provider.get_log_groups()
            
            assert len(log_groups) == 2
            assert "log-group-1" in log_groups
            assert "log-group-2" in log_groups

    def test_extract_log_level(self, mock_aws_config):
        """Test log level extraction from messages."""
        with patch('oncall_agent.integrations.aws.cloudwatch_logs.boto3.client'):
            provider = CloudWatchLogsProvider(mock_aws_config)
            
            assert provider._extract_log_level("ERROR: Something failed") == "ERROR"
            assert provider._extract_log_level("WARN: Warning message") == "WARN"
            assert provider._extract_log_level("INFO: Information") == "INFO"
            assert provider._extract_log_level("DEBUG: Debug info") == "DEBUG"
            assert provider._extract_log_level("Regular message") == "INFO"


class TestCloudWatchMetricsProvider:
    """Test cases for CloudWatchMetricsProvider."""

    @patch('oncall_agent.integrations.aws.cloudwatch_metrics.boto3.client')
    def test_init_success(self, mock_boto_client, mock_aws_config):
        """Test CloudWatch Metrics provider initialization."""
        mock_client = Mock()
        mock_boto_client.return_value = mock_client
        
        provider = CloudWatchMetricsProvider(mock_aws_config)
        
        assert provider.config == mock_aws_config
        assert provider.client == mock_client
        mock_boto_client.assert_called_once_with("cloudwatch", region_name="us-west-2")

    @pytest.mark.asyncio
    @patch('oncall_agent.integrations.aws.cloudwatch_metrics.boto3.client')
    @patch('oncall_agent.integrations.aws.cloudwatch_metrics.asyncio.to_thread')
    async def test_get_metric_data_success(self, mock_to_thread, mock_boto_client, mock_aws_config):
        """Test successful metric data retrieval."""
        mock_client = Mock()
        mock_boto_client.return_value = mock_client
        
        # Mock CloudWatch response
        mock_response = {
            "Datapoints": [
                {
                    "Timestamp": datetime.now(),
                    "Average": 75.5,
                    "Unit": "Percent"
                }
            ]
        }
        mock_to_thread.return_value = mock_response
        
        provider = CloudWatchMetricsProvider(mock_aws_config)
        
        start_time = datetime.now() - timedelta(hours=1)
        end_time = datetime.now()
        
        data_points = await provider.get_metric_data(
            metric_name="CPUUtilization",
            namespace="AWS/EC2",
            start_time=start_time,
            end_time=end_time
        )
        
        assert len(data_points) == 1
        assert data_points[0]["value"] == 75.5
        assert data_points[0]["unit"] == "Percent"
        assert data_points[0]["statistic"] == "Average"

    @pytest.mark.asyncio
    @patch('oncall_agent.integrations.aws.cloudwatch_metrics.boto3.client')
    @patch('oncall_agent.integrations.aws.cloudwatch_metrics.asyncio.to_thread')
    async def test_get_alarms(self, mock_to_thread, mock_boto_client, mock_aws_config):
        """Test getting alarms."""
        mock_client = Mock()
        mock_boto_client.return_value = mock_client
        
        # Mock CloudWatch response
        mock_response = {
            "MetricAlarms": [
                {
                    "AlarmName": "test-alarm",
                    "StateValue": "ALARM",
                    "StateReason": "Threshold exceeded",
                    "MetricName": "CPUUtilization",
                    "Namespace": "AWS/EC2",
                    "Threshold": 80.0,
                    "ComparisonOperator": "GreaterThanThreshold",
                    "Dimensions": [{"Name": "InstanceId", "Value": "i-123456"}]
                }
            ]
        }
        mock_to_thread.return_value = mock_response
        
        provider = CloudWatchMetricsProvider(mock_aws_config)
        alarms = await provider.get_alarms()
        
        assert len(alarms) == 1
        assert alarms[0]["name"] == "test-alarm"
        assert alarms[0]["state"] == "ALARM"
        assert alarms[0]["threshold"] == 80.0


class TestAWSClientManager:
    """Test cases for AWSClientManager."""

    def test_init_success(self, mock_aws_config):
        """Test AWS client manager initialization."""
        manager = AWSClientManager(mock_aws_config)
        
        assert manager.config == mock_aws_config

    def test_validate_config_missing_region(self):
        """Test configuration validation with missing region."""
        config = {"access_key_id": "test", "secret_access_key": "test"}
        
        with pytest.raises(ValueError) as exc_info:
            AWSClientManager(config)
        
        assert "AWS region is required" in str(exc_info.value)

    @patch('oncall_agent.integrations.aws.client.boto3.client')
    def test_create_client_success(self, mock_boto_client, mock_aws_config):
        """Test successful client creation."""
        mock_client = Mock()
        mock_boto_client.return_value = mock_client
        
        manager = AWSClientManager(mock_aws_config)
        client = manager.create_client("logs")
        
        assert client == mock_client
        mock_boto_client.assert_called_once_with(
            "logs",
            region_name="us-west-2",
            aws_access_key_id="test_access_key",
            aws_secret_access_key="test_secret_key"
        )

    @patch('oncall_agent.integrations.aws.client.boto3.Session')
    def test_create_client_with_profile(self, mock_session_class):
        """Test client creation with AWS profile."""
        config = {"region": "us-west-2", "profile_name": "test-profile"}
        
        mock_session = Mock()
        mock_client = Mock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session
        
        manager = AWSClientManager(config)
        client = manager.create_client("logs")
        
        assert client == mock_client
        mock_session_class.assert_called_once_with(
            profile_name="test-profile",
            region_name="us-west-2"
        )

    def test_create_aws_config(self):
        """Test AWS configuration creation utility."""
        config = create_aws_config(
            region="us-east-1",
            access_key_id="test_key",
            secret_access_key="test_secret"
        )
        
        assert config["region"] == "us-east-1"
        assert config["access_key_id"] == "test_key"
        assert config["secret_access_key"] == "test_secret"

    def test_create_aws_config_minimal(self):
        """Test AWS configuration creation with minimal parameters."""
        config = create_aws_config(region="us-east-1")
        
        assert config["region"] == "us-east-1"
        assert "access_key_id" not in config
        assert "secret_access_key" not in config
