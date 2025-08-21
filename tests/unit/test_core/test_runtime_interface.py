"""Unit tests for runtime interface."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock
from oncall_agent.core.runtime_interface import RuntimeInterface


class TestRuntimeInterface:
    """Test cases for RuntimeInterface."""

    @pytest.fixture
    def runtime_interface(self, mock_logs_provider, mock_code_provider, mock_llm_provider):
        """Create RuntimeInterface instance with required providers."""
        return RuntimeInterface(
            logs_provider=mock_logs_provider,
            code_provider=mock_code_provider,
            llm_provider=mock_llm_provider
        )

    @pytest.fixture
    def full_runtime_interface(
        self,
        mock_logs_provider,
        mock_code_provider,
        mock_llm_provider,
        mock_metrics_provider,
        mock_runbook_provider
    ):
        """Create RuntimeInterface instance with all providers."""
        return RuntimeInterface(
            logs_provider=mock_logs_provider,
            code_provider=mock_code_provider,
            llm_provider=mock_llm_provider,
            metrics_provider=mock_metrics_provider,
            runbook_provider=mock_runbook_provider
        )

    @pytest.mark.asyncio
    async def test_get_logs_function(self, runtime_interface, mock_logs_provider, sample_log_entries, time_range):
        """Test the unified get_logs function."""
        mock_logs_provider.fetch_logs.return_value = sample_log_entries
        
        logs = await runtime_interface.get_logs(
            query="ERROR",
            time_range=time_range,
            service_name="auth",
            limit=10
        )
        
        assert len(logs) == len(sample_log_entries)
        assert all("timestamp" in log for log in logs)
        assert all("message" in log for log in logs)
        assert all("level" in log for log in logs)
        mock_logs_provider.fetch_logs.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_logs_function_with_error(self, runtime_interface, mock_logs_provider, time_range):
        """Test get_logs function when provider fails."""
        mock_logs_provider.fetch_logs.side_effect = Exception("Connection failed")
        
        logs = await runtime_interface.get_logs(
            query="ERROR",
            time_range=time_range
        )
        
        assert len(logs) == 1
        assert logs[0]["level"] == "ERROR"
        assert "Error fetching logs" in logs[0]["message"]
        assert logs[0]["metadata"]["error"] is True

    @pytest.mark.asyncio
    async def test_get_metrics_function_available(self, full_runtime_interface, mock_metrics_provider, time_range):
        """Test get_metrics function when metrics provider is available."""
        mock_metrics_provider.get_metric_data.return_value = [
            {"timestamp": datetime.now(), "value": 95.0, "unit": "Percent", "statistic": "Average"}
        ]
        
        metrics = await full_runtime_interface.get_metrics(
            resource_info={"namespace": "test"},
            time_range=time_range,
            metric_names=["CPUUtilization"]
        )
        
        assert len(metrics) > 0
        assert all("timestamp" in metric for metric in metrics)
        assert all("metric_name" in metric for metric in metrics)
        assert all("value" in metric for metric in metrics)

    def test_get_metrics_function_not_available(self, runtime_interface):
        """Test get_metrics function when metrics provider is not available."""
        assert runtime_interface.get_metrics is None

    @pytest.mark.asyncio
    async def test_get_code_context_function(self, runtime_interface, mock_code_provider):
        """Test the unified get_code_context function."""
        mock_search_results = [
            {
                "repository": "org/repo",
                "file_path": "src/main.py",
                "matches": [{"line_number": 10, "content": "def test():", "context": {"before": [], "after": []}}],
                "url": "https://github.com/org/repo/blob/main/src/main.py"
            }
        ]
        mock_code_provider.search_code.return_value = mock_search_results
        mock_code_provider.get_recent_commits.return_value = []
        
        code_context = await runtime_interface.get_code_context(
            error_info={"message": "test error", "stack_trace": ""}
        )
        
        assert len(code_context) > 0
        assert all("repository" in item for item in code_context)
        assert all("file_path" in item for item in code_context)
        assert all("relevance" in item for item in code_context)

    @pytest.mark.asyncio
    async def test_get_llm_response_resolution(self, runtime_interface, mock_llm_provider, sample_incident_context):
        """Test get_llm_response function for resolution."""
        expected_response = {
            "resolution_summary": "Test resolution",
            "detailed_steps": "1. Step one\n2. Step two",
            "confidence_score": 0.8
        }
        mock_llm_provider.generate_resolution.return_value = expected_response
        
        response = await runtime_interface.get_llm_response(
            context=sample_incident_context,
            response_type="resolution"
        )
        
        assert response == expected_response
        mock_llm_provider.generate_resolution.assert_called_once_with(sample_incident_context)

    @pytest.mark.asyncio
    async def test_get_llm_response_log_analysis(self, runtime_interface, mock_llm_provider, sample_incident_context):
        """Test get_llm_response function for log analysis."""
        expected_response = {
            "error_patterns": ["connection timeout"],
            "severity_assessment": "High"
        }
        mock_llm_provider.analyze_logs.return_value = expected_response
        
        response = await runtime_interface.get_llm_response(
            context=sample_incident_context,
            response_type="log_analysis"
        )
        
        assert response == expected_response
        mock_llm_provider.analyze_logs.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_llm_response_invalid_type(self, runtime_interface, sample_incident_context):
        """Test get_llm_response function with invalid response type."""
        response = await runtime_interface.get_llm_response(
            context=sample_incident_context,
            response_type="invalid_type"
        )
        
        assert response["confidence_score"] == 0.0
        assert response["error"] is True
        assert "Unsupported response type" in response["resolution_summary"]

    @pytest.mark.asyncio
    async def test_get_runbook_guidance_available(self, full_runtime_interface, mock_runbook_provider):
        """Test get_runbook_guidance function when runbook provider is available."""
        mock_runbook_provider.find_relevant_runbooks.return_value = [
            {"id": "test.md", "title": "Test Runbook"}
        ]
        mock_runbook_provider.get_runbook_text.return_value = "Test runbook content"
        
        guidance = await full_runtime_interface.get_runbook_guidance(
            error_context="database error"
        )
        
        assert "Test runbook content" in guidance
        mock_runbook_provider.find_relevant_runbooks.assert_called_once()

    def test_get_runbook_guidance_not_available(self, runtime_interface):
        """Test get_runbook_guidance function when runbook provider is not available."""
        assert runtime_interface.get_runbook_guidance is None

    def test_adapt_logs_query_aws(self, runtime_interface):
        """Test logs query adaptation for AWS CloudWatch."""
        # Mock AWS provider
        runtime_interface._logs_provider.__class__.__name__ = "CloudWatchLogsProvider"
        
        adapted_query = runtime_interface._adapt_logs_query(
            "ERROR",
            service_name="auth",
            log_level="ERROR"
        )
        
        assert "fields @timestamp, @message" in adapted_query
        assert "filter @message like /ERROR/" in adapted_query
        assert "filter @logStream like /auth/" in adapted_query

    def test_adapt_logs_query_azure(self, runtime_interface):
        """Test logs query adaptation for Azure Monitor."""
        # Mock Azure provider
        runtime_interface._logs_provider.__class__.__name__ = "AzureMonitorLogsProvider"
        
        adapted_query = runtime_interface._adapt_logs_query(
            "ERROR",
            service_name="auth",
            log_level="ERROR"
        )
        
        assert 'search "ERROR"' in adapted_query
        assert 'contains "auth"' in adapted_query
        assert 'Level == "ERROR"' in adapted_query

    def test_adapt_logs_query_gcp(self, runtime_interface):
        """Test logs query adaptation for GCP Cloud Logging."""
        # Mock GCP provider
        runtime_interface._logs_provider.__class__.__name__ = "GCPCloudLoggingProvider"
        
        adapted_query = runtime_interface._adapt_logs_query(
            "ERROR",
            service_name="auth",
            log_level="ERROR"
        )
        
        assert 'textPayload:"ERROR"' in adapted_query
        assert 'service_name="auth"' in adapted_query
        assert 'severity >= "ERROR"' in adapted_query

    def test_normalize_log_level(self, runtime_interface):
        """Test log level normalization."""
        assert runtime_interface._normalize_log_level("ERROR") == "ERROR"
        assert runtime_interface._normalize_log_level("FATAL") == "ERROR"
        assert runtime_interface._normalize_log_level("WARNING") == "WARN"
        assert runtime_interface._normalize_log_level("INFORMATION") == "INFO"
        assert runtime_interface._normalize_log_level("DEBUG") == "DEBUG"
        assert runtime_interface._normalize_log_level("UNKNOWN") == "INFO"

    def test_normalize_logs_response(self, runtime_interface, sample_log_entries):
        """Test logs response normalization."""
        normalized = runtime_interface._normalize_logs_response(sample_log_entries)
        
        assert len(normalized) == len(sample_log_entries)
        for log in normalized:
            assert "timestamp" in log
            assert "message" in log
            assert "level" in log
            assert "source" in log
            assert "metadata" in log
            assert "provider" in log["metadata"]

    def test_calculate_code_relevance(self, runtime_interface):
        """Test code relevance calculation."""
        search_result = {
            "matches": [{"line_number": 10}, {"line_number": 20}],
            "last_modified": datetime.now() - timedelta(days=5)
        }
        
        relevance = runtime_interface._calculate_code_relevance(search_result)
        
        assert 0 <= relevance <= 1
        assert relevance > 0  # Should have some relevance due to matches

    def test_get_provider_info(self, full_runtime_interface):
        """Test getting provider information."""
        info = full_runtime_interface.get_provider_info()
        
        assert "logs_provider" in info
        assert "code_provider" in info
        assert "llm_provider" in info
        assert "metrics_provider" in info
        assert "runbook_provider" in info
        
        assert info["logs_provider"]["available"] is True
        assert info["metrics_provider"]["available"] is True
        assert info["runbook_provider"]["available"] is True

    def test_get_runtime_functions(self, full_runtime_interface):
        """Test getting runtime functions."""
        functions = full_runtime_interface.get_runtime_functions()
        
        assert "get_logs" in functions
        assert "get_code_context" in functions
        assert "get_llm_response" in functions
        assert "get_metrics" in functions
        assert "get_runbook_guidance" in functions
        
        # Verify functions are callable
        assert callable(functions["get_logs"])
        assert callable(functions["get_code_context"])
        assert callable(functions["get_llm_response"])

    @pytest.mark.asyncio
    async def test_health_check_all_providers(self, full_runtime_interface):
        """Test health check for all providers."""
        health_results = await full_runtime_interface.health_check_all_providers()
        
        assert "logs" in health_results
        assert "code" in health_results
        assert "llm" in health_results
        assert "metrics" in health_results
        assert "runbooks" in health_results
