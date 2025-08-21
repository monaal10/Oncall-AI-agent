"""Pytest configuration and fixtures for OnCall AI Agent tests."""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock, MagicMock


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_log_entries():
    """Sample log entries for testing."""
    return [
        {
            "timestamp": datetime.now() - timedelta(minutes=5),
            "message": "Database connection timeout",
            "level": "ERROR",
            "source": "auth-service",
            "metadata": {"request_id": "req-123"}
        },
        {
            "timestamp": datetime.now() - timedelta(minutes=3),
            "message": "Connection pool exhausted",
            "level": "WARN",
            "source": "auth-service",
            "metadata": {"pool_size": 10}
        },
        {
            "timestamp": datetime.now() - timedelta(minutes=1),
            "message": "Request processed successfully",
            "level": "INFO",
            "source": "auth-service",
            "metadata": {"response_time": 150}
        }
    ]


@pytest.fixture
def sample_metric_data():
    """Sample metric data for testing."""
    return [
        {
            "timestamp": datetime.now() - timedelta(minutes=5),
            "value": 95.5,
            "unit": "Percent",
            "statistic": "Average"
        },
        {
            "timestamp": datetime.now() - timedelta(minutes=4),
            "value": 87.2,
            "unit": "Percent", 
            "statistic": "Average"
        },
        {
            "timestamp": datetime.now() - timedelta(minutes=3),
            "value": 92.1,
            "unit": "Percent",
            "statistic": "Average"
        }
    ]


@pytest.fixture
def sample_code_snippets():
    """Sample code snippets for testing."""
    return [
        {
            "repository": "myorg/backend",
            "file_path": "auth/database.py",
            "content": "def connect():\n    return psycopg2.connect(DATABASE_URL, timeout=30)",
            "language": "python",
            "relevance": 0.9
        },
        {
            "repository": "myorg/backend",
            "file_path": "auth/models.py",
            "content": "class User:\n    def authenticate(self, password):\n        # auth logic",
            "language": "python",
            "relevance": 0.7
        }
    ]


@pytest.fixture
def sample_incident_context(sample_log_entries, sample_metric_data, sample_code_snippets):
    """Sample incident context for testing."""
    return {
        "incident_description": "Database connection errors in authentication service",
        "log_data": sample_log_entries,
        "metric_data": sample_metric_data,
        "code_context": sample_code_snippets,
        "runbook_guidance": "Database Issues: Check connection pool settings and server status.",
        "additional_context": "Issue started after recent deployment"
    }


@pytest.fixture
def mock_aws_config():
    """Mock AWS configuration for testing."""
    return {
        "region": "us-west-2",
        "access_key_id": "test_access_key",
        "secret_access_key": "test_secret_key"
    }


@pytest.fixture
def mock_azure_config():
    """Mock Azure configuration for testing."""
    return {
        "subscription_id": "test-subscription-id",
        "workspace_id": "test-workspace-id",
        "tenant_id": "test-tenant-id",
        "client_id": "test-client-id",
        "client_secret": "test-client-secret"
    }


@pytest.fixture
def mock_gcp_config():
    """Mock GCP configuration for testing."""
    return {
        "project_id": "test-project-id",
        "credentials_path": "/path/to/test-credentials.json"
    }


@pytest.fixture
def mock_github_config():
    """Mock GitHub configuration for testing."""
    return {
        "token": "test_github_token",
        "repositories": ["testorg/repo1", "testorg/repo2"],
        "base_url": "https://api.github.com"
    }


@pytest.fixture
def mock_openai_config():
    """Mock OpenAI configuration for testing."""
    return {
        "api_key": "test_openai_key",
        "model": "gpt-4",
        "max_tokens": 2000,
        "temperature": 0.1
    }


@pytest.fixture
def mock_anthropic_config():
    """Mock Anthropic configuration for testing."""
    return {
        "api_key": "test_anthropic_key",
        "model": "claude-3-sonnet-20240229",
        "max_tokens": 2000,
        "temperature": 0.1
    }


@pytest.fixture
def mock_ollama_config():
    """Mock Ollama configuration for testing."""
    return {
        "model_name": "llama2",
        "base_url": "http://localhost:11434",
        "temperature": 0.1,
        "timeout": 120
    }


@pytest.fixture
def mock_huggingface_config():
    """Mock HuggingFace configuration for testing."""
    return {
        "model_name": "microsoft/DialoGPT-medium",
        "max_tokens": 1000,
        "temperature": 0.1,
        "device": "cpu"
    }


@pytest.fixture
def mock_gemini_config():
    """Mock Gemini configuration for testing."""
    return {
        "api_key": "test_google_api_key",
        "model_name": "gemini-pro",
        "max_tokens": 2000,
        "temperature": 0.1,
        "timeout": 60
    }


@pytest.fixture
def mock_azure_openai_config():
    """Mock Azure OpenAI configuration for testing."""
    return {
        "api_key": "test_azure_openai_key",
        "azure_endpoint": "https://test.openai.azure.com/",
        "deployment_name": "gpt-4-deployment",
        "api_version": "2024-02-15-preview",
        "max_tokens": 2000,
        "temperature": 0.1
    }


@pytest.fixture
def mock_bedrock_config():
    """Mock Bedrock configuration for testing."""
    return {
        "model_id": "anthropic.claude-3-sonnet-20240229-v1:0",
        "region": "us-east-1",
        "max_tokens": 2000,
        "temperature": 0.1
    }


@pytest.fixture
def mock_runbook_config(tmp_path):
    """Mock runbook configuration for testing."""
    # Create temporary runbook directory
    runbook_dir = tmp_path / "runbooks"
    runbook_dir.mkdir()
    
    # Create sample runbook files
    (runbook_dir / "database-issues.md").write_text("# Database Issues\n\nCheck connection pool settings.")
    (runbook_dir / "api-errors.md").write_text("# API Errors\n\nReview error logs and metrics.")
    
    return {
        "directory": str(runbook_dir),
        "recursive": True
    }


@pytest.fixture
def mock_user_config(mock_aws_config, mock_github_config, mock_openai_config, mock_runbook_config):
    """Complete mock user configuration for testing."""
    return {
        "aws": mock_aws_config,
        "github": mock_github_config,
        "openai": mock_openai_config,
        "runbooks": mock_runbook_config,
        "preferences": {
            "preferred_cloud_provider": "aws",
            "preferred_llm_provider": "openai"
        }
    }


@pytest.fixture
def mock_logs_provider():
    """Mock logs provider for testing."""
    mock_provider = AsyncMock()
    mock_provider.fetch_logs = AsyncMock(return_value=[])
    mock_provider.search_logs_by_pattern = AsyncMock(return_value=[])
    mock_provider.get_log_groups = AsyncMock(return_value=["log-group-1", "log-group-2"])
    mock_provider.health_check = AsyncMock(return_value=True)
    return mock_provider


@pytest.fixture
def mock_metrics_provider():
    """Mock metrics provider for testing."""
    mock_provider = AsyncMock()
    mock_provider.get_metric_data = AsyncMock(return_value=[])
    mock_provider.get_alarms = AsyncMock(return_value=[])
    mock_provider.list_metrics = AsyncMock(return_value=[])
    mock_provider.health_check = AsyncMock(return_value=True)
    return mock_provider


@pytest.fixture
def mock_code_provider():
    """Mock code provider for testing."""
    mock_provider = AsyncMock()
    mock_provider.search_code = AsyncMock(return_value=[])
    mock_provider.get_file_content = AsyncMock(return_value={"content": "test content"})
    mock_provider.get_recent_commits = AsyncMock(return_value=[])
    mock_provider.health_check = AsyncMock(return_value=True)
    return mock_provider


@pytest.fixture
def mock_llm_provider():
    """Mock LLM provider for testing."""
    mock_provider = AsyncMock()
    mock_provider.generate_resolution = AsyncMock(return_value={
        "resolution_summary": "Test resolution",
        "detailed_steps": "1. Check logs\n2. Fix issue",
        "confidence_score": 0.8
    })
    mock_provider.analyze_logs = AsyncMock(return_value={
        "error_patterns": ["connection timeout"],
        "severity_assessment": "High"
    })
    mock_provider.analyze_code_context = AsyncMock(return_value={
        "potential_issues": ["timeout configuration"],
        "suggested_fixes": "Increase timeout value"
    })
    mock_provider.health_check = AsyncMock(return_value={"healthy": True})
    mock_provider.get_model_info = Mock(return_value={"provider": "test", "model_name": "test-model"})
    mock_provider.get_langchain_model = Mock(return_value=Mock())
    return mock_provider


@pytest.fixture
def mock_runbook_provider():
    """Mock runbook provider for testing."""
    mock_provider = AsyncMock()
    mock_provider.get_runbook_text = AsyncMock(return_value="Test runbook content")
    mock_provider.search_runbooks = AsyncMock(return_value=[])
    mock_provider.list_runbooks = AsyncMock(return_value=[])
    mock_provider.find_relevant_runbooks = AsyncMock(return_value=[])
    mock_provider.health_check = AsyncMock(return_value=True)
    return mock_provider


@pytest.fixture
def time_range():
    """Standard time range for testing."""
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=1)
    return (start_time, end_time)
