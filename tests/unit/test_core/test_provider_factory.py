"""Unit tests for provider factory."""

import pytest
from unittest.mock import patch, Mock, AsyncMock
from oncall_agent.core.provider_factory import ProviderFactory


class TestProviderFactory:
    """Test cases for ProviderFactory."""

    @pytest.fixture
    def factory(self):
        """Create ProviderFactory instance."""
        return ProviderFactory()

    @patch('oncall_agent.core.provider_factory.importlib.import_module')
    def test_import_provider_class_success(self, mock_import, factory):
        """Test successful provider class import."""
        mock_module = Mock()
        mock_class = Mock()
        mock_module.TestProvider = mock_class
        mock_import.return_value = mock_module
        
        result = factory._import_provider_class("test.module", "TestProvider")
        
        assert result == mock_class
        mock_import.assert_called_once_with("test.module")

    @patch('oncall_agent.core.provider_factory.importlib.import_module')
    def test_import_provider_class_module_not_found(self, mock_import, factory):
        """Test provider class import with missing module."""
        mock_import.side_effect = ImportError("Module not found")
        
        with pytest.raises(ImportError) as exc_info:
            factory._import_provider_class("missing.module", "TestProvider")
        
        assert "Failed to import TestProvider from missing.module" in str(exc_info.value)

    @patch('oncall_agent.core.provider_factory.importlib.import_module')
    def test_import_provider_class_class_not_found(self, mock_import, factory):
        """Test provider class import with missing class."""
        mock_module = Mock()
        del mock_module.MissingClass  # Ensure class doesn't exist
        mock_import.return_value = mock_module
        
        with pytest.raises(ImportError) as exc_info:
            factory._import_provider_class("test.module", "MissingClass")
        
        assert "Class MissingClass not found in test.module" in str(exc_info.value)

    @patch.object(ProviderFactory, '_import_provider_class')
    def test_create_logs_provider_aws(self, mock_import, factory, mock_aws_config):
        """Test creating AWS logs provider."""
        mock_provider_class = Mock()
        mock_provider_instance = Mock()
        mock_provider_class.return_value = mock_provider_instance
        mock_import.return_value = mock_provider_class
        
        result = factory.create_logs_provider("aws_cloudwatch", mock_aws_config)
        
        assert result == mock_provider_instance
        mock_import.assert_called_once_with(
            "oncall_agent.integrations.aws",
            "CloudWatchLogsProvider"
        )
        mock_provider_class.assert_called_once_with(mock_aws_config)

    @patch.object(ProviderFactory, '_import_provider_class')
    def test_create_logs_provider_azure(self, mock_import, factory, mock_azure_config):
        """Test creating Azure logs provider."""
        mock_provider_class = Mock()
        mock_provider_instance = Mock()
        mock_provider_class.return_value = mock_provider_instance
        mock_import.return_value = mock_provider_class
        
        result = factory.create_logs_provider("azure_monitor", mock_azure_config)
        
        assert result == mock_provider_instance
        mock_import.assert_called_once_with(
            "oncall_agent.integrations.azure",
            "AzureMonitorLogsProvider"
        )

    def test_create_logs_provider_unsupported(self, factory):
        """Test creating unsupported logs provider."""
        with pytest.raises(ValueError) as exc_info:
            factory.create_logs_provider("unsupported_provider", {})
        
        assert "Unsupported logs provider: unsupported_provider" in str(exc_info.value)

    @patch.object(ProviderFactory, '_import_provider_class')
    def test_create_llm_provider_openai(self, mock_import, factory, mock_openai_config):
        """Test creating OpenAI LLM provider."""
        mock_provider_class = Mock()
        mock_provider_instance = Mock()
        mock_provider_class.return_value = mock_provider_instance
        mock_import.return_value = mock_provider_class
        
        result = factory.create_llm_provider("openai", mock_openai_config)
        
        assert result == mock_provider_instance
        mock_import.assert_called_once_with(
            "oncall_agent.integrations.llm",
            "OpenAIProvider"
        )

    @patch.object(ProviderFactory, '_import_provider_class')
    def test_create_llm_provider_huggingface(self, mock_import, factory, mock_huggingface_config):
        """Test creating HuggingFace LLM provider."""
        mock_provider_class = Mock()
        mock_provider_instance = Mock()
        mock_provider_class.return_value = mock_provider_instance
        mock_import.return_value = mock_provider_class
        
        result = factory.create_llm_provider("huggingface", mock_huggingface_config)
        
        assert result == mock_provider_instance
        mock_import.assert_called_once_with(
            "oncall_agent.integrations.llm",
            "HuggingFaceProvider"
        )

    def test_create_metrics_provider_none(self, factory):
        """Test creating metrics provider with None type."""
        result = factory.create_metrics_provider(None, {})
        assert result is None

    def test_create_runbook_provider_none(self, factory):
        """Test creating runbook provider with None type."""
        result = factory.create_runbook_provider(None, {})
        assert result is None

    @pytest.mark.asyncio
    async def test_validate_provider_connectivity_healthy(self, factory):
        """Test provider connectivity validation for healthy provider."""
        mock_provider = AsyncMock()
        mock_provider.health_check = AsyncMock(return_value=True)
        
        result = await factory.validate_provider_connectivity(mock_provider, "test_provider")
        
        assert result["provider_type"] == "test_provider"
        assert result["healthy"] is True
        assert result["error"] is None
        assert result["latency_ms"] is not None

    @pytest.mark.asyncio
    async def test_validate_provider_connectivity_unhealthy(self, factory):
        """Test provider connectivity validation for unhealthy provider."""
        mock_provider = AsyncMock()
        mock_provider.health_check = AsyncMock(side_effect=Exception("Connection failed"))
        
        result = await factory.validate_provider_connectivity(mock_provider, "test_provider")
        
        assert result["provider_type"] == "test_provider"
        assert result["healthy"] is False
        assert "Connection failed" in result["error"]

    @pytest.mark.asyncio
    async def test_validate_provider_connectivity_no_health_check(self, factory):
        """Test provider connectivity validation for provider without health_check method."""
        mock_provider = Mock()  # No health_check method
        
        result = await factory.validate_provider_connectivity(mock_provider, "test_provider")
        
        assert result["provider_type"] == "test_provider"
        assert result["healthy"] is True  # Assumes healthy if no health check

    def test_get_provider_capabilities_logs_aws(self, factory):
        """Test getting capabilities for AWS logs provider."""
        capabilities = factory.get_provider_capabilities("logs", "aws_cloudwatch")
        
        assert capabilities["provider_type"] == "logs"
        assert capabilities["provider_name"] == "aws_cloudwatch"
        assert "Log search" in capabilities["features"]
        assert "CloudWatch Insights queries" in capabilities["features"]
        assert "boto3" in capabilities["dependencies"]

    def test_get_provider_capabilities_llm_ollama(self, factory):
        """Test getting capabilities for Ollama LLM provider."""
        capabilities = factory.get_provider_capabilities("llm", "ollama")
        
        assert capabilities["provider_type"] == "llm"
        assert capabilities["provider_name"] == "ollama"
        assert "Local deployment" in capabilities["features"]
        assert "Privacy" in capabilities["features"]
        assert "langchain-community" in capabilities["dependencies"]

    def test_get_provider_capabilities_llm_huggingface(self, factory):
        """Test getting capabilities for HuggingFace LLM provider."""
        capabilities = factory.get_provider_capabilities("llm", "huggingface")
        
        assert capabilities["provider_type"] == "llm"
        assert capabilities["provider_name"] == "huggingface"
        # Should have basic LLM features
        assert "Text generation" in capabilities["features"]
