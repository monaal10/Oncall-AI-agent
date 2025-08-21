"""Unit tests for integration registry."""

import pytest
import os
from unittest.mock import patch, Mock
from oncall_agent.integrations.registry import IntegrationRegistry


class TestIntegrationRegistry:
    """Test cases for IntegrationRegistry."""

    @pytest.fixture
    def registry(self):
        """Create IntegrationRegistry instance."""
        return IntegrationRegistry()

    @pytest.mark.asyncio
    async def test_discover_integrations_with_aws_github_openai(self, registry):
        """Test discovery with AWS, GitHub, and OpenAI configuration."""
        config = {
            "aws": {"region": "us-west-2", "access_key_id": "test", "secret_access_key": "test"},
            "github": {"token": "test_token", "repositories": ["org/repo"]},
            "openai": {"api_key": "test_key"}
        }
        
        result = await registry.discover_integrations(config)
        
        assert "aws_cloudwatch" in result["available_integrations"]["logs"]
        assert "aws_cloudwatch" in result["available_integrations"]["metrics"]
        assert "github" in result["available_integrations"]["code"]
        assert "openai" in result["available_integrations"]["llm"]
        
        assert result["selected_integrations"]["logs_provider"] == "aws_cloudwatch"
        assert result["selected_integrations"]["metrics_provider"] == "aws_cloudwatch"
        assert result["selected_integrations"]["code_provider"] == "github"
        assert result["selected_integrations"]["llm_provider"] == "openai"

    @pytest.mark.asyncio
    async def test_discover_integrations_with_azure_github_anthropic(self, registry):
        """Test discovery with Azure, GitHub, and Anthropic configuration."""
        config = {
            "azure": {
                "subscription_id": "test-sub",
                "workspace_id": "test-workspace",
                "tenant_id": "test-tenant",
                "client_id": "test-client",
                "client_secret": "test-secret"
            },
            "github": {"token": "test_token", "repositories": ["org/repo"]},
            "anthropic": {"api_key": "test_key"}
        }
        
        result = await registry.discover_integrations(config)
        
        assert "azure_monitor" in result["available_integrations"]["logs"]
        assert "azure_monitor" in result["available_integrations"]["metrics"]
        assert result["selected_integrations"]["logs_provider"] == "azure_monitor"
        assert result["selected_integrations"]["llm_provider"] == "anthropic"

    @pytest.mark.asyncio
    async def test_discover_integrations_with_ollama_no_api_key(self, registry):
        """Test discovery with Ollama (no API key required)."""
        config = {
            "aws": {"region": "us-west-2"},
            "github": {"token": "test_token", "repositories": ["org/repo"]},
            "ollama": {"model_name": "llama2", "base_url": "http://localhost:11434"}
        }
        
        with patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "test", "AWS_SECRET_ACCESS_KEY": "test"}):
            result = await registry.discover_integrations(config)
        
        assert "ollama" in result["available_integrations"]["llm"]
        assert result["selected_integrations"]["llm_provider"] == "ollama"

    @pytest.mark.asyncio
    async def test_discover_integrations_with_huggingface_no_api_key(self, registry):
        """Test discovery with HuggingFace (no API key required)."""
        config = {
            "aws": {"region": "us-west-2"},
            "github": {"token": "test_token", "repositories": ["org/repo"]},
            "huggingface": {"model_name": "microsoft/DialoGPT-medium"}
        }
        
        with patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "test", "AWS_SECRET_ACCESS_KEY": "test"}):
            result = await registry.discover_integrations(config)
        
        assert "huggingface" in result["available_integrations"]["llm"]
        assert result["selected_integrations"]["llm_provider"] == "huggingface"

    @pytest.mark.asyncio
    async def test_discover_integrations_missing_requirements(self, registry):
        """Test discovery with missing required integrations."""
        config = {
            "aws": {"region": "us-west-2"}
            # Missing GitHub and LLM providers
        }
        
        with pytest.raises(ValueError) as exc_info:
            await registry.discover_integrations(config)
        
        assert "Missing required integrations" in str(exc_info.value)

    def test_check_aws_credentials_explicit(self, registry):
        """Test AWS credential checking with explicit credentials."""
        config = {
            "aws": {
                "region": "us-west-2",
                "access_key_id": "test_key",
                "secret_access_key": "test_secret"
            }
        }
        
        assert registry._check_aws_credentials(config) is True

    def test_check_aws_credentials_environment(self, registry):
        """Test AWS credential checking with environment variables."""
        config = {"aws": {"region": "us-west-2"}}
        
        with patch.dict(os.environ, {
            "AWS_ACCESS_KEY_ID": "test_key",
            "AWS_SECRET_ACCESS_KEY": "test_secret"
        }):
            assert registry._check_aws_credentials(config) is True

    def test_check_aws_credentials_missing(self, registry):
        """Test AWS credential checking with missing credentials."""
        config = {"aws": {"region": "us-west-2"}}
        
        with patch.dict(os.environ, {}, clear=True):
            assert registry._check_aws_credentials(config) is False

    def test_check_github_credentials_explicit(self, registry):
        """Test GitHub credential checking with explicit token."""
        config = {"github": {"token": "test_token"}}
        
        assert registry._check_github_credentials(config) is True

    def test_check_github_credentials_environment(self, registry):
        """Test GitHub credential checking with environment variable."""
        config = {"github": {}}
        
        with patch.dict(os.environ, {"GITHUB_TOKEN": "test_token"}):
            assert registry._check_github_credentials(config) is True

    def test_check_openai_credentials(self, registry):
        """Test OpenAI credential checking."""
        # Explicit API key
        config = {"openai": {"api_key": "test_key"}}
        assert registry._check_openai_credentials(config) is True
        
        # Environment variable
        config = {"openai": {}}
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
            assert registry._check_openai_credentials(config) is True
        
        # Missing
        config = {"openai": {}}
        with patch.dict(os.environ, {}, clear=True):
            assert registry._check_openai_credentials(config) is False

    def test_check_ollama_availability(self, registry):
        """Test Ollama availability checking."""
        # With model_name
        config = {"ollama": {"model_name": "llama2"}}
        assert registry._check_ollama_availability(config) is True
        
        # Without model_name
        config = {"ollama": {}}
        assert registry._check_ollama_availability(config) is False

    def test_check_huggingface_availability(self, registry):
        """Test HuggingFace availability checking."""
        # With model_name
        config = {"huggingface": {"model_name": "microsoft/DialoGPT-medium"}}
        assert registry._check_huggingface_availability(config) is True
        
        # Without model_name
        config = {"huggingface": {}}
        assert registry._check_huggingface_availability(config) is False

    def test_check_gemini_credentials(self, registry):
        """Test Gemini credential checking."""
        # Explicit API key
        config = {"gemini": {"api_key": "test_key"}}
        assert registry._check_gemini_credentials(config) is True
        
        # Environment variable
        config = {"gemini": {}}
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test_key"}):
            assert registry._check_gemini_credentials(config) is True
        
        # Missing
        config = {"gemini": {}}
        with patch.dict(os.environ, {}, clear=True):
            assert registry._check_gemini_credentials(config) is False

    def test_check_azure_openai_credentials(self, registry):
        """Test Azure OpenAI credential checking."""
        # Explicit config
        config = {
            "azure_openai": {
                "api_key": "test_key",
                "azure_endpoint": "https://test.openai.azure.com/",
                "deployment_name": "gpt-4-deployment"
            }
        }
        assert registry._check_azure_openai_credentials(config) is True
        
        # Environment variables
        config = {"azure_openai": {"deployment_name": "gpt-4-deployment"}}
        with patch.dict(os.environ, {
            "AZURE_OPENAI_API_KEY": "test_key",
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/"
        }):
            assert registry._check_azure_openai_credentials(config) is True
        
        # Missing
        config = {"azure_openai": {}}
        with patch.dict(os.environ, {}, clear=True):
            assert registry._check_azure_openai_credentials(config) is False

    def test_check_bedrock_credentials(self, registry):
        """Test Bedrock credential checking."""
        # With model_id and explicit credentials
        config = {
            "bedrock": {
                "model_id": "anthropic.claude-3-sonnet-20240229-v1:0",
                "aws_access_key_id": "test_key",
                "aws_secret_access_key": "test_secret"
            }
        }
        assert registry._check_bedrock_credentials(config) is True
        
        # With model_id and environment variables
        config = {"bedrock": {"model_id": "anthropic.claude-3-sonnet-20240229-v1:0"}}
        with patch.dict(os.environ, {
            "AWS_ACCESS_KEY_ID": "test_key",
            "AWS_SECRET_ACCESS_KEY": "test_secret"
        }):
            assert registry._check_bedrock_credentials(config) is True
        
        # With model_id only (assumes IAM role)
        config = {"bedrock": {"model_id": "anthropic.claude-3-sonnet-20240229-v1:0"}}
        with patch.dict(os.environ, {}, clear=True):
            assert registry._check_bedrock_credentials(config) is True
        
        # Without model_id
        config = {"bedrock": {}}
        assert registry._check_bedrock_credentials(config) is False

    def test_validate_minimum_requirements_valid(self, registry):
        """Test minimum requirements validation with valid configuration."""
        available_integrations = {
            "logs": ["aws_cloudwatch"],
            "code": ["github"],
            "llm": ["openai"]
        }
        
        missing = registry._validate_minimum_requirements(available_integrations)
        assert missing == []

    def test_validate_minimum_requirements_missing_logs(self, registry):
        """Test minimum requirements validation with missing logs provider."""
        available_integrations = {
            "logs": [],
            "code": ["github"],
            "llm": ["openai"]
        }
        
        missing = registry._validate_minimum_requirements(available_integrations)
        assert any("logs provider" in item for item in missing)

    def test_validate_minimum_requirements_missing_llm(self, registry):
        """Test minimum requirements validation with missing LLM provider."""
        available_integrations = {
            "logs": ["aws_cloudwatch"],
            "code": ["github"],
            "llm": []
        }
        
        missing = registry._validate_minimum_requirements(available_integrations)
        assert any("LLM provider" in item for item in missing)

    def test_select_logs_provider_with_preference(self, registry):
        """Test logs provider selection with user preference."""
        options = ["aws_cloudwatch", "azure_monitor", "gcp_logging"]
        preferences = {"preferred_cloud_provider": "azure"}
        
        selected = registry._select_logs_provider(options, preferences)
        assert selected == "azure_monitor"

    def test_select_logs_provider_default_priority(self, registry):
        """Test logs provider selection with default priority."""
        options = ["gcp_logging", "azure_monitor", "aws_cloudwatch"]
        preferences = {}
        
        selected = registry._select_logs_provider(options, preferences)
        assert selected == "aws_cloudwatch"  # AWS has highest default priority

    def test_select_llm_provider_with_preference(self, registry):
        """Test LLM provider selection with user preference."""
        options = ["openai", "anthropic", "ollama", "huggingface"]
        preferences = {"preferred_llm_provider": "anthropic"}
        
        selected = registry._select_llm_provider(options, preferences)
        assert selected == "anthropic"

    def test_select_llm_provider_default_priority(self, registry):
        """Test LLM provider selection with default priority."""
        options = ["huggingface", "bedrock", "ollama", "gemini", "azure_openai", "anthropic", "openai"]
        preferences = {}
        
        selected = registry._select_llm_provider(options, preferences)
        assert selected == "openai"  # OpenAI has highest default priority

    def test_select_llm_provider_azure_openai_priority(self, registry):
        """Test LLM provider selection with Azure OpenAI available."""
        options = ["huggingface", "ollama", "azure_openai", "bedrock"]
        preferences = {}
        
        selected = registry._select_llm_provider(options, preferences)
        assert selected == "azure_openai"  # Azure OpenAI has higher priority than Bedrock, Ollama, HF

    def test_select_llm_provider_gemini_priority(self, registry):
        """Test LLM provider selection with Gemini available."""
        options = ["huggingface", "ollama", "bedrock", "gemini"]
        preferences = {}
        
        selected = registry._select_llm_provider(options, preferences)
        assert selected == "gemini"  # Gemini has higher priority than Bedrock, Ollama, HF

    def test_get_provider_config_aws(self, registry, mock_aws_config):
        """Test getting AWS provider configuration."""
        user_config = {"aws": mock_aws_config}
        
        config = registry.get_provider_config("logs", "aws_cloudwatch", user_config)
        
        assert config["region"] == "us-west-2"
        assert config["access_key_id"] == "test_access_key"
        assert config["secret_access_key"] == "test_secret_key"

    def test_get_provider_config_github(self, registry, mock_github_config):
        """Test getting GitHub provider configuration."""
        user_config = {"github": mock_github_config}
        
        config = registry.get_provider_config("code", "github", user_config)
        
        assert config["token"] == "test_github_token"
        assert config["repositories"] == ["testorg/repo1", "testorg/repo2"]

    def test_get_provider_config_with_environment_variables(self, registry):
        """Test getting provider configuration from environment variables."""
        user_config = {"github": {"repositories": ["org/repo"]}}
        
        with patch.dict(os.environ, {"GITHUB_TOKEN": "env_token"}):
            config = registry.get_provider_config("code", "github", user_config)
        
        assert config["token"] == "env_token"
        assert config["repositories"] == ["org/repo"]

    @patch("os.path.exists")
    @patch("os.walk")
    def test_has_files_with_extensions(self, mock_walk, mock_exists, registry):
        """Test file extension checking."""
        mock_exists.return_value = True
        mock_walk.return_value = [
            ("/test", [], ["file1.pdf", "file2.md", "file3.txt"])
        ]
        
        assert registry._has_files_with_extensions("/test", [".pdf"]) is True
        assert registry._has_files_with_extensions("/test", [".docx"]) is False

    def test_get_integration_summary(self, registry):
        """Test integration summary generation."""
        selected_integrations = {
            "logs_provider": "aws_cloudwatch",
            "metrics_provider": "aws_cloudwatch",
            "code_provider": "github",
            "llm_provider": "openai",
            "runbook_provider": "unified"
        }
        
        summary = registry.get_integration_summary(selected_integrations)
        
        assert summary["integration_count"] == 5
        assert summary["required_integrations"]["logs"] == "aws_cloudwatch"
        assert summary["required_integrations"]["code"] == "github"
        assert summary["required_integrations"]["llm"] == "openai"
        assert summary["optional_integrations"]["metrics"] == "aws_cloudwatch"
        assert summary["optional_integrations"]["runbooks"] == "unified"
        assert len(summary["capabilities"]) > 0
