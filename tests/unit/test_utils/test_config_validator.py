"""Unit tests for configuration validator."""

import pytest
import os
from unittest.mock import patch
from oncall_agent.utils.config_validator import ConfigValidator


class TestConfigValidator:
    """Test cases for ConfigValidator."""

    @pytest.fixture
    def validator(self):
        """Create ConfigValidator instance."""
        return ConfigValidator()

    def test_validate_config_valid_minimal(self, validator):
        """Test validation with valid minimal configuration."""
        config = {
            "aws": {"region": "us-west-2"},
            "github": {"token": "test", "repositories": ["org/repo"]},
            "openai": {"api_key": "test"}
        }
        
        with patch.dict('os.environ', {
            "AWS_ACCESS_KEY_ID": "test",
            "AWS_SECRET_ACCESS_KEY": "test"
        }):
            result = validator.validate_config(config)
        
        assert result["valid"] is True
        assert len(result["errors"]) == 0

    def test_validate_config_valid_with_all_providers(self, validator):
        """Test validation with all providers configured."""
        config = {
            "aws": {"region": "us-west-2"},
            "github": {"token": "test", "repositories": ["org/repo"]},
            "openai": {"api_key": "test"},
            "runbooks": {"directory": "/test/runbooks"}
        }
        
        with patch.dict('os.environ', {
            "AWS_ACCESS_KEY_ID": "test",
            "AWS_SECRET_ACCESS_KEY": "test"
        }), patch('os.path.exists', return_value=True), patch('os.path.isdir', return_value=True):
            result = validator.validate_config(config)
        
        assert result["valid"] is True
        assert result["integration_status"]["logs"]["available"] is True
        assert result["integration_status"]["metrics"]["available"] is True
        assert result["integration_status"]["runbooks"]["available"] is True

    def test_validate_config_missing_llm(self, validator):
        """Test validation with missing LLM provider."""
        config = {
            "aws": {"region": "us-west-2"},
            "github": {"token": "test", "repositories": ["org/repo"]}
            # Missing LLM provider
        }
        
        with patch.dict('os.environ', {
            "AWS_ACCESS_KEY_ID": "test",
            "AWS_SECRET_ACCESS_KEY": "test"
        }):
            result = validator.validate_config(config)
        
        assert result["valid"] is False
        assert any("LLM provider" in error for error in result["errors"])

    def test_validate_config_missing_github(self, validator):
        """Test validation with missing GitHub configuration."""
        config = {
            "aws": {"region": "us-west-2"},
            "openai": {"api_key": "test"}
            # Missing GitHub
        }
        
        result = validator.validate_config(config)
        
        assert result["valid"] is False
        assert any("GitHub" in error for error in result["errors"])

    def test_validate_config_missing_logs(self, validator):
        """Test validation with missing logs provider."""
        config = {
            "github": {"token": "test", "repositories": ["org/repo"]},
            "openai": {"api_key": "test"}
            # Missing logs provider
        }
        
        result = validator.validate_config(config)
        
        assert result["valid"] is False
        assert any("logs provider" in error for error in result["errors"])

    def test_validate_logs_config_aws_explicit(self, validator):
        """Test logs configuration validation with explicit AWS credentials."""
        config = {
            "aws": {
                "region": "us-west-2",
                "access_key_id": "test",
                "secret_access_key": "test"
            }
        }
        
        status = validator._validate_logs_config(config)
        
        assert status["available"] is True
        assert "aws_cloudwatch" in status["providers"]

    def test_validate_logs_config_aws_environment(self, validator):
        """Test logs configuration validation with AWS environment variables."""
        config = {"aws": {"region": "us-west-2"}}
        
        with patch.dict('os.environ', {
            "AWS_ACCESS_KEY_ID": "test",
            "AWS_SECRET_ACCESS_KEY": "test"
        }):
            status = validator._validate_logs_config(config)
        
        assert status["available"] is True
        assert "aws_cloudwatch" in status["providers"]

    def test_validate_logs_config_azure(self, validator):
        """Test logs configuration validation with Azure."""
        config = {
            "azure": {
                "subscription_id": "test-sub",
                "workspace_id": "test-workspace",
                "tenant_id": "test-tenant",
                "client_id": "test-client",
                "client_secret": "test-secret"
            }
        }
        
        status = validator._validate_logs_config(config)
        
        assert status["available"] is True
        assert "azure_monitor" in status["providers"]

    def test_validate_logs_config_gcp(self, validator):
        """Test logs configuration validation with GCP."""
        config = {
            "gcp": {
                "project_id": "test-project",
                "credentials_path": "/test/creds.json"
            }
        }
        
        with patch('os.path.exists', return_value=True):
            status = validator._validate_logs_config(config)
        
        assert status["available"] is True
        assert "gcp_logging" in status["providers"]

    def test_validate_code_config_valid(self, validator):
        """Test code configuration validation with valid GitHub config."""
        config = {
            "github": {
                "token": "test_token",
                "repositories": ["org/repo1", "org/repo2"]
            }
        }
        
        status = validator._validate_code_config(config)
        
        assert status["available"] is True
        assert "github" in status["providers"]

    def test_validate_code_config_missing_token(self, validator):
        """Test code configuration validation with missing GitHub token."""
        config = {
            "github": {
                "repositories": ["org/repo1", "org/repo2"]
            }
        }
        
        status = validator._validate_code_config(config)
        
        assert status["available"] is False
        assert any("GitHub token" in error for error in status["errors"])

    def test_validate_code_config_missing_repositories(self, validator):
        """Test code configuration validation with missing repositories."""
        config = {
            "github": {
                "token": "test_token"
            }
        }
        
        status = validator._validate_code_config(config)
        
        assert status["available"] is False
        assert any("repositories" in error for error in status["errors"])

    def test_validate_llm_config_openai(self, validator):
        """Test LLM configuration validation with OpenAI."""
        config = {"openai": {"api_key": "test_key"}}
        
        status = validator._validate_llm_config(config)
        
        assert status["available"] is True
        assert "openai" in status["providers"]

    def test_validate_llm_config_anthropic(self, validator):
        """Test LLM configuration validation with Anthropic."""
        config = {"anthropic": {"api_key": "test_key"}}
        
        status = validator._validate_llm_config(config)
        
        assert status["available"] is True
        assert "anthropic" in status["providers"]

    def test_validate_llm_config_ollama(self, validator):
        """Test LLM configuration validation with Ollama."""
        config = {"ollama": {"model_name": "llama2"}}
        
        status = validator._validate_llm_config(config)
        
        assert status["available"] is True
        assert "ollama" in status["providers"]

    def test_validate_llm_config_huggingface(self, validator):
        """Test LLM configuration validation with HuggingFace."""
        config = {"huggingface": {"model_name": "microsoft/DialoGPT-medium"}}
        
        status = validator._validate_llm_config(config)
        
        assert status["available"] is True
        assert "huggingface" in status["providers"]

    def test_validate_llm_config_multiple_providers(self, validator):
        """Test LLM configuration validation with multiple providers."""
        config = {
            "openai": {"api_key": "test_key"},
            "ollama": {"model_name": "llama2"},
            "huggingface": {"model_name": "microsoft/DialoGPT-medium"}
        }
        
        status = validator._validate_llm_config(config)
        
        assert status["available"] is True
        assert "openai" in status["providers"]
        assert "ollama" in status["providers"]
        assert "huggingface" in status["providers"]

    def test_validate_llm_config_none_available(self, validator):
        """Test LLM configuration validation with no providers."""
        config = {}
        
        status = validator._validate_llm_config(config)
        
        assert status["available"] is False
        assert any("No LLM provider configured" in error for error in status["errors"])

    @patch('os.path.exists')
    @patch('os.path.isdir')
    def test_validate_runbooks_config_directory(self, mock_isdir, mock_exists, validator):
        """Test runbooks configuration validation with directory."""
        mock_exists.return_value = True
        mock_isdir.return_value = True
        
        config = {"runbooks": {"directory": "/test/runbooks"}}
        
        with patch.object(validator, '_has_files_with_extensions', return_value=True):
            status = validator._validate_runbooks_config(config)
        
        assert status["available"] is True

    def test_validate_runbooks_config_web_urls(self, validator):
        """Test runbooks configuration validation with web URLs."""
        config = {
            "runbooks": {
                "web_urls": ["https://docs.example.com"]
            }
        }
        
        status = validator._validate_runbooks_config(config)
        
        assert status["available"] is True

    def test_generate_sample_config_aws(self, validator):
        """Test sample configuration generation with AWS preference."""
        sample_config = validator.generate_sample_config(["aws", "github", "openai"])
        
        assert "aws:" in sample_config
        assert "github:" in sample_config
        assert "openai:" in sample_config
        assert "AWS_ACCESS_KEY_ID" in sample_config
        assert "GITHUB_TOKEN" in sample_config
        assert "OPENAI_API_KEY" in sample_config

    def test_generate_sample_config_azure(self, validator):
        """Test sample configuration generation with Azure preference."""
        sample_config = validator.generate_sample_config(["azure", "github", "anthropic"])
        
        assert "azure:" in sample_config
        assert "github:" in sample_config
        assert "anthropic:" in sample_config
        assert "AZURE_SUBSCRIPTION_ID" in sample_config

    def test_get_setup_checklist(self, validator):
        """Test setup checklist generation."""
        config = {
            "aws": {"region": "us-west-2"},
            "github": {"token": "test", "repositories": ["org/repo"]},
            "openai": {"api_key": "test"}
        }
        
        with patch.dict('os.environ', {
            "AWS_ACCESS_KEY_ID": "test",
            "AWS_SECRET_ACCESS_KEY": "test"
        }):
            checklist = validator.get_setup_checklist(config)
        
        assert len(checklist) == 5  # 3 required + 2 optional
        
        # Check required items
        required_items = [item for item in checklist if item["required"]]
        assert len(required_items) == 3
        assert all(item["status"] for item in required_items)  # All should be configured
        
        # Check optional items
        optional_items = [item for item in checklist if not item["required"]]
        assert len(optional_items) == 2

    def test_get_environment_variables_needed(self, validator):
        """Test getting needed environment variables."""
        config = {
            "aws": {"region": "us-west-2"},
            "github": {"repositories": ["org/repo"]},
            "openai": {}
        }
        
        env_vars = validator.get_environment_variables_needed(config)
        
        assert "required" in env_vars
        assert "missing" in env_vars
        assert "GITHUB_TOKEN" in str(env_vars["required"])

    @patch('os.walk')
    def test_has_files_with_extensions(self, mock_walk, validator):
        """Test file extension checking utility."""
        mock_walk.return_value = [
            ("/test", [], ["file1.pdf", "file2.md", "file3.txt"])
        ]
        
        assert validator._has_files_with_extensions("/test", [".pdf"]) is True
        assert validator._has_files_with_extensions("/test", [".docx"]) is False
        assert validator._has_files_with_extensions("/test", [".md", ".txt"]) is True
