"""Unit tests for setup manager."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from oncall_agent.core.setup_manager import SetupManager


class TestSetupManager:
    """Test cases for SetupManager."""

    @pytest.fixture
    def setup_manager(self):
        """Create SetupManager instance."""
        return SetupManager()

    @pytest.mark.asyncio
    @patch('oncall_agent.core.setup_manager.IntegrationRegistry')
    @patch('oncall_agent.core.setup_manager.ProviderFactory')
    async def test_setup_from_config_success(
        self,
        mock_factory_class,
        mock_registry_class,
        setup_manager,
        mock_user_config,
        mock_logs_provider,
        mock_code_provider,
        mock_llm_provider
    ):
        """Test successful setup from configuration."""
        # Mock registry
        mock_registry = Mock()
        mock_registry.discover_integrations = AsyncMock(return_value={
            "available_integrations": {
                "logs": ["aws_cloudwatch"],
                "code": ["github"],
                "llm": ["openai"],
                "metrics": ["aws_cloudwatch"],
                "runbooks": []
            },
            "selected_integrations": {
                "logs_provider": "aws_cloudwatch",
                "code_provider": "github",
                "llm_provider": "openai",
                "metrics_provider": "aws_cloudwatch"
            }
        })
        mock_registry.get_provider_config = Mock(return_value={"test": "config"})
        mock_registry_class.return_value = mock_registry
        
        # Mock factory
        mock_factory = Mock()
        mock_factory.create_logs_provider = Mock(return_value=mock_logs_provider)
        mock_factory.create_code_provider = Mock(return_value=mock_code_provider)
        mock_factory.create_llm_provider = Mock(return_value=mock_llm_provider)
        mock_factory.create_metrics_provider = Mock(return_value=None)
        mock_factory.create_runbook_provider = Mock(return_value=None)
        mock_factory.validate_provider_connectivity = AsyncMock(return_value={"healthy": True})
        mock_factory_class.return_value = mock_factory
        
        # Mock RuntimeInterface
        with patch('oncall_agent.core.setup_manager.RuntimeInterface') as mock_runtime_class:
            mock_runtime = Mock()
            mock_runtime.health_check_all_providers = AsyncMock(return_value={
                "logs": {"healthy": True},
                "code": {"healthy": True},
                "llm": {"healthy": True}
            })
            mock_runtime.get_runtime_functions = Mock(return_value={
                "get_logs": Mock(),
                "get_code_context": Mock(),
                "get_llm_response": Mock()
            })
            mock_runtime_class.return_value = mock_runtime
            
            result = await setup_manager.setup_from_config(mock_user_config)
            
            assert result == mock_runtime
            mock_registry.discover_integrations.assert_called_once_with(mock_user_config)

    @pytest.mark.asyncio
    @patch('oncall_agent.core.setup_manager.IntegrationRegistry')
    async def test_setup_from_config_missing_requirements(
        self,
        mock_registry_class,
        setup_manager,
        mock_user_config
    ):
        """Test setup failure due to missing requirements."""
        # Mock registry to raise validation error
        mock_registry = Mock()
        mock_registry.discover_integrations = AsyncMock(
            side_effect=ValueError("Missing required integrations: logs provider")
        )
        mock_registry_class.return_value = mock_registry
        
        with pytest.raises(ValueError) as exc_info:
            await setup_manager.setup_from_config(mock_user_config)
        
        assert "Missing required integrations" in str(exc_info.value)

    def test_validate_minimum_config_valid(self, setup_manager):
        """Test minimum configuration validation with valid config."""
        config = {
            "aws": {"region": "us-west-2"},
            "github": {"token": "test", "repositories": ["org/repo"]},
            "openai": {"api_key": "test"}
        }
        
        with patch.dict('os.environ', {
            "AWS_ACCESS_KEY_ID": "test",
            "AWS_SECRET_ACCESS_KEY": "test"
        }):
            result = setup_manager.validate_minimum_config(config)
        
        assert result["valid"] is True
        assert len(result["errors"]) == 0

    def test_validate_minimum_config_missing_llm(self, setup_manager):
        """Test minimum configuration validation with missing LLM."""
        config = {
            "aws": {"region": "us-west-2"},
            "github": {"token": "test", "repositories": ["org/repo"]}
            # Missing LLM provider
        }
        
        with patch.dict('os.environ', {
            "AWS_ACCESS_KEY_ID": "test",
            "AWS_SECRET_ACCESS_KEY": "test"
        }):
            result = setup_manager.validate_minimum_config(config)
        
        assert result["valid"] is False
        assert any("LLM provider" in error for error in result["errors"])

    def test_validate_minimum_config_missing_github(self, setup_manager):
        """Test minimum configuration validation with missing GitHub."""
        config = {
            "aws": {"region": "us-west-2"},
            "openai": {"api_key": "test"}
            # Missing GitHub
        }
        
        result = setup_manager.validate_minimum_config(config)
        
        assert result["valid"] is False
        assert any("GitHub" in error for error in result["errors"])

    def test_check_aws_config(self, setup_manager):
        """Test AWS configuration checking."""
        # Valid config
        config = {"aws": {"region": "us-west-2"}}
        assert setup_manager._check_aws_config(config) is True
        
        # Missing region
        config = {"aws": {}}
        assert setup_manager._check_aws_config(config) is False

    def test_check_azure_config(self, setup_manager):
        """Test Azure configuration checking."""
        # Valid config
        config = {
            "azure": {
                "subscription_id": "test-sub",
                "workspace_id": "test-workspace"
            }
        }
        assert setup_manager._check_azure_config(config) is True
        
        # Missing workspace_id
        config = {"azure": {"subscription_id": "test-sub"}}
        assert setup_manager._check_azure_config(config) is False

    def test_check_gcp_config(self, setup_manager):
        """Test GCP configuration checking."""
        # Valid config
        config = {"gcp": {"project_id": "test-project"}}
        assert setup_manager._check_gcp_config(config) is True
        
        # Missing project_id
        config = {"gcp": {}}
        assert setup_manager._check_gcp_config(config) is False

    @pytest.mark.asyncio
    @patch('oncall_agent.core.setup_manager.asyncio.gather')
    async def test_create_provider_instances(self, mock_gather, setup_manager):
        """Test provider instance creation."""
        provider_configs = {
            "logs_provider": {"type": "aws_cloudwatch", "config": {"region": "us-west-2"}},
            "code_provider": {"type": "github", "config": {"token": "test"}},
            "llm_provider": {"type": "openai", "config": {"api_key": "test"}}
        }
        
        mock_logs_provider = Mock()
        mock_code_provider = Mock()
        mock_llm_provider = Mock()
        
        setup_manager.factory = Mock()
        setup_manager.factory.create_logs_provider = Mock(return_value=mock_logs_provider)
        setup_manager.factory.create_code_provider = Mock(return_value=mock_code_provider)
        setup_manager.factory.create_llm_provider = Mock(return_value=mock_llm_provider)
        setup_manager.factory.create_metrics_provider = Mock(return_value=None)
        setup_manager.factory.create_runbook_provider = Mock(return_value=None)
        
        providers = await setup_manager._create_provider_instances(provider_configs)
        
        assert "logs" in providers
        assert "code" in providers
        assert "llm" in providers
        assert providers["logs"] == mock_logs_provider
        assert providers["code"] == mock_code_provider
        assert providers["llm"] == mock_llm_provider

    @pytest.mark.asyncio
    async def test_validate_all_providers(self, setup_manager):
        """Test provider validation."""
        providers = {
            "logs": Mock(),
            "code": Mock(),
            "llm": Mock()
        }
        
        setup_manager.factory = Mock()
        setup_manager.factory.validate_provider_connectivity = AsyncMock(
            return_value={"healthy": True, "latency_ms": 100}
        )
        
        results = await setup_manager._validate_all_providers(providers)
        
        assert len(results) == 3
        assert all(result["healthy"] for result in results.values())

    @pytest.mark.asyncio
    async def test_test_setup(self, setup_manager):
        """Test setup testing functionality."""
        # Create mock runtime interface
        mock_runtime = Mock()
        mock_runtime.get_logs = AsyncMock(return_value=[])
        mock_runtime.get_code_context = AsyncMock(return_value=[])
        mock_runtime.get_llm_response = AsyncMock(return_value={"confidence_score": 0.8})
        mock_runtime.get_metrics = None
        mock_runtime.get_runbook_guidance = None
        
        test_results = await setup_manager.test_setup(mock_runtime)
        
        assert "logs_test" in test_results
        assert "code_test" in test_results
        assert "llm_test" in test_results
        assert "metrics_test" in test_results
        assert "runbooks_test" in test_results

    def test_get_integration_status_not_configured(self, setup_manager):
        """Test getting integration status when not configured."""
        status = setup_manager.get_integration_status()
        
        assert status["configured"] is False
        assert "Setup not completed" in status["message"]

    def test_get_integration_status_configured(self, setup_manager):
        """Test getting integration status when configured."""
        # Mock runtime interface
        mock_runtime = Mock()
        mock_runtime.get_provider_info = Mock(return_value={"test": "info"})
        mock_runtime.get_runtime_functions = Mock(return_value={"get_logs": Mock()})
        setup_manager.runtime_interface = mock_runtime
        
        status = setup_manager.get_integration_status()
        
        assert status["configured"] is True
        assert "provider_info" in status
        assert "runtime_functions" in status
