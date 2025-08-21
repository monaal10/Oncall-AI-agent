"""Unit tests for LLM providers."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from oncall_agent.integrations.llm.openai_provider import OpenAIProvider
from oncall_agent.integrations.llm.anthropic_provider import AnthropicProvider
from oncall_agent.integrations.llm.ollama_provider import OllamaProvider
from oncall_agent.integrations.llm.huggingface_provider import HuggingFaceProvider
from oncall_agent.integrations.llm.manager import LLMManager


class TestOpenAIProvider:
    """Test cases for OpenAIProvider."""

    @patch('oncall_agent.integrations.llm.openai_provider.ChatOpenAI')
    def test_init_with_gpt4(self, mock_chat_openai):
        """Test OpenAI provider initialization with GPT-4."""
        config = {
            "api_key": "test_key",
            "model": "gpt-4",
            "max_tokens": 2000,
            "temperature": 0.1
        }
        
        provider = OpenAIProvider(config)
        
        assert provider.config == config
        mock_chat_openai.assert_called_once()

    @patch('oncall_agent.integrations.llm.openai_provider.OpenAI')
    def test_init_with_completion_model(self, mock_openai):
        """Test OpenAI provider initialization with completion model."""
        config = {
            "api_key": "test_key",
            "model": "text-davinci-003",
            "max_tokens": 2000
        }
        
        provider = OpenAIProvider(config)
        
        mock_openai.assert_called_once()

    def test_validate_config_missing_api_key(self):
        """Test configuration validation with missing API key."""
        config = {"model": "gpt-4"}
        
        with pytest.raises(ValueError) as exc_info:
            OpenAIProvider(config)
        
        assert "api_key is required" in str(exc_info.value)

    @patch('oncall_agent.integrations.llm.openai_provider.ChatOpenAI')
    def test_get_model_info(self, mock_chat_openai):
        """Test getting model information."""
        config = {
            "api_key": "test_key",
            "model": "gpt-4",
            "max_tokens": 2000,
            "temperature": 0.1
        }
        
        provider = OpenAIProvider(config)
        info = provider.get_model_info()
        
        assert info["provider"] == "openai"
        assert info["model_name"] == "gpt-4"
        assert info["max_tokens"] == 2000
        assert info["supports_streaming"] is True
        assert info["supports_functions"] is True

    def test_get_context_window_gpt4(self):
        """Test context window calculation for GPT-4."""
        config = {"api_key": "test", "model": "gpt-4"}
        
        with patch('oncall_agent.integrations.llm.openai_provider.ChatOpenAI'):
            provider = OpenAIProvider(config)
            context_window = provider._get_context_window()
        
        assert context_window == 8192

    def test_get_context_window_gpt4_turbo(self):
        """Test context window calculation for GPT-4 Turbo."""
        config = {"api_key": "test", "model": "gpt-4-turbo"}
        
        with patch('oncall_agent.integrations.llm.openai_provider.ChatOpenAI'):
            provider = OpenAIProvider(config)
            context_window = provider._get_context_window()
        
        assert context_window == 128000

    @patch('oncall_agent.integrations.llm.openai_provider.ChatOpenAI')
    def test_extract_section(self, mock_chat_openai):
        """Test section extraction from response."""
        config = {"api_key": "test", "model": "gpt-4"}
        provider = OpenAIProvider(config)
        
        response_text = """
        **Summary:**
        This is the summary section.
        
        **Resolution Steps:**
        1. Step one
        2. Step two
        
        **Code Changes:**
        No code changes needed.
        """
        
        summary = provider._extract_section(response_text, "Summary")
        assert summary == "This is the summary section."
        
        steps = provider._extract_section(response_text, "Resolution Steps")
        assert "Step one" in steps
        assert "Step two" in steps


class TestAnthropicProvider:
    """Test cases for AnthropicProvider."""

    @patch('oncall_agent.integrations.llm.anthropic_provider.ChatAnthropic')
    def test_init_success(self, mock_chat_anthropic):
        """Test Anthropic provider initialization."""
        config = {
            "api_key": "test_key",
            "model": "claude-3-sonnet-20240229",
            "max_tokens": 2000
        }
        
        provider = AnthropicProvider(config)
        
        assert provider.config == config
        mock_chat_anthropic.assert_called_once()

    def test_validate_config_missing_api_key(self):
        """Test configuration validation with missing API key."""
        config = {"model": "claude-3-sonnet-20240229"}
        
        with pytest.raises(ValueError) as exc_info:
            AnthropicProvider(config)
        
        assert "api_key is required" in str(exc_info.value)

    @patch('oncall_agent.integrations.llm.anthropic_provider.ChatAnthropic')
    def test_get_model_info(self, mock_chat_anthropic):
        """Test getting model information."""
        config = {
            "api_key": "test_key",
            "model": "claude-3-sonnet-20240229"
        }
        
        provider = AnthropicProvider(config)
        info = provider.get_model_info()
        
        assert info["provider"] == "anthropic"
        assert info["model_name"] == "claude-3-sonnet-20240229"
        assert info["supports_streaming"] is True
        assert info["supports_functions"] is False
        assert info["context_window"] == 200000


class TestOllamaProvider:
    """Test cases for OllamaProvider."""

    @patch('oncall_agent.integrations.llm.ollama_provider.ChatOllama')
    def test_init_success(self, mock_chat_ollama):
        """Test Ollama provider initialization."""
        config = {
            "model_name": "llama2",
            "base_url": "http://localhost:11434",
            "temperature": 0.1
        }
        
        provider = OllamaProvider(config)
        
        assert provider.config == config
        mock_chat_ollama.assert_called_once()

    def test_validate_config_missing_model_name(self):
        """Test configuration validation with missing model name."""
        config = {"base_url": "http://localhost:11434"}
        
        with pytest.raises(ValueError) as exc_info:
            OllamaProvider(config)
        
        assert "model_name is required" in str(exc_info.value)

    @patch('oncall_agent.integrations.llm.ollama_provider.ChatOllama')
    def test_get_model_info(self, mock_chat_ollama):
        """Test getting model information."""
        config = {
            "model_name": "llama2",
            "base_url": "http://localhost:11434"
        }
        
        provider = OllamaProvider(config)
        info = provider.get_model_info()
        
        assert info["provider"] == "ollama"
        assert info["model_name"] == "llama2"
        assert info["local_model"] is True
        assert info["supports_functions"] is False

    @patch('oncall_agent.integrations.llm.ollama_provider.ChatOllama')
    def test_get_context_window_llama2(self, mock_chat_ollama):
        """Test context window calculation for Llama2."""
        config = {"model_name": "llama2"}
        
        provider = OllamaProvider(config)
        context_window = provider._get_context_window()
        
        assert context_window == 4096

    @patch('oncall_agent.integrations.llm.ollama_provider.ChatOllama')
    def test_get_context_window_codellama(self, mock_chat_ollama):
        """Test context window calculation for CodeLlama."""
        config = {"model_name": "codellama"}
        
        provider = OllamaProvider(config)
        context_window = provider._get_context_window()
        
        assert context_window == 16384


class TestHuggingFaceProvider:
    """Test cases for HuggingFaceProvider."""

    @patch('oncall_agent.integrations.llm.huggingface_provider.HuggingFacePipeline')
    def test_init_success(self, mock_pipeline):
        """Test HuggingFace provider initialization."""
        config = {
            "model_name": "microsoft/DialoGPT-medium",
            "max_tokens": 1000,
            "temperature": 0.1
        }
        
        mock_pipeline.from_model_id.return_value = Mock()
        
        provider = HuggingFaceProvider(config)
        
        assert provider.config == config
        mock_pipeline.from_model_id.assert_called_once()

    def test_validate_config_missing_model_name(self):
        """Test configuration validation with missing model name."""
        config = {"temperature": 0.1}
        
        with pytest.raises(ValueError) as exc_info:
            HuggingFaceProvider(config)
        
        assert "model_name is required" in str(exc_info.value)

    @patch('oncall_agent.integrations.llm.huggingface_provider.HuggingFacePipeline')
    def test_get_model_info(self, mock_pipeline):
        """Test getting model information."""
        config = {
            "model_name": "microsoft/DialoGPT-medium",
            "device": "cpu",
            "api_key": "test_key"
        }
        
        mock_pipeline.from_model_id.return_value = Mock()
        
        provider = HuggingFaceProvider(config)
        info = provider.get_model_info()
        
        assert info["provider"] == "huggingface"
        assert info["model_name"] == "microsoft/DialoGPT-medium"
        assert info["device"] == "cpu"
        assert info["local_model"] is True
        assert info["hosted_inference"] is True

    @patch('oncall_agent.integrations.llm.huggingface_provider.HuggingFacePipeline')
    def test_get_context_window_gpt2(self, mock_pipeline):
        """Test context window calculation for GPT-2 based model."""
        config = {"model_name": "gpt2-medium"}
        
        mock_pipeline.from_model_id.return_value = Mock()
        
        provider = HuggingFaceProvider(config)
        context_window = provider._get_context_window()
        
        assert context_window == 1024


class TestLLMManager:
    """Test cases for LLMManager."""

    @patch('oncall_agent.integrations.llm.manager.OpenAIProvider')
    def test_init_success(self, mock_openai_provider):
        """Test LLM manager initialization."""
        config = {
            "primary_provider": {
                "type": "openai",
                "config": {"api_key": "test"}
            }
        }
        
        mock_provider_instance = Mock()
        mock_openai_provider.return_value = mock_provider_instance
        
        manager = LLMManager(config)
        
        assert manager.primary_provider == mock_provider_instance
        assert len(manager.fallback_providers) == 0

    @patch('oncall_agent.integrations.llm.manager.OpenAIProvider')
    @patch('oncall_agent.integrations.llm.manager.AnthropicProvider')
    def test_init_with_fallbacks(self, mock_anthropic_provider, mock_openai_provider):
        """Test LLM manager initialization with fallback providers."""
        config = {
            "primary_provider": {
                "type": "openai",
                "config": {"api_key": "test"}
            },
            "fallback_providers": [
                {
                    "type": "anthropic",
                    "config": {"api_key": "test"}
                }
            ]
        }
        
        mock_openai_instance = Mock()
        mock_anthropic_instance = Mock()
        mock_openai_provider.return_value = mock_openai_instance
        mock_anthropic_provider.return_value = mock_anthropic_instance
        
        manager = LLMManager(config)
        
        assert manager.primary_provider == mock_openai_instance
        assert len(manager.fallback_providers) == 1
        assert manager.fallback_providers[0] == mock_anthropic_instance

    def test_validate_config_missing_primary(self):
        """Test configuration validation with missing primary provider."""
        config = {}
        
        with pytest.raises(ValueError) as exc_info:
            LLMManager(config)
        
        assert "primary_provider configuration is required" in str(exc_info.value)

    def test_validate_config_unsupported_provider(self):
        """Test configuration validation with unsupported provider type."""
        config = {
            "primary_provider": {
                "type": "unsupported",
                "config": {}
            }
        }
        
        with pytest.raises(ValueError) as exc_info:
            LLMManager(config)
        
        assert "Unsupported provider type: unsupported" in str(exc_info.value)

    @patch('oncall_agent.integrations.llm.manager.OpenAIProvider')
    def test_get_langchain_model(self, mock_openai_provider):
        """Test getting LangChain model for LangGraph integration."""
        config = {
            "primary_provider": {
                "type": "openai",
                "config": {"api_key": "test"}
            }
        }
        
        mock_provider_instance = Mock()
        mock_langchain_model = Mock()
        mock_provider_instance.get_langchain_model.return_value = mock_langchain_model
        mock_openai_provider.return_value = mock_provider_instance
        
        manager = LLMManager(config)
        model = manager.get_langchain_model()
        
        assert model == mock_langchain_model

    @pytest.mark.asyncio
    @patch('oncall_agent.integrations.llm.manager.OpenAIProvider')
    async def test_generate_resolution_success(self, mock_openai_provider, sample_incident_context):
        """Test successful resolution generation."""
        config = {
            "primary_provider": {
                "type": "openai",
                "config": {"api_key": "test"}
            }
        }
        
        expected_resolution = {
            "resolution_summary": "Test resolution",
            "confidence_score": 0.8
        }
        
        mock_provider_instance = Mock()
        mock_provider_instance.generate_resolution = AsyncMock(return_value=expected_resolution)
        mock_openai_provider.return_value = mock_provider_instance
        
        manager = LLMManager(config)
        resolution = await manager.generate_resolution(sample_incident_context)
        
        assert resolution == expected_resolution
        mock_provider_instance.generate_resolution.assert_called_once_with(
            sample_incident_context
        )

    @pytest.mark.asyncio
    @patch('oncall_agent.integrations.llm.manager.OpenAIProvider')
    @patch('oncall_agent.integrations.llm.manager.AnthropicProvider')
    async def test_generate_resolution_with_fallback(
        self,
        mock_anthropic_provider,
        mock_openai_provider,
        sample_incident_context
    ):
        """Test resolution generation with fallback when primary fails."""
        config = {
            "primary_provider": {
                "type": "openai",
                "config": {"api_key": "test"}
            },
            "fallback_providers": [
                {
                    "type": "anthropic",
                    "config": {"api_key": "test"}
                }
            ]
        }
        
        expected_resolution = {
            "resolution_summary": "Fallback resolution",
            "confidence_score": 0.7
        }
        
        # Primary provider fails
        mock_openai_instance = Mock()
        mock_openai_instance.generate_resolution = AsyncMock(
            side_effect=Exception("Primary failed")
        )
        mock_openai_provider.return_value = mock_openai_instance
        
        # Fallback provider succeeds
        mock_anthropic_instance = Mock()
        mock_anthropic_instance.generate_resolution = AsyncMock(return_value=expected_resolution)
        mock_anthropic_provider.return_value = mock_anthropic_instance
        
        manager = LLMManager(config)
        resolution = await manager.generate_resolution(sample_incident_context)
        
        assert resolution == expected_resolution
        mock_anthropic_instance.generate_resolution.assert_called_once()

    @pytest.mark.asyncio
    @patch('oncall_agent.integrations.llm.manager.OpenAIProvider')
    async def test_generate_resolution_all_providers_fail(
        self,
        mock_openai_provider,
        sample_incident_context
    ):
        """Test resolution generation when all providers fail."""
        config = {
            "primary_provider": {
                "type": "openai",
                "config": {"api_key": "test"}
            }
        }
        
        mock_provider_instance = Mock()
        mock_provider_instance.generate_resolution = AsyncMock(
            side_effect=Exception("Provider failed")
        )
        mock_openai_provider.return_value = mock_provider_instance
        
        manager = LLMManager(config)
        
        with pytest.raises(ConnectionError) as exc_info:
            await manager.generate_resolution(sample_incident_context)
        
        assert "All LLM providers failed" in str(exc_info.value)

    @pytest.mark.asyncio
    @patch('oncall_agent.integrations.llm.manager.OpenAIProvider')
    async def test_health_check(self, mock_openai_provider):
        """Test health check functionality."""
        config = {
            "primary_provider": {
                "type": "openai",
                "config": {"api_key": "test"}
            }
        }
        
        mock_provider_instance = Mock()
        mock_provider_instance.health_check = AsyncMock(return_value={
            "healthy": True,
            "latency": 100
        })
        mock_openai_provider.return_value = mock_provider_instance
        
        manager = LLMManager(config)
        health_results = await manager.health_check()
        
        assert "primary_provider" in health_results
        assert health_results["primary_provider"]["healthy"] is True
        assert "fallback_providers" in health_results

    @patch('oncall_agent.integrations.llm.manager.OpenAIProvider')
    def test_get_provider_info(self, mock_openai_provider):
        """Test getting provider information."""
        config = {
            "primary_provider": {
                "type": "openai",
                "config": {"api_key": "test"}
            }
        }
        
        mock_provider_instance = Mock()
        mock_provider_instance.get_model_info.return_value = {
            "provider": "openai",
            "model_name": "gpt-4"
        }
        mock_openai_provider.return_value = mock_provider_instance
        
        manager = LLMManager(config)
        info = manager.get_provider_info()
        
        assert "primary_provider" in info
        assert "fallback_providers" in info
        assert info["primary_provider"]["provider"] == "openai"
