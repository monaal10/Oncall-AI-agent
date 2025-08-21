"""Unit tests for new LLM providers (Gemini, Azure OpenAI, Bedrock)."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from oncall_agent.integrations.llm.gemini_provider import GeminiProvider
from oncall_agent.integrations.llm.azure_openai_provider import AzureOpenAIProvider
from oncall_agent.integrations.llm.bedrock_provider import BedrockProvider


class TestGeminiProvider:
    """Test cases for GeminiProvider."""

    @patch('oncall_agent.integrations.llm.gemini_provider.ChatGoogleGenerativeAI')
    def test_init_success(self, mock_chat_gemini):
        """Test Gemini provider initialization."""
        config = {
            "api_key": "test_google_api_key",
            "model_name": "gemini-pro",
            "max_tokens": 2000,
            "temperature": 0.1
        }
        
        provider = GeminiProvider(config)
        
        assert provider.config == config
        mock_chat_gemini.assert_called_once()

    def test_validate_config_missing_api_key(self):
        """Test configuration validation with missing API key."""
        config = {"model_name": "gemini-pro"}
        
        with pytest.raises(ValueError) as exc_info:
            GeminiProvider(config)
        
        assert "Google API key is required" in str(exc_info.value)

    @patch('oncall_agent.integrations.llm.gemini_provider.ChatGoogleGenerativeAI')
    def test_get_model_info(self, mock_chat_gemini):
        """Test getting model information."""
        config = {
            "api_key": "test_key",
            "model_name": "gemini-pro-vision",
            "max_tokens": 2000,
            "temperature": 0.1,
            "safety_settings": {"harassment": "block_none"}
        }
        
        provider = GeminiProvider(config)
        info = provider.get_model_info()
        
        assert info["provider"] == "gemini"
        assert info["model_name"] == "gemini-pro-vision"
        assert info["max_tokens"] == 2000
        assert info["supports_streaming"] is True
        assert info["supports_functions"] is False
        assert info["supports_vision"] is True
        assert info["safety_settings"] is True

    @patch('oncall_agent.integrations.llm.gemini_provider.ChatGoogleGenerativeAI')
    def test_get_context_window_gemini_pro(self, mock_chat_gemini):
        """Test context window calculation for Gemini Pro."""
        config = {"api_key": "test", "model_name": "gemini-pro"}
        
        provider = GeminiProvider(config)
        context_window = provider._get_context_window()
        
        assert context_window == 32768

    @patch('oncall_agent.integrations.llm.gemini_provider.ChatGoogleGenerativeAI')
    def test_get_context_window_gemini_15_pro(self, mock_chat_gemini):
        """Test context window calculation for Gemini 1.5 Pro."""
        config = {"api_key": "test", "model_name": "gemini-1.5-pro"}
        
        provider = GeminiProvider(config)
        context_window = provider._get_context_window()
        
        assert context_window == 1000000

    @pytest.mark.asyncio
    @patch('oncall_agent.integrations.llm.gemini_provider.ChatGoogleGenerativeAI')
    async def test_generate_resolution_success(self, mock_chat_gemini, sample_incident_context):
        """Test successful resolution generation."""
        mock_model = AsyncMock()
        mock_response = Mock()
        mock_response.content = "**Summary:**\nTest resolution summary\n**Resolution Steps:**\n1. Step one\n2. Step two"
        mock_model.ainvoke.return_value = mock_response
        mock_chat_gemini.return_value = mock_model
        
        provider = GeminiProvider({"api_key": "test"})
        
        resolution = await provider.generate_resolution(sample_incident_context)
        
        assert "resolution_summary" in resolution
        assert "detailed_steps" in resolution
        assert resolution["confidence_score"] == 0.8


class TestAzureOpenAIProvider:
    """Test cases for AzureOpenAIProvider."""

    @patch('oncall_agent.integrations.llm.azure_openai_provider.AzureChatOpenAI')
    def test_init_success(self, mock_azure_chat_openai):
        """Test Azure OpenAI provider initialization."""
        config = {
            "api_key": "test_azure_openai_key",
            "azure_endpoint": "https://test.openai.azure.com/",
            "deployment_name": "gpt-4-deployment",
            "api_version": "2024-02-15-preview",
            "max_tokens": 2000,
            "temperature": 0.1
        }
        
        provider = AzureOpenAIProvider(config)
        
        assert provider.config == config
        mock_azure_chat_openai.assert_called_once()

    def test_validate_config_missing_api_key(self):
        """Test configuration validation with missing API key."""
        config = {
            "azure_endpoint": "https://test.openai.azure.com/",
            "deployment_name": "gpt-4-deployment"
        }
        
        with pytest.raises(ValueError) as exc_info:
            AzureOpenAIProvider(config)
        
        assert "api_key is required" in str(exc_info.value)

    def test_validate_config_missing_endpoint(self):
        """Test configuration validation with missing endpoint."""
        config = {
            "api_key": "test_key",
            "deployment_name": "gpt-4-deployment"
        }
        
        with pytest.raises(ValueError) as exc_info:
            AzureOpenAIProvider(config)
        
        assert "azure_endpoint is required" in str(exc_info.value)

    def test_validate_config_missing_deployment(self):
        """Test configuration validation with missing deployment name."""
        config = {
            "api_key": "test_key",
            "azure_endpoint": "https://test.openai.azure.com/"
        }
        
        with pytest.raises(ValueError) as exc_info:
            AzureOpenAIProvider(config)
        
        assert "deployment_name is required" in str(exc_info.value)

    @patch('oncall_agent.integrations.llm.azure_openai_provider.AzureChatOpenAI')
    def test_get_model_info(self, mock_azure_chat_openai):
        """Test getting model information."""
        config = {
            "api_key": "test_key",
            "azure_endpoint": "https://test.openai.azure.com/",
            "deployment_name": "gpt-4-deployment",
            "api_version": "2024-02-15-preview"
        }
        
        provider = AzureOpenAIProvider(config)
        info = provider.get_model_info()
        
        assert info["provider"] == "azure_openai"
        assert info["deployment_name"] == "gpt-4-deployment"
        assert info["azure_endpoint"] == "https://test.openai.azure.com/"
        assert info["api_version"] == "2024-02-15-preview"
        assert info["supports_streaming"] is True
        assert info["supports_functions"] is True

    @patch('oncall_agent.integrations.llm.azure_openai_provider.AzureChatOpenAI')
    def test_get_context_window_gpt4(self, mock_azure_chat_openai):
        """Test context window calculation for GPT-4 deployment."""
        config = {
            "api_key": "test",
            "azure_endpoint": "https://test.openai.azure.com/",
            "deployment_name": "gpt-4-deployment"
        }
        
        provider = AzureOpenAIProvider(config)
        context_window = provider._get_context_window()
        
        assert context_window == 8192

    @patch('oncall_agent.integrations.llm.azure_openai_provider.AzureChatOpenAI')
    def test_get_context_window_gpt35_turbo(self, mock_azure_chat_openai):
        """Test context window calculation for GPT-3.5-turbo deployment."""
        config = {
            "api_key": "test",
            "azure_endpoint": "https://test.openai.azure.com/",
            "deployment_name": "gpt-35-turbo-deployment"
        }
        
        provider = AzureOpenAIProvider(config)
        context_window = provider._get_context_window()
        
        assert context_window == 4096

    @pytest.mark.asyncio
    @patch('oncall_agent.integrations.llm.azure_openai_provider.AzureChatOpenAI')
    async def test_generate_resolution_success(self, mock_azure_chat_openai, sample_incident_context):
        """Test successful resolution generation."""
        mock_model = AsyncMock()
        mock_response = Mock()
        mock_response.content = "**Summary:**\nAzure OpenAI resolution\n**Resolution Steps:**\n1. Check deployment\n2. Verify endpoint"
        mock_model.ainvoke.return_value = mock_response
        mock_azure_chat_openai.return_value = mock_model
        
        config = {
            "api_key": "test",
            "azure_endpoint": "https://test.openai.azure.com/",
            "deployment_name": "gpt-4-deployment"
        }
        
        provider = AzureOpenAIProvider(config)
        
        resolution = await provider.generate_resolution(sample_incident_context)
        
        assert "resolution_summary" in resolution
        assert "detailed_steps" in resolution
        assert resolution["confidence_score"] == 0.8


class TestBedrockProvider:
    """Test cases for BedrockProvider."""

    @patch('oncall_agent.integrations.llm.bedrock_provider.ChatBedrock')
    def test_init_success_chat_model(self, mock_chat_bedrock):
        """Test Bedrock provider initialization with chat model."""
        config = {
            "model_id": "anthropic.claude-3-sonnet-20240229-v1:0",
            "region": "us-east-1",
            "max_tokens": 2000,
            "temperature": 0.1
        }
        
        provider = BedrockProvider(config)
        
        assert provider.config == config
        mock_chat_bedrock.assert_called_once()

    @patch('oncall_agent.integrations.llm.bedrock_provider.BedrockLLM')
    def test_init_success_completion_model(self, mock_bedrock_llm):
        """Test Bedrock provider initialization with completion model."""
        config = {
            "model_id": "amazon.titan-text-lite-v1",
            "region": "us-west-2"
        }
        
        provider = BedrockProvider(config)
        
        mock_bedrock_llm.assert_called_once()

    def test_validate_config_missing_model_id(self):
        """Test configuration validation with missing model ID."""
        config = {"region": "us-east-1"}
        
        with pytest.raises(ValueError) as exc_info:
            BedrockProvider(config)
        
        assert "model_id is required" in str(exc_info.value)

    @patch('oncall_agent.integrations.llm.bedrock_provider.ChatBedrock')
    def test_get_model_info(self, mock_chat_bedrock):
        """Test getting model information."""
        config = {
            "model_id": "anthropic.claude-3-sonnet-20240229-v1:0",
            "region": "us-east-1",
            "max_tokens": 2000,
            "temperature": 0.1
        }
        
        provider = BedrockProvider(config)
        info = provider.get_model_info()
        
        assert info["provider"] == "bedrock"
        assert info["model_id"] == "anthropic.claude-3-sonnet-20240229-v1:0"
        assert info["region"] == "us-east-1"
        assert info["max_tokens"] == 2000
        assert info["supports_streaming"] is True
        assert info["supports_functions"] is False
        assert info["model_provider"] == "anthropic"

    @patch('oncall_agent.integrations.llm.bedrock_provider.ChatBedrock')
    def test_get_context_window_claude3(self, mock_chat_bedrock):
        """Test context window calculation for Claude-3."""
        config = {"model_id": "anthropic.claude-3-sonnet-20240229-v1:0"}
        
        provider = BedrockProvider(config)
        context_window = provider._get_context_window()
        
        assert context_window == 200000

    @patch('oncall_agent.integrations.llm.bedrock_provider.ChatBedrock')
    def test_get_context_window_llama2(self, mock_chat_bedrock):
        """Test context window calculation for Llama2."""
        config = {"model_id": "meta.llama2-13b-chat-v1"}
        
        provider = BedrockProvider(config)
        context_window = provider._get_context_window()
        
        assert context_window == 4096

    @patch('oncall_agent.integrations.llm.bedrock_provider.ChatBedrock')
    def test_get_model_provider_anthropic(self, mock_chat_bedrock):
        """Test model provider detection for Anthropic models."""
        config = {"model_id": "anthropic.claude-3-sonnet-20240229-v1:0"}
        
        provider = BedrockProvider(config)
        model_provider = provider._get_model_provider()
        
        assert model_provider == "anthropic"

    @patch('oncall_agent.integrations.llm.bedrock_provider.ChatBedrock')
    def test_get_model_provider_meta(self, mock_chat_bedrock):
        """Test model provider detection for Meta models."""
        config = {"model_id": "meta.llama2-13b-chat-v1"}
        
        provider = BedrockProvider(config)
        model_provider = provider._get_model_provider()
        
        assert model_provider == "meta"

    @patch('oncall_agent.integrations.llm.bedrock_provider.ChatBedrock')
    def test_get_model_provider_amazon(self, mock_chat_bedrock):
        """Test model provider detection for Amazon models."""
        config = {"model_id": "amazon.titan-text-express-v1"}
        
        provider = BedrockProvider(config)
        model_provider = provider._get_model_provider()
        
        assert model_provider == "amazon"

    @pytest.mark.asyncio
    @patch('oncall_agent.integrations.llm.bedrock_provider.ChatBedrock')
    async def test_generate_resolution_success(self, mock_chat_bedrock, sample_incident_context):
        """Test successful resolution generation."""
        mock_model = AsyncMock()
        mock_response = Mock()
        mock_response.content = "**Summary:**\nBedrock resolution\n**Steps:**\n1. Check model\n2. Verify region"
        mock_model.ainvoke.return_value = mock_response
        mock_chat_bedrock.return_value = mock_model
        
        config = {
            "model_id": "anthropic.claude-3-sonnet-20240229-v1:0",
            "region": "us-east-1"
        }
        
        provider = BedrockProvider(config)
        
        resolution = await provider.generate_resolution(sample_incident_context)
        
        assert "resolution_summary" in resolution
        assert "detailed_steps" in resolution
        assert resolution["confidence_score"] == 0.75

    @pytest.mark.asyncio
    @patch('oncall_agent.integrations.llm.bedrock_provider.ChatBedrock')
    @patch('oncall_agent.integrations.llm.bedrock_provider.boto3.client')
    @patch('oncall_agent.integrations.llm.bedrock_provider.asyncio.to_thread')
    async def test_list_available_models(self, mock_to_thread, mock_boto_client, mock_chat_bedrock):
        """Test listing available Bedrock models."""
        config = {
            "model_id": "anthropic.claude-3-sonnet-20240229-v1:0",
            "region": "us-east-1"
        }
        
        # Mock Bedrock client response
        mock_client = Mock()
        mock_boto_client.return_value = mock_client
        
        mock_response = {
            "modelSummaries": [
                {
                    "modelId": "anthropic.claude-3-sonnet-20240229-v1:0",
                    "modelName": "Claude 3 Sonnet",
                    "providerName": "Anthropic",
                    "inputModalities": ["TEXT"],
                    "outputModalities": ["TEXT"],
                    "responseStreamingSupported": True,
                    "customizationsSupported": []
                }
            ]
        }
        mock_to_thread.return_value = mock_response
        
        provider = BedrockProvider(config)
        
        models = await provider.list_available_models()
        
        assert len(models) == 1
        assert models[0]["model_id"] == "anthropic.claude-3-sonnet-20240229-v1:0"
        assert models[0]["provider_name"] == "Anthropic"
        assert models[0]["response_streaming_supported"] is True

    @pytest.mark.asyncio
    @patch('oncall_agent.integrations.llm.bedrock_provider.ChatBedrock')
    async def test_check_model_availability_success(self, mock_chat_bedrock):
        """Test model availability checking when model is available."""
        config = {
            "model_id": "anthropic.claude-3-sonnet-20240229-v1:0",
            "region": "us-east-1"
        }
        
        provider = BedrockProvider(config)
        
        # Mock list_available_models
        provider.list_available_models = AsyncMock(return_value=[
            {"model_id": "anthropic.claude-3-sonnet-20240229-v1:0", "model_name": "Claude 3 Sonnet"}
        ])
        
        availability = await provider.check_model_availability()
        
        assert availability["available"] is True
        assert availability["model_id"] == "anthropic.claude-3-sonnet-20240229-v1:0"
        assert availability["error"] is None

    @pytest.mark.asyncio
    @patch('oncall_agent.integrations.llm.bedrock_provider.ChatBedrock')
    async def test_check_model_availability_not_found(self, mock_chat_bedrock):
        """Test model availability checking when model is not available."""
        config = {
            "model_id": "nonexistent.model-v1:0",
            "region": "us-east-1"
        }
        
        provider = BedrockProvider(config)
        
        # Mock list_available_models
        provider.list_available_models = AsyncMock(return_value=[
            {"model_id": "other.model-v1:0", "model_name": "Other Model"}
        ])
        
        availability = await provider.check_model_availability()
        
        assert availability["available"] is False
        assert "not found" in availability["error"]


class TestNewLLMProvidersIntegration:
    """Integration tests for new LLM providers."""

    @pytest.fixture
    def gemini_config(self):
        """Gemini configuration for testing."""
        return {
            "api_key": "test_google_api_key",
            "model_name": "gemini-pro",
            "max_tokens": 2000,
            "temperature": 0.1
        }

    @pytest.fixture
    def azure_openai_config(self):
        """Azure OpenAI configuration for testing."""
        return {
            "api_key": "test_azure_openai_key",
            "azure_endpoint": "https://test.openai.azure.com/",
            "deployment_name": "gpt-4-deployment",
            "api_version": "2024-02-15-preview",
            "max_tokens": 2000,
            "temperature": 0.1
        }

    @pytest.fixture
    def bedrock_config(self):
        """Bedrock configuration for testing."""
        return {
            "model_id": "anthropic.claude-3-sonnet-20240229-v1:0",
            "region": "us-east-1",
            "max_tokens": 2000,
            "temperature": 0.1
        }

    @patch('oncall_agent.integrations.llm.gemini_provider.ChatGoogleGenerativeAI')
    @patch('oncall_agent.integrations.llm.azure_openai_provider.AzureChatOpenAI')
    @patch('oncall_agent.integrations.llm.bedrock_provider.ChatBedrock')
    def test_all_providers_implement_interface(
        self,
        mock_bedrock,
        mock_azure_openai,
        mock_gemini,
        gemini_config,
        azure_openai_config,
        bedrock_config
    ):
        """Test that all new providers implement the required interface."""
        providers = [
            GeminiProvider(gemini_config),
            AzureOpenAIProvider(azure_openai_config),
            BedrockProvider(bedrock_config)
        ]
        
        for provider in providers:
            # Check required methods exist
            assert hasattr(provider, 'generate_resolution')
            assert hasattr(provider, 'analyze_logs')
            assert hasattr(provider, 'analyze_code_context')
            assert hasattr(provider, 'stream_response')
            assert hasattr(provider, 'get_model_info')
            assert hasattr(provider, 'get_langchain_model')
            
            # Check model info structure
            info = provider.get_model_info()
            required_info_fields = [
                "provider", "max_tokens", "temperature", "supports_streaming",
                "supports_functions", "context_window"
            ]
            for field in required_info_fields:
                assert field in info

    @patch('oncall_agent.integrations.llm.gemini_provider.ChatGoogleGenerativeAI')
    def test_gemini_extract_section(self, mock_gemini):
        """Test section extraction for Gemini responses."""
        config = {"api_key": "test"}
        provider = GeminiProvider(config)
        
        response_text = """
        **Error Patterns:**
        - Connection timeout
        - Pool exhaustion
        
        **Severity Assessment:**
        High priority issue
        """
        
        patterns = provider._extract_list_items(response_text, "Error Patterns")
        severity = provider._extract_section(response_text, "Severity Assessment")
        
        assert "Connection timeout" in patterns
        assert "Pool exhaustion" in patterns
        assert severity == "High priority issue"

    @patch('oncall_agent.integrations.llm.azure_openai_provider.AzureChatOpenAI')
    def test_azure_openai_extract_section(self, mock_azure_openai):
        """Test section extraction for Azure OpenAI responses."""
        config = {
            "api_key": "test",
            "azure_endpoint": "https://test.openai.azure.com/",
            "deployment_name": "gpt-4-deployment"
        }
        provider = AzureOpenAIProvider(config)
        
        response_text = """
        **Potential issues:**
        - Database connection timeout
        - Connection pool exhausted
        
        **Fix suggestions:**
        Increase connection timeout and pool size
        """
        
        issues = provider._extract_list_items(response_text, "Potential issues")
        fixes = provider._extract_section(response_text, "Fix suggestions")
        
        assert "Database connection timeout" in issues
        assert "Connection pool exhausted" in issues
        assert "Increase connection timeout" in fixes

    @patch('oncall_agent.integrations.llm.bedrock_provider.ChatBedrock')
    def test_bedrock_extract_section_flexible(self, mock_bedrock):
        """Test flexible section extraction for Bedrock responses."""
        config = {"model_id": "anthropic.claude-3-sonnet-20240229-v1:0"}
        provider = BedrockProvider(config)
        
        # Test different formatting styles that Bedrock models might use
        response_text = """
        1. issues:
        - Connection problem
        - Timeout error
        
        2. fix:
        Adjust timeout settings
        """
        
        issues = provider._extract_list_items(response_text, "issues")
        fixes = provider._extract_section(response_text, "fix")
        
        assert "Connection problem" in issues
        assert "Timeout error" in issues
        assert "Adjust timeout settings" in fixes
