"""Dynamic provider factory for creating integration instances."""

import importlib
from typing import Dict, Optional, Any, Type

from ..integrations.base.log_provider import LogProvider
from ..integrations.base.metrics_provider import MetricsProvider
from ..integrations.base.code_provider import CodeProvider
from ..integrations.base.llm_provider import LLMProvider
from ..integrations.base.runbook_provider import RunbookProvider


class ProviderFactory:
    """Factory for creating provider instances dynamically based on configuration.
    
    Creates the appropriate provider instances based on the selected integrations
    and their configurations, handling dynamic imports and initialization.
    """

    def __init__(self):
        """Initialize the provider factory."""
        self._provider_cache = {}

    def create_logs_provider(self, provider_type: str, config: Dict[str, Any]) -> LogProvider:
        """Create logs provider instance.
        
        Args:
            provider_type: Type of logs provider (aws_cloudwatch, azure_monitor, gcp_logging)
            config: Provider configuration
            
        Returns:
            Initialized logs provider instance
            
        Raises:
            ValueError: If provider type is not supported
            ImportError: If required dependencies are not available
            ConnectionError: If provider initialization fails
        """
        provider_mapping = {
            "aws_cloudwatch": ("oncall_agent.integrations.aws", "CloudWatchLogsProvider"),
            "azure_monitor": ("oncall_agent.integrations.azure", "AzureMonitorLogsProvider"),
            "gcp_logging": ("oncall_agent.integrations.gcp", "GCPCloudLoggingProvider")
        }
        
        if provider_type not in provider_mapping:
            raise ValueError(f"Unsupported logs provider: {provider_type}")
        
        module_name, class_name = provider_mapping[provider_type]
        provider_class = self._import_provider_class(module_name, class_name)
        
        try:
            return provider_class(config)
        except Exception as e:
            raise ConnectionError(f"Failed to initialize {provider_type} logs provider: {e}")

    def create_metrics_provider(
        self,
        provider_type: str,
        config: Dict[str, Any]
    ) -> Optional[MetricsProvider]:
        """Create metrics provider instance.
        
        Args:
            provider_type: Type of metrics provider
            config: Provider configuration
            
        Returns:
            Initialized metrics provider instance or None if not configured
            
        Raises:
            ValueError: If provider type is not supported
            ImportError: If required dependencies are not available
            ConnectionError: If provider initialization fails
        """
        if not provider_type:
            return None
        
        provider_mapping = {
            "aws_cloudwatch": ("oncall_agent.integrations.aws", "CloudWatchMetricsProvider"),
            "azure_monitor": ("oncall_agent.integrations.azure", "AzureMonitorMetricsProvider"),
            "gcp_monitoring": ("oncall_agent.integrations.gcp", "GCPCloudMonitoringProvider")
        }
        
        if provider_type not in provider_mapping:
            raise ValueError(f"Unsupported metrics provider: {provider_type}")
        
        module_name, class_name = provider_mapping[provider_type]
        provider_class = self._import_provider_class(module_name, class_name)
        
        try:
            return provider_class(config)
        except Exception as e:
            raise ConnectionError(f"Failed to initialize {provider_type} metrics provider: {e}")

    def create_code_provider(self, provider_type: str, config: Dict[str, Any]) -> CodeProvider:
        """Create code provider instance.
        
        Args:
            provider_type: Type of code provider (currently only github)
            config: Provider configuration
            
        Returns:
            Initialized code provider instance
            
        Raises:
            ValueError: If provider type is not supported
            ImportError: If required dependencies are not available
            ConnectionError: If provider initialization fails
        """
        provider_mapping = {
            "github": ("oncall_agent.integrations.github", "GitHubRepositoryProvider")
        }
        
        if provider_type not in provider_mapping:
            raise ValueError(f"Unsupported code provider: {provider_type}")
        
        module_name, class_name = provider_mapping[provider_type]
        provider_class = self._import_provider_class(module_name, class_name)
        
        try:
            return provider_class(config)
        except Exception as e:
            raise ConnectionError(f"Failed to initialize {provider_type} code provider: {e}")

    def create_llm_provider(self, provider_type: str, config: Dict[str, Any]) -> LLMProvider:
        """Create LLM provider instance.
        
        Args:
            provider_type: Type of LLM provider (openai, anthropic, ollama)
            config: Provider configuration
            
        Returns:
            Initialized LLM provider instance
            
        Raises:
            ValueError: If provider type is not supported
            ImportError: If required dependencies are not available
            ConnectionError: If provider initialization fails
        """
        provider_mapping = {
            "openai": ("oncall_agent.integrations.llm", "OpenAIProvider"),
            "anthropic": ("oncall_agent.integrations.llm", "AnthropicProvider"),
            "ollama": ("oncall_agent.integrations.llm", "OllamaProvider"),
            "huggingface": ("oncall_agent.integrations.llm", "HuggingFaceProvider"),
            "gemini": ("oncall_agent.integrations.llm", "GeminiProvider"),
            "azure_openai": ("oncall_agent.integrations.llm", "AzureOpenAIProvider"),
            "bedrock": ("oncall_agent.integrations.llm", "BedrockProvider")
        }
        
        if provider_type not in provider_mapping:
            raise ValueError(f"Unsupported LLM provider: {provider_type}")
        
        module_name, class_name = provider_mapping[provider_type]
        provider_class = self._import_provider_class(module_name, class_name)
        
        try:
            return provider_class(config)
        except Exception as e:
            raise ConnectionError(f"Failed to initialize {provider_type} LLM provider: {e}")

    def create_runbook_provider(
        self,
        provider_type: str,
        config: Dict[str, Any]
    ) -> Optional[RunbookProvider]:
        """Create runbook provider instance.
        
        Args:
            provider_type: Type of runbook provider
            config: Provider configuration
            
        Returns:
            Initialized runbook provider instance or None if not configured
            
        Raises:
            ValueError: If provider type is not supported
            ImportError: If required dependencies are not available
            ConnectionError: If provider initialization fails
        """
        if not provider_type:
            return None
        
        provider_mapping = {
            "pdf": ("oncall_agent.integrations.runbooks", "PDFRunbookProvider"),
            "markdown": ("oncall_agent.integrations.runbooks", "MarkdownRunbookProvider"),
            "docx": ("oncall_agent.integrations.runbooks", "DocxRunbookProvider"),
            "web_link": ("oncall_agent.integrations.runbooks", "WebRunbookProvider"),
            "unified": ("oncall_agent.integrations.runbooks", "UnifiedRunbookProvider")
        }
        
        if provider_type not in provider_mapping:
            raise ValueError(f"Unsupported runbook provider: {provider_type}")
        
        module_name, class_name = provider_mapping[provider_type]
        provider_class = self._import_provider_class(module_name, class_name)
        
        try:
            return provider_class(config)
        except Exception as e:
            raise ConnectionError(f"Failed to initialize {provider_type} runbook provider: {e}")

    def _import_provider_class(self, module_name: str, class_name: str) -> Type:
        """Dynamically import a provider class.
        
        Args:
            module_name: Module name to import from
            class_name: Class name to import
            
        Returns:
            Imported class
            
        Raises:
            ImportError: If module or class cannot be imported
        """
        cache_key = f"{module_name}.{class_name}"
        
        if cache_key in self._provider_cache:
            return self._provider_cache[cache_key]
        
        try:
            module = importlib.import_module(module_name)
            provider_class = getattr(module, class_name)
            self._provider_cache[cache_key] = provider_class
            return provider_class
        except ImportError as e:
            raise ImportError(f"Failed to import {class_name} from {module_name}: {e}")
        except AttributeError as e:
            raise ImportError(f"Class {class_name} not found in {module_name}: {e}")

    async def validate_provider_connectivity(
        self,
        provider_instance: Any,
        provider_type: str
    ) -> Dict[str, Any]:
        """Validate that a provider instance can connect successfully.
        
        Args:
            provider_instance: Provider instance to validate
            provider_type: Type of provider for context
            
        Returns:
            Dictionary containing validation results
        """
        validation_result = {
            "provider_type": provider_type,
            "healthy": False,
            "error": None,
            "latency_ms": None,
            "additional_info": {}
        }
        
        try:
            import time
            start_time = time.time()
            
            # Call provider-specific health check
            if hasattr(provider_instance, 'health_check'):
                health_result = await provider_instance.health_check()
                
                end_time = time.time()
                validation_result["latency_ms"] = (end_time - start_time) * 1000
                
                if isinstance(health_result, bool):
                    validation_result["healthy"] = health_result
                elif isinstance(health_result, dict):
                    validation_result["healthy"] = health_result.get("healthy", False)
                    validation_result["additional_info"] = health_result
                else:
                    validation_result["healthy"] = True
            else:
                # If no health check method, assume healthy if instantiation succeeded
                validation_result["healthy"] = True
                validation_result["latency_ms"] = (time.time() - start_time) * 1000
                
        except Exception as e:
            validation_result["error"] = str(e)
            validation_result["healthy"] = False
        
        return validation_result

    def get_provider_capabilities(self, provider_type: str, provider_name: str) -> Dict[str, Any]:
        """Get capabilities information for a provider.
        
        Args:
            provider_type: Type of provider
            provider_name: Name of specific provider
            
        Returns:
            Dictionary containing provider capabilities
        """
        capabilities = {
            "provider_type": provider_type,
            "provider_name": provider_name,
            "features": [],
            "limitations": [],
            "dependencies": []
        }
        
        # Define capabilities by provider
        if provider_type == "logs":
            capabilities["features"] = ["Log search", "Pattern matching", "Time-based filtering"]
            
            if provider_name == "aws_cloudwatch":
                capabilities["features"].extend(["CloudWatch Insights queries", "Log group discovery"])
                capabilities["limitations"] = ["AWS region-specific", "CloudWatch Logs pricing"]
                capabilities["dependencies"] = ["boto3"]
                
            elif provider_name == "azure_monitor":
                capabilities["features"].extend(["KQL queries", "Application Insights integration"])
                capabilities["limitations"] = ["Requires Log Analytics workspace", "Azure-specific"]
                capabilities["dependencies"] = ["azure-monitor-query", "azure-identity"]
                
            elif provider_name == "gcp_logging":
                capabilities["features"].extend(["Cloud Logging filters", "Resource-based filtering"])
                capabilities["limitations"] = ["GCP project-specific", "Cloud Logging quotas"]
                capabilities["dependencies"] = ["google-cloud-logging"]
        
        elif provider_type == "llm":
            capabilities["features"] = ["Text generation", "Code analysis", "Incident resolution"]
            
            if provider_name == "openai":
                capabilities["features"].extend(["Function calling", "Vision (GPT-4V)", "High-quality reasoning"])
                capabilities["limitations"] = ["API costs", "Rate limits", "Internet required"]
                capabilities["dependencies"] = ["langchain-openai"]
                
            elif provider_name == "anthropic":
                capabilities["features"].extend(["Large context window", "Superior reasoning", "Safety-focused"])
                capabilities["limitations"] = ["API costs", "No function calling", "Internet required"]
                capabilities["dependencies"] = ["langchain-anthropic"]
                
            elif provider_name == "ollama":
                capabilities["features"].extend(["Local deployment", "Privacy", "No API costs"])
                capabilities["limitations"] = ["Local resources", "Model management", "Potentially slower"]
                capabilities["dependencies"] = ["langchain-community", "ollama server"]
        
        return capabilities
