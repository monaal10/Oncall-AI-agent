"""Integration registry for dynamic provider discovery and selection."""

import os
from typing import Dict, List, Optional, Any, Set
from datetime import datetime


class IntegrationRegistry:
    """Discovers and validates available integrations based on user configuration.
    
    Analyzes user-provided credentials and configuration to determine which
    integrations can be activated and creates the optimal integration setup.
    """

    def __init__(self):
        """Initialize the integration registry."""
        self.supported_integrations = {
            "logs_providers": {
                "aws_cloudwatch": {
                    "required_config": ["region"],
                    "optional_config": ["access_key_id", "secret_access_key", "session_token", "profile_name"],
                    "env_vars": ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION"],
                    "class_path": "oncall_agent.integrations.aws.CloudWatchLogsProvider"
                },
                "azure_monitor": {
                    "required_config": ["subscription_id", "workspace_id"],
                    "optional_config": ["tenant_id", "client_id", "client_secret"],
                    "env_vars": ["AZURE_SUBSCRIPTION_ID", "AZURE_WORKSPACE_ID", "AZURE_CLIENT_ID"],
                    "class_path": "oncall_agent.integrations.azure.AzureMonitorLogsProvider"
                },
                "gcp_logging": {
                    "required_config": ["project_id"],
                    "optional_config": ["credentials_path"],
                    "env_vars": ["GCP_PROJECT_ID", "GOOGLE_APPLICATION_CREDENTIALS"],
                    "class_path": "oncall_agent.integrations.gcp.GCPCloudLoggingProvider"
                }
            },
            "metrics_providers": {
                "aws_cloudwatch": {
                    "required_config": ["region"],
                    "optional_config": ["access_key_id", "secret_access_key", "session_token"],
                    "env_vars": ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION"],
                    "class_path": "oncall_agent.integrations.aws.CloudWatchMetricsProvider"
                },
                "azure_monitor": {
                    "required_config": ["subscription_id"],
                    "optional_config": ["tenant_id", "client_id", "client_secret"],
                    "env_vars": ["AZURE_SUBSCRIPTION_ID", "AZURE_CLIENT_ID"],
                    "class_path": "oncall_agent.integrations.azure.AzureMonitorMetricsProvider"
                },
                "gcp_monitoring": {
                    "required_config": ["project_id"],
                    "optional_config": ["credentials_path"],
                    "env_vars": ["GCP_PROJECT_ID", "GOOGLE_APPLICATION_CREDENTIALS"],
                    "class_path": "oncall_agent.integrations.gcp.GCPCloudMonitoringProvider"
                }
            },
            "code_providers": {
                "github": {
                    "required_config": ["token", "repositories"],
                    "optional_config": ["base_url"],
                    "env_vars": ["GITHUB_TOKEN"],
                    "class_path": "oncall_agent.integrations.github.GitHubRepositoryProvider"
                }
            },
            "llm_providers": {
                "openai": {
                    "required_config": ["api_key"],
                    "optional_config": ["model", "max_tokens", "temperature", "base_url", "organization"],
                    "env_vars": ["OPENAI_API_KEY"],
                    "class_path": "oncall_agent.integrations.llm.OpenAIProvider"
                },
                "anthropic": {
                    "required_config": ["api_key"],
                    "optional_config": ["model", "max_tokens", "temperature"],
                    "env_vars": ["ANTHROPIC_API_KEY"],
                    "class_path": "oncall_agent.integrations.llm.AnthropicProvider"
                },
                "ollama": {
                    "required_config": ["model_name"],
                    "optional_config": ["base_url", "temperature", "timeout", "use_chat_model"],
                    "env_vars": ["OLLAMA_BASE_URL"],
                    "class_path": "oncall_agent.integrations.llm.OllamaProvider"
                },
                "huggingface": {
                    "required_config": ["model_name"],
                    "optional_config": ["api_key", "max_tokens", "temperature", "device", "model_kwargs", "pipeline_kwargs", "use_chat_model"],
                    "env_vars": ["HUGGINGFACE_API_KEY"],
                    "class_path": "oncall_agent.integrations.llm.HuggingFaceProvider"
                },
                "gemini": {
                    "required_config": ["api_key"],
                    "optional_config": ["model_name", "max_tokens", "temperature", "timeout", "safety_settings"],
                    "env_vars": ["GOOGLE_API_KEY"],
                    "class_path": "oncall_agent.integrations.llm.GeminiProvider"
                },
                "azure_openai": {
                    "required_config": ["api_key", "azure_endpoint", "deployment_name"],
                    "optional_config": ["api_version", "max_tokens", "temperature", "timeout"],
                    "env_vars": ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"],
                    "class_path": "oncall_agent.integrations.llm.AzureOpenAIProvider"
                },
                "bedrock": {
                    "required_config": ["model_id"],
                    "optional_config": ["region", "aws_access_key_id", "aws_secret_access_key", "aws_session_token", "max_tokens", "temperature", "timeout"],
                    "env_vars": ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION"],
                    "class_path": "oncall_agent.integrations.llm.BedrockProvider"
                }
            },
            "runbook_providers": {
                "pdf": {
                    "required_config": ["runbook_directory"],
                    "optional_config": ["recursive", "cache_enabled"],
                    "env_vars": [],
                    "class_path": "oncall_agent.integrations.runbooks.PDFRunbookProvider"
                },
                "markdown": {
                    "required_config": ["runbook_directory"],
                    "optional_config": ["recursive", "file_extensions", "cache_enabled"],
                    "env_vars": [],
                    "class_path": "oncall_agent.integrations.runbooks.MarkdownRunbookProvider"
                },
                "docx": {
                    "required_config": ["runbook_directory"],
                    "optional_config": ["recursive", "cache_enabled"],
                    "env_vars": [],
                    "class_path": "oncall_agent.integrations.runbooks.DocxRunbookProvider"
                },
                "web_link": {
                    "required_config": ["base_urls"],
                    "optional_config": ["timeout", "max_pages", "cache_ttl", "user_agent", "headers"],
                    "env_vars": [],
                    "class_path": "oncall_agent.integrations.runbooks.WebRunbookProvider"
                },
                "unified": {
                    "required_config": ["providers"],
                    "optional_config": ["default_search_limit"],
                    "env_vars": [],
                    "class_path": "oncall_agent.integrations.runbooks.UnifiedRunbookProvider"
                }
            }
        }

    async def discover_integrations(self, user_config: Dict[str, Any]) -> Dict[str, Any]:
        """Discover available integrations based on user configuration and environment.
        
        Args:
            user_config: User-provided configuration dictionary
            
        Returns:
            Dictionary containing:
            - available_integrations: Dict of available integration types
            - selected_integrations: Recommended integration selection
            - missing_requirements: List of missing required configurations
            - validation_results: Detailed validation results for each provider
            
        Raises:
            ValueError: If no valid integrations can be discovered
        """
        discovery_results = {
            "available_integrations": {},
            "selected_integrations": {},
            "missing_requirements": [],
            "validation_results": {}
        }
        
        # Discover each integration type
        discovery_results["available_integrations"]["logs"] = await self._discover_logs_providers(user_config)
        discovery_results["available_integrations"]["metrics"] = await self._discover_metrics_providers(user_config)
        discovery_results["available_integrations"]["code"] = await self._discover_code_providers(user_config)
        discovery_results["available_integrations"]["llm"] = await self._discover_llm_providers(user_config)
        discovery_results["available_integrations"]["runbooks"] = await self._discover_runbook_providers(user_config)
        
        # Validate minimum requirements
        validation_errors = self._validate_minimum_requirements(discovery_results["available_integrations"])
        if validation_errors:
            discovery_results["missing_requirements"] = validation_errors
            raise ValueError(f"Missing required integrations: {', '.join(validation_errors)}")
        
        # Select optimal integrations
        discovery_results["selected_integrations"] = self._select_optimal_integrations(
            discovery_results["available_integrations"],
            user_config.get("preferences", {})
        )
        
        return discovery_results

    async def _discover_logs_providers(self, user_config: Dict[str, Any]) -> List[str]:
        """Discover available logs providers.
        
        Args:
            user_config: User configuration
            
        Returns:
            List of available logs provider names
        """
        available_providers = []
        
        # Check AWS CloudWatch
        if self._check_aws_credentials(user_config):
            available_providers.append("aws_cloudwatch")
        
        # Check Azure Monitor
        if self._check_azure_credentials(user_config):
            available_providers.append("azure_monitor")
        
        # Check GCP Cloud Logging
        if self._check_gcp_credentials(user_config):
            available_providers.append("gcp_logging")
        
        return available_providers

    async def _discover_metrics_providers(self, user_config: Dict[str, Any]) -> List[str]:
        """Discover available metrics providers.
        
        Args:
            user_config: User configuration
            
        Returns:
            List of available metrics provider names
        """
        available_providers = []
        
        # Metrics providers typically match logs providers
        if self._check_aws_credentials(user_config):
            available_providers.append("aws_cloudwatch")
        
        if self._check_azure_credentials(user_config):
            available_providers.append("azure_monitor")
        
        if self._check_gcp_credentials(user_config):
            available_providers.append("gcp_monitoring")
        
        return available_providers

    async def _discover_code_providers(self, user_config: Dict[str, Any]) -> List[str]:
        """Discover available code providers.
        
        Args:
            user_config: User configuration
            
        Returns:
            List of available code provider names
        """
        available_providers = []
        
        # Check GitHub
        if self._check_github_credentials(user_config):
            available_providers.append("github")
        
        return available_providers

    async def _discover_llm_providers(self, user_config: Dict[str, Any]) -> List[str]:
        """Discover available LLM providers.
        
        Args:
            user_config: User configuration
            
        Returns:
            List of available LLM provider names
        """
        available_providers = []
        
        # Check OpenAI
        if self._check_openai_credentials(user_config):
            available_providers.append("openai")
        
        # Check Anthropic
        if self._check_anthropic_credentials(user_config):
            available_providers.append("anthropic")
        
        # Check Ollama
        if self._check_ollama_availability(user_config):
            available_providers.append("ollama")
        
        # Check HuggingFace
        if self._check_huggingface_availability(user_config):
            available_providers.append("huggingface")
        
        # Check Gemini
        if self._check_gemini_credentials(user_config):
            available_providers.append("gemini")
        
        # Check Azure OpenAI
        if self._check_azure_openai_credentials(user_config):
            available_providers.append("azure_openai")
        
        # Check Bedrock
        if self._check_bedrock_credentials(user_config):
            available_providers.append("bedrock")
        
        return available_providers

    async def _discover_runbook_providers(self, user_config: Dict[str, Any]) -> List[str]:
        """Discover available runbook providers.
        
        Args:
            user_config: User configuration
            
        Returns:
            List of available runbook provider names
        """
        available_providers = []
        
        # Check for runbook directories
        runbook_config = user_config.get("runbooks", {})
        
        if "directory" in runbook_config and os.path.exists(runbook_config["directory"]):
            # Auto-detect runbook types in directory
            directory = runbook_config["directory"]
            
            if self._has_files_with_extensions(directory, [".pdf"]):
                available_providers.append("pdf")
            
            if self._has_files_with_extensions(directory, [".md", ".markdown", ".txt"]):
                available_providers.append("markdown")
            
            if self._has_files_with_extensions(directory, [".docx"]):
                available_providers.append("docx")
        
        # Check for web runbooks
        if "web_urls" in runbook_config and runbook_config["web_urls"]:
            available_providers.append("web_link")
        
        # If multiple types available, suggest unified provider
        if len(available_providers) > 1:
            available_providers = ["unified"]
        
        return available_providers

    def _check_aws_credentials(self, user_config: Dict[str, Any]) -> bool:
        """Check if AWS credentials are available."""
        aws_config = user_config.get("aws", {})
        
        # Check explicit credentials
        if "access_key_id" in aws_config and "secret_access_key" in aws_config and "region" in aws_config:
            return True
        
        # Check environment variables
        if all(var in os.environ for var in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]):
            if "AWS_DEFAULT_REGION" in os.environ or "region" in aws_config:
                return True
        
        # Check for AWS profile or IAM role (simplified check)
        if "region" in aws_config:
            # Assume IAM role or profile is available if region is specified
            return True
        
        return False

    def _check_azure_credentials(self, user_config: Dict[str, Any]) -> bool:
        """Check if Azure credentials are available."""
        azure_config = user_config.get("azure", {})
        
        # Check for required subscription and workspace
        if "subscription_id" not in azure_config:
            return False
        
        # For logs, workspace_id is required
        if "workspace_id" not in azure_config:
            return False
        
        # Check explicit service principal credentials
        if all(key in azure_config for key in ["tenant_id", "client_id", "client_secret"]):
            return True
        
        # Check environment variables
        if all(var in os.environ for var in ["AZURE_CLIENT_ID", "AZURE_CLIENT_SECRET", "AZURE_TENANT_ID"]):
            return True
        
        # Check for default credentials (Azure CLI, managed identity)
        # This is a simplified check - in practice, you might want to test actual connectivity
        return True

    def _check_gcp_credentials(self, user_config: Dict[str, Any]) -> bool:
        """Check if GCP credentials are available."""
        gcp_config = user_config.get("gcp", {})
        
        # Check for required project_id
        if "project_id" not in gcp_config:
            return False
        
        # Check explicit service account file
        if "credentials_path" in gcp_config and os.path.exists(gcp_config["credentials_path"]):
            return True
        
        # Check environment variable
        if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
            return os.path.exists(os.environ["GOOGLE_APPLICATION_CREDENTIALS"])
        
        # Check for default credentials (gcloud CLI, service account on GCE)
        # This is a simplified check
        return True

    def _check_github_credentials(self, user_config: Dict[str, Any]) -> bool:
        """Check if GitHub credentials are available."""
        github_config = user_config.get("github", {})
        
        # Check explicit token
        if "token" in github_config:
            return True
        
        # Check environment variable
        if "GITHUB_TOKEN" in os.environ:
            return True
        
        return False

    def _check_openai_credentials(self, user_config: Dict[str, Any]) -> bool:
        """Check if OpenAI credentials are available."""
        openai_config = user_config.get("openai", {})
        
        # Check explicit API key
        if "api_key" in openai_config:
            return True
        
        # Check environment variable
        if "OPENAI_API_KEY" in os.environ:
            return True
        
        return False

    def _check_anthropic_credentials(self, user_config: Dict[str, Any]) -> bool:
        """Check if Anthropic credentials are available."""
        anthropic_config = user_config.get("anthropic", {})
        
        # Check explicit API key
        if "api_key" in anthropic_config:
            return True
        
        # Check environment variable
        if "ANTHROPIC_API_KEY" in os.environ:
            return True
        
        return False

    def _check_ollama_availability(self, user_config: Dict[str, Any]) -> bool:
        """Check if Ollama is available."""
        ollama_config = user_config.get("ollama", {})
        
        # Check if model_name is specified
        if "model_name" not in ollama_config:
            return False
        
        # TODO: Could add actual connectivity check to Ollama server
        # For now, assume it's available if model_name is specified
        return True

    def _check_huggingface_availability(self, user_config: Dict[str, Any]) -> bool:
        """Check if HuggingFace is available."""
        huggingface_config = user_config.get("huggingface", {})
        
        # Check if model_name is specified
        if "model_name" not in huggingface_config:
            return False
        
        # HuggingFace doesn't require API key for many models
        return True

    def _check_gemini_credentials(self, user_config: Dict[str, Any]) -> bool:
        """Check if Gemini credentials are available."""
        gemini_config = user_config.get("gemini", {})
        
        # Check explicit API key
        if "api_key" in gemini_config:
            return True
        
        # Check environment variable
        if "GOOGLE_API_KEY" in os.environ:
            return True
        
        return False

    def _check_azure_openai_credentials(self, user_config: Dict[str, Any]) -> bool:
        """Check if Azure OpenAI credentials are available."""
        azure_openai_config = user_config.get("azure_openai", {})
        
        # Check required fields
        required_fields = ["api_key", "azure_endpoint", "deployment_name"]
        has_explicit_config = all(field in azure_openai_config for field in required_fields)
        
        if has_explicit_config:
            return True
        
        # Check environment variables
        has_env_config = all(
            var in os.environ 
            for var in ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"]
        ) and "deployment_name" in azure_openai_config
        
        return has_env_config

    def _check_bedrock_credentials(self, user_config: Dict[str, Any]) -> bool:
        """Check if Bedrock credentials are available."""
        bedrock_config = user_config.get("bedrock", {})
        
        # Check if model_id is specified
        if "model_id" not in bedrock_config:
            return False
        
        # Check explicit AWS credentials
        has_explicit_creds = all(
            key in bedrock_config 
            for key in ["aws_access_key_id", "aws_secret_access_key"]
        )
        
        if has_explicit_creds:
            return True
        
        # Check environment variables
        has_env_creds = all(
            var in os.environ 
            for var in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
        )
        
        if has_env_creds:
            return True
        
        # Assume IAM role is available if model_id is specified but no explicit creds
        return True

    def _has_files_with_extensions(self, directory: str, extensions: List[str]) -> bool:
        """Check if directory contains files with specified extensions."""
        try:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in extensions):
                        return True
            return False
        except Exception:
            return False

    def _validate_minimum_requirements(self, available_integrations: Dict[str, List[str]]) -> List[str]:
        """Validate that minimum required integrations are available.
        
        Args:
            available_integrations: Dictionary of available integrations by type
            
        Returns:
            List of missing required integration types
        """
        missing_requirements = []
        
        # Required: At least one logs provider
        if not available_integrations.get("logs"):
            missing_requirements.append("logs provider (AWS CloudWatch, Azure Monitor, or GCP Logging)")
        
        # Required: At least one LLM provider
        if not available_integrations.get("llm"):
            missing_requirements.append("LLM provider (OpenAI, Anthropic, or Ollama)")
        
        # Required: GitHub integration
        if not available_integrations.get("code"):
            missing_requirements.append("GitHub integration (GITHUB_TOKEN required)")
        
        return missing_requirements

    def _select_optimal_integrations(
        self,
        available_integrations: Dict[str, List[str]],
        user_preferences: Dict[str, Any]
    ) -> Dict[str, str]:
        """Select the optimal integration combination.
        
        Args:
            available_integrations: Available integrations by type
            user_preferences: User preferences for provider selection
            
        Returns:
            Dictionary mapping integration types to selected providers
        """
        selected = {}
        
        # Select logs provider
        logs_options = available_integrations.get("logs", [])
        selected["logs_provider"] = self._select_logs_provider(logs_options, user_preferences)
        
        # Select metrics provider (match logs provider if possible)
        metrics_options = available_integrations.get("metrics", [])
        selected["metrics_provider"] = self._select_metrics_provider(
            metrics_options, 
            selected["logs_provider"],
            user_preferences
        )
        
        # Select code provider (currently only GitHub)
        code_options = available_integrations.get("code", [])
        if code_options:
            selected["code_provider"] = code_options[0]
        
        # Select LLM provider
        llm_options = available_integrations.get("llm", [])
        selected["llm_provider"] = self._select_llm_provider(llm_options, user_preferences)
        
        # Select runbook provider (optional)
        runbook_options = available_integrations.get("runbooks", [])
        if runbook_options:
            selected["runbook_provider"] = runbook_options[0]
        
        return selected

    def _select_logs_provider(self, options: List[str], preferences: Dict[str, Any]) -> str:
        """Select the best logs provider based on options and preferences."""
        if not options:
            raise ValueError("No logs providers available")
        
        # Priority order based on preferences or defaults
        preferred_cloud = preferences.get("preferred_cloud_provider")
        if preferred_cloud:
            cloud_mapping = {
                "aws": "aws_cloudwatch",
                "azure": "azure_monitor", 
                "gcp": "gcp_logging"
            }
            preferred_provider = cloud_mapping.get(preferred_cloud)
            if preferred_provider in options:
                return preferred_provider
        
        # Default priority: AWS > Azure > GCP
        priority_order = ["aws_cloudwatch", "azure_monitor", "gcp_logging"]
        for provider in priority_order:
            if provider in options:
                return provider
        
        return options[0]

    def _select_metrics_provider(
        self,
        options: List[str],
        logs_provider: str,
        preferences: Dict[str, Any]
    ) -> Optional[str]:
        """Select metrics provider, preferring to match the logs provider."""
        if not options:
            return None
        
        # Try to match logs provider
        logs_to_metrics_mapping = {
            "aws_cloudwatch": "aws_cloudwatch",
            "azure_monitor": "azure_monitor",
            "gcp_logging": "gcp_monitoring"
        }
        
        preferred_metrics = logs_to_metrics_mapping.get(logs_provider)
        if preferred_metrics in options:
            return preferred_metrics
        
        # Otherwise, use first available
        return options[0]

    def _select_llm_provider(self, options: List[str], preferences: Dict[str, Any]) -> str:
        """Select the best LLM provider based on options and preferences."""
        if not options:
            raise ValueError("No LLM providers available")
        
        # Check user preference
        preferred_llm = preferences.get("preferred_llm_provider")
        if preferred_llm in options:
            return preferred_llm
        
        # Default priority: OpenAI > Anthropic > Azure OpenAI > Gemini > Bedrock > Ollama > HuggingFace
        priority_order = ["openai", "anthropic", "azure_openai", "gemini", "bedrock", "ollama", "huggingface"]
        for provider in priority_order:
            if provider in options:
                return provider
        
        return options[0]

    def get_provider_config(
        self,
        provider_type: str,
        provider_name: str,
        user_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get the configuration for a specific provider.
        
        Args:
            provider_type: Type of provider (logs, metrics, code, llm, runbooks)
            provider_name: Name of the specific provider
            user_config: User configuration
            
        Returns:
            Configuration dictionary for the provider
            
        Raises:
            ValueError: If provider is not supported or configuration is invalid
        """
        provider_key = f"{provider_type}_providers"
        if provider_key not in self.supported_integrations:
            raise ValueError(f"Unsupported provider type: {provider_type}")
        
        if provider_name not in self.supported_integrations[provider_key]:
            raise ValueError(f"Unsupported provider: {provider_name}")
        
        provider_spec = self.supported_integrations[provider_key][provider_name]
        config = {}
        
        # Get configuration from user config or environment variables
        provider_user_config = user_config.get(provider_name.split('_')[0], {})  # e.g., 'aws' from 'aws_cloudwatch'
        
        # Add required configuration
        for req_field in provider_spec["required_config"]:
            if req_field in provider_user_config:
                config[req_field] = provider_user_config[req_field]
            else:
                # Try to get from environment variables
                for env_var in provider_spec["env_vars"]:
                    if env_var in os.environ and req_field.upper() in env_var:
                        config[req_field] = os.environ[env_var]
                        break
        
        # Add optional configuration
        for opt_field in provider_spec["optional_config"]:
            if opt_field in provider_user_config:
                config[opt_field] = provider_user_config[opt_field]
        
        # Handle special cases
        config = self._handle_special_config_cases(provider_name, config, user_config)
        
        return config

    def _handle_special_config_cases(
        self,
        provider_name: str,
        config: Dict[str, Any],
        user_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle special configuration cases for specific providers.
        
        Args:
            provider_name: Name of the provider
            config: Current configuration
            user_config: Full user configuration
            
        Returns:
            Updated configuration
        """
        # GitHub repositories
        if provider_name == "github":
            github_config = user_config.get("github", {})
            if "repositories" in github_config:
                config["repositories"] = github_config["repositories"]
            
            # Get token from environment if not in config
            if "token" not in config and "GITHUB_TOKEN" in os.environ:
                config["token"] = os.environ["GITHUB_TOKEN"]
        
        # Runbook unified provider
        elif provider_name == "unified":
            runbook_config = user_config.get("runbooks", {})
            providers = {}
            
            if "directory" in runbook_config:
                directory = runbook_config["directory"]
                
                # Add providers based on file types found
                if self._has_files_with_extensions(directory, [".pdf"]):
                    providers["pdf_runbooks"] = {
                        "type": "pdf",
                        "config": {"runbook_directory": directory, "recursive": True}
                    }
                
                if self._has_files_with_extensions(directory, [".md", ".markdown", ".txt"]):
                    providers["markdown_runbooks"] = {
                        "type": "markdown", 
                        "config": {"runbook_directory": directory, "recursive": True}
                    }
                
                if self._has_files_with_extensions(directory, [".docx"]):
                    providers["docx_runbooks"] = {
                        "type": "docx",
                        "config": {"runbook_directory": directory, "recursive": True}
                    }
            
            if "web_urls" in runbook_config:
                providers["web_runbooks"] = {
                    "type": "web_link",
                    "config": {"base_urls": runbook_config["web_urls"]}
                }
            
            config["providers"] = providers
        
        # LLM provider defaults
        elif provider_name in ["openai", "anthropic", "ollama"]:
            # Set reasonable defaults
            if "max_tokens" not in config:
                config["max_tokens"] = 2000
            if "temperature" not in config:
                config["temperature"] = 0.1
            
            # Provider-specific defaults
            if provider_name == "openai" and "model" not in config:
                config["model"] = "gpt-4"
            elif provider_name == "anthropic" and "model" not in config:
                config["model"] = "claude-3-sonnet-20240229"
            elif provider_name == "ollama":
                if "base_url" not in config:
                    config["base_url"] = "http://localhost:11434"
                if "model" not in config:
                    config["model"] = "llama2"
        
        return config

    def get_integration_summary(self, selected_integrations: Dict[str, str]) -> Dict[str, Any]:
        """Get a summary of selected integrations for user confirmation.
        
        Args:
            selected_integrations: Selected integration mapping
            
        Returns:
            Summary dictionary with integration details
        """
        summary = {
            "integration_count": len([v for v in selected_integrations.values() if v]),
            "required_integrations": {},
            "optional_integrations": {},
            "capabilities": []
        }
        
        # Required integrations
        summary["required_integrations"]["logs"] = selected_integrations.get("logs_provider")
        summary["required_integrations"]["code"] = selected_integrations.get("code_provider") 
        summary["required_integrations"]["llm"] = selected_integrations.get("llm_provider")
        
        # Optional integrations
        if selected_integrations.get("metrics_provider"):
            summary["optional_integrations"]["metrics"] = selected_integrations["metrics_provider"]
        
        if selected_integrations.get("runbook_provider"):
            summary["optional_integrations"]["runbooks"] = selected_integrations["runbook_provider"]
        
        # Determine capabilities
        if summary["required_integrations"]["logs"]:
            summary["capabilities"].append("Log analysis and error detection")
        
        if summary["optional_integrations"].get("metrics"):
            summary["capabilities"].append("Metrics monitoring and alerting analysis")
        
        if summary["required_integrations"]["code"]:
            summary["capabilities"].append("Code analysis and fix suggestions")
        
        if summary["optional_integrations"].get("runbooks"):
            summary["capabilities"].append("Runbook-guided resolution")
        
        if summary["required_integrations"]["llm"]:
            summary["capabilities"].append("AI-powered incident resolution")
        
        return summary
