"""Configuration validation utilities."""

import os
from typing import Dict, List, Optional, Any, Tuple


class ConfigValidator:
    """Validates OnCall AI Agent configuration for completeness and correctness.
    
    Ensures that user configuration meets minimum requirements and provides
    helpful error messages and suggestions for fixing configuration issues.
    """

    def __init__(self):
        """Initialize the configuration validator."""
        self.required_integrations = ["logs", "code", "llm"]
        self.optional_integrations = ["metrics", "runbooks"]

    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate complete configuration.
        
        Args:
            config: User configuration dictionary
            
        Returns:
            Dictionary containing:
            - valid: Whether configuration is valid
            - errors: List of error messages
            - warnings: List of warning messages
            - suggestions: List of improvement suggestions
            - missing_env_vars: List of missing environment variables
            - integration_status: Status of each integration type
        """
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "suggestions": [],
            "missing_env_vars": [],
            "integration_status": {}
        }
        
        # Validate each integration type
        validation_result["integration_status"]["logs"] = self._validate_logs_config(config)
        validation_result["integration_status"]["code"] = self._validate_code_config(config)
        validation_result["integration_status"]["llm"] = self._validate_llm_config(config)
        validation_result["integration_status"]["metrics"] = self._validate_metrics_config(config)
        validation_result["integration_status"]["runbooks"] = self._validate_runbooks_config(config)
        
        # Collect errors and warnings
        for integration_type, status in validation_result["integration_status"].items():
            if integration_type in self.required_integrations and not status["available"]:
                validation_result["errors"].extend(status["errors"])
                validation_result["valid"] = False
            elif not status["available"]:
                validation_result["warnings"].extend(status["warnings"])
            
            validation_result["missing_env_vars"].extend(status.get("missing_env_vars", []))
            validation_result["suggestions"].extend(status.get("suggestions", []))
        
        # Add general suggestions
        if validation_result["valid"]:
            validation_result["suggestions"].extend(self._get_general_suggestions(config))
        
        return validation_result

    def _validate_logs_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate logs provider configuration.
        
        Args:
            config: User configuration
            
        Returns:
            Logs validation status
        """
        status = {
            "available": False,
            "providers": [],
            "errors": [],
            "warnings": [],
            "suggestions": [],
            "missing_env_vars": []
        }
        
        # Check AWS CloudWatch
        aws_status = self._check_aws_logs_config(config)
        if aws_status["available"]:
            status["providers"].append("aws_cloudwatch")
            status["available"] = True
        else:
            status["missing_env_vars"].extend(aws_status["missing_env_vars"])
        
        # Check Azure Monitor
        azure_status = self._check_azure_logs_config(config)
        if azure_status["available"]:
            status["providers"].append("azure_monitor")
            status["available"] = True
        else:
            status["missing_env_vars"].extend(azure_status["missing_env_vars"])
        
        # Check GCP Cloud Logging
        gcp_status = self._check_gcp_logs_config(config)
        if gcp_status["available"]:
            status["providers"].append("gcp_logging")
            status["available"] = True
        else:
            status["missing_env_vars"].extend(gcp_status["missing_env_vars"])
        
        if not status["available"]:
            status["errors"].append("No logs provider configured. Configure AWS, Azure, or GCP credentials.")
            status["suggestions"].extend([
                "For AWS: Set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION",
                "For Azure: Set AZURE_SUBSCRIPTION_ID, AZURE_WORKSPACE_ID, and auth credentials",
                "For GCP: Set GCP_PROJECT_ID and GOOGLE_APPLICATION_CREDENTIALS"
            ])
        
        return status

    def _validate_code_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate code provider configuration.
        
        Args:
            config: User configuration
            
        Returns:
            Code validation status
        """
        status = {
            "available": False,
            "providers": [],
            "errors": [],
            "warnings": [],
            "suggestions": [],
            "missing_env_vars": []
        }
        
        github_config = config.get("github", {})
        
        # Check for GitHub token
        if github_config.get("token") or os.getenv("GITHUB_TOKEN"):
            # Check for repositories
            if github_config.get("repositories"):
                status["available"] = True
                status["providers"].append("github")
            else:
                status["errors"].append("GitHub repositories list is required")
                status["suggestions"].append("Add 'repositories' list to github configuration")
        else:
            status["errors"].append("GitHub token is required")
            status["missing_env_vars"].append("GITHUB_TOKEN")
            status["suggestions"].append("Create GitHub Personal Access Token and set GITHUB_TOKEN environment variable")
        
        return status

    def _validate_llm_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate LLM provider configuration.
        
        Args:
            config: User configuration
            
        Returns:
            LLM validation status
        """
        status = {
            "available": False,
            "providers": [],
            "errors": [],
            "warnings": [],
            "suggestions": [],
            "missing_env_vars": []
        }
        
        # Check OpenAI
        if config.get("openai", {}).get("api_key") or os.getenv("OPENAI_API_KEY"):
            status["providers"].append("openai")
            status["available"] = True
        else:
            status["missing_env_vars"].append("OPENAI_API_KEY")
        
        # Check Anthropic
        if config.get("anthropic", {}).get("api_key") or os.getenv("ANTHROPIC_API_KEY"):
            status["providers"].append("anthropic")
            status["available"] = True
        else:
            status["missing_env_vars"].append("ANTHROPIC_API_KEY")
        
        # Check Ollama
        if config.get("ollama", {}).get("model_name"):
            status["providers"].append("ollama")
            status["available"] = True
        
        # Check HuggingFace
        if config.get("huggingface", {}).get("model_name"):
            status["providers"].append("huggingface")
            status["available"] = True
        
        # Check Gemini
        if config.get("gemini", {}).get("api_key") or os.getenv("GOOGLE_API_KEY"):
            status["providers"].append("gemini")
            status["available"] = True
        else:
            status["missing_env_vars"].append("GOOGLE_API_KEY")
        
        # Check Azure OpenAI
        azure_openai_config = config.get("azure_openai", {})
        if (azure_openai_config.get("api_key") and 
            azure_openai_config.get("azure_endpoint") and 
            azure_openai_config.get("deployment_name")):
            status["providers"].append("azure_openai")
            status["available"] = True
        elif os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"):
            status["providers"].append("azure_openai")
            status["available"] = True
        else:
            status["missing_env_vars"].extend(["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"])
        
        # Check Bedrock
        if config.get("bedrock", {}).get("model_id"):
            status["providers"].append("bedrock")
            status["available"] = True
        
        if not status["available"]:
            status["errors"].append("No LLM provider configured. Configure at least one LLM provider.")
            status["suggestions"].extend([
                "For OpenAI: Set OPENAI_API_KEY environment variable",
                "For Anthropic: Set ANTHROPIC_API_KEY environment variable", 
                "For Azure OpenAI: Set AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, and deployment_name",
                "For Gemini: Set GOOGLE_API_KEY environment variable",
                "For Bedrock: Specify model_id and AWS credentials",
                "For Ollama: Install Ollama and specify model_name in configuration",
                "For HuggingFace: Specify model_name (API key optional for many models)"
            ])
        
        return status

    def _validate_metrics_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate metrics provider configuration (optional).
        
        Args:
            config: User configuration
            
        Returns:
            Metrics validation status
        """
        status = {
            "available": False,
            "providers": [],
            "errors": [],
            "warnings": [],
            "suggestions": [],
            "missing_env_vars": []
        }
        
        # Metrics providers typically match logs providers
        if self._check_aws_logs_config(config)["available"]:
            status["providers"].append("aws_cloudwatch")
            status["available"] = True
        
        if self._check_azure_logs_config(config)["available"]:
            status["providers"].append("azure_monitor")
            status["available"] = True
        
        if self._check_gcp_logs_config(config)["available"]:
            status["providers"].append("gcp_monitoring")
            status["available"] = True
        
        if not status["available"]:
            status["warnings"].append("No metrics provider available - enhanced monitoring features will be disabled")
            status["suggestions"].append("Metrics providers use the same credentials as logs providers")
        
        return status

    def _validate_runbooks_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate runbooks provider configuration (optional).
        
        Args:
            config: User configuration
            
        Returns:
            Runbooks validation status
        """
        status = {
            "available": False,
            "providers": [],
            "errors": [],
            "warnings": [],
            "suggestions": [],
            "missing_env_vars": []
        }
        
        runbooks_config = config.get("runbooks", {})
        
        # Check for runbook directory
        if "directory" in runbooks_config:
            directory = runbooks_config["directory"]
            if os.path.exists(directory) and os.path.isdir(directory):
                # Check for runbook files
                has_files = False
                try:
                    for root, dirs, files in os.walk(directory):
                        if any(f.endswith(('.pdf', '.md', '.markdown', '.docx', '.txt')) for f in files):
                            has_files = True
                            break
                    
                    if has_files:
                        status["available"] = True
                        status["providers"].append("unified")
                    else:
                        status["warnings"].append(f"Runbook directory exists but contains no runbook files: {directory}")
                        
                except Exception as e:
                    status["warnings"].append(f"Could not scan runbook directory: {e}")
            else:
                status["warnings"].append(f"Runbook directory does not exist: {directory}")
        
        # Check for web runbooks
        if "web_urls" in runbooks_config and runbooks_config["web_urls"]:
            status["available"] = True
            if "unified" not in status["providers"]:
                status["providers"].append("web_link")
        
        if not status["available"]:
            status["warnings"].append("No runbook provider configured - runbook guidance will not be available")
            status["suggestions"].extend([
                "Create a runbooks directory with PDF, Markdown, or Word documents",
                "Or configure web_urls for online documentation"
            ])
        
        return status

    def _check_aws_logs_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Check AWS logs configuration."""
        aws_config = config.get("aws", {})
        missing_env_vars = []
        
        # Check region
        if not aws_config.get("region") and not os.getenv("AWS_DEFAULT_REGION"):
            missing_env_vars.append("AWS_DEFAULT_REGION")
        
        # Check credentials (explicit or environment)
        has_explicit_creds = aws_config.get("access_key_id") and aws_config.get("secret_access_key")
        has_env_creds = os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY")
        
        if not has_explicit_creds and not has_env_creds:
            missing_env_vars.extend(["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"])
        
        return {
            "available": bool(
                (aws_config.get("region") or os.getenv("AWS_DEFAULT_REGION")) and
                (has_explicit_creds or has_env_creds)
            ),
            "missing_env_vars": missing_env_vars
        }

    def _check_azure_logs_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Check Azure logs configuration."""
        azure_config = config.get("azure", {})
        missing_env_vars = []
        
        if not azure_config.get("subscription_id") and not os.getenv("AZURE_SUBSCRIPTION_ID"):
            missing_env_vars.append("AZURE_SUBSCRIPTION_ID")
        
        if not azure_config.get("workspace_id") and not os.getenv("AZURE_WORKSPACE_ID"):
            missing_env_vars.append("AZURE_WORKSPACE_ID")
        
        # Check auth (service principal or default)
        has_service_principal = all(
            azure_config.get(key) or os.getenv(f"AZURE_{key.upper()}")
            for key in ["tenant_id", "client_id", "client_secret"]
        )
        
        if not has_service_principal:
            missing_env_vars.extend(["AZURE_CLIENT_ID", "AZURE_CLIENT_SECRET", "AZURE_TENANT_ID"])
        
        return {
            "available": bool(
                (azure_config.get("subscription_id") or os.getenv("AZURE_SUBSCRIPTION_ID")) and
                (azure_config.get("workspace_id") or os.getenv("AZURE_WORKSPACE_ID"))
            ),
            "missing_env_vars": missing_env_vars
        }

    def _check_gcp_logs_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Check GCP logs configuration."""
        gcp_config = config.get("gcp", {})
        missing_env_vars = []
        
        if not gcp_config.get("project_id") and not os.getenv("GCP_PROJECT_ID"):
            missing_env_vars.append("GCP_PROJECT_ID")
        
        # Check credentials
        has_creds_file = (
            gcp_config.get("credentials_path") and os.path.exists(gcp_config["credentials_path"])
        ) or (
            os.getenv("GOOGLE_APPLICATION_CREDENTIALS") and 
            os.path.exists(os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
        )
        
        if not has_creds_file:
            missing_env_vars.append("GOOGLE_APPLICATION_CREDENTIALS")
        
        return {
            "available": bool(
                (gcp_config.get("project_id") or os.getenv("GCP_PROJECT_ID")) and
                has_creds_file
            ),
            "missing_env_vars": missing_env_vars
        }

    def generate_sample_config(self, integration_preferences: Optional[List[str]] = None) -> str:
        """Generate a sample configuration file.
        
        Args:
            integration_preferences: Preferred integration types
            
        Returns:
            Sample configuration as YAML string
        """
        preferences = integration_preferences or ["aws", "github", "openai"]
        
        config_parts = []
        config_parts.append("# OnCall AI Agent Configuration")
        config_parts.append("# Minimal configuration - only configure what you have access to\n")
        
        # Add preferred cloud provider
        if "aws" in preferences:
            config_parts.append("""# AWS Configuration (for logs and metrics)
aws:
  region: "us-west-2"
  # Optional: explicit credentials (if not using AWS CLI/IAM roles)
  # access_key_id: "${AWS_ACCESS_KEY_ID}"
  # secret_access_key: "${AWS_SECRET_ACCESS_KEY}"
""")
        
        if "azure" in preferences:
            config_parts.append("""# Azure Configuration (for logs and metrics)
azure:
  subscription_id: "${AZURE_SUBSCRIPTION_ID}"
  workspace_id: "${AZURE_WORKSPACE_ID}"
  # Optional: service principal (if not using Azure CLI/managed identity)
  # tenant_id: "${AZURE_TENANT_ID}"
  # client_id: "${AZURE_CLIENT_ID}"
  # client_secret: "${AZURE_CLIENT_SECRET}"
""")
        
        if "gcp" in preferences:
            config_parts.append("""# GCP Configuration (for logs and metrics)
gcp:
  project_id: "${GCP_PROJECT_ID}"
  # Optional: service account file (if not using gcloud CLI/service account)
  # credentials_path: "/path/to/service-account.json"
""")
        
        # GitHub (always required)
        config_parts.append("""# GitHub Configuration (required)
github:
  token: "${GITHUB_TOKEN}"
  repositories:
    - "your-org/backend-service"
    - "your-org/frontend-app"
    # Add all repositories you want the agent to analyze
""")
        
        # LLM providers
        if "openai" in preferences:
            config_parts.append("""# OpenAI Configuration
openai:
  api_key: "${OPENAI_API_KEY}"
  model: "gpt-4"  # or "gpt-3.5-turbo" for lower cost
  max_tokens: 2000
  temperature: 0.1
""")
        
        if "anthropic" in preferences:
            config_parts.append("""# Anthropic Configuration  
anthropic:
  api_key: "${ANTHROPIC_API_KEY}"
  model: "claude-3-sonnet-20240229"
  max_tokens: 2000
  temperature: 0.1
""")
        
        if "ollama" in preferences:
            config_parts.append("""# Ollama Configuration (local models)
ollama:
  model: "llama2"  # or "codellama", "mistral"
  base_url: "http://localhost:11434"
  temperature: 0.1
""")
        
        # Optional runbooks
        config_parts.append("""# Runbooks Configuration (optional)
runbooks:
  directory: "/path/to/your/runbooks"
  # Or for web-based runbooks:
  # web_urls:
  #   - "https://docs.yourcompany.com/runbooks"
""")
        
        return "\n".join(config_parts)

    def get_setup_checklist(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get a setup checklist for the user.
        
        Args:
            config: User configuration
            
        Returns:
            List of checklist items with status
        """
        validation_result = self.validate_config(config)
        checklist = []
        
        # Required items
        checklist.append({
            "item": "Configure logs provider (AWS/Azure/GCP)",
            "required": True,
            "status": validation_result["integration_status"]["logs"]["available"],
            "details": "At least one cloud provider for log access"
        })
        
        checklist.append({
            "item": "Configure GitHub integration",
            "required": True,
            "status": validation_result["integration_status"]["code"]["available"],
            "details": "GitHub token and repository list"
        })
        
        checklist.append({
            "item": "Configure LLM provider",
            "required": True,
            "status": validation_result["integration_status"]["llm"]["available"],
            "details": "OpenAI, Anthropic, or Ollama"
        })
        
        # Optional items
        checklist.append({
            "item": "Configure metrics provider",
            "required": False,
            "status": validation_result["integration_status"]["metrics"]["available"],
            "details": "Enhanced monitoring (uses same credentials as logs)"
        })
        
        checklist.append({
            "item": "Configure runbooks",
            "required": False,
            "status": validation_result["integration_status"]["runbooks"]["available"],
            "details": "Local directory or web URLs for guidance documents"
        })
        
        return checklist

    def get_environment_variables_needed(self, config: Dict[str, Any]) -> Dict[str, List[str]]:
        """Get list of environment variables needed for configuration.
        
        Args:
            config: User configuration
            
        Returns:
            Dictionary mapping categories to required environment variables
        """
        validation_result = self.validate_config(config)
        
        env_vars = {
            "required": [],
            "optional": [],
            "missing": validation_result["missing_env_vars"]
        }
        
        # Always required
        env_vars["required"].extend(["GITHUB_TOKEN"])
        
        # LLM providers (at least one required)
        llm_vars = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
        env_vars["required"].append(f"At least one of: {', '.join(llm_vars)}")
        
        # Cloud providers (at least one required)
        cloud_vars = [
            "AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY + AWS_DEFAULT_REGION",
            "AZURE_SUBSCRIPTION_ID + AZURE_WORKSPACE_ID + AZURE_CLIENT_ID + AZURE_CLIENT_SECRET + AZURE_TENANT_ID",
            "GCP_PROJECT_ID + GOOGLE_APPLICATION_CREDENTIALS"
        ]
        env_vars["required"].append(f"At least one cloud provider: {' OR '.join(cloud_vars)}")
        
        return env_vars
