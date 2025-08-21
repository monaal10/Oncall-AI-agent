"""Setup manager for dynamic integration configuration."""

import asyncio
from typing import Dict, List, Optional, Any

from .provider_factory import ProviderFactory
from .runtime_interface import RuntimeInterface
from ..integrations.registry import IntegrationRegistry


class SetupManager:
    """Manages the complete setup process for OnCall AI Agent.
    
    Orchestrates integration discovery, provider creation, validation,
    and runtime interface setup based on user configuration.
    """

    def __init__(self):
        """Initialize the setup manager."""
        self.registry = IntegrationRegistry()
        self.factory = ProviderFactory()
        self.runtime_interface: Optional[RuntimeInterface] = None

    async def setup_from_config(self, user_config: Dict[str, Any]) -> RuntimeInterface:
        """Set up the complete OnCall AI Agent from user configuration.
        
        Args:
            user_config: User-provided configuration dictionary
            
        Returns:
            Configured RuntimeInterface with unified functions
            
        Raises:
            ValueError: If configuration is invalid or missing requirements
            ConnectionError: If provider initialization or validation fails
        """
        print("🔧 Starting OnCall AI Agent setup...")
        
        # Step 1: Discover available integrations
        print("📡 Discovering available integrations...")
        discovery_results = await self.registry.discover_integrations(user_config)
        
        available = discovery_results["available_integrations"]
        selected = discovery_results["selected_integrations"]
        
        print(f"✓ Found integrations:")
        print(f"  📊 Logs: {', '.join(available['logs'])}")
        print(f"  📈 Metrics: {', '.join(available['metrics']) if available['metrics'] else 'None'}")
        print(f"  💻 Code: {', '.join(available['code'])}")
        print(f"  🤖 LLM: {', '.join(available['llm'])}")
        print(f"  📖 Runbooks: {', '.join(available['runbooks']) if available['runbooks'] else 'None'}")
        
        print(f"\n🎯 Selected optimal combination:")
        for integration_type, provider_name in selected.items():
            if provider_name:
                print(f"  {integration_type}: {provider_name}")
        
        # Step 2: Create provider configurations
        print("\n⚙️  Creating provider configurations...")
        provider_configs = {}
        
        for integration_type, provider_name in selected.items():
            if provider_name:
                try:
                    config = self.registry.get_provider_config(
                        integration_type.replace("_provider", ""),
                        provider_name,
                        user_config
                    )
                    provider_configs[integration_type] = {
                        "type": provider_name,
                        "config": config
                    }
                    print(f"  ✓ {integration_type}: {provider_name}")
                except Exception as e:
                    print(f"  ✗ {integration_type}: {provider_name} - {e}")
                    raise
        
        # Step 3: Create provider instances
        print("\n🏗️  Creating provider instances...")
        providers = await self._create_provider_instances(provider_configs)
        
        # Step 4: Validate provider connectivity
        print("\n🔍 Validating provider connectivity...")
        validation_results = await self._validate_all_providers(providers)
        
        failed_providers = [
            name for name, result in validation_results.items()
            if not result.get("healthy", False)
        ]
        
        if failed_providers:
            print(f"⚠️  Some providers failed validation: {', '.join(failed_providers)}")
            for provider_name in failed_providers:
                error = validation_results[provider_name].get("error", "Unknown error")
                print(f"  {provider_name}: {error}")
            
            # Check if critical providers failed
            critical_providers = ["logs", "code", "llm"]
            failed_critical = [p for p in failed_providers if any(c in p for c in critical_providers)]
            
            if failed_critical:
                raise ConnectionError(f"Critical providers failed: {', '.join(failed_critical)}")
        
        # Step 5: Create runtime interface
        print("\n🚀 Creating runtime interface...")
        self.runtime_interface = RuntimeInterface(
            logs_provider=providers["logs"],
            code_provider=providers["code"],
            llm_provider=providers["llm"],
            metrics_provider=providers.get("metrics"),
            runbook_provider=providers.get("runbooks")
        )
        
        # Step 6: Final validation
        print("\n✅ Running final health checks...")
        final_health = await self.runtime_interface.health_check_all_providers()
        
        healthy_count = sum(1 for h in final_health.values() if h.get("healthy", False))
        total_count = len(final_health)
        
        print(f"🎉 Setup completed! {healthy_count}/{total_count} providers healthy")
        
        # Display runtime functions available
        runtime_functions = self.runtime_interface.get_runtime_functions()
        print(f"\n📋 Available runtime functions:")
        for func_name in runtime_functions.keys():
            print(f"  • {func_name}")
        
        return self.runtime_interface

    async def _create_provider_instances(
        self,
        provider_configs: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create all provider instances.
        
        Args:
            provider_configs: Configuration for each provider
            
        Returns:
            Dictionary of created provider instances
            
        Raises:
            ConnectionError: If critical provider creation fails
        """
        providers = {}
        
        # Create logs provider (required)
        if "logs_provider" in provider_configs:
            config = provider_configs["logs_provider"]
            providers["logs"] = self.factory.create_logs_provider(
                config["type"],
                config["config"]
            )
            print(f"  ✓ Logs provider: {config['type']}")
        
        # Create code provider (required)
        if "code_provider" in provider_configs:
            config = provider_configs["code_provider"]
            providers["code"] = self.factory.create_code_provider(
                config["type"],
                config["config"]
            )
            print(f"  ✓ Code provider: {config['type']}")
        
        # Create LLM provider (required)
        if "llm_provider" in provider_configs:
            config = provider_configs["llm_provider"]
            providers["llm"] = self.factory.create_llm_provider(
                config["type"],
                config["config"]
            )
            print(f"  ✓ LLM provider: {config['type']}")
        
        # Create optional providers
        if "metrics_provider" in provider_configs:
            config = provider_configs["metrics_provider"]
            try:
                providers["metrics"] = self.factory.create_metrics_provider(
                    config["type"],
                    config["config"]
                )
                print(f"  ✓ Metrics provider: {config['type']}")
            except Exception as e:
                print(f"  ⚠️  Metrics provider failed: {e}")
        
        if "runbook_provider" in provider_configs:
            config = provider_configs["runbook_provider"]
            try:
                providers["runbooks"] = self.factory.create_runbook_provider(
                    config["type"],
                    config["config"]
                )
                print(f"  ✓ Runbook provider: {config['type']}")
            except Exception as e:
                print(f"  ⚠️  Runbook provider failed: {e}")
        
        return providers

    async def _validate_all_providers(self, providers: Dict[str, Any]) -> Dict[str, Any]:
        """Validate connectivity for all providers.
        
        Args:
            providers: Dictionary of provider instances
            
        Returns:
            Dictionary of validation results
        """
        validation_tasks = []
        provider_names = []
        
        for provider_name, provider_instance in providers.items():
            task = asyncio.create_task(
                self.factory.validate_provider_connectivity(
                    provider_instance,
                    provider_name
                )
            )
            validation_tasks.append(task)
            provider_names.append(provider_name)
        
        # Run all validations in parallel
        validation_results = await asyncio.gather(*validation_tasks, return_exceptions=True)
        
        # Process results
        results = {}
        for i, provider_name in enumerate(provider_names):
            result = validation_results[i]
            if isinstance(result, Exception):
                results[provider_name] = {
                    "healthy": False,
                    "error": str(result)
                }
            else:
                results[provider_name] = result
                
            # Display result
            if results[provider_name].get("healthy", False):
                latency = results[provider_name].get("latency_ms", 0)
                print(f"  ✓ {provider_name}: healthy ({latency:.0f}ms)")
            else:
                error = results[provider_name].get("error", "Unknown error")
                print(f"  ✗ {provider_name}: failed - {error}")
        
        return results

    def get_setup_summary(self) -> Dict[str, Any]:
        """Get a summary of the current setup.
        
        Returns:
            Dictionary containing setup summary
        """
        if not self.runtime_interface:
            return {"status": "not_configured", "message": "Setup not completed"}
        
        provider_info = self.runtime_interface.get_provider_info()
        runtime_functions = self.runtime_interface.get_runtime_functions()
        
        return {
            "status": "configured",
            "providers": provider_info,
            "runtime_functions": list(runtime_functions.keys()),
            "capabilities": self._determine_capabilities(provider_info),
            "setup_time": datetime.now()
        }

    def _determine_capabilities(self, provider_info: Dict[str, Any]) -> List[str]:
        """Determine system capabilities based on configured providers.
        
        Args:
            provider_info: Information about configured providers
            
        Returns:
            List of available capabilities
        """
        capabilities = []
        
        if provider_info["logs_provider"]["available"]:
            capabilities.append("Log analysis and error detection")
        
        if provider_info["metrics_provider"]["available"]:
            capabilities.append("Metrics monitoring and anomaly detection")
        
        if provider_info["code_provider"]["available"]:
            capabilities.append("Code analysis and fix suggestions")
        
        if provider_info["llm_provider"]["available"]:
            capabilities.append("AI-powered incident resolution")
        
        if provider_info["runbook_provider"]["available"]:
            capabilities.append("Runbook-guided troubleshooting")
        
        # Combined capabilities
        if len(capabilities) >= 3:
            capabilities.append("Comprehensive incident analysis")
        
        if provider_info["metrics_provider"]["available"] and provider_info["logs_provider"]["available"]:
            capabilities.append("Correlated logs and metrics analysis")
        
        return capabilities

    async def quick_setup(self, **kwargs) -> RuntimeInterface:
        """Quick setup using environment variables and defaults.
        
        Args:
            **kwargs: Override configuration parameters
            
        Returns:
            Configured RuntimeInterface
            
        Raises:
            ValueError: If minimum requirements not met
        """
        print("⚡ Quick setup using environment variables...")
        
        # Build configuration from environment variables and kwargs
        config = {
            "aws": {},
            "azure": {},
            "gcp": {},
            "github": {},
            "openai": {},
            "anthropic": {},
            "ollama": {},
            "runbooks": {}
        }
        
        # Override with any provided kwargs
        for key, value in kwargs.items():
            if "." in key:
                # Handle nested keys like "aws.region"
                parts = key.split(".")
                if parts[0] in config:
                    config[parts[0]][parts[1]] = value
            else:
                config[key] = value
        
        return await self.setup_from_config(config)

    def validate_minimum_config(self, user_config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that minimum configuration requirements are met.
        
        Args:
            user_config: User configuration to validate
            
        Returns:
            Dictionary containing validation results
        """
        validation = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "recommendations": []
        }
        
        # Check for at least one LLM provider
        llm_available = (
            user_config.get("openai", {}).get("api_key") or
            user_config.get("anthropic", {}).get("api_key") or
            user_config.get("ollama", {}).get("model")
        )
        
        if not llm_available:
            validation["errors"].append("At least one LLM provider (OpenAI, Anthropic, or Ollama) is required")
            validation["valid"] = False
        
        # Check for GitHub access
        github_available = (
            user_config.get("github", {}).get("token") or
            user_config.get("github", {}).get("repositories")
        )
        
        if not github_available:
            validation["errors"].append("GitHub integration is required (token and repositories)")
            validation["valid"] = False
        
        # Check for at least one logs provider
        logs_available = (
            self._check_aws_config(user_config) or
            self._check_azure_config(user_config) or
            self._check_gcp_config(user_config)
        )
        
        if not logs_available:
            validation["errors"].append("At least one logs provider (AWS, Azure, or GCP) is required")
            validation["valid"] = False
        
        # Warnings for missing optional components
        if not user_config.get("runbooks", {}).get("directory"):
            validation["warnings"].append("No runbook directory configured - runbook guidance will not be available")
        
        # Recommendations
        if not self._check_metrics_available(user_config):
            validation["recommendations"].append("Configure metrics provider for enhanced incident analysis")
        
        if len([p for p in [user_config.get("openai"), user_config.get("anthropic"), user_config.get("ollama")] if p]) == 1:
            validation["recommendations"].append("Consider configuring multiple LLM providers for fallback support")
        
        return validation

    def _check_aws_config(self, config: Dict[str, Any]) -> bool:
        """Check if AWS configuration is available."""
        aws_config = config.get("aws", {})
        return bool(aws_config.get("region"))

    def _check_azure_config(self, config: Dict[str, Any]) -> bool:
        """Check if Azure configuration is available."""
        azure_config = config.get("azure", {})
        return bool(azure_config.get("subscription_id") and azure_config.get("workspace_id"))

    def _check_gcp_config(self, config: Dict[str, Any]) -> bool:
        """Check if GCP configuration is available."""
        gcp_config = config.get("gcp", {})
        return bool(gcp_config.get("project_id"))

    def _check_metrics_available(self, config: Dict[str, Any]) -> bool:
        """Check if any metrics provider is available."""
        return (
            self._check_aws_config(config) or
            self._check_azure_config(config) or
            self._check_gcp_config(config)
        )

    async def test_setup(self, runtime_interface: RuntimeInterface) -> Dict[str, Any]:
        """Test the setup by running sample operations.
        
        Args:
            runtime_interface: Configured runtime interface
            
        Returns:
            Dictionary containing test results
        """
        print("\n🧪 Testing setup with sample operations...")
        
        test_results = {
            "logs_test": {"success": False, "error": None},
            "code_test": {"success": False, "error": None},
            "llm_test": {"success": False, "error": None},
            "metrics_test": {"success": False, "error": None},
            "runbooks_test": {"success": False, "error": None}
        }
        
        # Test logs function
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=1)
            
            logs = await runtime_interface.get_logs(
                query="ERROR",
                time_range=(start_time, end_time),
                limit=5
            )
            test_results["logs_test"]["success"] = True
            test_results["logs_test"]["result_count"] = len(logs)
            print("  ✓ Logs function: working")
            
        except Exception as e:
            test_results["logs_test"]["error"] = str(e)
            print(f"  ✗ Logs function: {e}")
        
        # Test code function
        try:
            code_context = await runtime_interface.get_code_context(
                error_info={"message": "test error", "stack_trace": ""}
            )
            test_results["code_test"]["success"] = True
            test_results["code_test"]["result_count"] = len(code_context)
            print("  ✓ Code function: working")
            
        except Exception as e:
            test_results["code_test"]["error"] = str(e)
            print(f"  ✗ Code function: {e}")
        
        # Test LLM function
        try:
            test_context = {
                "incident_description": "Test connectivity check",
                "log_data": [],
                "metric_data": [],
                "code_context": [],
                "runbook_guidance": ""
            }
            
            llm_response = await runtime_interface.get_llm_response(
                context=test_context,
                response_type="resolution"
            )
            test_results["llm_test"]["success"] = True
            test_results["llm_test"]["confidence"] = llm_response.get("confidence_score", 0)
            print("  ✓ LLM function: working")
            
        except Exception as e:
            test_results["llm_test"]["error"] = str(e)
            print(f"  ✗ LLM function: {e}")
        
        # Test optional functions
        if runtime_interface.get_metrics:
            try:
                metrics = await runtime_interface.get_metrics(
                    resource_info={"namespace": "test"},
                    time_range=(start_time, end_time)
                )
                test_results["metrics_test"]["success"] = True
                test_results["metrics_test"]["result_count"] = len(metrics)
                print("  ✓ Metrics function: working")
                
            except Exception as e:
                test_results["metrics_test"]["error"] = str(e)
                print(f"  ⚠️  Metrics function: {e}")
        else:
            print("  - Metrics function: not configured")
        
        if runtime_interface.get_runbook_guidance:
            try:
                guidance = await runtime_interface.get_runbook_guidance(
                    error_context="test error"
                )
                test_results["runbooks_test"]["success"] = True
                test_results["runbooks_test"]["guidance_length"] = len(guidance)
                print("  ✓ Runbooks function: working")
                
            except Exception as e:
                test_results["runbooks_test"]["error"] = str(e)
                print(f"  ⚠️  Runbooks function: {e}")
        else:
            print("  - Runbooks function: not configured")
        
        return test_results

    def get_integration_status(self) -> Dict[str, Any]:
        """Get current integration status.
        
        Returns:
            Dictionary containing current status
        """
        if not self.runtime_interface:
            return {
                "configured": False,
                "message": "Setup not completed. Call setup_from_config() first."
            }
        
        return {
            "configured": True,
            "provider_info": self.runtime_interface.get_provider_info(),
            "runtime_functions": list(self.runtime_interface.get_runtime_functions().keys()),
            "setup_summary": self.get_setup_summary()
        }
