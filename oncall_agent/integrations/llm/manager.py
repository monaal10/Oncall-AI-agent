"""Unified LLM manager for LangGraph integration."""

from typing import Dict, List, Optional, Any, AsyncGenerator, Union
from datetime import datetime

try:
    from langchain_core.language_models import BaseLLM, BaseChatModel
    from langchain_core.messages import BaseMessage
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

from ..base.llm_provider import LLMProvider
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider
from .ollama_provider import OllamaProvider
from .huggingface_provider import HuggingFaceProvider
from .gemini_provider import GeminiProvider
from .azure_openai_provider import AzureOpenAIProvider
from .bedrock_provider import BedrockProvider


class LLMManager:
    """Unified LLM manager for OnCall AI Agent.
    
    Manages multiple LLM providers and provides a unified interface
    for LangGraph workflows and incident resolution.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize LLM manager.
        
        Args:
            config: Configuration dictionary containing:
                - primary_provider: Primary LLM provider configuration
                - fallback_providers: List of fallback provider configurations (optional)
                - retry_attempts: Number of retry attempts (optional, default: 3)
                - timeout: Global timeout in seconds (optional, default: 120)
        """
        if not LANGGRAPH_AVAILABLE:
            raise ImportError(
                "LangGraph not available. Install with: pip install langgraph langchain-core"
            )
        
        self.config = config
        self._validate_config()
        self._setup_providers()

    def _validate_config(self) -> None:
        """Validate LLM manager configuration.
        
        Raises:
            ValueError: If required configuration is missing
        """
        if "primary_provider" not in self.config:
            raise ValueError("primary_provider configuration is required")

    def _setup_providers(self) -> None:
        """Set up LLM providers.
        
        Raises:
            ValueError: If provider setup fails
        """
        provider_map = {
            "openai": OpenAIProvider,
            "anthropic": AnthropicProvider,
            "ollama": OllamaProvider,
            "huggingface": HuggingFaceProvider,
            "gemini": GeminiProvider,
            "azure_openai": AzureOpenAIProvider,
            "bedrock": BedrockProvider
        }
        
        # Setup primary provider
        primary_config = self.config["primary_provider"]
        provider_type = primary_config["type"]
        
        if provider_type not in provider_map:
            raise ValueError(f"Unsupported provider type: {provider_type}")
        
        self.primary_provider = provider_map[provider_type](primary_config["config"])
        
        # Setup fallback providers
        self.fallback_providers = []
        for fallback_config in self.config.get("fallback_providers", []):
            fallback_type = fallback_config["type"]
            if fallback_type in provider_map:
                try:
                    fallback_provider = provider_map[fallback_type](fallback_config["config"])
                    self.fallback_providers.append(fallback_provider)
                except Exception as e:
                    print(f"Warning: Failed to setup fallback provider {fallback_type}: {e}")

    def get_langchain_model(self) -> Union[BaseLLM, BaseChatModel]:
        """Get the primary LangChain model for LangGraph integration.
        
        This is the main method for LangGraph workflows.
        
        Returns:
            LangChain model instance that can be used directly in LangGraph
        """
        return self.primary_provider.get_langchain_model()

    async def generate_resolution(
        self,
        incident_context: Dict[str, Any],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate incident resolution with fallback support.
        
        Args:
            incident_context: Incident information and context
            system_prompt: Optional system prompt override
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing resolution information
            
        Raises:
            ConnectionError: If all providers fail
        """
        providers_to_try = [self.primary_provider] + self.fallback_providers
        last_error = None
        
        for provider in providers_to_try:
            try:
                return await provider.generate_resolution(
                    incident_context,
                    system_prompt,
                    **kwargs
                )
            except Exception as e:
                last_error = e
                continue
        
        raise ConnectionError(f"All LLM providers failed. Last error: {last_error}")

    async def analyze_logs(
        self,
        log_entries: List[Dict[str, Any]],
        context: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Analyze logs with fallback support.
        
        Args:
            log_entries: List of log entries to analyze
            context: Additional context
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing log analysis results
            
        Raises:
            ConnectionError: If all providers fail
        """
        providers_to_try = [self.primary_provider] + self.fallback_providers
        last_error = None
        
        for provider in providers_to_try:
            try:
                return await provider.analyze_logs(log_entries, context, **kwargs)
            except Exception as e:
                last_error = e
                continue
        
        raise ConnectionError(f"All LLM providers failed for log analysis. Last error: {last_error}")

    async def analyze_code_context(
        self,
        code_snippets: List[Dict[str, Any]],
        error_message: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Analyze code context with fallback support.
        
        Args:
            code_snippets: List of relevant code snippets
            error_message: Error message or stack trace
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing code analysis results
            
        Raises:
            ConnectionError: If all providers fail
        """
        providers_to_try = [self.primary_provider] + self.fallback_providers
        last_error = None
        
        for provider in providers_to_try:
            try:
                return await provider.analyze_code_context(code_snippets, error_message, **kwargs)
            except Exception as e:
                last_error = e
                continue
        
        raise ConnectionError(f"All LLM providers failed for code analysis. Last error: {last_error}")

    async def stream_response(
        self,
        prompt: str,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream response with fallback support.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters
            
        Yields:
            Chunks of the response as they are generated
            
        Raises:
            ConnectionError: If all providers fail
        """
        providers_to_try = [self.primary_provider] + self.fallback_providers
        
        for provider in providers_to_try:
            try:
                async for chunk in provider.stream_response(prompt, **kwargs):
                    yield chunk
                return  # Success, don't try other providers
            except Exception as e:
                if provider == providers_to_try[-1]:  # Last provider
                    raise ConnectionError(f"All LLM providers failed for streaming. Last error: {e}")
                continue

    async def health_check(self) -> Dict[str, Any]:
        """Check health of all configured providers.
        
        Returns:
            Dictionary containing health status for all providers
        """
        health_results = {
            "primary_provider": await self.primary_provider.health_check(),
            "fallback_providers": []
        }
        
        for i, provider in enumerate(self.fallback_providers):
            try:
                fallback_health = await provider.health_check()
                fallback_health["provider_index"] = i
                health_results["fallback_providers"].append(fallback_health)
            except Exception as e:
                health_results["fallback_providers"].append({
                    "healthy": False,
                    "provider_index": i,
                    "error": str(e)
                })
        
        return health_results

    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about all configured providers.
        
        Returns:
            Dictionary containing information about all providers
        """
        info = {
            "primary_provider": self.primary_provider.get_model_info(),
            "fallback_providers": []
        }
        
        for provider in self.fallback_providers:
            info["fallback_providers"].append(provider.get_model_info())
        
        return info

    def create_langgraph_workflow(self) -> StateGraph:
        """Create a basic LangGraph workflow for incident resolution.
        
        This provides a foundation for building more complex LangGraph workflows.
        
        Returns:
            Configured LangGraph StateGraph
        """
        # Define the state structure
        class IncidentState:
            incident_description: str
            log_data: List[Dict[str, Any]]
            metric_data: List[Dict[str, Any]]
            code_context: List[Dict[str, Any]]
            runbook_guidance: str
            resolution: Optional[Dict[str, Any]] = None
            analysis_steps: List[str] = []

        # Create workflow
        workflow = StateGraph(IncidentState)
        
        # Add nodes
        workflow.add_node("analyze_logs", self._analyze_logs_node)
        workflow.add_node("analyze_code", self._analyze_code_node)
        workflow.add_node("generate_resolution", self._generate_resolution_node)
        
        # Add edges
        workflow.add_edge("analyze_logs", "analyze_code")
        workflow.add_edge("analyze_code", "generate_resolution")
        workflow.add_edge("generate_resolution", END)
        
        # Set entry point
        workflow.set_entry_point("analyze_logs")
        
        # Compile with memory
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)

    async def _analyze_logs_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """LangGraph node for log analysis.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with log analysis
        """
        if state.get("log_data"):
            try:
                log_analysis = await self.analyze_logs(
                    state["log_data"],
                    context=state.get("incident_description")
                )
                state["log_analysis"] = log_analysis
                state["analysis_steps"].append("Log analysis completed")
            except Exception as e:
                state["analysis_steps"].append(f"Log analysis failed: {e}")
        
        return state

    async def _analyze_code_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """LangGraph node for code analysis.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with code analysis
        """
        if state.get("code_context"):
            try:
                code_analysis = await self.analyze_code_context(
                    state["code_context"],
                    error_message=state.get("incident_description", "")
                )
                state["code_analysis"] = code_analysis
                state["analysis_steps"].append("Code analysis completed")
            except Exception as e:
                state["analysis_steps"].append(f"Code analysis failed: {e}")
        
        return state

    async def _generate_resolution_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """LangGraph node for resolution generation.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with resolution
        """
        try:
            # Combine all context for resolution
            incident_context = {
                "incident_description": state.get("incident_description", ""),
                "log_data": state.get("log_data", []),
                "metric_data": state.get("metric_data", []),
                "code_context": state.get("code_context", []),
                "runbook_guidance": state.get("runbook_guidance", ""),
                "log_analysis": state.get("log_analysis", {}),
                "code_analysis": state.get("code_analysis", {})
            }
            
            resolution = await self.generate_resolution(incident_context)
            state["resolution"] = resolution
            state["analysis_steps"].append("Resolution generated")
            
        except Exception as e:
            state["analysis_steps"].append(f"Resolution generation failed: {e}")
            state["resolution"] = {
                "resolution_summary": "Failed to generate resolution",
                "detailed_steps": f"Error: {e}",
                "confidence_score": 0.0
            }
        
        return state
