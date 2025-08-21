"""Abstract base class for LLM providers using LangChain."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, AsyncGenerator
from datetime import datetime
from enum import Enum

try:
    from langchain_core.language_models import BaseLLM, BaseChatModel
    from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
    from langchain_core.outputs import LLMResult, ChatResult
    from langchain_core.callbacks import CallbackManagerForLLMRun
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    # Fallback types for when LangChain is not available
    BaseLLM = object
    BaseChatModel = object
    BaseMessage = object
    LLMResult = object
    ChatResult = object


class LLMProvider(ABC):
    """Abstract base class for all LLM providers.
    
    This class defines the interface that all LLM providers must implement
    to integrate with the OnCall AI Agent system and LangGraph workflows.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the LLM provider with configuration.
        
        Args:
            config: Provider-specific configuration dictionary
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain not available. Install with: pip install langchain langchain-core"
            )
        
        self.config = config
        self._validate_config()
        self._setup_model()

    @abstractmethod
    def _validate_config(self) -> None:
        """Validate the provider configuration.
        
        Raises:
            ValueError: If configuration is invalid or missing required fields
        """
        pass

    @abstractmethod
    def _setup_model(self) -> None:
        """Set up the LangChain model instance.
        
        This should create self.model as a LangChain LLM instance.
        """
        pass

    @property
    @abstractmethod
    def model(self) -> BaseLLM:
        """Get the LangChain model instance.
        
        Returns:
            LangChain LLM instance for use in LangGraph workflows
        """
        pass

    @abstractmethod
    async def generate_resolution(
        self,
        incident_context: Dict[str, Any],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate incident resolution using the LLM.
        
        Args:
            incident_context: Dictionary containing:
                - incident_description: Description of the incident
                - log_data: Relevant log entries
                - metric_data: Relevant metrics and alarms
                - code_context: Relevant code snippets
                - runbook_guidance: Relevant runbook content
                - additional_context: Any other relevant information
            system_prompt: Optional system prompt override
            **kwargs: Provider-specific parameters
            
        Returns:
            Dictionary containing:
            - resolution_summary: High-level resolution summary
            - detailed_steps: Step-by-step resolution instructions
            - code_changes: Suggested code changes (if any)
            - root_cause_analysis: Analysis of the root cause
            - confidence_score: Confidence in the resolution (0-1)
            - reasoning: Explanation of the reasoning
            - additional_recommendations: Additional suggestions
            
        Raises:
            ConnectionError: If unable to connect to LLM service
            ValueError: If input parameters are invalid
        """
        pass

    @abstractmethod
    async def analyze_logs(
        self,
        log_entries: List[Dict[str, Any]],
        context: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Analyze log entries to identify patterns and issues.
        
        Args:
            log_entries: List of log entries to analyze
            context: Additional context about the incident
            **kwargs: Provider-specific parameters
            
        Returns:
            Dictionary containing:
            - error_patterns: Identified error patterns
            - severity_assessment: Assessment of issue severity
            - timeline: Timeline of events from logs
            - affected_components: Components affected by the issue
            - suggested_queries: Suggested follow-up queries
            
        Raises:
            ConnectionError: If unable to connect to LLM service
        """
        pass

    @abstractmethod
    async def analyze_code_context(
        self,
        code_snippets: List[Dict[str, Any]],
        error_message: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Analyze code context to suggest fixes.
        
        Args:
            code_snippets: List of relevant code snippets
            error_message: Error message or stack trace
            **kwargs: Provider-specific parameters
            
        Returns:
            Dictionary containing:
            - potential_issues: Identified potential issues in code
            - suggested_fixes: Specific code fix suggestions
            - best_practices: Relevant best practices
            - testing_recommendations: Testing suggestions
            
        Raises:
            ConnectionError: If unable to connect to LLM service
        """
        pass

    @abstractmethod
    async def stream_response(
        self,
        prompt: str,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream LLM response for real-time updates.
        
        Args:
            prompt: Input prompt
            **kwargs: Provider-specific parameters
            
        Yields:
            Chunks of the response as they are generated
            
        Raises:
            ConnectionError: If unable to connect to LLM service
        """
        pass

    async def health_check(self) -> Dict[str, Any]:
        """Check if the LLM provider is accessible and functional.
        
        Returns:
            Dictionary containing:
            - healthy: Whether the provider is accessible
            - model_info: Information about the model
            - latency: Response latency in milliseconds
            - error: Error message if unhealthy
        """
        start_time = datetime.now()
        
        try:
            # Simple test prompt
            test_context = {
                "incident_description": "Test connectivity",
                "log_data": [],
                "metric_data": [],
                "code_context": [],
                "runbook_guidance": ""
            }
            
            result = await self.generate_resolution(test_context)
            
            end_time = datetime.now()
            latency = (end_time - start_time).total_seconds() * 1000
            
            return {
                "healthy": True,
                "model_info": self.get_model_info(),
                "latency": latency,
                "error": None
            }
            
        except Exception as e:
            end_time = datetime.now()
            latency = (end_time - start_time).total_seconds() * 1000
            
            return {
                "healthy": False,
                "model_info": self.get_model_info(),
                "latency": latency,
                "error": str(e)
            }

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model.
        
        Returns:
            Dictionary containing model information:
            - provider: Provider name (openai, anthropic, etc.)
            - model_name: Model name/identifier
            - max_tokens: Maximum token limit
            - supports_streaming: Whether streaming is supported
            - supports_functions: Whether function calling is supported
        """
        pass

    def get_langchain_model(self) -> BaseLLM:
        """Get the underlying LangChain model for LangGraph integration.
        
        This is the key method for LangGraph integration.
        
        Returns:
            LangChain model instance that can be used directly in LangGraph workflows
        """
        return self.model

    async def create_chat_messages(
        self,
        incident_context: Dict[str, Any],
        system_prompt: Optional[str] = None
    ) -> List[BaseMessage]:
        """Create LangChain chat messages from incident context.
        
        Args:
            incident_context: Incident information
            system_prompt: Optional system prompt override
            
        Returns:
            List of LangChain message objects for chat models
        """
        messages = []
        
        # System message
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        else:
            default_system = self._get_default_system_prompt()
            messages.append(SystemMessage(content=default_system))
        
        # Human message with incident context
        human_content = self._format_incident_context(incident_context)
        messages.append(HumanMessage(content=human_content))
        
        return messages

    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for incident resolution.
        
        Returns:
            Default system prompt
        """
        return """You are an expert DevOps engineer and incident response specialist. 
Your role is to analyze incidents and provide clear, actionable resolution steps.

When analyzing an incident, consider:
1. Log patterns and error messages
2. Metric anomalies and trends
3. Recent code changes that might be related
4. Runbook procedures and best practices
5. Root cause analysis

Provide responses in this format:
- **Summary**: Brief description of the issue
- **Root Cause**: Analysis of what caused the issue
- **Resolution Steps**: Clear, numbered steps to resolve
- **Code Changes**: Specific code fixes if needed
- **Prevention**: How to prevent this issue in the future

Be specific, actionable, and include relevant commands or code snippets."""

    def _format_incident_context(self, context: Dict[str, Any]) -> str:
        """Format incident context into a structured prompt.
        
        Args:
            context: Incident context dictionary
            
        Returns:
            Formatted context string
        """
        parts = []
        
        # Incident description
        if context.get("incident_description"):
            parts.append(f"**Incident Description:**\n{context['incident_description']}")
        
        # Log data
        if context.get("log_data"):
            log_summary = "\n".join([
                f"[{log.get('timestamp', 'unknown')}] {log.get('level', 'INFO')}: {log.get('message', '')}"
                for log in context["log_data"][:10]  # Limit to 10 most recent
            ])
            parts.append(f"**Recent Logs:**\n{log_summary}")
        
        # Metric data
        if context.get("metric_data"):
            metric_summary = "\n".join([
                f"- {metric.get('name', 'unknown')}: {metric.get('value', 'N/A')} {metric.get('unit', '')}"
                for metric in context["metric_data"][:5]  # Limit to 5 key metrics
            ])
            parts.append(f"**Key Metrics:**\n{metric_summary}")
        
        # Code context
        if context.get("code_context"):
            code_summary = "\n".join([
                f"File: {code.get('file_path', 'unknown')}\n```\n{code.get('content', '')[:500]}...\n```"
                for code in context["code_context"][:3]  # Limit to 3 code snippets
            ])
            parts.append(f"**Relevant Code:**\n{code_summary}")
        
        # Runbook guidance
        if context.get("runbook_guidance"):
            guidance = context["runbook_guidance"]
            if len(guidance) > 2000:  # Truncate if too long
                guidance = guidance[:2000] + "\n...[truncated]"
            parts.append(f"**Runbook Guidance:**\n{guidance}")
        
        # Additional context
        if context.get("additional_context"):
            parts.append(f"**Additional Context:**\n{context['additional_context']}")
        
        return "\n\n".join(parts)

    def get_token_count(self, text: str) -> int:
        """Estimate token count for text.
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        # Simple estimation: ~4 characters per token for English text
        return len(text) // 4

    def truncate_context(
        self,
        context: Dict[str, Any],
        max_tokens: int
    ) -> Dict[str, Any]:
        """Truncate context to fit within token limits.
        
        Args:
            context: Incident context
            max_tokens: Maximum tokens allowed
            
        Returns:
            Truncated context dictionary
        """
        # Estimate current token usage
        formatted_context = self._format_incident_context(context)
        current_tokens = self.get_token_count(formatted_context)
        
        if current_tokens <= max_tokens:
            return context
        
        # Truncate in order of priority (keep most important data)
        truncated_context = context.copy()
        
        # 1. Truncate additional context first
        if "additional_context" in truncated_context:
            del truncated_context["additional_context"]
        
        # 2. Truncate runbook guidance
        if "runbook_guidance" in truncated_context and len(truncated_context["runbook_guidance"]) > 1000:
            truncated_context["runbook_guidance"] = truncated_context["runbook_guidance"][:1000] + "...[truncated]"
        
        # 3. Limit code context
        if "code_context" in truncated_context and len(truncated_context["code_context"]) > 2:
            truncated_context["code_context"] = truncated_context["code_context"][:2]
        
        # 4. Limit log data
        if "log_data" in truncated_context and len(truncated_context["log_data"]) > 5:
            truncated_context["log_data"] = truncated_context["log_data"][:5]
        
        # 5. Limit metric data
        if "metric_data" in truncated_context and len(truncated_context["metric_data"]) > 3:
            truncated_context["metric_data"] = truncated_context["metric_data"][:3]
        
        return truncated_context
