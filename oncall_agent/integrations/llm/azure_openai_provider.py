"""Azure OpenAI LLM provider implementation using LangChain."""

import asyncio
from typing import Dict, List, Optional, Any, AsyncGenerator

try:
    from langchain_openai import AzureChatOpenAI, AzureOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
    LANGCHAIN_AZURE_OPENAI_AVAILABLE = True
except ImportError:
    LANGCHAIN_AZURE_OPENAI_AVAILABLE = False

from ..base.llm_provider import LLMProvider


class AzureOpenAIProvider(LLMProvider):
    """Azure OpenAI LLM provider implementation using LangChain.
    
    Provides integration with Azure OpenAI Service models
    through LangChain for use in LangGraph workflows.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize Azure OpenAI provider.
        
        Args:
            config: Configuration dictionary containing:
                - api_key: Azure OpenAI API key (required)
                - azure_endpoint: Azure OpenAI endpoint URL (required)
                - deployment_name: Deployment name (required)
                - api_version: API version (optional, default: 2024-02-15-preview)
                - max_tokens: Maximum tokens for responses (optional, default: 2000)
                - temperature: Temperature for generation (optional, default: 0.1)
                - timeout: Request timeout in seconds (optional, default: 60)
        """
        if not LANGCHAIN_AZURE_OPENAI_AVAILABLE:
            raise ImportError(
                "LangChain Azure OpenAI not available. Install with: pip install langchain-openai"
            )
        
        super().__init__(config)

    def _validate_config(self) -> None:
        """Validate Azure OpenAI configuration.
        
        Raises:
            ValueError: If required configuration is missing
        """
        required_fields = ["api_key", "azure_endpoint", "deployment_name"]
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Azure OpenAI {field} is required")

    def _setup_model(self) -> None:
        """Set up the LangChain Azure OpenAI model instance."""
        deployment_name = self.config["deployment_name"]
        
        # Determine if this is a chat model based on deployment name
        if any(model in deployment_name.lower() for model in ["gpt-4", "gpt-35-turbo", "gpt-3.5-turbo"]):
            # Use chat model for GPT-4 and GPT-3.5-turbo deployments
            self._model = AzureChatOpenAI(
                deployment_name=deployment_name,
                api_key=self.config["api_key"],
                azure_endpoint=self.config["azure_endpoint"],
                api_version=self.config.get("api_version", "2024-02-15-preview"),
                max_tokens=self.config.get("max_tokens", 2000),
                temperature=self.config.get("temperature", 0.1),
                timeout=self.config.get("timeout", 60),
                streaming=True
            )
        else:
            # Use completion model for other deployments
            self._model = AzureOpenAI(
                deployment_name=deployment_name,
                api_key=self.config["api_key"],
                azure_endpoint=self.config["azure_endpoint"],
                api_version=self.config.get("api_version", "2024-02-15-preview"),
                max_tokens=self.config.get("max_tokens", 2000),
                temperature=self.config.get("temperature", 0.1),
                timeout=self.config.get("timeout", 60)
            )

    @property
    def model(self):
        """Get the LangChain model instance for LangGraph integration.
        
        Returns:
            LangChain Azure OpenAI model instance
        """
        return self._model

    async def generate_resolution(
        self,
        incident_context: Dict[str, Any],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate incident resolution using Azure OpenAI.
        
        Args:
            incident_context: Incident information and context
            system_prompt: Optional system prompt override
            **kwargs: Additional parameters:
                - max_tokens: Override max tokens
                - temperature: Override temperature
                
        Returns:
            Dictionary containing resolution information
            
        Raises:
            ConnectionError: If unable to connect to Azure OpenAI
            ValueError: If input parameters are invalid
        """
        try:
            # Azure OpenAI has similar context windows to OpenAI
            max_tokens = kwargs.get("max_tokens", self.config.get("max_tokens", 2000))
            context_limit = max_tokens * 3
            truncated_context = self.truncate_context(incident_context, context_limit)
            
            # Create messages
            messages = await self.create_chat_messages(truncated_context, system_prompt)
            
            # Override model parameters if specified
            model_kwargs = {}
            if "max_tokens" in kwargs:
                model_kwargs["max_tokens"] = kwargs["max_tokens"]
            if "temperature" in kwargs:
                model_kwargs["temperature"] = kwargs["temperature"]
            
            # Generate response
            if hasattr(self._model, 'ainvoke'):
                response = await self._model.ainvoke(messages, **model_kwargs)
                response_text = response.content if hasattr(response, 'content') else str(response)
            else:
                prompt_text = self._format_incident_context(truncated_context)
                response = await self._model.ainvoke(prompt_text, **model_kwargs)
                response_text = response
            
            # Parse structured response
            return self._parse_resolution_response(response_text)
            
        except Exception as e:
            raise ConnectionError(f"Azure OpenAI generation failed: {e}")

    async def analyze_logs(
        self,
        log_entries: List[Dict[str, Any]],
        context: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Analyze log entries using Azure OpenAI.
        
        Args:
            log_entries: List of log entries to analyze
            context: Additional context about the incident
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing log analysis results
            
        Raises:
            ConnectionError: If unable to connect to Azure OpenAI
        """
        try:
            # Format logs for analysis
            log_text = "\n".join([
                f"[{log.get('timestamp', 'unknown')}] {log.get('level', 'INFO')}: {log.get('message', '')}"
                for log in log_entries[:20]  # Standard limit
            ])
            
            # Create analysis prompt
            analysis_prompt = f"""
            Analyze the following log entries to identify patterns and issues:
            
            {f"Context: {context}" if context else ""}
            
            Log Entries:
            {log_text}
            
            Please provide:
            1. Error patterns identified
            2. Severity assessment
            3. Timeline of events
            4. Affected components
            5. Suggested follow-up queries
            """
            
            # Generate analysis
            if hasattr(self._model, 'ainvoke'):
                messages = [HumanMessage(content=analysis_prompt)]
                response = await self._model.ainvoke(messages)
                response_text = response.content if hasattr(response, 'content') else str(response)
            else:
                response_text = await self._model.ainvoke(analysis_prompt)
            
            return self._parse_log_analysis_response(response_text)
            
        except Exception as e:
            raise ConnectionError(f"Azure OpenAI log analysis failed: {e}")

    async def analyze_code_context(
        self,
        code_snippets: List[Dict[str, Any]],
        error_message: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Analyze code context using Azure OpenAI.
        
        Args:
            code_snippets: List of relevant code snippets
            error_message: Error message or stack trace
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing code analysis results
            
        Raises:
            ConnectionError: If unable to connect to Azure OpenAI
        """
        try:
            # Format code for analysis
            code_text = "\n\n".join([
                f"File: {code.get('file_path', 'unknown')}\n```{code.get('language', '')}\n{code.get('content', '')[:500]}...\n```"
                for code in code_snippets[:5]  # Standard limit
            ])
            
            # Create analysis prompt
            analysis_prompt = f"""
            Analyze this code error and suggest fixes:
            
            Error: {error_message}
            
            Code Context:
            {code_text}
            
            Please provide:
            1. Potential issues in the code
            2. Specific fix suggestions
            3. Best practices recommendations
            4. Testing recommendations
            """
            
            # Generate analysis
            if hasattr(self._model, 'ainvoke'):
                messages = [HumanMessage(content=analysis_prompt)]
                response = await self._model.ainvoke(messages)
                response_text = response.content if hasattr(response, 'content') else str(response)
            else:
                response_text = await self._model.ainvoke(analysis_prompt)
            
            return self._parse_code_analysis_response(response_text)
            
        except Exception as e:
            raise ConnectionError(f"Azure OpenAI code analysis failed: {e}")

    async def stream_response(
        self,
        prompt: str,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream Azure OpenAI response for real-time updates.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters
            
        Yields:
            Chunks of the response as they are generated
            
        Raises:
            ConnectionError: If unable to connect to Azure OpenAI
        """
        try:
            if hasattr(self._model, 'astream'):
                messages = [HumanMessage(content=prompt)]
                async for chunk in self._model.astream(messages):
                    if hasattr(chunk, 'content'):
                        yield chunk.content
                    else:
                        yield str(chunk)
            else:
                # Fallback for models that don't support streaming
                response = await self._model.ainvoke(prompt)
                yield response
                
        except Exception as e:
            raise ConnectionError(f"Azure OpenAI streaming failed: {e}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the Azure OpenAI model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "provider": "azure_openai",
            "deployment_name": self.config["deployment_name"],
            "azure_endpoint": self.config["azure_endpoint"],
            "api_version": self.config.get("api_version", "2024-02-15-preview"),
            "max_tokens": self.config.get("max_tokens", 2000),
            "temperature": self.config.get("temperature", 0.1),
            "supports_streaming": True,
            "supports_functions": True,  # Azure OpenAI supports function calling
            "supports_vision": "gpt-4" in self.config["deployment_name"].lower() and "vision" in self.config["deployment_name"].lower(),
            "context_window": self._get_context_window()
        }

    def _get_context_window(self) -> int:
        """Get the context window size for the deployment.
        
        Returns:
            Context window size in tokens
        """
        deployment_name = self.config["deployment_name"].lower()
        
        # Map deployment names to context windows (similar to OpenAI)
        if "gpt-4" in deployment_name:
            if "32k" in deployment_name:
                return 32768
            elif "turbo" in deployment_name:
                return 128000
            else:
                return 8192
        elif "gpt-35-turbo" in deployment_name or "gpt-3.5-turbo" in deployment_name:
            if "16k" in deployment_name:
                return 16384
            else:
                return 4096
        elif "text-davinci" in deployment_name:
            return 4097
        else:
            return 4096  # Conservative default

    def _parse_resolution_response(self, response_text: str) -> Dict[str, Any]:
        """Parse structured resolution response.
        
        Args:
            response_text: Raw response from Azure OpenAI
            
        Returns:
            Structured resolution dictionary
        """
        return {
            "resolution_summary": self._extract_section(response_text, "Summary") or "Resolution provided",
            "detailed_steps": self._extract_section(response_text, "Resolution Steps") or response_text,
            "code_changes": self._extract_section(response_text, "Code Changes") or "",
            "root_cause_analysis": self._extract_section(response_text, "Root Cause") or "",
            "confidence_score": 0.8,  # Similar quality to OpenAI
            "reasoning": response_text,
            "additional_recommendations": self._extract_section(response_text, "Prevention") or ""
        }

    def _parse_log_analysis_response(self, response_text: str) -> Dict[str, Any]:
        """Parse log analysis response.
        
        Args:
            response_text: Raw response from Azure OpenAI
            
        Returns:
            Structured log analysis dictionary
        """
        return {
            "error_patterns": self._extract_list_items(response_text, "Error patterns") or [],
            "severity_assessment": self._extract_section(response_text, "Severity") or "Medium",
            "timeline": self._extract_section(response_text, "Timeline") or "",
            "affected_components": self._extract_list_items(response_text, "Affected components") or [],
            "suggested_queries": self._extract_list_items(response_text, "Suggested queries") or [],
            "full_analysis": response_text
        }

    def _parse_code_analysis_response(self, response_text: str) -> Dict[str, Any]:
        """Parse code analysis response.
        
        Args:
            response_text: Raw response from Azure OpenAI
            
        Returns:
            Structured code analysis dictionary
        """
        return {
            "potential_issues": self._extract_list_items(response_text, "Potential issues") or [],
            "suggested_fixes": self._extract_section(response_text, "Fix suggestions") or "",
            "best_practices": self._extract_list_items(response_text, "Best practices") or [],
            "testing_recommendations": self._extract_list_items(response_text, "Testing") or [],
            "full_analysis": response_text
        }

    def _extract_section(self, text: str, section_name: str) -> Optional[str]:
        """Extract a specific section from formatted response.
        
        Args:
            text: Response text
            section_name: Name of section to extract
            
        Returns:
            Extracted section content or None
        """
        import re
        
        # Look for section headers (markdown style)
        pattern = rf'\*\*{re.escape(section_name)}[:\*]*\s*\*\*\s*\n(.*?)(?=\n\*\*|\n#|\Z)'
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        
        if match:
            return match.group(1).strip()
        
        # Fallback: look for simpler patterns
        pattern = rf'{re.escape(section_name)}[:\s]*\n(.*?)(?=\n[A-Z][a-z]+:|\Z)'
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        
        if match:
            return match.group(1).strip()
        
        return None

    def _extract_list_items(self, text: str, section_name: str) -> List[str]:
        """Extract list items from a section.
        
        Args:
            text: Response text
            section_name: Name of section containing the list
            
        Returns:
            List of extracted items
        """
        section_content = self._extract_section(text, section_name)
        if not section_content:
            return []
        
        # Extract list items (markdown style)
        import re
        items = re.findall(r'^[\s]*[-*+]\s+(.+)$', section_content, re.MULTILINE)
        
        # Also try numbered lists
        if not items:
            items = re.findall(r'^[\s]*\d+\.\s+(.+)$', section_content, re.MULTILINE)
        
        return [item.strip() for item in items if item.strip()]
