"""Google Gemini LLM provider implementation using LangChain."""

import asyncio
from typing import Dict, List, Optional, Any, AsyncGenerator

try:
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAI
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
    LANGCHAIN_GEMINI_AVAILABLE = True
except ImportError:
    LANGCHAIN_GEMINI_AVAILABLE = False

from ..base.llm_provider import LLMProvider


class GeminiProvider(LLMProvider):
    """Google Gemini LLM provider implementation using LangChain.
    
    Provides integration with Google Gemini models (Gemini Pro, Gemini Pro Vision)
    through LangChain for use in LangGraph workflows.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize Gemini provider.
        
        Args:
            config: Configuration dictionary containing:
                - api_key: Google API key (required)
                - model_name: Model name (optional, default: gemini-pro)
                - max_tokens: Maximum tokens for responses (optional, default: 2000)
                - temperature: Temperature for generation (optional, default: 0.1)
                - timeout: Request timeout in seconds (optional, default: 60)
                - safety_settings: Safety filter settings (optional)
        """
        if not LANGCHAIN_GEMINI_AVAILABLE:
            raise ImportError(
                "LangChain Google GenAI not available. Install with: pip install langchain-google-genai"
            )
        
        super().__init__(config)

    def _validate_config(self) -> None:
        """Validate Gemini configuration.
        
        Raises:
            ValueError: If required configuration is missing
        """
        if "api_key" not in self.config:
            raise ValueError("Google API key is required for Gemini provider")

    def _setup_model(self) -> None:
        """Set up the LangChain Gemini model instance."""
        model_name = self.config.get("model_name", "gemini-pro")
        
        # Use chat model for Gemini Pro
        self._model = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=self.config["api_key"],
            max_tokens=self.config.get("max_tokens", 2000),
            temperature=self.config.get("temperature", 0.1),
            timeout=self.config.get("timeout", 60),
            safety_settings=self.config.get("safety_settings", {}),
            convert_system_message_to_human=True  # Gemini doesn't support system messages directly
        )

    @property
    def model(self):
        """Get the LangChain model instance for LangGraph integration.
        
        Returns:
            LangChain Gemini model instance
        """
        return self._model

    async def generate_resolution(
        self,
        incident_context: Dict[str, Any],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate incident resolution using Gemini.
        
        Args:
            incident_context: Incident information and context
            system_prompt: Optional system prompt override
            **kwargs: Additional parameters:
                - max_tokens: Override max tokens
                - temperature: Override temperature
                
        Returns:
            Dictionary containing resolution information
            
        Raises:
            ConnectionError: If unable to connect to Gemini
            ValueError: If input parameters are invalid
        """
        try:
            # Gemini has large context windows but we'll be conservative
            max_tokens = kwargs.get("max_tokens", self.config.get("max_tokens", 2000))
            context_limit = max_tokens * 4  # Gemini can handle substantial context
            truncated_context = self.truncate_context(incident_context, context_limit)
            
            # Create messages (Gemini converts system message to human message)
            messages = await self.create_chat_messages(truncated_context, system_prompt)
            
            # Override model parameters if specified
            model_kwargs = {}
            if "max_tokens" in kwargs:
                model_kwargs["max_tokens"] = kwargs["max_tokens"]
            if "temperature" in kwargs:
                model_kwargs["temperature"] = kwargs["temperature"]
            
            # Generate response
            response = await self._model.ainvoke(messages, **model_kwargs)
            response_text = response.content
            
            # Parse structured response
            return self._parse_resolution_response(response_text)
            
        except Exception as e:
            raise ConnectionError(f"Gemini generation failed: {e}")

    async def analyze_logs(
        self,
        log_entries: List[Dict[str, Any]],
        context: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Analyze log entries using Gemini.
        
        Args:
            log_entries: List of log entries to analyze
            context: Additional context about the incident
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing log analysis results
            
        Raises:
            ConnectionError: If unable to connect to Gemini
        """
        try:
            # Format logs for analysis
            log_text = "\n".join([
                f"[{log.get('timestamp', 'unknown')}] {log.get('level', 'INFO')}: {log.get('message', '')}"
                for log in log_entries[:25]  # Gemini can handle more logs
            ])
            
            # Create analysis prompt
            analysis_prompt = f"""
            Analyze these log entries for an incident investigation:
            
            {f"Context: {context}" if context else ""}
            
            Log Entries:
            {log_text}
            
            Please provide a structured analysis:
            1. **Error Patterns**: What error patterns do you identify?
            2. **Severity Assessment**: How severe is this issue?
            3. **Timeline**: What sequence of events occurred?
            4. **Affected Components**: Which systems are affected?
            5. **Suggested Queries**: What additional queries would help?
            """
            
            # Generate analysis
            messages = [HumanMessage(content=analysis_prompt)]
            response = await self._model.ainvoke(messages)
            response_text = response.content
            
            return self._parse_log_analysis_response(response_text)
            
        except Exception as e:
            raise ConnectionError(f"Gemini log analysis failed: {e}")

    async def analyze_code_context(
        self,
        code_snippets: List[Dict[str, Any]],
        error_message: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Analyze code context using Gemini.
        
        Args:
            code_snippets: List of relevant code snippets
            error_message: Error message or stack trace
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing code analysis results
            
        Raises:
            ConnectionError: If unable to connect to Gemini
        """
        try:
            # Format code for analysis
            code_text = "\n\n".join([
                f"**File:** {code.get('file_path', 'unknown')}\n```{code.get('language', '')}\n{code.get('content', '')}\n```"
                for code in code_snippets[:6]  # Gemini can handle more code
            ])
            
            # Create analysis prompt
            analysis_prompt = f"""
            Please analyze this code in relation to the following error:
            
            **Error:**
            {error_message}
            
            **Code Context:**
            {code_text}
            
            Please provide:
            1. **Potential Issues**: What could be causing this error?
            2. **Suggested Fixes**: Specific code changes to resolve the issue
            3. **Best Practices**: How can this code be improved?
            4. **Testing Recommendations**: How should fixes be tested?
            """
            
            # Generate analysis
            messages = [HumanMessage(content=analysis_prompt)]
            response = await self._model.ainvoke(messages)
            response_text = response.content
            
            return self._parse_code_analysis_response(response_text)
            
        except Exception as e:
            raise ConnectionError(f"Gemini code analysis failed: {e}")

    async def stream_response(
        self,
        prompt: str,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream Gemini response for real-time updates.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters
            
        Yields:
            Chunks of the response as they are generated
            
        Raises:
            ConnectionError: If unable to connect to Gemini
        """
        try:
            messages = [HumanMessage(content=prompt)]
            async for chunk in self._model.astream(messages):
                if hasattr(chunk, 'content'):
                    yield chunk.content
                else:
                    yield str(chunk)
                    
        except Exception as e:
            raise ConnectionError(f"Gemini streaming failed: {e}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the Gemini model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "provider": "gemini",
            "model_name": self.config.get("model_name", "gemini-pro"),
            "max_tokens": self.config.get("max_tokens", 2000),
            "temperature": self.config.get("temperature", 0.1),
            "supports_streaming": True,
            "supports_functions": False,  # Gemini doesn't support function calling like OpenAI
            "supports_vision": "vision" in self.config.get("model_name", "gemini-pro"),
            "context_window": self._get_context_window(),
            "safety_settings": bool(self.config.get("safety_settings"))
        }

    def _get_context_window(self) -> int:
        """Get the context window size for the model.
        
        Returns:
            Context window size in tokens
        """
        model_name = self.config.get("model_name", "gemini-pro")
        
        # Gemini model context windows
        context_windows = {
            "gemini-pro": 32768,
            "gemini-pro-vision": 16384,
            "gemini-1.5-pro": 1000000,  # 1M token context window
            "gemini-1.5-flash": 1000000
        }
        
        return context_windows.get(model_name, 32768)

    def _parse_resolution_response(self, response_text: str) -> Dict[str, Any]:
        """Parse structured resolution response.
        
        Args:
            response_text: Raw response from Gemini
            
        Returns:
            Structured resolution dictionary
        """
        return {
            "resolution_summary": self._extract_section(response_text, "Summary") or "Resolution provided",
            "detailed_steps": self._extract_section(response_text, "Resolution Steps") or response_text,
            "code_changes": self._extract_section(response_text, "Code Changes") or "",
            "root_cause_analysis": self._extract_section(response_text, "Root Cause") or "",
            "confidence_score": 0.8,  # Gemini provides good quality responses
            "reasoning": response_text,
            "additional_recommendations": self._extract_section(response_text, "Prevention") or ""
        }

    def _parse_log_analysis_response(self, response_text: str) -> Dict[str, Any]:
        """Parse log analysis response.
        
        Args:
            response_text: Raw response from Gemini
            
        Returns:
            Structured log analysis dictionary
        """
        return {
            "error_patterns": self._extract_list_items(response_text, "Error Patterns") or [],
            "severity_assessment": self._extract_section(response_text, "Severity Assessment") or "Medium",
            "timeline": self._extract_section(response_text, "Timeline") or "",
            "affected_components": self._extract_list_items(response_text, "Affected Components") or [],
            "suggested_queries": self._extract_list_items(response_text, "Suggested Queries") or [],
            "full_analysis": response_text
        }

    def _parse_code_analysis_response(self, response_text: str) -> Dict[str, Any]:
        """Parse code analysis response.
        
        Args:
            response_text: Raw response from Gemini
            
        Returns:
            Structured code analysis dictionary
        """
        return {
            "potential_issues": self._extract_list_items(response_text, "Potential Issues") or [],
            "suggested_fixes": self._extract_section(response_text, "Suggested Fixes") or "",
            "best_practices": self._extract_list_items(response_text, "Best Practices") or [],
            "testing_recommendations": self._extract_list_items(response_text, "Testing Recommendations") or [],
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
        
        # Look for numbered sections
        pattern = rf'\d+\.\s*\*\*{re.escape(section_name)}[:\*]*\s*\*\*\s*\n(.*?)(?=\n\d+\.|\n\*\*|\Z)'
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
