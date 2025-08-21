"""Ollama LLM provider implementation using LangChain."""

import asyncio
from typing import Dict, List, Optional, Any, AsyncGenerator

try:
    from langchain_community.llms import Ollama
    from langchain_community.chat_models import ChatOllama
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
    LANGCHAIN_OLLAMA_AVAILABLE = True
except ImportError:
    LANGCHAIN_OLLAMA_AVAILABLE = False

from ..base.llm_provider import LLMProvider


class OllamaProvider(LLMProvider):
    """Ollama LLM provider implementation using LangChain.
    
    Provides integration with locally hosted Ollama models
    through LangChain for use in LangGraph workflows.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize Ollama provider.
        
        Args:
            config: Configuration dictionary containing:
                - model: Model name (required, e.g., 'llama2', 'codellama', 'mistral')
                - base_url: Ollama server URL (optional, default: http://localhost:11434)
                - temperature: Temperature for generation (optional, default: 0.1)
                - timeout: Request timeout in seconds (optional, default: 120)
                - use_chat_model: Whether to use chat format (optional, default: True)
        """
        if not LANGCHAIN_OLLAMA_AVAILABLE:
            raise ImportError(
                "LangChain Ollama not available. Install with: pip install langchain-community"
            )
        
        super().__init__(config)

    def _validate_config(self) -> None:
        """Validate Ollama configuration.
        
        Raises:
            ValueError: If required configuration is missing
        """
        if "model_name" not in self.config:
            raise ValueError("Ollama model_name is required")

    def _setup_model(self) -> None:
        """Set up the LangChain Ollama model instance."""
        base_url = self.config.get("base_url", "http://localhost:11434")
        model_name = self.config["model_name"]
        temperature = self.config.get("temperature", 0.1)
        timeout = self.config.get("timeout", 120)
        
        # Use chat model by default for better conversation handling
        if self.config.get("use_chat_model", True):
            self._model = ChatOllama(
                model=model_name,
                base_url=base_url,
                temperature=temperature,
                timeout=timeout
            )
        else:
            self._model = Ollama(
                model=model_name,
                base_url=base_url,
                temperature=temperature,
                timeout=timeout
            )

    @property
    def model(self):
        """Get the LangChain model instance for LangGraph integration.
        
        Returns:
            LangChain Ollama model instance
        """
        return self._model

    async def generate_resolution(
        self,
        incident_context: Dict[str, Any],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate incident resolution using Ollama.
        
        Args:
            incident_context: Incident information and context
            system_prompt: Optional system prompt override
            **kwargs: Additional parameters:
                - temperature: Override temperature
                
        Returns:
            Dictionary containing resolution information
            
        Raises:
            ConnectionError: If unable to connect to Ollama
            ValueError: If input parameters are invalid
        """
        try:
            # Local models typically have smaller context windows
            context_limit = 4000  # Conservative limit for local models
            truncated_context = self.truncate_context(incident_context, context_limit)
            
            # Create messages or prompt based on model type
            if hasattr(self._model, 'ainvoke') and hasattr(self._model, '_llm_type'):
                if 'chat' in self._model._llm_type.lower():
                    # Chat model
                    messages = await self.create_chat_messages(truncated_context, system_prompt)
                    response = await self._model.ainvoke(messages)
                    response_text = response.content if hasattr(response, 'content') else str(response)
                else:
                    # Completion model
                    prompt_text = self._format_incident_context(truncated_context)
                    if system_prompt:
                        prompt_text = f"{system_prompt}\n\n{prompt_text}"
                    response = await self._model.ainvoke(prompt_text)
                    response_text = response
            else:
                # Fallback approach
                prompt_text = self._format_incident_context(truncated_context)
                if system_prompt:
                    prompt_text = f"{system_prompt}\n\n{prompt_text}"
                response = await self._model.ainvoke(prompt_text)
                response_text = response if isinstance(response, str) else str(response)
            
            # Parse structured response
            return self._parse_resolution_response(response_text)
            
        except Exception as e:
            raise ConnectionError(f"Ollama generation failed: {e}")

    async def analyze_logs(
        self,
        log_entries: List[Dict[str, Any]],
        context: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Analyze log entries using Ollama.
        
        Args:
            log_entries: List of log entries to analyze
            context: Additional context about the incident
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing log analysis results
            
        Raises:
            ConnectionError: If unable to connect to Ollama
        """
        try:
            # Format logs for analysis (limit for local models)
            log_text = "\n".join([
                f"[{log.get('timestamp', 'unknown')}] {log.get('level', 'INFO')}: {log.get('message', '')}"
                for log in log_entries[:15]  # Limit for local models
            ])
            
            # Create analysis prompt
            analysis_prompt = f"""
            Analyze these log entries to identify issues and patterns:
            
            {f"Context: {context}" if context else ""}
            
            Logs:
            {log_text}
            
            Provide:
            1. Error patterns found
            2. Severity level
            3. Event timeline
            4. Affected systems
            5. Next investigation steps
            """
            
            # Generate analysis
            if hasattr(self._model, 'ainvoke') and 'chat' in getattr(self._model, '_llm_type', '').lower():
                messages = [HumanMessage(content=analysis_prompt)]
                response = await self._model.ainvoke(messages)
                response_text = response.content if hasattr(response, 'content') else str(response)
            else:
                response = await self._model.ainvoke(analysis_prompt)
                response_text = response if isinstance(response, str) else str(response)
            
            return self._parse_log_analysis_response(response_text)
            
        except Exception as e:
            raise ConnectionError(f"Ollama log analysis failed: {e}")

    async def analyze_code_context(
        self,
        code_snippets: List[Dict[str, Any]],
        error_message: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Analyze code context using Ollama.
        
        Args:
            code_snippets: List of relevant code snippets
            error_message: Error message or stack trace
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing code analysis results
            
        Raises:
            ConnectionError: If unable to connect to Ollama
        """
        try:
            # Format code for analysis (limit for local models)
            code_text = "\n\n".join([
                f"File: {code.get('file_path', 'unknown')}\n```\n{code.get('content', '')[:1000]}...\n```"
                for code in code_snippets[:3]  # Limit for local models
            ])
            
            # Create analysis prompt
            analysis_prompt = f"""
            Analyze this code error and suggest fixes:
            
            Error: {error_message}
            
            Code:
            {code_text}
            
            Provide:
            1. What's causing the error
            2. How to fix it
            3. Code improvements
            4. Testing approach
            """
            
            # Generate analysis
            if hasattr(self._model, 'ainvoke') and 'chat' in getattr(self._model, '_llm_type', '').lower():
                messages = [HumanMessage(content=analysis_prompt)]
                response = await self._model.ainvoke(messages)
                response_text = response.content if hasattr(response, 'content') else str(response)
            else:
                response = await self._model.ainvoke(analysis_prompt)
                response_text = response if isinstance(response, str) else str(response)
            
            return self._parse_code_analysis_response(response_text)
            
        except Exception as e:
            raise ConnectionError(f"Ollama code analysis failed: {e}")

    async def stream_response(
        self,
        prompt: str,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream Ollama response for real-time updates.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters
            
        Yields:
            Chunks of the response as they are generated
            
        Raises:
            ConnectionError: If unable to connect to Ollama
        """
        try:
            if hasattr(self._model, 'astream'):
                if hasattr(self._model, '_llm_type') and 'chat' in self._model._llm_type.lower():
                    messages = [HumanMessage(content=prompt)]
                    async for chunk in self._model.astream(messages):
                        if hasattr(chunk, 'content'):
                            yield chunk.content
                        else:
                            yield str(chunk)
                else:
                    async for chunk in self._model.astream(prompt):
                        yield str(chunk)
            else:
                # Fallback for models that don't support streaming
                response = await self._model.ainvoke(prompt)
                yield response if isinstance(response, str) else str(response)
                
        except Exception as e:
            raise ConnectionError(f"Ollama streaming failed: {e}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the Ollama model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "provider": "ollama",
            "model_name": self.config["model_name"],
            "base_url": self.config.get("base_url", "http://localhost:11434"),
            "max_tokens": self.config.get("max_tokens", 2000),
            "temperature": self.config.get("temperature", 0.1),
            "supports_streaming": True,
            "supports_functions": False,  # Most local models don't support function calling
            "supports_vision": "vision" in self.config["model"].lower(),
            "context_window": self._get_context_window(),
            "local_model": True
        }

    def _get_context_window(self) -> int:
        """Get the context window size for the model.
        
        Returns:
            Context window size in tokens
        """
        model_name = self.config["model_name"].lower()
        
        # Common Ollama model context windows
        context_windows = {
            "llama2": 4096,
            "llama2:13b": 4096,
            "llama2:70b": 4096,
            "codellama": 16384,
            "codellama:13b": 16384,
            "codellama:34b": 16384,
            "mistral": 8192,
            "mistral:7b": 8192,
            "mixtral": 32768,
            "neural-chat": 4096,
            "starling-lm": 8192,
            "orca-mini": 4096
        }
        
        # Try exact match first
        if model_name in context_windows:
            return context_windows[model_name]
        
        # Try partial match
        for model_key, context_size in context_windows.items():
            if model_key in model_name:
                return context_size
        
        # Default conservative estimate
        return 4096

    async def check_model_availability(self) -> Dict[str, Any]:
        """Check if the specified model is available in Ollama.
        
        Returns:
            Dictionary containing availability information
        """
        try:
            # Try to make a simple request to test model availability
            test_prompt = "Hello"
            
            if hasattr(self._model, 'ainvoke'):
                response = await self._model.ainvoke(test_prompt)
                
                return {
                    "available": True,
                    "model": self.config["model_name"],
                    "base_url": self.config.get("base_url", "http://localhost:11434"),
                    "error": None
                }
            else:
                return {
                    "available": False,
                    "model": self.config["model_name"],
                    "base_url": self.config.get("base_url", "http://localhost:11434"),
                    "error": "Model interface not available"
                }
                
        except Exception as e:
            return {
                "available": False,
                "model": self.config["model_name"],
                "base_url": self.config.get("base_url", "http://localhost:11434"),
                "error": str(e)
            }

    def _parse_resolution_response(self, response_text: str) -> Dict[str, Any]:
        """Parse structured resolution response.
        
        Args:
            response_text: Raw response from Ollama
            
        Returns:
            Structured resolution dictionary
        """
        # Local models may have less structured output, so be more flexible
        return {
            "resolution_summary": self._extract_section(response_text, "Summary") or response_text.split('\n')[0],
            "detailed_steps": self._extract_section(response_text, "Steps") or response_text,
            "code_changes": self._extract_section(response_text, "Code") or "",
            "root_cause_analysis": self._extract_section(response_text, "Cause") or "",
            "confidence_score": 0.7,  # Local models may be less confident
            "reasoning": response_text,
            "additional_recommendations": self._extract_section(response_text, "Recommendations") or ""
        }

    def _parse_log_analysis_response(self, response_text: str) -> Dict[str, Any]:
        """Parse log analysis response.
        
        Args:
            response_text: Raw response from Ollama
            
        Returns:
            Structured log analysis dictionary
        """
        return {
            "error_patterns": self._extract_list_items(response_text, "patterns") or [],
            "severity_assessment": self._extract_section(response_text, "severity") or "Medium",
            "timeline": self._extract_section(response_text, "timeline") or "",
            "affected_components": self._extract_list_items(response_text, "components") or [],
            "suggested_queries": self._extract_list_items(response_text, "queries") or [],
            "full_analysis": response_text
        }

    def _parse_code_analysis_response(self, response_text: str) -> Dict[str, Any]:
        """Parse code analysis response.
        
        Args:
            response_text: Raw response from Ollama
            
        Returns:
            Structured code analysis dictionary
        """
        return {
            "potential_issues": self._extract_list_items(response_text, "issues") or [],
            "suggested_fixes": self._extract_section(response_text, "fix") or "",
            "best_practices": self._extract_list_items(response_text, "practices") or [],
            "testing_recommendations": self._extract_list_items(response_text, "testing") or [],
            "full_analysis": response_text
        }

    def _extract_section(self, text: str, section_name: str) -> Optional[str]:
        """Extract a specific section from response (flexible parsing for local models).
        
        Args:
            text: Response text
            section_name: Name of section to extract
            
        Returns:
            Extracted section content or None
        """
        import re
        
        # Try multiple patterns since local models may format differently
        patterns = [
            rf'\*\*{re.escape(section_name)}[:\*]*\s*\*\*\s*\n(.*?)(?=\n\*\*|\n#|\Z)',
            rf'{re.escape(section_name)}[:\s]*\n(.*?)(?=\n[A-Z][a-z]+:|\Z)',
            rf'\d+\.\s*{re.escape(section_name)}[:\s]*\n(.*?)(?=\n\d+\.|\Z)',
            rf'{re.escape(section_name).lower()}[:\s]+(.*?)(?=\n[a-z]+:|\Z)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None

    def _extract_list_items(self, text: str, section_name: str) -> List[str]:
        """Extract list items from a section (flexible parsing).
        
        Args:
            text: Response text
            section_name: Name of section containing the list
            
        Returns:
            List of extracted items
        """
        section_content = self._extract_section(text, section_name)
        if not section_content:
            # Try to find the section name and extract following items
            import re
            pattern = rf'{re.escape(section_name)}[:\s]*\n((?:[-*+]\s+.+\n?)+)'
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                section_content = match.group(1)
            else:
                return []
        
        # Extract list items
        import re
        items = re.findall(r'^[\s]*[-*+]\s+(.+)$', section_content, re.MULTILINE)
        
        # Also try numbered lists
        if not items:
            items = re.findall(r'^[\s]*\d+\.\s+(.+)$', section_content, re.MULTILINE)
        
        return [item.strip() for item in items if item.strip()]
