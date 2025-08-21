"""HuggingFace LLM provider implementation using LangChain."""

import asyncio
from typing import Dict, List, Optional, Any, AsyncGenerator

try:
    from langchain_community.llms import HuggingFacePipeline
    from langchain_community.chat_models import ChatHuggingFace
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
    LANGCHAIN_HUGGINGFACE_AVAILABLE = True
except ImportError:
    LANGCHAIN_HUGGINGFACE_AVAILABLE = False

from ..base.llm_provider import LLMProvider


class HuggingFaceProvider(LLMProvider):
    """HuggingFace LLM provider implementation using LangChain.
    
    Provides integration with HuggingFace models through LangChain
    for use in LangGraph workflows. Supports both local and hosted models.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize HuggingFace provider.
        
        Args:
            config: Configuration dictionary containing:
                - model_name: HuggingFace model name (required, e.g., 'microsoft/DialoGPT-medium')
                - api_key: HuggingFace API key (optional, for hosted inference)
                - max_tokens: Maximum tokens for responses (optional, default: 1000)
                - temperature: Temperature for generation (optional, default: 0.1)
                - device: Device to run model on (optional, default: 'auto')
                - model_kwargs: Additional model parameters (optional)
                - pipeline_kwargs: Additional pipeline parameters (optional)
                - use_chat_model: Whether to use chat format (optional, default: False)
        """
        if not LANGCHAIN_HUGGINGFACE_AVAILABLE:
            raise ImportError(
                "LangChain HuggingFace not available. Install with: pip install langchain-community transformers torch"
            )
        
        super().__init__(config)

    def _validate_config(self) -> None:
        """Validate HuggingFace configuration.
        
        Raises:
            ValueError: If required configuration is missing
        """
        if "model_name" not in self.config:
            raise ValueError("HuggingFace model_name is required")

    def _setup_model(self) -> None:
        """Set up the LangChain HuggingFace model instance."""
        model_name = self.config["model_name"]
        
        # Model parameters
        model_kwargs = {
            "temperature": self.config.get("temperature", 0.1),
            "max_length": self.config.get("max_tokens", 1000),
            **self.config.get("model_kwargs", {})
        }
        
        # Pipeline parameters
        pipeline_kwargs = {
            "device": self.config.get("device", "auto"),
            **self.config.get("pipeline_kwargs", {})
        }
        
        # Add API key if provided (for hosted inference)
        if "api_key" in self.config:
            pipeline_kwargs["huggingfacehub_api_token"] = self.config["api_key"]
        
        try:
            # Create HuggingFace pipeline
            self._pipeline = HuggingFacePipeline.from_model_id(
                model_id=model_name,
                task="text-generation",
                model_kwargs=model_kwargs,
                pipeline_kwargs=pipeline_kwargs
            )
            
            # Use chat model if requested
            if self.config.get("use_chat_model", False):
                self._model = ChatHuggingFace(llm=self._pipeline)
            else:
                self._model = self._pipeline
                
        except Exception as e:
            raise ConnectionError(f"Failed to initialize HuggingFace model {model_name}: {e}")

    @property
    def model(self):
        """Get the LangChain model instance for LangGraph integration.
        
        Returns:
            LangChain HuggingFace model instance
        """
        return self._model

    async def generate_resolution(
        self,
        incident_context: Dict[str, Any],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate incident resolution using HuggingFace model.
        
        Args:
            incident_context: Incident information and context
            system_prompt: Optional system prompt override
            **kwargs: Additional parameters:
                - max_tokens: Override max tokens
                - temperature: Override temperature
                
        Returns:
            Dictionary containing resolution information
            
        Raises:
            ConnectionError: If unable to generate response
            ValueError: If input parameters are invalid
        """
        try:
            # HuggingFace models often have smaller context windows
            context_limit = 2000  # Conservative limit for most HF models
            truncated_context = self.truncate_context(incident_context, context_limit)
            
            # Create prompt
            if hasattr(self._model, 'ainvoke') and hasattr(self._model, '_llm_type'):
                if 'chat' in str(type(self._model)).lower():
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
                response = await asyncio.to_thread(self._model.invoke, prompt_text)
                response_text = response if isinstance(response, str) else str(response)
            
            # Parse structured response
            return self._parse_resolution_response(response_text)
            
        except Exception as e:
            raise ConnectionError(f"HuggingFace generation failed: {e}")

    async def analyze_logs(
        self,
        log_entries: List[Dict[str, Any]],
        context: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Analyze log entries using HuggingFace model.
        
        Args:
            log_entries: List of log entries to analyze
            context: Additional context about the incident
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing log analysis results
            
        Raises:
            ConnectionError: If unable to generate response
        """
        try:
            # Format logs for analysis (limit for HF models)
            log_text = "\n".join([
                f"[{log.get('timestamp', 'unknown')}] {log.get('level', 'INFO')}: {log.get('message', '')}"
                for log in log_entries[:10]  # Limit for HF models
            ])
            
            # Create analysis prompt
            analysis_prompt = f"""
            Analyze these log entries to find issues:
            
            {f"Context: {context}" if context else ""}
            
            Logs:
            {log_text}
            
            What error patterns do you see? What is the severity level?
            """
            
            # Generate analysis
            if hasattr(self._model, 'ainvoke'):
                if 'chat' in str(type(self._model)).lower():
                    messages = [HumanMessage(content=analysis_prompt)]
                    response = await self._model.ainvoke(messages)
                    response_text = response.content if hasattr(response, 'content') else str(response)
                else:
                    response = await self._model.ainvoke(analysis_prompt)
                    response_text = response if isinstance(response, str) else str(response)
            else:
                response_text = await asyncio.to_thread(self._model.invoke, analysis_prompt)
                response_text = response_text if isinstance(response_text, str) else str(response_text)
            
            return self._parse_log_analysis_response(response_text)
            
        except Exception as e:
            raise ConnectionError(f"HuggingFace log analysis failed: {e}")

    async def analyze_code_context(
        self,
        code_snippets: List[Dict[str, Any]],
        error_message: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Analyze code context using HuggingFace model.
        
        Args:
            code_snippets: List of relevant code snippets
            error_message: Error message or stack trace
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing code analysis results
            
        Raises:
            ConnectionError: If unable to generate response
        """
        try:
            # Format code for analysis (limit for HF models)
            code_text = "\n\n".join([
                f"File: {code.get('file_path', 'unknown')}\n{code.get('content', '')[:800]}..."
                for code in code_snippets[:2]  # Limit for HF models
            ])
            
            # Create analysis prompt
            analysis_prompt = f"""
            Analyze this code error:
            
            Error: {error_message}
            
            Code:
            {code_text}
            
            What could be causing this error? How to fix it?
            """
            
            # Generate analysis
            if hasattr(self._model, 'ainvoke'):
                if 'chat' in str(type(self._model)).lower():
                    messages = [HumanMessage(content=analysis_prompt)]
                    response = await self._model.ainvoke(messages)
                    response_text = response.content if hasattr(response, 'content') else str(response)
                else:
                    response = await self._model.ainvoke(analysis_prompt)
                    response_text = response if isinstance(response, str) else str(response)
            else:
                response_text = await asyncio.to_thread(self._model.invoke, analysis_prompt)
                response_text = response_text if isinstance(response_text, str) else str(response_text)
            
            return self._parse_code_analysis_response(response_text)
            
        except Exception as e:
            raise ConnectionError(f"HuggingFace code analysis failed: {e}")

    async def stream_response(
        self,
        prompt: str,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream HuggingFace response for real-time updates.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters
            
        Yields:
            Chunks of the response as they are generated
            
        Raises:
            ConnectionError: If unable to generate response
        """
        try:
            if hasattr(self._model, 'astream'):
                if 'chat' in str(type(self._model)).lower():
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
                response = await asyncio.to_thread(self._model.invoke, prompt)
                yield response if isinstance(response, str) else str(response)
                
        except Exception as e:
            raise ConnectionError(f"HuggingFace streaming failed: {e}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the HuggingFace model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "provider": "huggingface",
            "model_name": self.config["model_name"],
            "max_tokens": self.config.get("max_tokens", 1000),
            "temperature": self.config.get("temperature", 0.1),
            "device": self.config.get("device", "auto"),
            "supports_streaming": hasattr(self._model, 'astream'),
            "supports_functions": False,  # Most HF models don't support function calling
            "supports_vision": "vision" in self.config["model_name"].lower(),
            "context_window": self._get_context_window(),
            "local_model": True,
            "hosted_inference": "api_key" in self.config
        }

    def _get_context_window(self) -> int:
        """Get the context window size for the model.
        
        Returns:
            Context window size in tokens
        """
        model_name = self.config["model_name"].lower()
        
        # Common HuggingFace model context windows
        context_windows = {
            "gpt2": 1024,
            "gpt-j": 2048,
            "gpt-neo": 2048,
            "bloom": 2048,
            "flan-t5": 512,
            "t5": 512,
            "bart": 1024,
            "pegasus": 1024,
            "dialogpt": 1024,
            "blenderbot": 128,
            "opt": 2048,
            "llama": 2048,
            "alpaca": 2048,
            "vicuna": 2048,
            "codegen": 2048
        }
        
        # Try to match model name patterns
        for model_key, context_size in context_windows.items():
            if model_key in model_name:
                return context_size
        
        # Default conservative estimate
        return 1024

    async def check_model_availability(self) -> Dict[str, Any]:
        """Check if the specified model is available.
        
        Returns:
            Dictionary containing availability information
        """
        try:
            # Try to make a simple request to test model availability
            test_prompt = "Hello"
            
            if hasattr(self._model, 'invoke'):
                response = await asyncio.to_thread(self._model.invoke, test_prompt)
                
                return {
                    "available": True,
                    "model": self.config["model_name"],
                    "device": self.config.get("device", "auto"),
                    "hosted": "api_key" in self.config,
                    "error": None
                }
            else:
                return {
                    "available": False,
                    "model": self.config["model_name"],
                    "device": self.config.get("device", "auto"),
                    "hosted": "api_key" in self.config,
                    "error": "Model interface not available"
                }
                
        except Exception as e:
            return {
                "available": False,
                "model": self.config["model_name"],
                "device": self.config.get("device", "auto"),
                "hosted": "api_key" in self.config,
                "error": str(e)
            }

    def _parse_resolution_response(self, response_text: str) -> Dict[str, Any]:
        """Parse structured resolution response.
        
        Args:
            response_text: Raw response from HuggingFace
            
        Returns:
            Structured resolution dictionary
        """
        # HuggingFace models may have less structured output, so be flexible
        return {
            "resolution_summary": self._extract_section(response_text, "Summary") or response_text.split('\n')[0],
            "detailed_steps": self._extract_section(response_text, "Steps") or response_text,
            "code_changes": self._extract_section(response_text, "Code") or "",
            "root_cause_analysis": self._extract_section(response_text, "Cause") or "",
            "confidence_score": 0.6,  # HF models may be less confident than commercial models
            "reasoning": response_text,
            "additional_recommendations": self._extract_section(response_text, "Recommendations") or ""
        }

    def _parse_log_analysis_response(self, response_text: str) -> Dict[str, Any]:
        """Parse log analysis response.
        
        Args:
            response_text: Raw response from HuggingFace
            
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
            response_text: Raw response from HuggingFace
            
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
        """Extract a specific section from response (flexible parsing for HF models).
        
        Args:
            text: Response text
            section_name: Name of section to extract
            
        Returns:
            Extracted section content or None
        """
        import re
        
        # Try multiple patterns since HF models may format differently
        patterns = [
            rf'{re.escape(section_name)}[:\s]+(.*?)(?=\n[A-Z][a-z]+:|\Z)',
            rf'\*\*{re.escape(section_name)}[:\*]*\s*\*\*\s*\n(.*?)(?=\n\*\*|\Z)',
            rf'\d+\.\s*{re.escape(section_name)}[:\s]*\n(.*?)(?=\n\d+\.|\Z)',
            rf'{re.escape(section_name).lower()}[:\s]+(.*?)(?=\n|\Z)'
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
            # Try to find items directly in text
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
        
        # Try simple line-based extraction
        if not items:
            lines = section_content.split('\n')
            items = [line.strip() for line in lines if line.strip() and not line.strip().endswith(':')]
        
        return [item.strip() for item in items if item.strip()][:5]  # Limit to 5 items
