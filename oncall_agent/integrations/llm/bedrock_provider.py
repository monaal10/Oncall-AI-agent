"""AWS Bedrock LLM provider implementation using LangChain."""

import asyncio
from typing import Dict, List, Optional, Any, AsyncGenerator

try:
    from langchain_aws import ChatBedrock, BedrockLLM
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
    LANGCHAIN_BEDROCK_AVAILABLE = True
except ImportError:
    LANGCHAIN_BEDROCK_AVAILABLE = False

from ..base.llm_provider import LLMProvider


class BedrockProvider(LLMProvider):
    """AWS Bedrock LLM provider implementation using LangChain.
    
    Provides integration with AWS Bedrock models (Claude, Llama2, Titan, etc.)
    through LangChain for use in LangGraph workflows.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize Bedrock provider.
        
        Args:
            config: Configuration dictionary containing:
                - model_id: Bedrock model ID (required, e.g., 'anthropic.claude-3-sonnet-20240229-v1:0')
                - region: AWS region (optional, default: us-east-1)
                - aws_access_key_id: AWS access key (optional, uses default credentials)
                - aws_secret_access_key: AWS secret key (optional, uses default credentials)
                - aws_session_token: AWS session token (optional)
                - max_tokens: Maximum tokens for responses (optional, default: 2000)
                - temperature: Temperature for generation (optional, default: 0.1)
                - timeout: Request timeout in seconds (optional, default: 60)
        """
        if not LANGCHAIN_BEDROCK_AVAILABLE:
            raise ImportError(
                "LangChain AWS not available. Install with: pip install langchain-aws boto3"
            )
        
        super().__init__(config)

    def _validate_config(self) -> None:
        """Validate Bedrock configuration.
        
        Raises:
            ValueError: If required configuration is missing
        """
        if "model_id" not in self.config:
            raise ValueError("Bedrock model_id is required")

    def _setup_model(self) -> None:
        """Set up the LangChain Bedrock model instance."""
        model_id = self.config["model_id"]
        
        # Setup AWS credentials if provided
        credentials = {}
        if "aws_access_key_id" in self.config:
            credentials["aws_access_key_id"] = self.config["aws_access_key_id"]
        if "aws_secret_access_key" in self.config:
            credentials["aws_secret_access_key"] = self.config["aws_secret_access_key"]
        if "aws_session_token" in self.config:
            credentials["aws_session_token"] = self.config["aws_session_token"]
        
        # Determine if this is a chat model based on model ID
        chat_models = ["anthropic.claude", "meta.llama2-chat", "amazon.titan-text-express"]
        is_chat_model = any(chat_model in model_id for chat_model in chat_models)
        
        if is_chat_model:
            # Use chat model for conversational models
            self._model = ChatBedrock(
                model_id=model_id,
                region_name=self.config.get("region", "us-east-1"),
                model_kwargs={
                    "max_tokens": self.config.get("max_tokens", 2000),
                    "temperature": self.config.get("temperature", 0.1),
                },
                streaming=True,
                **credentials
            )
        else:
            # Use completion model for other models
            self._model = BedrockLLM(
                model_id=model_id,
                region_name=self.config.get("region", "us-east-1"),
                model_kwargs={
                    "max_tokens_to_sample": self.config.get("max_tokens", 2000),
                    "temperature": self.config.get("temperature", 0.1),
                },
                **credentials
            )

    @property
    def model(self):
        """Get the LangChain model instance for LangGraph integration.
        
        Returns:
            LangChain Bedrock model instance
        """
        return self._model

    async def generate_resolution(
        self,
        incident_context: Dict[str, Any],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate incident resolution using Bedrock.
        
        Args:
            incident_context: Incident information and context
            system_prompt: Optional system prompt override
            **kwargs: Additional parameters:
                - max_tokens: Override max tokens
                - temperature: Override temperature
                
        Returns:
            Dictionary containing resolution information
            
        Raises:
            ConnectionError: If unable to connect to Bedrock
            ValueError: If input parameters are invalid
        """
        try:
            # Bedrock models have varying context windows
            max_tokens = kwargs.get("max_tokens", self.config.get("max_tokens", 2000))
            context_limit = max_tokens * 3
            truncated_context = self.truncate_context(incident_context, context_limit)
            
            # Create messages or prompt based on model type
            if hasattr(self._model, 'ainvoke') and 'Chat' in type(self._model).__name__:
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
            
            # Parse structured response
            return self._parse_resolution_response(response_text)
            
        except Exception as e:
            raise ConnectionError(f"Bedrock generation failed: {e}")

    async def analyze_logs(
        self,
        log_entries: List[Dict[str, Any]],
        context: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Analyze log entries using Bedrock.
        
        Args:
            log_entries: List of log entries to analyze
            context: Additional context about the incident
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing log analysis results
            
        Raises:
            ConnectionError: If unable to connect to Bedrock
        """
        try:
            # Format logs for analysis
            log_text = "\n".join([
                f"[{log.get('timestamp', 'unknown')}] {log.get('level', 'INFO')}: {log.get('message', '')}"
                for log in log_entries[:20]  # Standard limit
            ])
            
            # Create analysis prompt
            analysis_prompt = f"""
            Human: Analyze these log entries to identify patterns and issues:
            
            {f"Context: {context}" if context else ""}
            
            Log Entries:
            {log_text}
            
            Please provide:
            1. Error patterns identified
            2. Severity assessment
            3. Timeline of events
            4. Affected components
            5. Suggested follow-up queries
            
            Assistant: I'll analyze these log entries for you.
            """
            
            # Generate analysis
            if hasattr(self._model, 'ainvoke') and 'Chat' in type(self._model).__name__:
                messages = [HumanMessage(content=analysis_prompt)]
                response = await self._model.ainvoke(messages)
                response_text = response.content if hasattr(response, 'content') else str(response)
            else:
                response_text = await self._model.ainvoke(analysis_prompt)
            
            return self._parse_log_analysis_response(response_text)
            
        except Exception as e:
            raise ConnectionError(f"Bedrock log analysis failed: {e}")

    async def analyze_code_context(
        self,
        code_snippets: List[Dict[str, Any]],
        error_message: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Analyze code context using Bedrock.
        
        Args:
            code_snippets: List of relevant code snippets
            error_message: Error message or stack trace
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing code analysis results
            
        Raises:
            ConnectionError: If unable to connect to Bedrock
        """
        try:
            # Format code for analysis
            code_text = "\n\n".join([
                f"File: {code.get('file_path', 'unknown')}\n```{code.get('language', '')}\n{code.get('content', '')[:800]}...\n```"
                for code in code_snippets[:4]  # Conservative limit for Bedrock
            ])
            
            # Create analysis prompt (Bedrock often works better with Human/Assistant format)
            analysis_prompt = f"""
            Human: Analyze this code error and suggest fixes:
            
            Error: {error_message}
            
            Code Context:
            {code_text}
            
            Please provide:
            1. Potential issues in the code
            2. Specific fix suggestions
            3. Best practices recommendations
            4. Testing recommendations
            
            Assistant: I'll analyze this code error for you.
            """
            
            # Generate analysis
            if hasattr(self._model, 'ainvoke') and 'Chat' in type(self._model).__name__:
                messages = [HumanMessage(content=analysis_prompt)]
                response = await self._model.ainvoke(messages)
                response_text = response.content if hasattr(response, 'content') else str(response)
            else:
                response_text = await self._model.ainvoke(analysis_prompt)
            
            return self._parse_code_analysis_response(response_text)
            
        except Exception as e:
            raise ConnectionError(f"Bedrock code analysis failed: {e}")

    async def stream_response(
        self,
        prompt: str,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream Bedrock response for real-time updates.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters
            
        Yields:
            Chunks of the response as they are generated
            
        Raises:
            ConnectionError: If unable to connect to Bedrock
        """
        try:
            if hasattr(self._model, 'astream'):
                if 'Chat' in type(self._model).__name__:
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
                yield response
                
        except Exception as e:
            raise ConnectionError(f"Bedrock streaming failed: {e}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the Bedrock model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "provider": "bedrock",
            "model_id": self.config["model_id"],
            "region": self.config.get("region", "us-east-1"),
            "max_tokens": self.config.get("max_tokens", 2000),
            "temperature": self.config.get("temperature", 0.1),
            "supports_streaming": True,
            "supports_functions": False,  # Most Bedrock models don't support function calling
            "supports_vision": "vision" in self.config["model_id"].lower(),
            "context_window": self._get_context_window(),
            "model_provider": self._get_model_provider()
        }

    def _get_context_window(self) -> int:
        """Get the context window size for the model.
        
        Returns:
            Context window size in tokens
        """
        model_id = self.config["model_id"].lower()
        
        # Bedrock model context windows by provider
        if "anthropic.claude-3" in model_id:
            return 200000  # Claude-3 models
        elif "anthropic.claude" in model_id:
            return 100000  # Claude-2 models
        elif "meta.llama2" in model_id:
            return 4096    # Llama2 models
        elif "amazon.titan" in model_id:
            return 8192    # Titan models
        elif "ai21.j2" in model_id:
            return 8192    # Jurassic models
        elif "cohere.command" in model_id:
            return 4096    # Cohere models
        else:
            return 4096    # Conservative default

    def _get_model_provider(self) -> str:
        """Get the underlying model provider.
        
        Returns:
            Model provider name (anthropic, meta, amazon, etc.)
        """
        model_id = self.config["model_id"].lower()
        
        if "anthropic" in model_id:
            return "anthropic"
        elif "meta" in model_id:
            return "meta"
        elif "amazon" in model_id:
            return "amazon"
        elif "ai21" in model_id:
            return "ai21"
        elif "cohere" in model_id:
            return "cohere"
        else:
            return "unknown"

    def _parse_resolution_response(self, response_text: str) -> Dict[str, Any]:
        """Parse structured resolution response.
        
        Args:
            response_text: Raw response from Bedrock
            
        Returns:
            Structured resolution dictionary
        """
        return {
            "resolution_summary": self._extract_section(response_text, "Summary") or response_text.split('\n')[0],
            "detailed_steps": self._extract_section(response_text, "Steps") or response_text,
            "code_changes": self._extract_section(response_text, "Code") or "",
            "root_cause_analysis": self._extract_section(response_text, "Cause") or "",
            "confidence_score": 0.75,  # Bedrock models vary in confidence
            "reasoning": response_text,
            "additional_recommendations": self._extract_section(response_text, "Recommendations") or ""
        }

    def _parse_log_analysis_response(self, response_text: str) -> Dict[str, Any]:
        """Parse log analysis response.
        
        Args:
            response_text: Raw response from Bedrock
            
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
            response_text: Raw response from Bedrock
            
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
        """Extract a specific section from response (flexible parsing for Bedrock models).
        
        Args:
            text: Response text
            section_name: Name of section to extract
            
        Returns:
            Extracted section content or None
        """
        import re
        
        # Try multiple patterns since Bedrock models may format differently
        patterns = [
            rf'\*\*{re.escape(section_name)}[:\*]*\s*\*\*\s*\n(.*?)(?=\n\*\*|\n#|\Z)',
            rf'{re.escape(section_name)}[:\s]*\n(.*?)(?=\n[A-Z][a-z]+:|\Z)',
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
        
        return [item.strip() for item in items if item.strip()][:5]  # Limit to 5 items

    async def list_available_models(self) -> List[Dict[str, Any]]:
        """List available models in Bedrock.
        
        Returns:
            List of available models with their capabilities
            
        Raises:
            ConnectionError: If unable to connect to Bedrock
        """
        try:
            import boto3
            
            # Create Bedrock client
            session_kwargs = {"region_name": self.config.get("region", "us-east-1")}
            if "aws_access_key_id" in self.config:
                session_kwargs["aws_access_key_id"] = self.config["aws_access_key_id"]
            if "aws_secret_access_key" in self.config:
                session_kwargs["aws_secret_access_key"] = self.config["aws_secret_access_key"]
            if "aws_session_token" in self.config:
                session_kwargs["aws_session_token"] = self.config["aws_session_token"]
            
            bedrock_client = boto3.client("bedrock", **session_kwargs)
            
            # List foundation models
            response = await asyncio.to_thread(bedrock_client.list_foundation_models)
            
            models = []
            for model in response.get("modelSummaries", []):
                model_info = {
                    "model_id": model["modelId"],
                    "model_name": model["modelName"],
                    "provider_name": model["providerName"],
                    "input_modalities": model.get("inputModalities", []),
                    "output_modalities": model.get("outputModalities", []),
                    "response_streaming_supported": model.get("responseStreamingSupported", False),
                    "customizations_supported": model.get("customizationsSupported", [])
                }
                models.append(model_info)
            
            return models
            
        except Exception as e:
            raise ConnectionError(f"Failed to list Bedrock models: {e}")

    async def check_model_availability(self) -> Dict[str, Any]:
        """Check if the specified model is available in Bedrock.
        
        Returns:
            Dictionary containing availability information
        """
        try:
            available_models = await self.list_available_models()
            model_id = self.config["model_id"]
            
            is_available = any(model["model_id"] == model_id for model in available_models)
            
            if is_available:
                model_info = next(
                    (model for model in available_models if model["model_id"] == model_id),
                    {}
                )
                return {
                    "available": True,
                    "model_id": model_id,
                    "region": self.config.get("region", "us-east-1"),
                    "model_info": model_info,
                    "error": None
                }
            else:
                return {
                    "available": False,
                    "model_id": model_id,
                    "region": self.config.get("region", "us-east-1"),
                    "model_info": {},
                    "error": f"Model {model_id} not found in available models"
                }
                
        except Exception as e:
            return {
                "available": False,
                "model_id": self.config["model_id"],
                "region": self.config.get("region", "us-east-1"),
                "model_info": {},
                "error": str(e)
            }
