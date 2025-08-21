"""Unified runtime interface for AI agent integration."""

from typing import Dict, List, Optional, Any, Callable, AsyncGenerator
from datetime import datetime, timedelta

from ..integrations.base.log_provider import LogProvider
from ..integrations.base.metrics_provider import MetricsProvider
from ..integrations.base.code_provider import CodeProvider
from ..integrations.base.llm_provider import LLMProvider
from ..integrations.base.runbook_provider import RunbookProvider


class RuntimeInterface:
    """Provides unified runtime functions for the AI agent.
    
    This class creates standardized functions that abstract away the specific
    cloud provider implementations, giving the AI agent a consistent interface
    regardless of which providers are configured.
    """

    def __init__(
        self,
        logs_provider: LogProvider,
        code_provider: CodeProvider,
        llm_provider: LLMProvider,
        metrics_provider: Optional[MetricsProvider] = None,
        runbook_provider: Optional[RunbookProvider] = None
    ):
        """Initialize the runtime interface.
        
        Args:
            logs_provider: Logs provider instance (required)
            code_provider: Code provider instance (required)
            llm_provider: LLM provider instance (required)
            metrics_provider: Metrics provider instance (optional)
            runbook_provider: Runbook provider instance (optional)
        """
        self._logs_provider = logs_provider
        self._code_provider = code_provider
        self._llm_provider = llm_provider
        self._metrics_provider = metrics_provider
        self._runbook_provider = runbook_provider
        
        # Create unified runtime functions
        self.get_logs = self._create_logs_function()
        self.get_metrics = self._create_metrics_function()
        self.get_code_context = self._create_code_function()
        self.get_llm_response = self._create_llm_function()
        self.get_runbook_guidance = self._create_runbook_function()

    def _create_logs_function(self) -> Callable:
        """Create unified logs function.
        
        Returns:
            Async function that provides standardized logs interface
        """
        async def get_logs(
            query: str,
            time_range: tuple[datetime, datetime],
            service_name: Optional[str] = None,
            log_level: Optional[str] = None,
            limit: int = 1000,
            **kwargs
        ) -> List[Dict[str, Any]]:
            """Get logs from the configured provider.
            
            Args:
                query: Search query (will be adapted to provider format)
                time_range: Tuple of (start_time, end_time)
                service_name: Optional service name filter
                log_level: Optional log level filter (ERROR, WARN, INFO, etc.)
                limit: Maximum number of log entries to return
                **kwargs: Additional provider-specific parameters
                
            Returns:
                Standardized list of log entries with consistent format:
                - timestamp: datetime object
                - message: log message string
                - level: standardized log level
                - source: source identifier
                - metadata: additional metadata dict
            """
            try:
                start_time, end_time = time_range
                
                # Adapt query based on provider type
                adapted_query = self._adapt_logs_query(query, service_name, log_level)
                
                # Get logs from provider
                raw_logs = await self._logs_provider.fetch_logs(
                    query=adapted_query,
                    start_time=start_time,
                    end_time=end_time,
                    limit=limit,
                    **kwargs
                )
                
                # Normalize the response format
                return self._normalize_logs_response(raw_logs)
                
            except Exception as e:
                # Return empty list with error info rather than failing
                return [{
                    "timestamp": datetime.now(),
                    "message": f"Error fetching logs: {e}",
                    "level": "ERROR",
                    "source": "oncall_agent",
                    "metadata": {"error": True, "error_type": type(e).__name__}
                }]
        
        return get_logs

    def _create_metrics_function(self) -> Optional[Callable]:
        """Create unified metrics function.
        
        Returns:
            Async function for metrics or None if no metrics provider
        """
        if not self._metrics_provider:
            return None
        
        async def get_metrics(
            resource_info: Dict[str, Any],
            time_range: tuple[datetime, datetime],
            metric_names: Optional[List[str]] = None,
            **kwargs
        ) -> List[Dict[str, Any]]:
            """Get metrics from the configured provider.
            
            Args:
                resource_info: Information about the resource to monitor
                time_range: Tuple of (start_time, end_time)
                metric_names: Specific metrics to retrieve
                **kwargs: Additional provider-specific parameters
                
            Returns:
                Standardized list of metric data points:
                - timestamp: datetime object
                - metric_name: name of the metric
                - value: numeric value
                - unit: unit of measurement
                - dimensions: relevant dimensions/tags
            """
            try:
                start_time, end_time = time_range
                metrics_data = []
                
                # Get metrics based on provider type
                if metric_names:
                    for metric_name in metric_names:
                        try:
                            data = await self._metrics_provider.get_metric_data(
                                metric_name=metric_name,
                                namespace=resource_info.get("namespace", ""),
                                start_time=start_time,
                                end_time=end_time,
                                dimensions=resource_info.get("dimensions", {}),
                                **kwargs
                            )
                            metrics_data.extend(self._normalize_metrics_response(data, metric_name))
                        except Exception:
                            continue  # Skip metrics that can't be retrieved
                else:
                    # Get alarming metrics if no specific metrics requested
                    try:
                        alarming_metrics = await self._metrics_provider.get_alarming_metrics()
                        metrics_data.extend(self._normalize_alarming_metrics(alarming_metrics))
                    except Exception:
                        pass
                
                return metrics_data
                
            except Exception as e:
                return [{
                    "timestamp": datetime.now(),
                    "metric_name": "error",
                    "value": 1,
                    "unit": "count",
                    "dimensions": {"error": str(e), "error_type": type(e).__name__}
                }]
        
        return get_metrics

    def _create_code_function(self) -> Callable:
        """Create unified code context function.
        
        Returns:
            Async function that provides standardized code interface
        """
        async def get_code_context(
            error_info: Dict[str, Any],
            repositories: Optional[List[str]] = None,
            **kwargs
        ) -> List[Dict[str, Any]]:
            """Get code context related to an error.
            
            Args:
                error_info: Information about the error (message, stack trace, etc.)
                repositories: Specific repositories to search
                **kwargs: Additional parameters
                
            Returns:
                Standardized list of code context:
                - repository: repository name
                - file_path: path to relevant file
                - content: code content
                - relevance: relevance score
                - analysis: code analysis results
            """
            try:
                code_context = []
                error_message = error_info.get("message", "")
                stack_trace = error_info.get("stack_trace", "")
                
                # Search for relevant code
                if error_message:
                    search_results = await self._code_provider.search_code(
                        query=error_message[:100],  # Limit query length
                        repositories=repositories,
                        limit=10
                    )
                    code_context.extend(self._normalize_code_search_results(search_results))
                
                # Analyze stack trace if available
                if stack_trace:
                    try:
                        from ..integrations.github import GitHubCodeAnalyzer
                        analyzer = GitHubCodeAnalyzer(self._code_provider)
                        stack_analysis = await analyzer.analyze_stack_trace(
                            stack_trace, repositories
                        )
                        code_context.extend(self._normalize_stack_trace_results(stack_analysis))
                    except Exception:
                        pass  # Skip if analyzer not available
                
                # Get recent commits that might be related
                if repositories:
                    for repo in repositories[:3]:  # Limit to 3 repos
                        try:
                            recent_commits = await self._code_provider.get_recent_commits(
                                repository=repo,
                                since=datetime.now() - timedelta(hours=24),
                                limit=5
                            )
                            code_context.extend(self._normalize_commit_results(recent_commits, repo))
                        except Exception:
                            continue
                
                return code_context[:20]  # Limit total results
                
            except Exception as e:
                return [{
                    "repository": "error",
                    "file_path": "error",
                    "content": f"Error getting code context: {e}",
                    "relevance": 0.0,
                    "analysis": {"error": True, "error_type": type(e).__name__}
                }]
        
        return get_code_context

    def _create_llm_function(self) -> Callable:
        """Create unified LLM function.
        
        Returns:
            Async function that provides standardized LLM interface
        """
        async def get_llm_response(
            context: Dict[str, Any],
            response_type: str = "resolution",
            stream: bool = False,
            **kwargs
        ) -> Dict[str, Any]:
            """Get LLM response for incident analysis.
            
            Args:
                context: Complete incident context
                response_type: Type of response (resolution, log_analysis, code_analysis)
                stream: Whether to stream the response
                **kwargs: Additional LLM parameters
                
            Returns:
                Standardized LLM response format
            """
            try:
                if response_type == "resolution":
                    return await self._llm_provider.generate_resolution(context, **kwargs)
                elif response_type == "log_analysis":
                    return await self._llm_provider.analyze_logs(
                        context.get("log_data", []),
                        context.get("incident_description"),
                        **kwargs
                    )
                elif response_type == "code_analysis":
                    return await self._llm_provider.analyze_code_context(
                        context.get("code_context", []),
                        context.get("incident_description", ""),
                        **kwargs
                    )
                else:
                    raise ValueError(f"Unsupported response type: {response_type}")
                    
            except Exception as e:
                return {
                    "resolution_summary": f"LLM analysis failed: {e}",
                    "detailed_steps": "Unable to generate resolution due to LLM error",
                    "confidence_score": 0.0,
                    "error": True,
                    "error_type": type(e).__name__
                }
        
        return get_llm_response

    def _create_runbook_function(self) -> Optional[Callable]:
        """Create unified runbook function.
        
        Returns:
            Async function for runbooks or None if no runbook provider
        """
        if not self._runbook_provider:
            return None
        
        async def get_runbook_guidance(
            error_context: str,
            limit: int = 3,
            **kwargs
        ) -> str:
            """Get runbook guidance for an incident.
            
            Args:
                error_context: Error description or incident context
                limit: Maximum number of runbooks to include
                **kwargs: Additional parameters
                
            Returns:
                Combined runbook text guidance
            """
            try:
                # Find relevant runbooks
                relevant_runbooks = await self._runbook_provider.find_relevant_runbooks(
                    error_context=error_context,
                    limit=limit
                )
                
                if not relevant_runbooks:
                    return "No relevant runbooks found for this incident."
                
                # Get comprehensive context
                if hasattr(self._runbook_provider, 'get_comprehensive_runbook_context'):
                    context = await self._runbook_provider.get_comprehensive_runbook_context(
                        error_context=error_context,
                        limit=limit
                    )
                    return context.get("combined_text", "")
                else:
                    # Fallback: get individual runbook texts
                    guidance_parts = []
                    for runbook in relevant_runbooks:
                        try:
                            text = await self._runbook_provider.get_runbook_text(runbook["id"])
                            guidance_parts.append(f"--- {runbook['title']} ---\n{text}")
                        except Exception:
                            continue
                    
                    return "\n\n".join(guidance_parts)
                
            except Exception as e:
                return f"Error retrieving runbook guidance: {e}"
        
        return get_runbook_guidance

    def _adapt_logs_query(
        self,
        query: str,
        service_name: Optional[str] = None,
        log_level: Optional[str] = None
    ) -> str:
        """Adapt generic query to provider-specific format.
        
        Args:
            query: Generic search query
            service_name: Service name filter
            log_level: Log level filter
            
        Returns:
            Provider-specific query string
        """
        provider_type = type(self._logs_provider).__name__
        
        if "CloudWatch" in provider_type:
            # AWS CloudWatch Logs Insights format
            adapted_query = f'fields @timestamp, @message | filter @message like /{query}/'
            if service_name:
                adapted_query += f' | filter @logStream like /{service_name}/'
            if log_level:
                adapted_query += f' | filter @message like /{log_level}/'
            adapted_query += ' | sort @timestamp desc'
            
        elif "Azure" in provider_type:
            # Azure Monitor KQL format
            adapted_query = f'search "{query}"'
            if service_name:
                adapted_query += f' | where * contains "{service_name}"'
            if log_level:
                adapted_query += f' | where Level == "{log_level}"'
            adapted_query += ' | order by TimeGenerated desc'
            
        elif "GCP" in provider_type:
            # GCP Cloud Logging filter format
            adapted_query = f'textPayload:"{query}" OR jsonPayload:"{query}"'
            if service_name:
                adapted_query += f' AND resource.labels.service_name="{service_name}"'
            if log_level:
                adapted_query += f' AND severity >= "{log_level}"'
                
        else:
            # Fallback: use query as-is
            adapted_query = query
        
        return adapted_query

    def _normalize_logs_response(self, logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize logs response to standard format.
        
        Args:
            logs: Raw logs from provider
            
        Returns:
            Normalized logs with consistent format
        """
        normalized_logs = []
        
        for log in logs:
            normalized_log = {
                "timestamp": log.get("timestamp", datetime.now()),
                "message": log.get("message", ""),
                "level": self._normalize_log_level(log.get("level", "INFO")),
                "source": log.get("source", "unknown"),
                "metadata": log.get("metadata", {})
            }
            
            # Add provider type to metadata
            normalized_log["metadata"]["provider"] = type(self._logs_provider).__name__
            
            normalized_logs.append(normalized_log)
        
        return normalized_logs

    def _normalize_log_level(self, level: str) -> str:
        """Normalize log level to standard format.
        
        Args:
            level: Raw log level from provider
            
        Returns:
            Normalized log level (ERROR, WARN, INFO, DEBUG)
        """
        level_upper = level.upper()
        
        # Map various level formats to standard levels
        if level_upper in ["ERROR", "FATAL", "CRITICAL"]:
            return "ERROR"
        elif level_upper in ["WARN", "WARNING"]:
            return "WARN"
        elif level_upper in ["INFO", "INFORMATION"]:
            return "INFO"
        elif level_upper in ["DEBUG", "TRACE", "VERBOSE"]:
            return "DEBUG"
        else:
            return "INFO"

    def _normalize_metrics_response(
        self,
        metrics: List[Dict[str, Any]],
        metric_name: str
    ) -> List[Dict[str, Any]]:
        """Normalize metrics response to standard format.
        
        Args:
            metrics: Raw metrics from provider
            metric_name: Name of the metric
            
        Returns:
            Normalized metrics with consistent format
        """
        normalized_metrics = []
        
        for metric in metrics:
            normalized_metric = {
                "timestamp": metric.get("timestamp", datetime.now()),
                "metric_name": metric_name,
                "value": float(metric.get("value", 0)),
                "unit": metric.get("unit", ""),
                "dimensions": metric.get("dimensions", {}),
                "statistic": metric.get("statistic", "Average")
            }
            
            # Add provider type to dimensions
            normalized_metric["dimensions"]["provider"] = type(self._metrics_provider).__name__
            
            normalized_metrics.append(normalized_metric)
        
        return normalized_metrics

    def _normalize_alarming_metrics(self, alarming_metrics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize alarming metrics to standard format.
        
        Args:
            alarming_metrics: Raw alarming metrics from provider
            
        Returns:
            Normalized alarming metrics
        """
        normalized_metrics = []
        
        for metric in alarming_metrics:
            normalized_metric = {
                "timestamp": metric.get("state_timestamp", datetime.now()),
                "metric_name": metric.get("metric_name", "unknown"),
                "value": metric.get("threshold", 0),
                "unit": "threshold",
                "dimensions": metric.get("dimensions", {}),
                "statistic": "Alarm",
                "alarm_info": {
                    "alarm_name": metric.get("alarm_name", ""),
                    "alarm_reason": metric.get("alarm_reason", ""),
                    "comparison_operator": metric.get("comparison_operator", ""),
                    "current_state": metric.get("current_state", "")
                }
            }
            
            normalized_metrics.append(normalized_metric)
        
        return normalized_metrics

    def _normalize_code_search_results(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize code search results to standard format.
        
        Args:
            search_results: Raw search results from code provider
            
        Returns:
            Normalized code context
        """
        normalized_results = []
        
        for result in search_results:
            normalized_result = {
                "repository": result.get("repository", "unknown"),
                "file_path": result.get("file_path", "unknown"),
                "content": self._extract_relevant_code_content(result),
                "relevance": self._calculate_code_relevance(result),
                "analysis": {
                    "type": "search_result",
                    "matches": result.get("matches", []),
                    "url": result.get("url", ""),
                    "last_modified": result.get("last_modified")
                }
            }
            
            normalized_results.append(normalized_result)
        
        return normalized_results

    def _normalize_stack_trace_results(self, stack_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Normalize stack trace analysis results.
        
        Args:
            stack_analysis: Stack trace analysis from code analyzer
            
        Returns:
            Normalized code context from stack trace
        """
        normalized_results = []
        
        for code_info in stack_analysis.get("related_code", []):
            normalized_result = {
                "repository": code_info.get("repository", "unknown"),
                "file_path": code_info.get("file_path", "unknown"),
                "content": code_info.get("code_snippet", {}).get("lines", []),
                "relevance": 0.9,  # High relevance for stack trace matches
                "analysis": {
                    "type": "stack_trace_match",
                    "line_number": code_info.get("line_number"),
                    "url": code_info.get("url", "")
                }
            }
            
            normalized_results.append(normalized_result)
        
        return normalized_results

    def _normalize_commit_results(self, commits: List[Dict[str, Any]], repository: str) -> List[Dict[str, Any]]:
        """Normalize recent commit results.
        
        Args:
            commits: Recent commits from code provider
            repository: Repository name
            
        Returns:
            Normalized code context from commits
        """
        normalized_results = []
        
        for commit in commits[:5]:  # Limit to 5 most recent
            normalized_result = {
                "repository": repository,
                "file_path": "recent_changes",
                "content": f"Commit: {commit.get('message', '')}\nFiles: {len(commit.get('files_changed', []))}",
                "relevance": 0.5,  # Medium relevance for recent commits
                "analysis": {
                    "type": "recent_commit",
                    "sha": commit.get("sha", ""),
                    "author": commit.get("author", {}),
                    "date": commit.get("date"),
                    "files_changed": commit.get("files_changed", [])
                }
            }
            
            normalized_results.append(normalized_result)
        
        return normalized_results

    def _extract_relevant_code_content(self, search_result: Dict[str, Any]) -> str:
        """Extract relevant code content from search result.
        
        Args:
            search_result: Code search result
            
        Returns:
            Relevant code content string
        """
        matches = search_result.get("matches", [])
        if not matches:
            return ""
        
        # Combine match content with context
        content_parts = []
        for match in matches[:3]:  # Limit to 3 matches per file
            content_parts.append(f"Line {match.get('line_number', '?')}: {match.get('content', '')}")
            
            # Add context if available
            context = match.get("context", {})
            if context.get("before"):
                content_parts.append("Context before: " + "\n".join(context["before"][-2:]))
            if context.get("after"):
                content_parts.append("Context after: " + "\n".join(context["after"][:2]))
        
        return "\n".join(content_parts)

    def _calculate_code_relevance(self, search_result: Dict[str, Any]) -> float:
        """Calculate relevance score for code search result.
        
        Args:
            search_result: Code search result
            
        Returns:
            Relevance score between 0 and 1
        """
        matches = search_result.get("matches", [])
        if not matches:
            return 0.0
        
        # Base score from number of matches
        base_score = min(len(matches) / 5, 1.0)
        
        # Boost for recent files
        last_modified = search_result.get("last_modified")
        if last_modified and isinstance(last_modified, datetime):
            days_old = (datetime.now() - last_modified).days
            recency_boost = max(0, (30 - days_old) / 30 * 0.3)  # Up to 0.3 boost for files < 30 days old
        else:
            recency_boost = 0
        
        return min(base_score + recency_boost, 1.0)

    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about all configured providers.
        
        Returns:
            Dictionary containing provider information
        """
        provider_info = {
            "logs_provider": {
                "type": type(self._logs_provider).__name__,
                "available": True
            },
            "code_provider": {
                "type": type(self._code_provider).__name__,
                "available": True
            },
            "llm_provider": {
                "type": type(self._llm_provider).__name__,
                "available": True,
                "model_info": self._llm_provider.get_model_info()
            }
        }
        
        if self._metrics_provider:
            provider_info["metrics_provider"] = {
                "type": type(self._metrics_provider).__name__,
                "available": True
            }
        else:
            provider_info["metrics_provider"] = {"available": False}
        
        if self._runbook_provider:
            provider_info["runbook_provider"] = {
                "type": type(self._runbook_provider).__name__,
                "available": True
            }
        else:
            provider_info["runbook_provider"] = {"available": False}
        
        return provider_info

    async def health_check_all_providers(self) -> Dict[str, Any]:
        """Check health of all configured providers.
        
        Returns:
            Dictionary containing health status for all providers
        """
        health_results = {}
        
        # Check logs provider
        try:
            health_results["logs"] = await self._logs_provider.health_check()
        except Exception as e:
            health_results["logs"] = {"healthy": False, "error": str(e)}
        
        # Check code provider
        try:
            health_results["code"] = await self._code_provider.health_check()
        except Exception as e:
            health_results["code"] = {"healthy": False, "error": str(e)}
        
        # Check LLM provider
        try:
            health_results["llm"] = await self._llm_provider.health_check()
        except Exception as e:
            health_results["llm"] = {"healthy": False, "error": str(e)}
        
        # Check optional providers
        if self._metrics_provider:
            try:
                health_results["metrics"] = await self._metrics_provider.health_check()
            except Exception as e:
                health_results["metrics"] = {"healthy": False, "error": str(e)}
        
        if self._runbook_provider:
            try:
                health_results["runbooks"] = await self._runbook_provider.health_check()
            except Exception as e:
                health_results["runbooks"] = {"healthy": False, "error": str(e)}
        
        return health_results

    def get_runtime_functions(self) -> Dict[str, Callable]:
        """Get all runtime functions for the AI agent.
        
        Returns:
            Dictionary mapping function names to callable functions
        """
        functions = {
            "get_logs": self.get_logs,
            "get_code_context": self.get_code_context,
            "get_llm_response": self.get_llm_response
        }
        
        if self.get_metrics:
            functions["get_metrics"] = self.get_metrics
        
        if self.get_runbook_guidance:
            functions["get_runbook_guidance"] = self.get_runbook_guidance
        
        return functions

    def get_langchain_model(self):
        """Get the LangChain model for direct LangGraph integration.
        
        Returns:
            LangChain model instance
        """
        return self._llm_provider.get_langchain_model()
