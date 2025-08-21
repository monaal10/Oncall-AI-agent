"""Unified runbook manager for multiple runbook types."""

import asyncio
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

from ..base.runbook_provider import RunbookProvider, RunbookType
from .pdf_parser import PDFRunbookProvider
from .markdown_parser import MarkdownRunbookProvider
from .docx_parser import DocxRunbookProvider
from .web_parser import WebRunbookProvider


class UnifiedRunbookProvider(RunbookProvider):
    """Unified runbook provider that can handle multiple runbook types.
    
    This provider can work with PDF files, Markdown files, Word documents,
    and web links simultaneously, providing a single interface for all
    runbook operations.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize unified runbook provider.
        
        Args:
            config: Configuration dictionary containing:
                - providers: Dictionary of provider configurations, each with:
                    - type: Provider type (pdf, markdown, docx, web_link)
                    - config: Provider-specific configuration
                - default_search_limit: Default limit for searches (optional, default: 50)
        """
        super().__init__(config)
        self.providers: Dict[str, RunbookProvider] = {}
        self._setup_providers()

    def _validate_config(self) -> None:
        """Validate unified runbook configuration.
        
        Raises:
            ValueError: If required configuration is missing
        """
        if "providers" not in self.config or not self.config["providers"]:
            raise ValueError("At least one provider configuration is required")
        
        for provider_name, provider_config in self.config["providers"].items():
            if "type" not in provider_config:
                raise ValueError(f"Provider type is required for {provider_name}")
            if "config" not in provider_config:
                raise ValueError(f"Provider config is required for {provider_name}")

    def _setup_providers(self) -> None:
        """Set up individual runbook providers.
        
        Raises:
            ValueError: If provider type is not supported
        """
        provider_map = {
            RunbookType.PDF.value: PDFRunbookProvider,
            RunbookType.MARKDOWN.value: MarkdownRunbookProvider,
            RunbookType.DOCX.value: DocxRunbookProvider,
            RunbookType.WEB_LINK.value: WebRunbookProvider
        }
        
        for provider_name, provider_config in self.config["providers"].items():
            provider_type = provider_config["type"]
            
            if provider_type not in provider_map:
                raise ValueError(f"Unsupported provider type: {provider_type}")
            
            try:
                provider_class = provider_map[provider_type]
                self.providers[provider_name] = provider_class(provider_config["config"])
            except Exception as e:
                raise ValueError(f"Failed to initialize {provider_name} provider: {e}")

    async def get_runbook_text(
        self,
        runbook_id: str,
        provider_name: Optional[str] = None,
        **kwargs
    ) -> str:
        """Get the full text content of a runbook.
        
        Args:
            runbook_id: Runbook identifier (file path, URL, etc.)
            provider_name: Specific provider to use (if None, auto-detect)
            **kwargs: Provider-specific additional parameters
            
        Returns:
            Full text content of the runbook
            
        Raises:
            ConnectionError: If unable to access runbook
            ValueError: If runbook not found or provider not available
        """
        if provider_name:
            # Use specific provider
            if provider_name not in self.providers:
                raise ValueError(f"Provider {provider_name} not configured")
            
            return await self.providers[provider_name].get_runbook_text(runbook_id, **kwargs)
        
        else:
            # Auto-detect provider based on runbook_id
            provider = self._detect_provider(runbook_id)
            if not provider:
                raise ValueError(f"Could not determine provider for runbook: {runbook_id}")
            
            return await provider.get_runbook_text(runbook_id, **kwargs)

    async def search_runbooks(
        self,
        query: str,
        limit: Optional[int] = 10,
        provider_names: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Search for runbooks containing specific content across all providers.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            provider_names: Specific providers to search (None for all)
            
        Returns:
            List of runbook search results from all providers
            
        Raises:
            ConnectionError: If unable to search runbooks
        """
        try:
            target_providers = provider_names or list(self.providers.keys())
            all_results = []
            
            # Search across all specified providers in parallel
            search_tasks = []
            for provider_name in target_providers:
                if provider_name in self.providers:
                    provider = self.providers[provider_name]
                    task = asyncio.create_task(
                        self._search_with_provider(provider, provider_name, query, limit)
                    )
                    search_tasks.append(task)
            
            # Wait for all searches to complete
            provider_results = await asyncio.gather(*search_tasks, return_exceptions=True)
            
            # Combine results
            for results in provider_results:
                if isinstance(results, list):
                    all_results.extend(results)
                # Skip exceptions from individual providers
            
            # Sort by relevance score
            all_results.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            return all_results[:limit] if limit else all_results
            
        except Exception as e:
            raise ConnectionError(f"Failed to search runbooks: {e}")

    async def list_runbooks(self) -> List[Dict[str, Any]]:
        """List all available runbooks from all providers.
        
        Returns:
            List of all runbooks with provider information
            
        Raises:
            ConnectionError: If unable to list runbooks
        """
        try:
            all_runbooks = []
            
            # Get runbooks from all providers in parallel
            list_tasks = []
            for provider_name, provider in self.providers.items():
                task = asyncio.create_task(
                    self._list_with_provider(provider, provider_name)
                )
                list_tasks.append(task)
            
            # Wait for all listings to complete
            provider_results = await asyncio.gather(*list_tasks, return_exceptions=True)
            
            # Combine results
            for results in provider_results:
                if isinstance(results, list):
                    all_runbooks.extend(results)
                # Skip exceptions from individual providers
            
            # Sort by last modified date
            all_runbooks.sort(key=lambda x: x['last_modified'], reverse=True)
            
            return all_runbooks
            
        except Exception as e:
            raise ConnectionError(f"Failed to list runbooks: {e}")

    async def find_relevant_runbooks(
        self,
        error_context: str,
        limit: Optional[int] = 5
    ) -> List[Dict[str, Any]]:
        """Find runbooks most relevant to an error or incident context.
        
        Args:
            error_context: Error message or incident description
            limit: Maximum number of runbooks to return
            
        Returns:
            List of most relevant runbooks with relevance scores
            
        Raises:
            ConnectionError: If unable to search runbooks
        """
        try:
            # Extract keywords from error context
            keywords = self._extract_keywords(error_context.lower())
            
            # Search for each keyword and combine results
            all_results = []
            
            for keyword in keywords[:5]:  # Use top 5 keywords
                try:
                    results = await self.search_runbooks(query=keyword, limit=limit * 2)
                    all_results.extend(results)
                except Exception:
                    continue
            
            # Deduplicate and re-score based on error context
            unique_results = {}
            for result in all_results:
                runbook_id = result['id']
                if runbook_id in unique_results:
                    # Combine relevance scores
                    unique_results[runbook_id]['relevance_score'] = max(
                        unique_results[runbook_id]['relevance_score'],
                        result['relevance_score']
                    )
                else:
                    unique_results[runbook_id] = result
            
            # Sort by relevance and return top results
            final_results = list(unique_results.values())
            final_results.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            return final_results[:limit] if limit else final_results
            
        except Exception as e:
            raise ConnectionError(f"Failed to find relevant runbooks: {e}")

    async def get_comprehensive_runbook_context(
        self,
        error_context: str,
        limit: Optional[int] = 3
    ) -> Dict[str, Any]:
        """Get comprehensive runbook context for an error.
        
        Args:
            error_context: Error message or incident description
            limit: Maximum number of runbooks to analyze
            
        Returns:
            Dictionary containing:
            - relevant_runbooks: Most relevant runbooks
            - combined_text: Combined text from all relevant runbooks
            - sections: Relevant sections from all runbooks
            - summary: Summary of available guidance
            
        Raises:
            ConnectionError: If unable to access runbooks
        """
        try:
            # Find relevant runbooks
            relevant_runbooks = await self.find_relevant_runbooks(error_context, limit)
            
            if not relevant_runbooks:
                return {
                    "relevant_runbooks": [],
                    "combined_text": "",
                    "sections": [],
                    "summary": "No relevant runbooks found for this error context."
                }
            
            # Get detailed content from relevant runbooks
            combined_text_parts = []
            all_sections = []
            
            for runbook in relevant_runbooks:
                try:
                    # Get full text
                    text = await self.get_runbook_text(runbook['id'])
                    combined_text_parts.append(f"--- {runbook['title']} ---\n{text}\n")
                    
                    # Get relevant sections
                    provider = self._detect_provider(runbook['id'])
                    if provider:
                        sections = await provider.find_relevant_sections(
                            runbook['id'],
                            error_context
                        )
                        for section in sections:
                            section['source_runbook'] = runbook['title']
                            section['source_id'] = runbook['id']
                        all_sections.extend(sections)
                    
                except Exception:
                    # Skip runbooks that can't be processed
                    continue
            
            # Sort sections by relevance
            all_sections.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
            
            # Generate summary
            summary = self._generate_runbook_summary(relevant_runbooks, all_sections)
            
            return {
                "relevant_runbooks": relevant_runbooks,
                "combined_text": '\n'.join(combined_text_parts),
                "sections": all_sections[:10],  # Top 10 most relevant sections
                "summary": summary
            }
            
        except Exception as e:
            raise ConnectionError(f"Failed to get comprehensive runbook context: {e}")

    def _detect_provider(self, runbook_id: str) -> Optional[RunbookProvider]:
        """Detect the appropriate provider for a runbook ID.
        
        Args:
            runbook_id: Runbook identifier
            
        Returns:
            Appropriate provider or None if not found
        """
        # Check by file extension or URL pattern
        runbook_lower = runbook_id.lower()
        
        if runbook_lower.endswith('.pdf'):
            return next((p for p in self.providers.values() if isinstance(p, PDFRunbookProvider)), None)
        elif runbook_lower.endswith(('.md', '.markdown', '.txt')):
            return next((p for p in self.providers.values() if isinstance(p, MarkdownRunbookProvider)), None)
        elif runbook_lower.endswith('.docx'):
            return next((p for p in self.providers.values() if isinstance(p, DocxRunbookProvider)), None)
        elif runbook_lower.startswith(('http://', 'https://')):
            return next((p for p in self.providers.values() if isinstance(p, WebRunbookProvider)), None)
        
        return None

    async def _search_with_provider(
        self,
        provider: RunbookProvider,
        provider_name: str,
        query: str,
        limit: Optional[int]
    ) -> List[Dict[str, Any]]:
        """Search with a specific provider and add provider info to results.
        
        Args:
            provider: Runbook provider instance
            provider_name: Name of the provider
            query: Search query
            limit: Result limit
            
        Returns:
            Search results with provider information
        """
        try:
            results = await provider.search_runbooks(query, limit)
            
            # Add provider information to each result
            for result in results:
                result['provider'] = provider_name
                result['provider_type'] = result['type']
            
            return results
            
        except Exception:
            # Return empty list if provider fails
            return []

    async def _list_with_provider(
        self,
        provider: RunbookProvider,
        provider_name: str
    ) -> List[Dict[str, Any]]:
        """List runbooks with a specific provider and add provider info.
        
        Args:
            provider: Runbook provider instance
            provider_name: Name of the provider
            
        Returns:
            Runbook list with provider information
        """
        try:
            runbooks = await provider.list_runbooks()
            
            # Add provider information to each runbook
            for runbook in runbooks:
                runbook['provider'] = provider_name
                runbook['provider_type'] = runbook['type']
            
            return runbooks
            
        except Exception:
            # Return empty list if provider fails
            return []

    def _generate_runbook_summary(
        self,
        runbooks: List[Dict[str, Any]],
        sections: List[Dict[str, Any]]
    ) -> str:
        """Generate a summary of available runbook guidance.
        
        Args:
            runbooks: List of relevant runbooks
            sections: List of relevant sections
            
        Returns:
            Summary text
        """
        if not runbooks:
            return "No relevant runbooks found."
        
        summary_parts = []
        
        # Summary of runbooks found
        runbook_types = {}
        for runbook in runbooks:
            provider_type = runbook['type']
            if provider_type not in runbook_types:
                runbook_types[provider_type] = 0
            runbook_types[provider_type] += 1
        
        type_summary = ", ".join([f"{count} {type_name}" for type_name, count in runbook_types.items()])
        summary_parts.append(f"Found {len(runbooks)} relevant runbooks: {type_summary}")
        
        # Summary of sections
        if sections:
            high_relevance_sections = [s for s in sections if s.get('relevance_score', 0) > 0.5]
            if high_relevance_sections:
                summary_parts.append(f"Found {len(high_relevance_sections)} highly relevant sections")
            else:
                summary_parts.append(f"Found {len(sections)} potentially relevant sections")
        
        # Top runbook titles
        if len(runbooks) <= 3:
            titles = [r['title'] for r in runbooks]
            summary_parts.append(f"Available runbooks: {', '.join(titles)}")
        else:
            top_titles = [r['title'] for r in runbooks[:3]]
            summary_parts.append(f"Top runbooks: {', '.join(top_titles)} and {len(runbooks) - 3} more")
        
        return ". ".join(summary_parts) + "."

    async def health_check(self) -> Dict[str, bool]:
        """Check health of all configured providers.
        
        Returns:
            Dictionary mapping provider names to health status
        """
        health_results = {}
        
        health_tasks = []
        for provider_name, provider in self.providers.items():
            task = asyncio.create_task(
                self._check_provider_health(provider_name, provider)
            )
            health_tasks.append(task)
        
        # Wait for all health checks
        results = await asyncio.gather(*health_tasks, return_exceptions=True)
        
        for i, (provider_name, _) in enumerate(self.providers.items()):
            result = results[i]
            health_results[provider_name] = result if isinstance(result, bool) else False
        
        return health_results

    async def _check_provider_health(self, provider_name: str, provider: RunbookProvider) -> bool:
        """Check health of a specific provider.
        
        Args:
            provider_name: Name of the provider
            provider: Provider instance
            
        Returns:
            True if healthy, False otherwise
        """
        try:
            return await provider.health_check()
        except Exception:
            return False

    async def get_runbook_by_type(
        self,
        runbook_type: RunbookType,
        query: Optional[str] = None,
        limit: Optional[int] = 10
    ) -> List[Dict[str, Any]]:
        """Get runbooks of a specific type.
        
        Args:
            runbook_type: Type of runbooks to retrieve
            query: Optional search query
            limit: Maximum number of results
            
        Returns:
            List of runbooks of the specified type
            
        Raises:
            ConnectionError: If unable to access runbooks
        """
        # Find providers that handle this type
        matching_providers = [
            (name, provider) for name, provider in self.providers.items()
            if self._provider_handles_type(provider, runbook_type)
        ]
        
        if not matching_providers:
            return []
        
        all_results = []
        
        for provider_name, provider in matching_providers:
            try:
                if query:
                    results = await provider.search_runbooks(query, limit)
                else:
                    results = await provider.list_runbooks()
                
                # Add provider info
                for result in results:
                    result['provider'] = provider_name
                
                all_results.extend(results)
                
            except Exception:
                # Skip providers that fail
                continue
        
        # Sort and limit
        if query:
            all_results.sort(key=lambda x: x['relevance_score'], reverse=True)
        else:
            all_results.sort(key=lambda x: x['last_modified'], reverse=True)
        
        return all_results[:limit] if limit else all_results

    def _provider_handles_type(self, provider: RunbookProvider, runbook_type: RunbookType) -> bool:
        """Check if a provider handles a specific runbook type.
        
        Args:
            provider: Provider instance
            runbook_type: Runbook type to check
            
        Returns:
            True if provider handles the type
        """
        type_mapping = {
            RunbookType.PDF: PDFRunbookProvider,
            RunbookType.MARKDOWN: MarkdownRunbookProvider,
            RunbookType.DOCX: DocxRunbookProvider,
            RunbookType.WEB_LINK: WebRunbookProvider
        }
        
        return isinstance(provider, type_mapping.get(runbook_type, type(None)))
