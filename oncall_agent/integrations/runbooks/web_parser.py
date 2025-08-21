"""Web link runbook parser implementation."""

import asyncio
import re
from typing import Dict, List, Optional, Any
from datetime import datetime
from urllib.parse import urljoin, urlparse

try:
    import httpx
    from bs4 import BeautifulSoup
    WEB_AVAILABLE = True
except ImportError:
    WEB_AVAILABLE = False

from ..base.runbook_provider import RunbookProvider, RunbookType


class WebRunbookProvider(RunbookProvider):
    """Web link runbook provider implementation.
    
    Provides integration with web-based runbooks (documentation sites,
    wikis, etc.) for content extraction and search functionality.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize web runbook provider.
        
        Args:
            config: Configuration dictionary containing:
                - base_urls: List of base URLs to search (required)
                - timeout: Request timeout in seconds (optional, default: 30)
                - max_pages: Maximum pages to crawl per base URL (optional, default: 100)
                - cache_ttl: Cache TTL in seconds (optional, default: 3600)
                - user_agent: Custom user agent (optional)
                - headers: Additional HTTP headers (optional)
        """
        if not WEB_AVAILABLE:
            raise ImportError(
                "Web support not available. Install with: pip install httpx beautifulsoup4"
            )
        
        super().__init__(config)
        self._content_cache = {} if config.get("cache_ttl", 3600) > 0 else None
        self._cache_ttl = config.get("cache_ttl", 3600)
        
        # Setup HTTP client
        headers = {
            'User-Agent': config.get('user_agent', 'OnCall-AI-Agent/1.0'),
            **config.get('headers', {})
        }
        
        self._client = httpx.AsyncClient(
            timeout=config.get('timeout', 30),
            headers=headers,
            follow_redirects=True
        )

    def _validate_config(self) -> None:
        """Validate web runbook configuration.
        
        Raises:
            ValueError: If required configuration is missing
        """
        if "base_urls" not in self.config or not self.config["base_urls"]:
            raise ValueError("At least one base_url is required for web runbook provider")
        
        # Validate URLs
        for url in self.config["base_urls"]:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                raise ValueError(f"Invalid base URL: {url}")

    async def get_runbook_text(
        self,
        runbook_id: str,
        **kwargs
    ) -> str:
        """Get the full text content of a web runbook.
        
        Args:
            runbook_id: URL of the web page
            **kwargs: Additional parameters:
                - strip_html: Whether to strip HTML tags (default: True)
                - include_links: Whether to include link URLs (default: False)
                
        Returns:
            Full text content of the web page
            
        Raises:
            ConnectionError: If unable to fetch the web page
            ValueError: If URL is invalid
        """
        try:
            # Validate URL
            parsed_url = urlparse(runbook_id)
            if not parsed_url.scheme:
                # Try to resolve relative URL against base URLs
                for base_url in self.config["base_urls"]:
                    if runbook_id.startswith('/'):
                        runbook_id = urljoin(base_url, runbook_id)
                        break
                    elif not runbook_id.startswith('http'):
                        runbook_id = urljoin(base_url, runbook_id)
                        break
            
            # Check cache first
            if self._content_cache is not None and runbook_id in self._content_cache:
                cache_entry = self._content_cache[runbook_id]
                # Check if cache is still valid
                if (datetime.now() - cache_entry['cached_at']).seconds < self._cache_ttl:
                    return cache_entry['content']
            
            # Fetch web page
            response = await self._client.get(runbook_id)
            response.raise_for_status()
            
            # Parse HTML content
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract text
            text = self._extract_web_text(
                soup,
                kwargs.get('strip_html', True),
                kwargs.get('include_links', False)
            )
            
            # Cache the result
            if self._content_cache is not None:
                self._content_cache[runbook_id] = {
                    'content': text,
                    'cached_at': datetime.now()
                }
            
            return text
            
        except httpx.HTTPStatusError as e:
            raise ValueError(f"Web page not found or inaccessible: {runbook_id} ({e.response.status_code})")
        except httpx.RequestError as e:
            raise ConnectionError(f"Failed to fetch web page: {e}")
        except Exception as e:
            raise ConnectionError(f"Failed to parse web runbook: {e}")

    async def search_runbooks(
        self,
        query: str,
        limit: Optional[int] = 10
    ) -> List[Dict[str, Any]]:
        """Search for web runbooks containing specific content.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            
        Returns:
            List of web runbook search results
            
        Raises:
            ConnectionError: If unable to search runbooks
        """
        try:
            # Discover pages from base URLs
            web_pages = await self._discover_pages()
            results = []
            query_lower = query.lower()
            
            for page_url in web_pages[:limit * 3]:  # Process more pages than limit
                try:
                    # Get page content
                    text = await self.get_runbook_text(page_url)
                    text_lower = text.lower()
                    
                    # Check if query matches
                    if query_lower in text_lower:
                        # Extract excerpt around the match
                        excerpt = self._extract_excerpt(text, query, 200)
                        
                        # Calculate relevance score
                        relevance_score = self._calculate_web_relevance(text_lower, query_lower)
                        
                        # Get page metadata
                        metadata = await self._get_page_metadata(page_url)
                        
                        result = {
                            'id': page_url,
                            'title': metadata.get('title', self._get_title_from_url(page_url)),
                            'type': RunbookType.WEB_LINK.value,
                            'excerpt': excerpt,
                            'relevance_score': relevance_score,
                            'last_modified': metadata.get('last_modified', datetime.now()),
                            'url': page_url,
                            'description': metadata.get('description', '')
                        }
                        results.append(result)
                        
                except Exception:
                    # Skip pages that can't be processed
                    continue
                
                if len(results) >= (limit or 10):
                    break
            
            # Sort by relevance score
            results.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            return results[:limit] if limit else results
            
        except Exception as e:
            raise ConnectionError(f"Failed to search web runbooks: {e}")

    async def list_runbooks(self) -> List[Dict[str, Any]]:
        """List all discoverable web runbooks.
        
        Returns:
            List of web runbooks with metadata
            
        Raises:
            ConnectionError: If unable to list runbooks
        """
        try:
            web_pages = await self._discover_pages()
            runbooks = []
            
            for page_url in web_pages:
                try:
                    # Get page metadata
                    metadata = await self._get_page_metadata(page_url)
                    
                    runbook = {
                        'id': page_url,
                        'title': metadata.get('title', self._get_title_from_url(page_url)),
                        'type': RunbookType.WEB_LINK.value,
                        'description': metadata.get('description', 'Web-based runbook'),
                        'last_modified': metadata.get('last_modified', datetime.now()),
                        'url': page_url,
                        'content_type': metadata.get('content_type', 'text/html'),
                        'size': metadata.get('content_length', 0)
                    }
                    runbooks.append(runbook)
                    
                except Exception:
                    # Skip pages that can't be processed
                    continue
            
            return runbooks
            
        except Exception as e:
            raise ConnectionError(f"Failed to list web runbooks: {e}")

    async def _discover_pages(self) -> List[str]:
        """Discover pages from base URLs.
        
        Returns:
            List of discovered page URLs
        """
        discovered_pages = []
        max_pages = self.config.get('max_pages', 100)
        
        for base_url in self.config["base_urls"]:
            try:
                # Start with the base URL
                pages_to_visit = [base_url]
                visited_pages = set()
                
                while pages_to_visit and len(discovered_pages) < max_pages:
                    current_url = pages_to_visit.pop(0)
                    
                    if current_url in visited_pages:
                        continue
                    
                    visited_pages.add(current_url)
                    discovered_pages.append(current_url)
                    
                    try:
                        # Get page content
                        response = await self._client.get(current_url)
                        response.raise_for_status()
                        
                        # Parse HTML to find more links
                        soup = BeautifulSoup(response.content, 'html.parser')
                        
                        # Find internal links
                        for link in soup.find_all('a', href=True):
                            href = link['href']
                            
                            # Resolve relative URLs
                            if href.startswith('/'):
                                full_url = urljoin(base_url, href)
                            elif href.startswith('http'):
                                full_url = href
                            else:
                                full_url = urljoin(current_url, href)
                            
                            # Only include pages from the same domain
                            if urlparse(full_url).netloc == urlparse(base_url).netloc:
                                if full_url not in visited_pages and full_url not in pages_to_visit:
                                    pages_to_visit.append(full_url)
                        
                    except Exception:
                        # Skip pages that can't be processed
                        continue
                    
                    # Limit discovery to prevent infinite crawling
                    if len(pages_to_visit) > max_pages * 2:
                        break
                        
            except Exception:
                # Skip base URLs that can't be processed
                continue
        
        return discovered_pages[:max_pages]

    def _extract_web_text(
        self,
        soup: BeautifulSoup,
        strip_html: bool = True,
        include_links: bool = False
    ) -> str:
        """Extract text from BeautifulSoup object.
        
        Args:
            soup: BeautifulSoup parsed HTML
            strip_html: Whether to strip HTML tags
            include_links: Whether to include link URLs
            
        Returns:
            Extracted text content
        """
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        if strip_html:
            # Get plain text
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
        else:
            # Keep some HTML structure
            text = str(soup)
        
        # Add link URLs if requested
        if include_links:
            links = []
            for link in soup.find_all('a', href=True):
                link_text = link.get_text().strip()
                if link_text and link['href']:
                    links.append(f"[{link_text}]({link['href']})")
            
            if links:
                text += "\n\n--- Links ---\n" + '\n'.join(links)
        
        return text

    async def _get_page_metadata(self, url: str) -> Dict[str, Any]:
        """Get metadata for a web page.
        
        Args:
            url: Page URL
            
        Returns:
            Dictionary containing page metadata
        """
        metadata = {}
        
        try:
            response = await self._client.head(url)
            
            # Get headers
            headers = response.headers
            metadata.update({
                'content_type': headers.get('content-type', ''),
                'content_length': int(headers.get('content-length', 0)) if headers.get('content-length') else 0,
                'last_modified': datetime.fromisoformat(
                    headers.get('last-modified', datetime.now().isoformat())
                ) if headers.get('last-modified') else datetime.now(),
                'server': headers.get('server', ''),
                'etag': headers.get('etag', '')
            })
            
            # Get page content for title and description
            response = await self._client.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title_tag = soup.find('title')
            if title_tag:
                metadata['title'] = title_tag.get_text().strip()
            
            # Extract description
            desc_tag = soup.find('meta', attrs={'name': 'description'})
            if desc_tag:
                metadata['description'] = desc_tag.get('content', '').strip()
            
            # Extract Open Graph metadata
            og_title = soup.find('meta', attrs={'property': 'og:title'})
            if og_title:
                metadata['og_title'] = og_title.get('content', '').strip()
            
            og_desc = soup.find('meta', attrs={'property': 'og:description'})
            if og_desc:
                metadata['og_description'] = og_desc.get('content', '').strip()
            
        except Exception:
            # If metadata extraction fails, return basic info
            pass
        
        return metadata

    def _get_title_from_url(self, url: str) -> str:
        """Generate a title from URL.
        
        Args:
            url: Page URL
            
        Returns:
            Human-readable title
        """
        parsed = urlparse(url)
        path = parsed.path.strip('/')
        
        if path:
            # Use the last part of the path
            title = path.split('/')[-1]
            # Remove file extensions
            if '.' in title:
                title = title.split('.')[0]
            # Replace hyphens and underscores
            title = title.replace('-', ' ').replace('_', ' ')
            # Capitalize
            title = ' '.join(word.capitalize() for word in title.split())
            return title
        else:
            # Use domain name
            return parsed.netloc.replace('www.', '').capitalize()

    def _extract_excerpt(self, text: str, query: str, max_length: int = 200) -> str:
        """Extract an excerpt around the query match.
        
        Args:
            text: Full text content
            query: Search query
            max_length: Maximum excerpt length
            
        Returns:
            Text excerpt containing the query
        """
        text_lower = text.lower()
        query_lower = query.lower()
        
        # Find the first occurrence of the query
        match_pos = text_lower.find(query_lower)
        if match_pos == -1:
            return text[:max_length] + "..." if len(text) > max_length else text
        
        # Calculate excerpt boundaries
        start = max(0, match_pos - max_length // 2)
        end = min(len(text), match_pos + len(query) + max_length // 2)
        
        excerpt = text[start:end]
        
        # Add ellipsis if truncated
        if start > 0:
            excerpt = "..." + excerpt
        if end < len(text):
            excerpt = excerpt + "..."
        
        return excerpt

    def _calculate_web_relevance(self, text: str, query: str) -> float:
        """Calculate relevance score for web content.
        
        Args:
            text: Web page text content (lowercase)
            query: Search query (lowercase)
            
        Returns:
            Relevance score between 0 and 1
        """
        if not query:
            return 0.0
        
        # Count occurrences
        query_count = text.count(query)
        if query_count == 0:
            return 0.0
        
        # Base score from occurrence frequency
        text_length = len(text.split())
        frequency_score = min(query_count / max(text_length / 100, 1), 1.0)
        
        # Boost score for matches in different contexts
        words = text.split()
        contexts = set()
        for i, word in enumerate(words):
            if query in word:
                # Get surrounding context (10 words before and after)
                context_start = max(0, i - 10)
                context_end = min(len(words), i + 11)
                context = ' '.join(words[context_start:context_end])
                contexts.add(context[:100])  # Limit context length
        
        context_diversity = min(len(contexts) / 3, 1.0)  # Max 3 different contexts
        
        return min(frequency_score + context_diversity * 0.2, 1.0)

    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()

    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            if hasattr(self, '_client'):
                asyncio.create_task(self._client.aclose())
        except Exception:
            pass
