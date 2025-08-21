"""Markdown runbook parser implementation."""

import os
import asyncio
import re
from typing import Dict, List, Optional, Any
from datetime import datetime

from ..base.runbook_provider import RunbookProvider, RunbookType


class MarkdownRunbookProvider(RunbookProvider):
    """Markdown runbook provider implementation.
    
    Provides integration with Markdown files for runbook content extraction
    and search functionality with rich formatting support.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize Markdown runbook provider.
        
        Args:
            config: Configuration dictionary containing:
                - runbook_directory: Directory containing Markdown files (required)
                - recursive: Whether to search subdirectories (optional, default: True)
                - file_extensions: List of file extensions to include (optional, default: ['.md', '.markdown'])
                - cache_enabled: Whether to cache parsed content (optional, default: True)
        """
        super().__init__(config)
        self._content_cache = {} if config.get("cache_enabled", True) else None
        self._extensions = config.get("file_extensions", ['.md', '.markdown', '.txt'])

    def _validate_config(self) -> None:
        """Validate Markdown runbook configuration.
        
        Raises:
            ValueError: If required configuration is missing
        """
        if "runbook_directory" not in self.config:
            raise ValueError("runbook_directory is required for Markdown runbook provider")
        
        directory = self.config["runbook_directory"]
        if not os.path.exists(directory):
            raise ValueError(f"Runbook directory does not exist: {directory}")
        
        if not os.path.isdir(directory):
            raise ValueError(f"Runbook directory is not a directory: {directory}")

    async def get_runbook_text(
        self,
        runbook_id: str,
        **kwargs
    ) -> str:
        """Get the full text content of a Markdown runbook.
        
        Args:
            runbook_id: Markdown file path (relative to runbook_directory or absolute)
            **kwargs: Additional parameters:
                - strip_formatting: Whether to strip Markdown formatting (default: False)
                - encoding: File encoding (default: 'utf-8')
                
        Returns:
            Full text content of the Markdown file
            
        Raises:
            ConnectionError: If unable to read the file
            ValueError: If file not found
        """
        try:
            # Resolve file path
            file_path = self._resolve_file_path(runbook_id)
            
            # Check cache first
            if self._content_cache is not None and file_path in self._content_cache:
                cache_entry = self._content_cache[file_path]
                # Check if file has been modified since caching
                if cache_entry['modified'] >= os.path.getmtime(file_path):
                    return cache_entry['content']
            
            # Read file content
            encoding = kwargs.get('encoding', 'utf-8')
            with open(file_path, 'r', encoding=encoding) as file:
                content = file.read()
            
            # Strip formatting if requested
            if kwargs.get('strip_formatting', False):
                content = self._strip_markdown_formatting(content)
            
            # Cache the result
            if self._content_cache is not None:
                self._content_cache[file_path] = {
                    'content': content,
                    'modified': os.path.getmtime(file_path)
                }
            
            return content
            
        except FileNotFoundError:
            raise ValueError(f"Markdown runbook not found: {runbook_id}")
        except UnicodeDecodeError as e:
            raise ConnectionError(f"Failed to decode file {runbook_id}: {e}")
        except Exception as e:
            raise ConnectionError(f"Failed to read Markdown runbook: {e}")

    async def search_runbooks(
        self,
        query: str,
        limit: Optional[int] = 10
    ) -> List[Dict[str, Any]]:
        """Search for Markdown runbooks containing specific content.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            
        Returns:
            List of Markdown runbook search results
            
        Raises:
            ConnectionError: If unable to search runbooks
        """
        try:
            markdown_files = await self._find_markdown_files()
            results = []
            query_lower = query.lower()
            
            for md_file in markdown_files:
                try:
                    # Get text content
                    content = await self.get_runbook_text(md_file)
                    content_lower = content.lower()
                    
                    # Check if query matches
                    if query_lower in content_lower:
                        # Extract excerpt around the match
                        excerpt = self._extract_excerpt(content, query, 200)
                        
                        # Calculate relevance score
                        relevance_score = self._calculate_markdown_relevance(content_lower, query_lower)
                        
                        # Get file info
                        full_path = self._resolve_file_path(md_file)
                        file_stat = os.stat(full_path)
                        
                        # Extract title from content
                        title = self._extract_title_from_content(content) or self._get_title_from_filename(md_file)
                        
                        result = {
                            'id': md_file,
                            'title': title,
                            'type': RunbookType.MARKDOWN.value,
                            'excerpt': excerpt,
                            'relevance_score': relevance_score,
                            'last_modified': datetime.fromtimestamp(file_stat.st_mtime),
                            'url': f"file://{full_path}",
                            'file_size': file_stat.st_size
                        }
                        results.append(result)
                        
                except Exception:
                    # Skip files that can't be processed
                    continue
            
            # Sort by relevance score
            results.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            return results[:limit] if limit else results
            
        except Exception as e:
            raise ConnectionError(f"Failed to search Markdown runbooks: {e}")

    async def list_runbooks(self) -> List[Dict[str, Any]]:
        """List all available Markdown runbooks.
        
        Returns:
            List of Markdown runbooks with metadata
            
        Raises:
            ConnectionError: If unable to list runbooks
        """
        try:
            markdown_files = await self._find_markdown_files()
            runbooks = []
            
            for md_file in markdown_files:
                try:
                    full_path = self._resolve_file_path(md_file)
                    file_stat = os.stat(full_path)
                    
                    # Get content to extract metadata
                    content = await self.get_runbook_text(md_file)
                    
                    # Extract metadata from content
                    title = self._extract_title_from_content(content) or self._get_title_from_filename(md_file)
                    description = self._extract_description_from_content(content)
                    
                    runbook = {
                        'id': md_file,
                        'title': title,
                        'type': RunbookType.MARKDOWN.value,
                        'description': description,
                        'last_modified': datetime.fromtimestamp(file_stat.st_mtime),
                        'size': file_stat.st_size,
                        'url': f"file://{full_path}",
                        'sections': len(self._parse_sections(content)),
                        'word_count': len(content.split())
                    }
                    runbooks.append(runbook)
                    
                except Exception:
                    # Skip files that can't be processed
                    continue
            
            return runbooks
            
        except Exception as e:
            raise ConnectionError(f"Failed to list Markdown runbooks: {e}")

    async def get_runbook_sections(
        self,
        runbook_id: str,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Get structured sections from a Markdown runbook.
        
        Args:
            runbook_id: Markdown file identifier
            **kwargs: Additional parameters
            
        Returns:
            List of sections with enhanced Markdown metadata
            
        Raises:
            ConnectionError: If unable to access runbook
        """
        try:
            content = await self.get_runbook_text(runbook_id, **kwargs)
            sections = self._parse_sections(content)
            
            # Enhance sections with Markdown-specific metadata
            for section in sections:
                section['markdown_metadata'] = self._extract_section_metadata(section['content'])
            
            return sections
            
        except Exception as e:
            raise ConnectionError(f"Failed to get runbook sections: {e}")

    def _resolve_file_path(self, runbook_id: str) -> str:
        """Resolve runbook ID to full file path.
        
        Args:
            runbook_id: Runbook identifier (file path)
            
        Returns:
            Full path to the Markdown file
        """
        if os.path.isabs(runbook_id):
            return runbook_id
        else:
            return os.path.join(self.config["runbook_directory"], runbook_id)

    async def _find_markdown_files(self) -> List[str]:
        """Find all Markdown files in the runbook directory.
        
        Returns:
            List of Markdown file paths (relative to runbook_directory)
        """
        markdown_files = []
        directory = self.config["runbook_directory"]
        recursive = self.config.get("recursive", True)
        
        if recursive:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in self._extensions):
                        full_path = os.path.join(root, file)
                        relative_path = os.path.relpath(full_path, directory)
                        markdown_files.append(relative_path)
        else:
            for file in os.listdir(directory):
                if any(file.lower().endswith(ext) for ext in self._extensions):
                    full_path = os.path.join(directory, file)
                    if os.path.isfile(full_path):
                        markdown_files.append(file)
        
        return markdown_files

    def _strip_markdown_formatting(self, content: str) -> str:
        """Strip Markdown formatting to get plain text.
        
        Args:
            content: Markdown content
            
        Returns:
            Plain text content
        """
        # Remove headers
        content = re.sub(r'^#{1,6}\s+', '', content, flags=re.MULTILINE)
        
        # Remove bold and italic
        content = re.sub(r'\*\*([^*]+)\*\*', r'\1', content)
        content = re.sub(r'\*([^*]+)\*', r'\1', content)
        content = re.sub(r'__([^_]+)__', r'\1', content)
        content = re.sub(r'_([^_]+)_', r'\1', content)
        
        # Remove links
        content = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', content)
        
        # Remove code blocks
        content = re.sub(r'```[^`]*```', '', content, flags=re.DOTALL)
        content = re.sub(r'`([^`]+)`', r'\1', content)
        
        # Remove images
        content = re.sub(r'!\[[^\]]*\]\([^)]+\)', '', content)
        
        # Remove horizontal rules
        content = re.sub(r'^---+$', '', content, flags=re.MULTILINE)
        
        # Remove blockquotes
        content = re.sub(r'^>\s*', '', content, flags=re.MULTILINE)
        
        # Remove list markers
        content = re.sub(r'^[\s]*[-*+]\s+', '', content, flags=re.MULTILINE)
        content = re.sub(r'^[\s]*\d+\.\s+', '', content, flags=re.MULTILINE)
        
        # Clean up extra whitespace
        content = re.sub(r'\n\s*\n', '\n\n', content)
        content = content.strip()
        
        return content

    def _extract_title_from_content(self, content: str) -> Optional[str]:
        """Extract title from Markdown content.
        
        Args:
            content: Markdown content
            
        Returns:
            Extracted title or None
        """
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for H1 header
            if line.startswith('# '):
                return line[2:].strip()
            
            # Check for underlined header
            if len(lines) > lines.index(line.strip()) + 1:
                next_line = lines[lines.index(line.strip()) + 1].strip()
                if re.match(r'^=+$', next_line):
                    return line
            
            # If no header found, use first non-empty line (truncated)
            if not line.startswith('#') and len(line) > 10:
                return line[:50] + "..." if len(line) > 50 else line
        
        return None

    def _get_title_from_filename(self, filename: str) -> str:
        """Generate a title from filename.
        
        Args:
            filename: File name
            
        Returns:
            Human-readable title
        """
        # Get filename without extension
        name = os.path.splitext(os.path.basename(filename))[0]
        
        # Replace underscores and hyphens with spaces
        title = name.replace('_', ' ').replace('-', ' ')
        
        # Capitalize words
        title = ' '.join(word.capitalize() for word in title.split())
        
        return title

    def _extract_description_from_content(self, content: str) -> str:
        """Extract description from Markdown content.
        
        Args:
            content: Markdown content
            
        Returns:
            Brief description
        """
        lines = content.split('\n')
        description_lines = []
        
        # Skip title and find first paragraph
        skip_next = False
        for line in lines:
            line = line.strip()
            
            # Skip headers
            if line.startswith('#') or skip_next:
                skip_next = line.startswith('#')
                continue
            
            # Skip underlined headers
            if re.match(r'^[=-]+$', line):
                continue
            
            # Skip empty lines
            if not line:
                if description_lines:  # Stop at first empty line after content
                    break
                continue
            
            # Skip code blocks and other formatting
            if line.startswith('```') or line.startswith('---'):
                continue
            
            description_lines.append(line)
            
            # Stop after a reasonable amount of text
            if len(' '.join(description_lines)) > 200:
                break
        
        description = ' '.join(description_lines)
        
        # Truncate if too long
        if len(description) > 200:
            description = description[:197] + "..."
        
        return description or "Markdown runbook"

    def _extract_section_metadata(self, content: str) -> Dict[str, Any]:
        """Extract metadata from a Markdown section.
        
        Args:
            content: Section content
            
        Returns:
            Dictionary containing section metadata
        """
        metadata = {
            'has_code_blocks': bool(re.search(r'```', content)),
            'has_links': bool(re.search(r'\[([^\]]+)\]\([^)]+\)', content)),
            'has_images': bool(re.search(r'!\[[^\]]*\]\([^)]+\)', content)),
            'has_lists': bool(re.search(r'^[\s]*[-*+]\s+', content, re.MULTILINE)),
            'has_numbered_lists': bool(re.search(r'^[\s]*\d+\.\s+', content, re.MULTILINE)),
            'has_tables': bool(re.search(r'\|.*\|', content)),
            'has_blockquotes': bool(re.search(r'^>\s+', content, re.MULTILINE)),
            'word_count': len(content.split()),
            'line_count': len(content.split('\n'))
        }
        
        # Extract code languages
        code_blocks = re.findall(r'```(\w+)', content)
        if code_blocks:
            metadata['code_languages'] = list(set(code_blocks))
        
        return metadata

    def _extract_excerpt(self, content: str, query: str, max_length: int = 200) -> str:
        """Extract an excerpt around the query match.
        
        Args:
            content: Full content
            query: Search query
            max_length: Maximum excerpt length
            
        Returns:
            Text excerpt containing the query
        """
        content_lower = content.lower()
        query_lower = query.lower()
        
        # Find the first occurrence of the query
        match_pos = content_lower.find(query_lower)
        if match_pos == -1:
            return content[:max_length] + "..." if len(content) > max_length else content
        
        # Calculate excerpt boundaries
        start = max(0, match_pos - max_length // 2)
        end = min(len(content), match_pos + len(query) + max_length // 2)
        
        excerpt = content[start:end]
        
        # Add ellipsis if truncated
        if start > 0:
            excerpt = "..." + excerpt
        if end < len(content):
            excerpt = excerpt + "..."
        
        return excerpt

    def _calculate_markdown_relevance(self, content: str, query: str) -> float:
        """Calculate relevance score for Markdown content.
        
        Args:
            content: Markdown content (lowercase)
            query: Search query (lowercase)
            
        Returns:
            Relevance score between 0 and 1
        """
        if not query:
            return 0.0
        
        # Count occurrences
        query_count = content.count(query)
        if query_count == 0:
            return 0.0
        
        # Base score from occurrence frequency
        content_length = len(content.split())
        frequency_score = min(query_count / max(content_length / 100, 1), 1.0)
        
        # Boost score for matches in headers
        header_matches = len(re.findall(rf'^#{1,6}.*{re.escape(query)}.*$', content, re.MULTILINE))
        header_boost = min(header_matches * 0.3, 0.5)
        
        # Boost score for matches in code blocks
        code_matches = len(re.findall(rf'```[^`]*{re.escape(query)}[^`]*```', content, re.DOTALL))
        code_boost = min(code_matches * 0.2, 0.3)
        
        return min(frequency_score + header_boost + code_boost, 1.0)
