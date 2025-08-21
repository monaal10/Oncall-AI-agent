"""Abstract base class for runbook providers."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum


class RunbookType(Enum):
    """Supported runbook types."""
    PDF = "pdf"
    MARKDOWN = "markdown"
    DOCX = "docx"
    WEB_LINK = "web_link"
    CONFLUENCE = "confluence"
    NOTION = "notion"


class RunbookProvider(ABC):
    """Abstract base class for all runbook providers.
    
    This class defines the interface that all runbook providers must implement
    to integrate with the OnCall AI Agent system.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the runbook provider with configuration.
        
        Args:
            config: Provider-specific configuration dictionary
        """
        self.config = config
        self._validate_config()

    @abstractmethod
    def _validate_config(self) -> None:
        """Validate the provider configuration.
        
        Raises:
            ValueError: If configuration is invalid or missing required fields
        """
        pass

    @abstractmethod
    async def get_runbook_text(
        self,
        runbook_id: str,
        **kwargs
    ) -> str:
        """Get the full text content of a runbook.
        
        Args:
            runbook_id: Identifier for the runbook (file path, URL, etc.)
            **kwargs: Provider-specific additional parameters
            
        Returns:
            Full text content of the runbook
            
        Raises:
            ConnectionError: If unable to connect to runbook source
            ValueError: If runbook not found or invalid
        """
        pass

    @abstractmethod
    async def search_runbooks(
        self,
        query: str,
        limit: Optional[int] = 10
    ) -> List[Dict[str, Any]]:
        """Search for runbooks containing specific content.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            
        Returns:
            List of runbook search results, each containing:
            - id: Runbook identifier
            - title: Runbook title
            - type: Runbook type (pdf, markdown, etc.)
            - excerpt: Relevant excerpt containing the query
            - relevance_score: Relevance score (0-1)
            - last_modified: When runbook was last modified
            - url: URL to access the runbook (if applicable)
            
        Raises:
            ConnectionError: If unable to search runbooks
            ValueError: If query parameters are invalid
        """
        pass

    @abstractmethod
    async def list_runbooks(self) -> List[Dict[str, Any]]:
        """List all available runbooks.
        
        Returns:
            List of runbooks, each containing:
            - id: Runbook identifier
            - title: Runbook title
            - type: Runbook type
            - description: Brief description
            - last_modified: When runbook was last modified
            - size: Size in bytes (if applicable)
            - url: URL to access the runbook (if applicable)
            
        Raises:
            ConnectionError: If unable to list runbooks
        """
        pass

    async def get_runbook_sections(
        self,
        runbook_id: str,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Get structured sections from a runbook.
        
        Args:
            runbook_id: Identifier for the runbook
            **kwargs: Provider-specific additional parameters
            
        Returns:
            List of sections, each containing:
            - title: Section title
            - content: Section content
            - level: Heading level (1-6)
            - line_start: Starting line number (if applicable)
            - line_end: Ending line number (if applicable)
            
        Raises:
            ConnectionError: If unable to access runbook
            ValueError: If runbook not found
        """
        # Default implementation: get full text and parse basic sections
        try:
            full_text = await self.get_runbook_text(runbook_id, **kwargs)
            return self._parse_sections(full_text)
        except Exception as e:
            raise ConnectionError(f"Failed to get runbook sections: {e}")

    async def find_relevant_sections(
        self,
        runbook_id: str,
        error_context: str,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Find runbook sections relevant to an error or incident.
        
        Args:
            runbook_id: Identifier for the runbook
            error_context: Error message or incident context
            **kwargs: Provider-specific additional parameters
            
        Returns:
            List of relevant sections with relevance scores
            
        Raises:
            ConnectionError: If unable to access runbook
        """
        try:
            sections = await self.get_runbook_sections(runbook_id, **kwargs)
            relevant_sections = []
            
            # Simple relevance scoring based on keyword matching
            error_keywords = self._extract_keywords(error_context.lower())
            
            for section in sections:
                relevance_score = self._calculate_relevance(
                    section['content'].lower(),
                    error_keywords
                )
                
                if relevance_score > 0.1:  # Minimum relevance threshold
                    section['relevance_score'] = relevance_score
                    relevant_sections.append(section)
            
            # Sort by relevance score
            relevant_sections.sort(key=lambda x: x['relevance_score'], reverse=True)
            return relevant_sections
            
        except Exception as e:
            raise ConnectionError(f"Failed to find relevant sections: {e}")

    async def health_check(self) -> bool:
        """Check if the runbook provider is accessible.
        
        Returns:
            True if provider is healthy, False otherwise
        """
        try:
            await self.list_runbooks()
            return True
        except Exception:
            return False

    def _parse_sections(self, text: str) -> List[Dict[str, Any]]:
        """Parse text into sections based on headers.
        
        Args:
            text: Full text content
            
        Returns:
            List of parsed sections
        """
        import re
        
        lines = text.split('\n')
        sections = []
        current_section = None
        
        for i, line in enumerate(lines):
            # Check for markdown-style headers
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if header_match:
                # Save previous section
                if current_section:
                    current_section['line_end'] = i - 1
                    current_section['content'] = '\n'.join(
                        lines[current_section['line_start']:current_section['line_end'] + 1]
                    )
                    sections.append(current_section)
                
                # Start new section
                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                current_section = {
                    'title': title,
                    'level': level,
                    'line_start': i,
                    'line_end': len(lines) - 1,  # Will be updated when next section found
                    'content': ''
                }
            
            # Check for other header patterns (underlined headers)
            elif i + 1 < len(lines):
                next_line = lines[i + 1]
                if re.match(r'^=+$', next_line) or re.match(r'^-+$', next_line):
                    # Save previous section
                    if current_section:
                        current_section['line_end'] = i - 1
                        current_section['content'] = '\n'.join(
                            lines[current_section['line_start']:current_section['line_end'] + 1]
                        )
                        sections.append(current_section)
                    
                    # Start new section
                    level = 1 if re.match(r'^=+$', next_line) else 2
                    current_section = {
                        'title': line.strip(),
                        'level': level,
                        'line_start': i,
                        'line_end': len(lines) - 1,
                        'content': ''
                    }
        
        # Add final section
        if current_section:
            current_section['content'] = '\n'.join(
                lines[current_section['line_start']:current_section['line_end'] + 1]
            )
            sections.append(current_section)
        
        # If no sections found, create a single section with all content
        if not sections:
            sections.append({
                'title': 'Content',
                'level': 1,
                'line_start': 0,
                'line_end': len(lines) - 1,
                'content': text
            })
        
        return sections

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text for relevance matching.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted keywords
        """
        import re
        
        # Remove common stop words
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'an', 'a', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has',
            'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
            'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
            'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
        }
        
        # Extract words (alphanumeric + underscore)
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter out stop words and short words
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for keyword in keywords:
            if keyword not in seen:
                seen.add(keyword)
                unique_keywords.append(keyword)
        
        return unique_keywords

    def _calculate_relevance(self, content: str, keywords: List[str]) -> float:
        """Calculate relevance score between content and keywords.
        
        Args:
            content: Content to score
            keywords: List of keywords to match
            
        Returns:
            Relevance score between 0 and 1
        """
        if not keywords:
            return 0.0
        
        content_words = set(self._extract_keywords(content))
        keyword_set = set(keywords)
        
        # Calculate intersection ratio
        intersection = content_words.intersection(keyword_set)
        if not intersection:
            return 0.0
        
        # Score based on intersection ratio and frequency
        base_score = len(intersection) / len(keyword_set)
        
        # Boost score based on keyword frequency in content
        frequency_boost = 0
        for keyword in intersection:
            frequency_boost += content.count(keyword) * 0.1
        
        # Cap the frequency boost
        frequency_boost = min(frequency_boost, 0.5)
        
        return min(base_score + frequency_boost, 1.0)
