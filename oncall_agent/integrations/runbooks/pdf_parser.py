"""PDF runbook parser implementation."""

import os
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime

try:
    import PyPDF2
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

from ..base.runbook_provider import RunbookProvider, RunbookType


class PDFRunbookProvider(RunbookProvider):
    """PDF runbook provider implementation.
    
    Provides integration with PDF files for runbook content extraction
    and search functionality.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize PDF runbook provider.
        
        Args:
            config: Configuration dictionary containing:
                - runbook_directory: Directory containing PDF files (required)
                - recursive: Whether to search subdirectories (optional, default: True)
                - cache_enabled: Whether to cache extracted text (optional, default: True)
        """
        if not PDF_AVAILABLE:
            raise ImportError(
                "PDF support not available. Install with: pip install PyPDF2 pdfplumber"
            )
        
        super().__init__(config)
        self._text_cache = {} if config.get("cache_enabled", True) else None

    def _validate_config(self) -> None:
        """Validate PDF runbook configuration.
        
        Raises:
            ValueError: If required configuration is missing
        """
        if "runbook_directory" not in self.config:
            raise ValueError("runbook_directory is required for PDF runbook provider")
        
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
        """Get the full text content of a PDF runbook.
        
        Args:
            runbook_id: PDF file path (relative to runbook_directory or absolute)
            **kwargs: Additional parameters:
                - use_ocr: Whether to use OCR for scanned PDFs (default: False)
                - pages: Specific pages to extract (list of page numbers)
                
        Returns:
            Full text content of the PDF
            
        Raises:
            ConnectionError: If unable to read the PDF file
            ValueError: If PDF file not found
        """
        try:
            # Resolve file path
            file_path = self._resolve_file_path(runbook_id)
            
            # Check cache first
            if self._text_cache is not None and file_path in self._text_cache:
                cache_entry = self._text_cache[file_path]
                # Check if file has been modified since caching
                if cache_entry['modified'] >= os.path.getmtime(file_path):
                    return cache_entry['text']
            
            # Extract text from PDF
            text = await asyncio.to_thread(
                self._extract_pdf_text,
                file_path,
                kwargs.get('use_ocr', False),
                kwargs.get('pages')
            )
            
            # Cache the result
            if self._text_cache is not None:
                self._text_cache[file_path] = {
                    'text': text,
                    'modified': os.path.getmtime(file_path)
                }
            
            return text
            
        except FileNotFoundError:
            raise ValueError(f"PDF runbook not found: {runbook_id}")
        except Exception as e:
            raise ConnectionError(f"Failed to read PDF runbook: {e}")

    async def search_runbooks(
        self,
        query: str,
        limit: Optional[int] = 10
    ) -> List[Dict[str, Any]]:
        """Search for PDF runbooks containing specific content.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            
        Returns:
            List of PDF runbook search results
            
        Raises:
            ConnectionError: If unable to search runbooks
        """
        try:
            pdf_files = await self._find_pdf_files()
            results = []
            query_lower = query.lower()
            
            for pdf_file in pdf_files:
                try:
                    # Get text content
                    text = await self.get_runbook_text(pdf_file)
                    text_lower = text.lower()
                    
                    # Check if query matches
                    if query_lower in text_lower:
                        # Extract excerpt around the match
                        excerpt = self._extract_excerpt(text, query, 200)
                        
                        # Calculate relevance score
                        relevance_score = self._calculate_pdf_relevance(text_lower, query_lower)
                        
                        # Get file info
                        full_path = self._resolve_file_path(pdf_file)
                        file_stat = os.stat(full_path)
                        
                        result = {
                            'id': pdf_file,
                            'title': self._get_pdf_title(pdf_file),
                            'type': RunbookType.PDF.value,
                            'excerpt': excerpt,
                            'relevance_score': relevance_score,
                            'last_modified': datetime.fromtimestamp(file_stat.st_mtime),
                            'url': f"file://{full_path}",
                            'file_size': file_stat.st_size
                        }
                        results.append(result)
                        
                except Exception as e:
                    # Skip files that can't be processed
                    continue
            
            # Sort by relevance score
            results.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            return results[:limit] if limit else results
            
        except Exception as e:
            raise ConnectionError(f"Failed to search PDF runbooks: {e}")

    async def list_runbooks(self) -> List[Dict[str, Any]]:
        """List all available PDF runbooks.
        
        Returns:
            List of PDF runbooks with metadata
            
        Raises:
            ConnectionError: If unable to list runbooks
        """
        try:
            pdf_files = await self._find_pdf_files()
            runbooks = []
            
            for pdf_file in pdf_files:
                try:
                    full_path = self._resolve_file_path(pdf_file)
                    file_stat = os.stat(full_path)
                    
                    # Try to get PDF metadata
                    metadata = await asyncio.to_thread(self._get_pdf_metadata, full_path)
                    
                    runbook = {
                        'id': pdf_file,
                        'title': metadata.get('title') or self._get_pdf_title(pdf_file),
                        'type': RunbookType.PDF.value,
                        'description': metadata.get('subject', 'PDF runbook'),
                        'last_modified': datetime.fromtimestamp(file_stat.st_mtime),
                        'size': file_stat.st_size,
                        'url': f"file://{full_path}",
                        'pages': metadata.get('pages', 0),
                        'author': metadata.get('author', 'Unknown')
                    }
                    runbooks.append(runbook)
                    
                except Exception:
                    # Skip files that can't be processed
                    continue
            
            return runbooks
            
        except Exception as e:
            raise ConnectionError(f"Failed to list PDF runbooks: {e}")

    def _resolve_file_path(self, runbook_id: str) -> str:
        """Resolve runbook ID to full file path.
        
        Args:
            runbook_id: Runbook identifier (file path)
            
        Returns:
            Full path to the PDF file
        """
        if os.path.isabs(runbook_id):
            return runbook_id
        else:
            return os.path.join(self.config["runbook_directory"], runbook_id)

    async def _find_pdf_files(self) -> List[str]:
        """Find all PDF files in the runbook directory.
        
        Returns:
            List of PDF file paths (relative to runbook_directory)
        """
        pdf_files = []
        directory = self.config["runbook_directory"]
        recursive = self.config.get("recursive", True)
        
        if recursive:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.lower().endswith('.pdf'):
                        full_path = os.path.join(root, file)
                        relative_path = os.path.relpath(full_path, directory)
                        pdf_files.append(relative_path)
        else:
            for file in os.listdir(directory):
                if file.lower().endswith('.pdf'):
                    full_path = os.path.join(directory, file)
                    if os.path.isfile(full_path):
                        pdf_files.append(file)
        
        return pdf_files

    def _extract_pdf_text(
        self,
        file_path: str,
        use_ocr: bool = False,
        pages: Optional[List[int]] = None
    ) -> str:
        """Extract text from PDF file.
        
        Args:
            file_path: Path to PDF file
            use_ocr: Whether to use OCR for scanned PDFs
            pages: Specific pages to extract
            
        Returns:
            Extracted text content
        """
        text_parts = []
        
        try:
            # Try pdfplumber first (better text extraction)
            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)
                pages_to_process = pages if pages else range(total_pages)
                
                for page_num in pages_to_process:
                    if 0 <= page_num < total_pages:
                        page = pdf.pages[page_num]
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(f"--- Page {page_num + 1} ---\n{page_text}\n")
                        
        except Exception:
            # Fallback to PyPDF2
            try:
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    total_pages = len(pdf_reader.pages)
                    pages_to_process = pages if pages else range(total_pages)
                    
                    for page_num in pages_to_process:
                        if 0 <= page_num < total_pages:
                            page = pdf_reader.pages[page_num]
                            page_text = page.extract_text()
                            if page_text:
                                text_parts.append(f"--- Page {page_num + 1} ---\n{page_text}\n")
                                
            except Exception as e:
                raise ConnectionError(f"Failed to extract text from PDF: {e}")
        
        return '\n'.join(text_parts)

    def _get_pdf_metadata(self, file_path: str) -> Dict[str, Any]:
        """Get PDF metadata.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Dictionary containing PDF metadata
        """
        metadata = {}
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                if pdf_reader.metadata:
                    metadata.update({
                        'title': pdf_reader.metadata.get('/Title'),
                        'author': pdf_reader.metadata.get('/Author'),
                        'subject': pdf_reader.metadata.get('/Subject'),
                        'creator': pdf_reader.metadata.get('/Creator'),
                        'producer': pdf_reader.metadata.get('/Producer'),
                        'creation_date': pdf_reader.metadata.get('/CreationDate'),
                        'modification_date': pdf_reader.metadata.get('/ModDate')
                    })
                
                metadata['pages'] = len(pdf_reader.pages)
                
        except Exception:
            # If metadata extraction fails, return empty dict
            pass
        
        return metadata

    def _get_pdf_title(self, pdf_file: str) -> str:
        """Generate a title from PDF filename.
        
        Args:
            pdf_file: PDF file path
            
        Returns:
            Human-readable title
        """
        # Get filename without extension
        filename = os.path.splitext(os.path.basename(pdf_file))[0]
        
        # Replace underscores and hyphens with spaces
        title = filename.replace('_', ' ').replace('-', ' ')
        
        # Capitalize words
        title = ' '.join(word.capitalize() for word in title.split())
        
        return title

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

    def _calculate_pdf_relevance(self, text: str, query: str) -> float:
        """Calculate relevance score for PDF content.
        
        Args:
            text: PDF text content (lowercase)
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
        
        # Boost score if query appears in different contexts
        words = text.split()
        contexts = set()
        for i, word in enumerate(words):
            if query in word:
                # Get surrounding context (5 words before and after)
                context_start = max(0, i - 5)
                context_end = min(len(words), i + 6)
                context = ' '.join(words[context_start:context_end])
                contexts.add(context[:50])  # Limit context length
        
        context_diversity = min(len(contexts) / 5, 1.0)  # Max 5 different contexts
        
        return min(frequency_score + context_diversity * 0.3, 1.0)
