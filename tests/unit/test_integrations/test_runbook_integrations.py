"""Unit tests for runbook integrations."""

import pytest
import os
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch, mock_open
from oncall_agent.integrations.runbooks.markdown_parser import MarkdownRunbookProvider
from oncall_agent.integrations.runbooks.manager import UnifiedRunbookProvider
from oncall_agent.integrations.base.runbook_provider import RunbookType


class TestMarkdownRunbookProvider:
    """Test cases for MarkdownRunbookProvider."""

    @pytest.fixture
    def markdown_config(self, tmp_path):
        """Create markdown configuration with temporary directory."""
        runbook_dir = tmp_path / "runbooks"
        runbook_dir.mkdir()
        
        # Create test markdown file
        test_file = runbook_dir / "test-runbook.md"
        test_file.write_text("""# Database Issues

This is a test runbook for database troubleshooting.

## Connection Problems

Check the following:
- Database server status
- Connection pool settings
- Network connectivity

## Performance Issues

Monitor these metrics:
- CPU utilization
- Memory usage
- Disk I/O
""")
        
        return {
            "runbook_directory": str(runbook_dir),
            "recursive": True,
            "cache_enabled": True
        }

    def test_init_success(self, markdown_config):
        """Test Markdown provider initialization."""
        provider = MarkdownRunbookProvider(markdown_config)
        
        assert provider.config == markdown_config
        assert provider._content_cache is not None

    def test_validate_config_missing_directory(self):
        """Test configuration validation with missing directory."""
        config = {"recursive": True}
        
        with pytest.raises(ValueError) as exc_info:
            MarkdownRunbookProvider(config)
        
        assert "runbook_directory is required" in str(exc_info.value)

    def test_validate_config_nonexistent_directory(self):
        """Test configuration validation with nonexistent directory."""
        config = {"runbook_directory": "/nonexistent/path"}
        
        with pytest.raises(ValueError) as exc_info:
            MarkdownRunbookProvider(config)
        
        assert "does not exist" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_runbook_text_success(self, markdown_config):
        """Test successful runbook text retrieval."""
        provider = MarkdownRunbookProvider(markdown_config)
        
        text = await provider.get_runbook_text("test-runbook.md")
        
        assert "# Database Issues" in text
        assert "Connection Problems" in text
        assert "Performance Issues" in text

    @pytest.mark.asyncio
    async def test_get_runbook_text_not_found(self, markdown_config):
        """Test runbook text retrieval with file not found."""
        provider = MarkdownRunbookProvider(markdown_config)
        
        with pytest.raises(ValueError) as exc_info:
            await provider.get_runbook_text("nonexistent.md")
        
        assert "not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_search_runbooks_success(self, markdown_config):
        """Test successful runbook search."""
        provider = MarkdownRunbookProvider(markdown_config)
        
        results = await provider.search_runbooks(
            query="database",
            limit=10
        )
        
        assert len(results) == 1
        assert results[0]["title"] == "Database Issues"
        assert results[0]["type"] == RunbookType.MARKDOWN.value
        assert "database" in results[0]["excerpt"].lower()
        assert results[0]["relevance_score"] > 0

    @pytest.mark.asyncio
    async def test_list_runbooks_success(self, markdown_config):
        """Test successful runbook listing."""
        provider = MarkdownRunbookProvider(markdown_config)
        
        runbooks = await provider.list_runbooks()
        
        assert len(runbooks) == 1
        assert runbooks[0]["title"] == "Database Issues"
        assert runbooks[0]["type"] == RunbookType.MARKDOWN.value
        assert runbooks[0]["id"] == "test-runbook.md"

    def test_extract_title_from_content(self, markdown_config):
        """Test title extraction from markdown content."""
        provider = MarkdownRunbookProvider(markdown_config)
        
        content = "# Main Title\n\nSome content here"
        title = provider._extract_title_from_content(content)
        
        assert title == "Main Title"

    def test_extract_title_from_content_underlined(self, markdown_config):
        """Test title extraction from underlined header."""
        provider = MarkdownRunbookProvider(markdown_config)
        
        content = "Main Title\n==========\n\nSome content"
        title = provider._extract_title_from_content(content)
        
        assert title == "Main Title"

    def test_get_title_from_filename(self, markdown_config):
        """Test title generation from filename."""
        provider = MarkdownRunbookProvider(markdown_config)
        
        title = provider._get_title_from_filename("database-connection-issues.md")
        
        assert title == "Database Connection Issues"

    def test_strip_markdown_formatting(self, markdown_config):
        """Test markdown formatting removal."""
        provider = MarkdownRunbookProvider(markdown_config)
        
        content = """
# Header

**Bold text** and *italic text*

- List item 1
- List item 2

```python
code block
```

[Link](http://example.com)
"""
        
        stripped = provider._strip_markdown_formatting(content)
        
        assert "**Bold text**" not in stripped
        assert "Bold text" in stripped
        assert "*italic text*" not in stripped
        assert "italic text" in stripped
        assert "```python" not in stripped
        assert "[Link](http://example.com)" not in stripped
        assert "Link" in stripped


class TestUnifiedRunbookProvider:
    """Test cases for UnifiedRunbookProvider."""

    @pytest.fixture
    def unified_config(self, tmp_path):
        """Create unified runbook configuration."""
        # Create test directories and files
        md_dir = tmp_path / "markdown"
        md_dir.mkdir()
        (md_dir / "test.md").write_text("# Test\nContent")
        
        return {
            "providers": {
                "markdown_runbooks": {
                    "type": "markdown",
                    "config": {
                        "runbook_directory": str(md_dir),
                        "recursive": True
                    }
                }
            }
        }

    def test_init_success(self, unified_config):
        """Test unified provider initialization."""
        with patch('oncall_agent.integrations.runbooks.manager.MarkdownRunbookProvider'):
            provider = UnifiedRunbookProvider(unified_config)
            
            assert len(provider.providers) == 1
            assert "markdown_runbooks" in provider.providers

    def test_validate_config_missing_providers(self):
        """Test configuration validation with missing providers."""
        config = {}
        
        with pytest.raises(ValueError) as exc_info:
            UnifiedRunbookProvider(config)
        
        assert "At least one provider configuration is required" in str(exc_info.value)

    def test_validate_config_missing_provider_type(self):
        """Test configuration validation with missing provider type."""
        config = {
            "providers": {
                "test_provider": {
                    "config": {}
                }
            }
        }
        
        with pytest.raises(ValueError) as exc_info:
            UnifiedRunbookProvider(config)
        
        assert "Provider type is required" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_search_runbooks_success(self, unified_config):
        """Test successful runbook search across providers."""
        with patch('oncall_agent.integrations.runbooks.manager.MarkdownRunbookProvider') as mock_provider_class:
            mock_provider = AsyncMock()
            mock_provider.search_runbooks.return_value = [
                {
                    "id": "test.md",
                    "title": "Test Runbook",
                    "type": "markdown",
                    "relevance_score": 0.8
                }
            ]
            mock_provider_class.return_value = mock_provider
            
            provider = UnifiedRunbookProvider(unified_config)
            
            results = await provider.search_runbooks("test query", limit=5)
            
            assert len(results) == 1
            assert results[0]["title"] == "Test Runbook"
            assert results[0]["provider"] == "markdown_runbooks"

    @pytest.mark.asyncio
    async def test_find_relevant_runbooks(self, unified_config):
        """Test finding relevant runbooks for error context."""
        with patch('oncall_agent.integrations.runbooks.manager.MarkdownRunbookProvider') as mock_provider_class:
            mock_provider = AsyncMock()
            mock_provider.search_runbooks.return_value = [
                {
                    "id": "db-issues.md",
                    "title": "Database Issues",
                    "type": "markdown",
                    "relevance_score": 0.9
                }
            ]
            mock_provider_class.return_value = mock_provider
            
            provider = UnifiedRunbookProvider(unified_config)
            
            relevant = await provider.find_relevant_runbooks(
                error_context="database connection timeout",
                limit=3
            )
            
            assert len(relevant) == 1
            assert relevant[0]["title"] == "Database Issues"

    @pytest.mark.asyncio
    async def test_get_comprehensive_runbook_context(self, unified_config):
        """Test getting comprehensive runbook context."""
        with patch('oncall_agent.integrations.runbooks.manager.MarkdownRunbookProvider') as mock_provider_class:
            mock_provider = AsyncMock()
            mock_provider.search_runbooks.return_value = [
                {
                    "id": "test.md",
                    "title": "Test Runbook",
                    "type": "markdown",
                    "relevance_score": 0.8
                }
            ]
            mock_provider.find_relevant_sections.return_value = [
                {
                    "title": "Section 1",
                    "content": "Section content",
                    "relevance_score": 0.7
                }
            ]
            mock_provider_class.return_value = mock_provider
            
            provider = UnifiedRunbookProvider(unified_config)
            provider.get_runbook_text = AsyncMock(return_value="Test runbook content")
            
            context = await provider.get_comprehensive_runbook_context(
                error_context="test error",
                limit=2
            )
            
            assert "relevant_runbooks" in context
            assert "combined_text" in context
            assert "sections" in context
            assert "summary" in context
            assert len(context["relevant_runbooks"]) == 1

    def test_detect_provider_pdf(self, unified_config):
        """Test provider detection for PDF files."""
        with patch('oncall_agent.integrations.runbooks.manager.MarkdownRunbookProvider'):
            provider = UnifiedRunbookProvider(unified_config)
            
            # Mock PDF provider
            mock_pdf_provider = Mock()
            provider.providers["pdf_provider"] = mock_pdf_provider
            
            detected = provider._detect_provider("test-file.pdf")
            
            # Should return None since no PDF provider is configured in this test
            assert detected is None

    def test_detect_provider_markdown(self, unified_config):
        """Test provider detection for Markdown files."""
        with patch('oncall_agent.integrations.runbooks.manager.MarkdownRunbookProvider') as mock_provider_class:
            mock_provider = mock_provider_class.return_value
            provider = UnifiedRunbookProvider(unified_config)
            
            detected = provider._detect_provider("test-file.md")
            
            assert detected == mock_provider

    def test_detect_provider_web_url(self, unified_config):
        """Test provider detection for web URLs."""
        with patch('oncall_agent.integrations.runbooks.manager.MarkdownRunbookProvider'):
            provider = UnifiedRunbookProvider(unified_config)
            
            detected = provider._detect_provider("https://docs.example.com/runbook")
            
            # Should return None since no web provider is configured
            assert detected is None

    @pytest.mark.asyncio
    async def test_health_check(self, unified_config):
        """Test health check for all providers."""
        with patch('oncall_agent.integrations.runbooks.manager.MarkdownRunbookProvider') as mock_provider_class:
            mock_provider = AsyncMock()
            mock_provider.health_check.return_value = True
            mock_provider_class.return_value = mock_provider
            
            provider = UnifiedRunbookProvider(unified_config)
            
            health_results = await provider.health_check()
            
            assert "markdown_runbooks" in health_results
            assert health_results["markdown_runbooks"] is True

    def test_generate_runbook_summary_with_runbooks(self, unified_config):
        """Test runbook summary generation with available runbooks."""
        with patch('oncall_agent.integrations.runbooks.manager.MarkdownRunbookProvider'):
            provider = UnifiedRunbookProvider(unified_config)
            
            runbooks = [
                {"title": "Database Issues", "type": "markdown"},
                {"title": "API Errors", "type": "markdown"}
            ]
            sections = [
                {"title": "Section 1", "relevance_score": 0.8},
                {"title": "Section 2", "relevance_score": 0.6}
            ]
            
            summary = provider._generate_runbook_summary(runbooks, sections)
            
            assert "Found 2 relevant runbooks" in summary
            assert "2 markdown" in summary
            assert "2 highly relevant sections" in summary or "2 potentially relevant sections" in summary

    def test_generate_runbook_summary_no_runbooks(self, unified_config):
        """Test runbook summary generation with no runbooks."""
        with patch('oncall_agent.integrations.runbooks.manager.MarkdownRunbookProvider'):
            provider = UnifiedRunbookProvider(unified_config)
            
            summary = provider._generate_runbook_summary([], [])
            
            assert summary == "No relevant runbooks found."

    @pytest.mark.asyncio
    async def test_get_runbook_by_type(self, unified_config):
        """Test getting runbooks by specific type."""
        with patch('oncall_agent.integrations.runbooks.manager.MarkdownRunbookProvider') as mock_provider_class:
            mock_provider = AsyncMock()
            mock_provider.list_runbooks.return_value = [
                {"id": "test.md", "title": "Test", "type": "markdown"}
            ]
            mock_provider_class.return_value = mock_provider
            
            provider = UnifiedRunbookProvider(unified_config)
            
            runbooks = await provider.get_runbook_by_type(RunbookType.MARKDOWN)
            
            assert len(runbooks) == 1
            assert runbooks[0]["type"] == "markdown"
            assert runbooks[0]["provider"] == "markdown_runbooks"


class TestRunbookBaseProvider:
    """Test cases for base runbook provider functionality."""

    @pytest.fixture
    def base_provider_config(self):
        """Base provider configuration."""
        return {"test_setting": "test_value"}

    def test_extract_keywords(self, markdown_config):
        """Test keyword extraction from text."""
        provider = MarkdownRunbookProvider(markdown_config)
        
        text = "Database connection timeout error in production environment"
        keywords = provider._extract_keywords(text)
        
        assert "database" in keywords
        assert "connection" in keywords
        assert "timeout" in keywords
        assert "error" in keywords
        assert "production" in keywords
        assert "environment" in keywords
        
        # Stop words should be filtered out
        assert "in" not in keywords
        assert "the" not in keywords

    def test_calculate_relevance(self, markdown_config):
        """Test relevance score calculation."""
        provider = MarkdownRunbookProvider(markdown_config)
        
        content = "database connection timeout error database connection"
        keywords = ["database", "connection", "timeout"]
        
        relevance = provider._calculate_relevance(content, keywords)
        
        assert 0 <= relevance <= 1
        assert relevance > 0.5  # Should be high relevance

    def test_calculate_relevance_no_match(self, markdown_config):
        """Test relevance score calculation with no matches."""
        provider = MarkdownRunbookProvider(markdown_config)
        
        content = "completely unrelated content"
        keywords = ["database", "connection", "timeout"]
        
        relevance = provider._calculate_relevance(content, keywords)
        
        assert relevance == 0.0

    @pytest.mark.asyncio
    async def test_parse_sections(self, markdown_config):
        """Test section parsing from markdown content."""
        provider = MarkdownRunbookProvider(markdown_config)
        
        content = """# Main Title

Some introduction text.

## Section 1

Content for section 1.

### Subsection 1.1

Subsection content.

## Section 2

Content for section 2.
"""
        
        sections = provider._parse_sections(content)
        
        assert len(sections) >= 3  # Should have at least 3 sections
        
        # Check section structure
        main_section = next((s for s in sections if s["title"] == "Main Title"), None)
        assert main_section is not None
        assert main_section["level"] == 1
        
        section1 = next((s for s in sections if s["title"] == "Section 1"), None)
        assert section1 is not None
        assert section1["level"] == 2
