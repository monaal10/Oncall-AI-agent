"""Unit tests for base provider classes."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock
from oncall_agent.integrations.base.log_provider import LogProvider
from oncall_agent.integrations.base.metrics_provider import MetricsProvider
from oncall_agent.integrations.base.code_provider import CodeProvider
from oncall_agent.integrations.base.llm_provider import LLMProvider
from oncall_agent.integrations.base.runbook_provider import RunbookProvider, RunbookType


class ConcreteLogProvider(LogProvider):
    """Concrete implementation of LogProvider for testing."""
    
    def _validate_config(self):
        pass
    
    async def fetch_logs(self, query, start_time, end_time, limit=1000, **kwargs):
        return []
    
    async def search_logs_by_pattern(self, pattern, start_time, end_time, log_groups=None, limit=1000):
        return []
    
    async def get_log_groups(self):
        return ["test-group"]


class ConcreteMetricsProvider(MetricsProvider):
    """Concrete implementation of MetricsProvider for testing."""
    
    def _validate_config(self):
        pass
    
    async def get_metric_data(self, metric_name, namespace, start_time, end_time, dimensions=None, statistic="Average", period=300):
        return []
    
    async def get_alarms(self, alarm_names=None, state_value=None):
        return []
    
    async def get_alarm_history(self, alarm_name, start_time, end_time, history_item_type=None):
        return []
    
    async def list_metrics(self, namespace=None, metric_name=None, dimensions=None):
        return []


class ConcreteCodeProvider(CodeProvider):
    """Concrete implementation of CodeProvider for testing."""
    
    def _validate_config(self):
        pass
    
    async def search_code(self, query, repositories=None, file_extension=None, limit=50):
        return []
    
    async def get_file_content(self, repository, file_path, branch="main"):
        return {"content": "test"}
    
    async def get_recent_commits(self, repository, since=None, author=None, limit=50):
        return []
    
    async def get_repository_info(self, repository):
        return {"name": "test"}
    
    async def get_pull_requests(self, repository, state="open", limit=50):
        return []
    
    async def get_issues(self, repository, state="open", labels=None, limit=50):
        return []


class ConcreteLLMProvider(LLMProvider):
    """Concrete implementation of LLMProvider for testing."""
    
    def _validate_config(self):
        pass
    
    def _setup_model(self):
        self._model = Mock()
    
    @property
    def model(self):
        return self._model
    
    async def generate_resolution(self, incident_context, system_prompt=None, **kwargs):
        return {"resolution_summary": "test"}
    
    async def analyze_logs(self, log_entries, context=None, **kwargs):
        return {"error_patterns": []}
    
    async def analyze_code_context(self, code_snippets, error_message, **kwargs):
        return {"potential_issues": []}
    
    async def stream_response(self, prompt, **kwargs):
        yield "test response"
    
    def get_model_info(self):
        return {"provider": "test"}


class ConcreteRunbookProvider(RunbookProvider):
    """Concrete implementation of RunbookProvider for testing."""
    
    def _validate_config(self):
        pass
    
    async def get_runbook_text(self, runbook_id, **kwargs):
        return "test runbook content"
    
    async def search_runbooks(self, query, limit=10):
        return []
    
    async def list_runbooks(self):
        return []


class TestLogProvider:
    """Test cases for LogProvider base class."""

    @pytest.fixture
    def log_provider(self):
        """Create concrete log provider instance."""
        return ConcreteLogProvider({"test": "config"})

    @pytest.mark.asyncio
    async def test_health_check_success(self, log_provider):
        """Test health check when provider is healthy."""
        log_provider.get_log_groups = AsyncMock(return_value=["group1", "group2"])
        
        is_healthy = await log_provider.health_check()
        
        assert is_healthy is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self, log_provider):
        """Test health check when provider fails."""
        log_provider.get_log_groups = AsyncMock(side_effect=Exception("Connection failed"))
        
        is_healthy = await log_provider.health_check()
        
        assert is_healthy is False


class TestMetricsProvider:
    """Test cases for MetricsProvider base class."""

    @pytest.fixture
    def metrics_provider(self):
        """Create concrete metrics provider instance."""
        return ConcreteMetricsProvider({"test": "config"})

    @pytest.mark.asyncio
    async def test_health_check_success(self, metrics_provider):
        """Test health check when provider is healthy."""
        metrics_provider.list_metrics = AsyncMock(return_value=[])
        
        is_healthy = await metrics_provider.health_check()
        
        assert is_healthy is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self, metrics_provider):
        """Test health check when provider fails."""
        metrics_provider.list_metrics = AsyncMock(side_effect=Exception("Connection failed"))
        
        is_healthy = await metrics_provider.health_check()
        
        assert is_healthy is False


class TestCodeProvider:
    """Test cases for CodeProvider base class."""

    @pytest.fixture
    def code_provider(self):
        """Create concrete code provider instance."""
        return ConcreteCodeProvider({"repositories": ["org/repo"]})

    @pytest.mark.asyncio
    async def test_health_check_success(self, code_provider):
        """Test health check when provider is healthy."""
        code_provider.get_repository_info = AsyncMock(return_value={"name": "test"})
        
        is_healthy = await code_provider.health_check()
        
        assert is_healthy is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self, code_provider):
        """Test health check when provider fails."""
        code_provider.get_repository_info = AsyncMock(side_effect=Exception("Connection failed"))
        
        is_healthy = await code_provider.health_check()
        
        assert is_healthy is False

    @pytest.mark.asyncio
    async def test_analyze_error_context(self, code_provider):
        """Test error context analysis."""
        # Mock the search methods
        code_provider.search_code = AsyncMock(return_value=[
            {"repository": "org/repo", "file_path": "test.py", "matches": []}
        ])
        code_provider.get_recent_commits = AsyncMock(return_value=[
            {"sha": "abc123", "message": "fix: database connection issue"}
        ])
        
        analysis = await code_provider.analyze_error_context(
            error_message="def connect() failed with timeout",
            stack_trace="File test.py, line 10"
        )
        
        assert "related_files" in analysis
        assert "function_definitions" in analysis
        assert "recent_changes" in analysis
        assert "suggestions" in analysis


class TestLLMProvider:
    """Test cases for LLMProvider base class."""

    @pytest.fixture
    def llm_provider(self):
        """Create concrete LLM provider instance."""
        return ConcreteLLMProvider({"test": "config"})

    @pytest.mark.asyncio
    async def test_health_check_success(self, llm_provider):
        """Test health check when provider is healthy."""
        is_healthy_result = await llm_provider.health_check()
        
        assert is_healthy_result["healthy"] is True
        assert "latency" in is_healthy_result
        assert is_healthy_result["error"] is None

    @pytest.mark.asyncio
    async def test_health_check_failure(self, llm_provider):
        """Test health check when provider fails."""
        llm_provider.generate_resolution = AsyncMock(side_effect=Exception("LLM failed"))
        
        is_healthy_result = await llm_provider.health_check()
        
        assert is_healthy_result["healthy"] is False
        assert is_healthy_result["error"] is not None

    @pytest.mark.asyncio
    async def test_create_chat_messages(self, llm_provider, sample_incident_context):
        """Test chat message creation from incident context."""
        messages = await llm_provider.create_chat_messages(
            incident_context=sample_incident_context,
            system_prompt="Test system prompt"
        )
        
        assert len(messages) == 2  # System + Human message
        # Note: Would need to import actual message types to test properly

    def test_get_default_system_prompt(self, llm_provider):
        """Test default system prompt generation."""
        prompt = llm_provider._get_default_system_prompt()
        
        assert "DevOps engineer" in prompt
        assert "incident response" in prompt
        assert "Resolution Steps" in prompt

    def test_format_incident_context(self, llm_provider, sample_incident_context):
        """Test incident context formatting."""
        formatted = llm_provider._format_incident_context(sample_incident_context)
        
        assert "**Incident Description:**" in formatted
        assert "**Recent Logs:**" in formatted
        assert "**Key Metrics:**" in formatted
        assert "**Relevant Code:**" in formatted
        assert "**Runbook Guidance:**" in formatted

    def test_get_token_count(self, llm_provider):
        """Test token count estimation."""
        text = "This is a test message with some words"
        count = llm_provider.get_token_count(text)
        
        assert count > 0
        assert count == len(text) // 4  # Simple estimation

    def test_truncate_context(self, llm_provider, sample_incident_context):
        """Test context truncation for token limits."""
        # Test with very small limit to force truncation
        truncated = llm_provider.truncate_context(sample_incident_context, max_tokens=100)
        
        # Should still have incident description
        assert "incident_description" in truncated
        
        # Some fields might be truncated or removed
        original_keys = set(sample_incident_context.keys())
        truncated_keys = set(truncated.keys())
        assert truncated_keys.issubset(original_keys)


class TestRunbookProvider:
    """Test cases for RunbookProvider base class."""

    @pytest.fixture
    def runbook_provider(self):
        """Create concrete runbook provider instance."""
        return ConcreteRunbookProvider({"test": "config"})

    @pytest.mark.asyncio
    async def test_health_check_success(self, runbook_provider):
        """Test health check when provider is healthy."""
        runbook_provider.list_runbooks = AsyncMock(return_value=[])
        
        is_healthy = await runbook_provider.health_check()
        
        assert is_healthy is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self, runbook_provider):
        """Test health check when provider fails."""
        runbook_provider.list_runbooks = AsyncMock(side_effect=Exception("Connection failed"))
        
        is_healthy = await runbook_provider.health_check()
        
        assert is_healthy is False

    @pytest.mark.asyncio
    async def test_get_runbook_sections_default(self, runbook_provider):
        """Test default runbook sections implementation."""
        runbook_provider.get_runbook_text = AsyncMock(return_value="# Section 1\nContent 1\n## Section 2\nContent 2")
        
        sections = await runbook_provider.get_runbook_sections("test.md")
        
        assert len(sections) >= 1
        assert all("title" in section for section in sections)
        assert all("content" in section for section in sections)

    def test_extract_keywords(self, runbook_provider):
        """Test keyword extraction."""
        text = "Database connection timeout error in production environment"
        keywords = runbook_provider._extract_keywords(text)
        
        assert "database" in keywords
        assert "connection" in keywords
        assert "timeout" in keywords
        # Stop words should be filtered
        assert "in" not in keywords
        assert "the" not in keywords

    def test_calculate_relevance(self, runbook_provider):
        """Test relevance calculation."""
        content = "database connection timeout error database"
        keywords = ["database", "connection", "timeout"]
        
        relevance = runbook_provider._calculate_relevance(content, keywords)
        
        assert 0 <= relevance <= 1
        assert relevance > 0

    def test_parse_sections_markdown_headers(self, runbook_provider):
        """Test section parsing with markdown headers."""
        text = """# Main Title
Content for main section.

## Section 1
Content for section 1.

### Subsection 1.1
Content for subsection.

## Section 2
Content for section 2.
"""
        
        sections = runbook_provider._parse_sections(text)
        
        assert len(sections) >= 3
        
        # Check main title
        main_section = next((s for s in sections if s["title"] == "Main Title"), None)
        assert main_section is not None
        assert main_section["level"] == 1
        
        # Check subsection
        subsection = next((s for s in sections if s["title"] == "Subsection 1.1"), None)
        assert subsection is not None
        assert subsection["level"] == 3

    def test_parse_sections_underlined_headers(self, runbook_provider):
        """Test section parsing with underlined headers."""
        text = """Main Title
==========

Content for main section.

Section 1
---------

Content for section 1.
"""
        
        sections = runbook_provider._parse_sections(text)
        
        assert len(sections) >= 2
        
        # Check main title (level 1 for = underline)
        main_section = next((s for s in sections if s["title"] == "Main Title"), None)
        assert main_section is not None
        assert main_section["level"] == 1
        
        # Check section (level 2 for - underline)
        section1 = next((s for s in sections if s["title"] == "Section 1"), None)
        assert section1 is not None
        assert section1["level"] == 2

    def test_parse_sections_no_headers(self, runbook_provider):
        """Test section parsing with no headers."""
        text = "Just plain content without any headers."
        
        sections = runbook_provider._parse_sections(text)
        
        assert len(sections) == 1
        assert sections[0]["title"] == "Content"
        assert sections[0]["level"] == 1
        assert sections[0]["content"] == text
