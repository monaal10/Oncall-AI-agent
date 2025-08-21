"""Unit tests for GitHub integration."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from oncall_agent.integrations.github.repository import GitHubRepositoryProvider
from oncall_agent.integrations.github.code_analyzer import GitHubCodeAnalyzer


class TestGitHubRepositoryProvider:
    """Test cases for GitHubRepositoryProvider."""

    @patch('oncall_agent.integrations.github.repository.Github')
    def test_init_success(self, mock_github_class, mock_github_config):
        """Test GitHub repository provider initialization."""
        mock_github = Mock()
        mock_user = Mock()
        mock_user.login = "testuser"
        mock_github.get_user.return_value = mock_user
        mock_github_class.return_value = mock_github
        
        provider = GitHubRepositoryProvider(mock_github_config)
        
        assert provider.config == mock_github_config
        assert provider.github == mock_github

    def test_validate_config_missing_token(self):
        """Test configuration validation with missing token."""
        config = {"repositories": ["org/repo"]}
        
        with pytest.raises(ValueError) as exc_info:
            GitHubRepositoryProvider(config)
        
        assert "GitHub token is required" in str(exc_info.value)

    def test_validate_config_missing_repositories(self):
        """Test configuration validation with missing repositories."""
        config = {"token": "test_token"}
        
        with pytest.raises(ValueError) as exc_info:
            GitHubRepositoryProvider(config)
        
        assert "At least one repository must be specified" in str(exc_info.value)

    @patch('oncall_agent.integrations.github.repository.Github')
    def test_get_repository_success(self, mock_github_class, mock_github_config):
        """Test getting repository object."""
        mock_github = Mock()
        mock_repo = Mock()
        mock_github.get_repo.return_value = mock_repo
        mock_github.get_user.return_value = Mock(login="testuser")
        mock_github_class.return_value = mock_github
        
        provider = GitHubRepositoryProvider(mock_github_config)
        repo = provider._get_repository("org/repo")
        
        assert repo == mock_repo
        mock_github.get_repo.assert_called_once_with("org/repo")

    @pytest.mark.asyncio
    @patch('oncall_agent.integrations.github.repository.Github')
    @patch('oncall_agent.integrations.github.repository.asyncio.to_thread')
    async def test_search_code_success(self, mock_to_thread, mock_github_class, mock_github_config):
        """Test successful code search."""
        mock_github = Mock()
        mock_github.get_user.return_value = Mock(login="testuser")
        mock_github_class.return_value = mock_github
        
        # Mock search results
        mock_result = Mock()
        mock_result.content = "dGVzdCBjb250ZW50"  # base64 encoded "test content"
        mock_result.repository.full_name = "org/repo"
        mock_result.path = "src/main.py"
        mock_result.name = "main.py"
        mock_result.html_url = "https://github.com/org/repo/blob/main/src/main.py"
        mock_result.sha = "abc123"
        mock_result.repository.updated_at = datetime.now()
        
        mock_to_thread.return_value = [mock_result]
        
        provider = GitHubRepositoryProvider(mock_github_config)
        
        results = await provider.search_code(
            query="def test",
            limit=10
        )
        
        assert len(results) == 1
        assert results[0]["repository"] == "org/repo"
        assert results[0]["file_path"] == "src/main.py"
        assert results[0]["file_name"] == "main.py"

    @pytest.mark.asyncio
    @patch('oncall_agent.integrations.github.repository.Github')
    @patch('oncall_agent.integrations.github.repository.asyncio.to_thread')
    async def test_get_file_content_success(self, mock_to_thread, mock_github_class, mock_github_config):
        """Test successful file content retrieval."""
        mock_github = Mock()
        mock_github.get_user.return_value = Mock(login="testuser")
        mock_repo = Mock()
        mock_github.get_repo.return_value = mock_repo
        mock_github_class.return_value = mock_github
        
        # Mock file content
        mock_file_content = Mock()
        mock_file_content.content = "dGVzdCBjb250ZW50"  # base64 encoded "test content"
        mock_file_content.encoding = "base64"
        mock_file_content.size = 12
        mock_file_content.sha = "abc123"
        mock_file_content.html_url = "https://github.com/org/repo/blob/main/test.py"
        mock_file_content.last_modified = datetime.now()
        mock_file_content.download_url = "https://raw.githubusercontent.com/org/repo/main/test.py"
        
        mock_to_thread.return_value = mock_file_content
        
        provider = GitHubRepositoryProvider(mock_github_config)
        
        file_content = await provider.get_file_content(
            repository="org/repo",
            file_path="test.py"
        )
        
        assert file_content["content"] == "test content"
        assert file_content["encoding"] == "base64"
        assert file_content["size"] == 12
        assert file_content["sha"] == "abc123"

    @pytest.mark.asyncio
    @patch('oncall_agent.integrations.github.repository.Github')
    @patch('oncall_agent.integrations.github.repository.asyncio.to_thread')
    async def test_get_recent_commits_success(self, mock_to_thread, mock_github_class, mock_github_config):
        """Test successful recent commits retrieval."""
        mock_github = Mock()
        mock_github.get_user.return_value = Mock(login="testuser")
        mock_repo = Mock()
        mock_github.get_repo.return_value = mock_repo
        mock_github_class.return_value = mock_github
        
        # Mock commit
        mock_commit = Mock()
        mock_commit.sha = "abc123"
        mock_commit.commit.message = "Test commit"
        mock_commit.commit.author.name = "Test Author"
        mock_commit.commit.author.email = "test@example.com"
        mock_commit.commit.author.date = datetime.now()
        mock_commit.author.login = "testuser"
        mock_commit.html_url = "https://github.com/org/repo/commit/abc123"
        mock_commit.files = [
            Mock(filename="test.py", status="modified", additions=5, deletions=2, changes=7)
        ]
        
        mock_to_thread.return_value = [mock_commit]
        
        provider = GitHubRepositoryProvider(mock_github_config)
        
        commits = await provider.get_recent_commits(
            repository="org/repo",
            limit=10
        )
        
        assert len(commits) == 1
        assert commits[0]["sha"] == "abc123"
        assert commits[0]["message"] == "Test commit"
        assert commits[0]["author"]["name"] == "Test Author"
        assert len(commits[0]["files_changed"]) == 1

    @pytest.mark.asyncio
    @patch('oncall_agent.integrations.github.repository.Github')
    async def test_get_error_related_issues(self, mock_github_class, mock_github_config):
        """Test getting error-related issues."""
        mock_github = Mock()
        mock_github.get_user.return_value = Mock(login="testuser")
        mock_github_class.return_value = mock_github
        
        provider = GitHubRepositoryProvider(mock_github_config)
        
        # Mock search_issues method
        provider.search_issues = AsyncMock(return_value=[
            {
                "number": 123,
                "title": "Database connection timeout",
                "state": "open"
            }
        ])
        
        issues = await provider.get_error_related_issues(
            error_message="database connection timeout error",
            limit=5
        )
        
        assert len(issues) == 1
        assert issues[0]["number"] == 123
        provider.search_issues.assert_called_once()


class TestGitHubCodeAnalyzer:
    """Test cases for GitHubCodeAnalyzer."""

    @pytest.fixture
    def code_analyzer(self, mock_github_config):
        """Create GitHubCodeAnalyzer instance."""
        with patch('oncall_agent.integrations.github.repository.Github'):
            repo_provider = GitHubRepositoryProvider(mock_github_config)
            return GitHubCodeAnalyzer(repo_provider)

    @pytest.mark.asyncio
    async def test_analyze_stack_trace(self, code_analyzer):
        """Test stack trace analysis."""
        stack_trace = """
        Traceback (most recent call last):
          File "app/main.py", line 45, in process_request
            result = database.connect()
          File "app/database.py", line 23, in connect
            connection = psycopg2.connect(self.connection_string)
        psycopg2.OperationalError: could not connect to server
        """
        
        # Mock the repo provider methods
        code_analyzer.repo_provider.get_file_content = AsyncMock(return_value={
            "content": "def connect():\n    return psycopg2.connect(url)",
            "url": "https://github.com/org/repo/blob/main/app/database.py"
        })
        code_analyzer.repo_provider.get_recent_commits = AsyncMock(return_value=[])
        
        analysis = await code_analyzer.analyze_stack_trace(stack_trace)
        
        assert "files_involved" in analysis
        assert "functions_involved" in analysis
        assert "suggestions" in analysis
        
        # Check that files were extracted
        files = analysis["files_involved"]
        assert any("app/main.py" in str(f) for f in files)
        assert any("app/database.py" in str(f) for f in files)
        
        # Check that functions were extracted
        functions = analysis["functions_involved"]
        assert "connect" in functions

    def test_extract_files_from_stack_trace(self, code_analyzer):
        """Test file extraction from stack trace."""
        stack_trace = 'File "app/main.py", line 45, in process_request'
        
        files = code_analyzer._extract_files_from_stack_trace(stack_trace)
        
        assert len(files) == 1
        assert files[0]["file_path"] == "app/main.py"
        assert files[0]["line_number"] == 45

    def test_extract_functions_from_stack_trace(self, code_analyzer):
        """Test function extraction from stack trace."""
        stack_trace = "in process_request\nat connect\nin main"
        
        functions = code_analyzer._extract_functions_from_stack_trace(stack_trace)
        
        assert "process_request" in functions
        assert "connect" in functions
        # "main" should be filtered out as a common word

    def test_extract_code_around_line(self, code_analyzer):
        """Test code extraction around specific line."""
        content = "\n".join([
            "def function1():",
            "    pass",
            "",
            "def function2():",
            "    # This is line 5",
            "    return True",
            "",
            "def function3():",
            "    pass"
        ])
        
        result = code_analyzer._extract_code_around_line(content, 5, context_lines=2)
        
        assert result["target_line"] == 5
        assert result["start_line"] == 3
        assert result["end_line"] == 7
        assert "# This is line 5" in result["target_content"]

    def test_find_function_definition(self, code_analyzer):
        """Test function definition finding."""
        content = """
def function1():
    pass

def target_function():
    return True

def function3():
    pass
"""
        
        definition = code_analyzer._find_function_definition(content, "target_function")
        
        assert definition["start_line"] > 0
        assert "def target_function():" in definition["definition"]
        assert "def target_function():" in definition["signature"]

    def test_analyze_function_complexity_simple(self, code_analyzer):
        """Test function complexity analysis for simple function."""
        function_def = {
            "definition": "def simple_function():\n    return True"
        }
        
        analysis = code_analyzer._analyze_function_complexity(function_def)
        
        assert analysis["cyclomatic_complexity"] == 1
        assert analysis["complexity_level"] == "low"
        assert len(analysis["issues"]) == 0

    def test_analyze_function_complexity_complex(self, code_analyzer):
        """Test function complexity analysis for complex function."""
        function_def = {
            "definition": """
def complex_function():
    if condition1:
        for item in items:
            if condition2:
                while condition3:
                    try:
                        process()
                    except Exception:
                        handle_error()
    return result
"""
        }
        
        analysis = code_analyzer._analyze_function_complexity(function_def)
        
        assert analysis["cyclomatic_complexity"] > 5
        assert analysis["complexity_level"] in ["medium", "high"]
