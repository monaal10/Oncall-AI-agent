"""Abstract base class for code repository providers."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime


class CodeProvider(ABC):
    """Abstract base class for all code repository providers.
    
    This class defines the interface that all code repository providers must implement
    to integrate with the OnCall AI Agent system.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the code provider with configuration.
        
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
    async def search_code(
        self,
        query: str,
        repositories: Optional[List[str]] = None,
        file_extension: Optional[str] = None,
        limit: Optional[int] = 50
    ) -> List[Dict[str, Any]]:
        """Search for code across repositories.
        
        Args:
            query: Search query (code content, function names, etc.)
            repositories: Specific repositories to search (None for all configured)
            file_extension: Filter by file extension (e.g., "py", "js")
            limit: Maximum number of results to return
            
        Returns:
            List of code search results, each containing:
            - repository: Repository name
            - file_path: Path to the file
            - file_name: Name of the file
            - matches: List of matching code snippets with line numbers
            - url: URL to view the file
            - last_modified: When the file was last modified
            
        Raises:
            ConnectionError: If unable to connect to code repository service
            ValueError: If query parameters are invalid
        """
        pass

    @abstractmethod
    async def get_file_content(
        self,
        repository: str,
        file_path: str,
        branch: str = "main"
    ) -> Dict[str, Any]:
        """Get content of a specific file.
        
        Args:
            repository: Repository name (e.g., "owner/repo")
            file_path: Path to the file in the repository
            branch: Branch name (defaults to "main")
            
        Returns:
            Dictionary containing:
            - content: File content as string
            - encoding: File encoding
            - size: File size in bytes
            - sha: File SHA hash
            - url: URL to view the file
            - last_modified: When the file was last modified
            
        Raises:
            ConnectionError: If unable to connect to repository
            ValueError: If file not found or parameters invalid
        """
        pass

    @abstractmethod
    async def get_recent_commits(
        self,
        repository: str,
        since: Optional[datetime] = None,
        author: Optional[str] = None,
        limit: Optional[int] = 50
    ) -> List[Dict[str, Any]]:
        """Get recent commits from a repository.
        
        Args:
            repository: Repository name (e.g., "owner/repo")
            since: Only commits after this date
            author: Filter by commit author
            limit: Maximum number of commits to return
            
        Returns:
            List of commits, each containing:
            - sha: Commit SHA
            - message: Commit message
            - author: Author information
            - date: Commit date
            - url: URL to view the commit
            - files_changed: List of changed files
            
        Raises:
            ConnectionError: If unable to connect to repository
            ValueError: If repository not found
        """
        pass

    @abstractmethod
    async def get_repository_info(
        self,
        repository: str
    ) -> Dict[str, Any]:
        """Get information about a repository.
        
        Args:
            repository: Repository name (e.g., "owner/repo")
            
        Returns:
            Dictionary containing:
            - name: Repository name
            - full_name: Full repository name with owner
            - description: Repository description
            - language: Primary language
            - languages: All languages used
            - default_branch: Default branch name
            - created_at: Repository creation date
            - updated_at: Last update date
            - url: Repository URL
            
        Raises:
            ConnectionError: If unable to connect to repository
            ValueError: If repository not found
        """
        pass

    @abstractmethod
    async def get_pull_requests(
        self,
        repository: str,
        state: str = "open",
        limit: Optional[int] = 50
    ) -> List[Dict[str, Any]]:
        """Get pull requests from a repository.
        
        Args:
            repository: Repository name (e.g., "owner/repo")
            state: PR state ("open", "closed", "all")
            limit: Maximum number of PRs to return
            
        Returns:
            List of pull requests, each containing:
            - number: PR number
            - title: PR title
            - state: PR state
            - author: Author information
            - created_at: Creation date
            - updated_at: Last update date
            - url: URL to view the PR
            - mergeable: Whether PR can be merged
            
        Raises:
            ConnectionError: If unable to connect to repository
            ValueError: If repository not found
        """
        pass

    @abstractmethod
    async def get_issues(
        self,
        repository: str,
        state: str = "open",
        labels: Optional[List[str]] = None,
        limit: Optional[int] = 50
    ) -> List[Dict[str, Any]]:
        """Get issues from a repository.
        
        Args:
            repository: Repository name (e.g., "owner/repo")
            state: Issue state ("open", "closed", "all")
            labels: Filter by labels
            limit: Maximum number of issues to return
            
        Returns:
            List of issues, each containing:
            - number: Issue number
            - title: Issue title
            - state: Issue state
            - author: Author information
            - created_at: Creation date
            - updated_at: Last update date
            - labels: List of labels
            - url: URL to view the issue
            
        Raises:
            ConnectionError: If unable to connect to repository
            ValueError: If repository not found
        """
        pass

    async def health_check(self) -> bool:
        """Check if the code provider is accessible.
        
        Returns:
            True if provider is healthy, False otherwise
        """
        try:
            # Try to get repository info for the first configured repository
            repositories = self.config.get("repositories", [])
            if repositories:
                await self.get_repository_info(repositories[0])
            return True
        except Exception:
            return False

    async def analyze_error_context(
        self,
        error_message: str,
        stack_trace: Optional[str] = None,
        repositories: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Analyze error context by searching related code.
        
        Args:
            error_message: Error message to analyze
            stack_trace: Stack trace (if available)
            repositories: Repositories to search
            
        Returns:
            Dictionary containing:
            - related_files: Files that might be related to the error
            - function_definitions: Relevant function definitions
            - recent_changes: Recent commits that might be related
            - suggestions: Potential areas to investigate
            
        Raises:
            ConnectionError: If unable to connect to repository service
        """
        # Extract key terms from error message
        import re
        
        # Extract function names, class names, and file names from error
        function_pattern = r'(\w+)\('
        class_pattern = r'class\s+(\w+)'
        file_pattern = r'(\w+\.\w+)'
        
        functions = re.findall(function_pattern, error_message)
        classes = re.findall(class_pattern, error_message)
        files = re.findall(file_pattern, error_message)
        
        # Search for related code
        related_files = []
        function_definitions = []
        
        # Search for functions mentioned in error
        for func in functions:
            try:
                results = await self.search_code(
                    query=f"def {func}",
                    repositories=repositories,
                    limit=5
                )
                function_definitions.extend(results)
            except Exception:
                continue
        
        # Search for classes mentioned in error
        for cls in classes:
            try:
                results = await self.search_code(
                    query=f"class {cls}",
                    repositories=repositories,
                    limit=5
                )
                related_files.extend(results)
            except Exception:
                continue
        
        # Get recent changes that might be related
        recent_changes = []
        if repositories:
            for repo in repositories:
                try:
                    commits = await self.get_recent_commits(
                        repository=repo,
                        limit=10
                    )
                    # Filter commits that mention error-related terms
                    for commit in commits:
                        if any(term.lower() in commit['message'].lower() 
                               for term in functions + classes + ['fix', 'bug', 'error']):
                            recent_changes.append(commit)
                except Exception:
                    continue
        
        return {
            "related_files": related_files,
            "function_definitions": function_definitions,
            "recent_changes": recent_changes[:5],  # Limit to 5 most relevant
            "suggestions": [
                "Check recent commits for related changes",
                "Review function definitions for potential issues",
                "Look for similar error patterns in issues"
            ]
        }
