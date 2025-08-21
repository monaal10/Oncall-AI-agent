"""GitHub repository integration."""

import asyncio
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from github import Github, GithubException
from github.Repository import Repository

from ..base.code_provider import CodeProvider


class GitHubRepositoryProvider(CodeProvider):
    """GitHub repository provider implementation.
    
    Provides integration with GitHub repositories for code search,
    file access, and repository information.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize GitHub repository provider.
        
        Args:
            config: Configuration dictionary containing:
                - token: GitHub personal access token (required)
                - repositories: List of repositories to access (required)
                - base_url: GitHub API base URL (optional, for GitHub Enterprise)
        """
        super().__init__(config)
        self._setup_client()

    def _validate_config(self) -> None:
        """Validate GitHub configuration.
        
        Raises:
            ValueError: If required configuration is missing
        """
        if "token" not in self.config:
            raise ValueError("GitHub token is required")
        if "repositories" not in self.config or not self.config["repositories"]:
            raise ValueError("At least one repository must be specified")

    def _setup_client(self) -> None:
        """Set up GitHub client.
        
        Raises:
            ConnectionError: If unable to create GitHub client
        """
        try:
            base_url = self.config.get("base_url", "https://api.github.com")
            self.github = Github(
                login_or_token=self.config["token"],
                base_url=base_url
            )
            # Test the connection
            self.github.get_user().login
        except Exception as e:
            raise ConnectionError(f"Failed to create GitHub client: {e}")

    def _get_repository(self, repository: str) -> Repository:
        """Get GitHub repository object.
        
        Args:
            repository: Repository name (e.g., "owner/repo")
            
        Returns:
            GitHub Repository object
            
        Raises:
            ValueError: If repository not found or invalid
        """
        try:
            return self.github.get_repo(repository)
        except GithubException as e:
            raise ValueError(f"Repository {repository} not found or inaccessible: {e}")

    async def search_code(
        self,
        query: str,
        repositories: Optional[List[str]] = None,
        file_extension: Optional[str] = None,
        limit: Optional[int] = 50
    ) -> List[Dict[str, Any]]:
        """Search for code across GitHub repositories.
        
        Args:
            query: Search query (code content, function names, etc.)
            repositories: Specific repositories to search (None for all configured)
            file_extension: Filter by file extension (e.g., "py", "js")
            limit: Maximum number of results to return
            
        Returns:
            List of code search results with file info and matches
            
        Raises:
            ConnectionError: If unable to connect to GitHub
            ValueError: If query parameters are invalid
        """
        try:
            target_repos = repositories or self.config["repositories"]
            
            # Build search query
            search_query = query
            
            # Add repository filter
            repo_filter = " OR ".join([f"repo:{repo}" for repo in target_repos])
            search_query = f"({search_query}) AND ({repo_filter})"
            
            # Add file extension filter
            if file_extension:
                search_query += f" extension:{file_extension}"

            # Execute search
            search_results = await asyncio.to_thread(
                lambda: list(self.github.search_code(
                    query=search_query,
                    order="desc"
                )[:limit or 50])
            )

            results = []
            for result in search_results:
                try:
                    # Get file content to extract matches
                    content = base64.b64decode(result.content).decode('utf-8')
                    lines = content.split('\n')
                    
                    # Find matching lines
                    matches = []
                    for i, line in enumerate(lines, 1):
                        if query.lower() in line.lower():
                            matches.append({
                                "line_number": i,
                                "content": line.strip(),
                                "context": {
                                    "before": lines[max(0, i-3):i-1] if i > 1 else [],
                                    "after": lines[i:min(len(lines), i+3)]
                                }
                            })
                    
                    result_dict = {
                        "repository": result.repository.full_name,
                        "file_path": result.path,
                        "file_name": result.name,
                        "matches": matches[:5],  # Limit matches per file
                        "url": result.html_url,
                        "last_modified": result.repository.updated_at,
                        "sha": result.sha
                    }
                    results.append(result_dict)
                    
                except Exception as e:
                    # Skip files that can't be processed
                    continue

            return results

        except GithubException as e:
            raise ConnectionError(f"GitHub search error: {e}")
        except Exception as e:
            raise ConnectionError(f"Failed to search code: {e}")

    async def get_file_content(
        self,
        repository: str,
        file_path: str,
        branch: str = "main"
    ) -> Dict[str, Any]:
        """Get content of a specific file from GitHub.
        
        Args:
            repository: Repository name (e.g., "owner/repo")
            file_path: Path to the file in the repository
            branch: Branch name (defaults to "main")
            
        Returns:
            Dictionary containing file content and metadata
            
        Raises:
            ConnectionError: If unable to connect to GitHub
            ValueError: If file not found
        """
        try:
            repo = self._get_repository(repository)
            
            file_content = await asyncio.to_thread(
                repo.get_contents,
                file_path,
                ref=branch
            )
            
            # Decode content if it's base64 encoded
            if file_content.encoding == "base64":
                content = base64.b64decode(file_content.content).decode('utf-8')
            else:
                content = file_content.content

            return {
                "content": content,
                "encoding": file_content.encoding,
                "size": file_content.size,
                "sha": file_content.sha,
                "url": file_content.html_url,
                "last_modified": file_content.last_modified,
                "download_url": file_content.download_url
            }

        except GithubException as e:
            if e.status == 404:
                raise ValueError(f"File {file_path} not found in {repository}")
            raise ConnectionError(f"GitHub API error: {e}")
        except Exception as e:
            raise ConnectionError(f"Failed to get file content: {e}")

    async def get_recent_commits(
        self,
        repository: str,
        since: Optional[datetime] = None,
        author: Optional[str] = None,
        limit: Optional[int] = 50
    ) -> List[Dict[str, Any]]:
        """Get recent commits from a GitHub repository.
        
        Args:
            repository: Repository name (e.g., "owner/repo")
            since: Only commits after this date
            author: Filter by commit author
            limit: Maximum number of commits to return
            
        Returns:
            List of commits with metadata and changed files
            
        Raises:
            ConnectionError: If unable to connect to GitHub
            ValueError: If repository not found
        """
        try:
            repo = self._get_repository(repository)
            
            kwargs = {}
            if since:
                kwargs["since"] = since
            if author:
                kwargs["author"] = author

            commits = await asyncio.to_thread(
                lambda: list(repo.get_commits(**kwargs)[:limit or 50])
            )

            results = []
            for commit in commits:
                try:
                    # Get files changed in this commit
                    files_changed = []
                    for file in commit.files:
                        files_changed.append({
                            "filename": file.filename,
                            "status": file.status,
                            "additions": file.additions,
                            "deletions": file.deletions,
                            "changes": file.changes
                        })

                    commit_dict = {
                        "sha": commit.sha,
                        "message": commit.commit.message,
                        "author": {
                            "name": commit.commit.author.name,
                            "email": commit.commit.author.email,
                            "login": commit.author.login if commit.author else None
                        },
                        "date": commit.commit.author.date,
                        "url": commit.html_url,
                        "files_changed": files_changed
                    }
                    results.append(commit_dict)
                    
                except Exception:
                    # Skip commits that can't be processed
                    continue

            return results

        except GithubException as e:
            raise ConnectionError(f"GitHub API error: {e}")
        except Exception as e:
            raise ConnectionError(f"Failed to get commits: {e}")

    async def get_repository_info(
        self,
        repository: str
    ) -> Dict[str, Any]:
        """Get information about a GitHub repository.
        
        Args:
            repository: Repository name (e.g., "owner/repo")
            
        Returns:
            Dictionary containing repository metadata
            
        Raises:
            ConnectionError: If unable to connect to GitHub
            ValueError: If repository not found
        """
        try:
            repo = self._get_repository(repository)

            # Get languages
            languages = await asyncio.to_thread(repo.get_languages)

            return {
                "name": repo.name,
                "full_name": repo.full_name,
                "description": repo.description,
                "language": repo.language,
                "languages": languages,
                "default_branch": repo.default_branch,
                "created_at": repo.created_at,
                "updated_at": repo.updated_at,
                "pushed_at": repo.pushed_at,
                "url": repo.html_url,
                "clone_url": repo.clone_url,
                "stars": repo.stargazers_count,
                "forks": repo.forks_count,
                "open_issues": repo.open_issues_count,
                "size": repo.size,
                "private": repo.private
            }

        except GithubException as e:
            raise ConnectionError(f"GitHub API error: {e}")
        except Exception as e:
            raise ConnectionError(f"Failed to get repository info: {e}")

    async def get_pull_requests(
        self,
        repository: str,
        state: str = "open",
        limit: Optional[int] = 50
    ) -> List[Dict[str, Any]]:
        """Get pull requests from a GitHub repository.
        
        Args:
            repository: Repository name (e.g., "owner/repo")
            state: PR state ("open", "closed", "all")
            limit: Maximum number of PRs to return
            
        Returns:
            List of pull requests with metadata
            
        Raises:
            ConnectionError: If unable to connect to GitHub
            ValueError: If repository not found
        """
        try:
            repo = self._get_repository(repository)
            
            pulls = await asyncio.to_thread(
                lambda: list(repo.get_pulls(state=state)[:limit or 50])
            )

            results = []
            for pr in pulls:
                pr_dict = {
                    "number": pr.number,
                    "title": pr.title,
                    "state": pr.state,
                    "author": {
                        "login": pr.user.login,
                        "name": pr.user.name if pr.user.name else pr.user.login
                    },
                    "created_at": pr.created_at,
                    "updated_at": pr.updated_at,
                    "merged_at": pr.merged_at,
                    "url": pr.html_url,
                    "mergeable": pr.mergeable,
                    "mergeable_state": pr.mergeable_state,
                    "head_branch": pr.head.ref,
                    "base_branch": pr.base.ref,
                    "additions": pr.additions,
                    "deletions": pr.deletions,
                    "changed_files": pr.changed_files
                }
                results.append(pr_dict)

            return results

        except GithubException as e:
            raise ConnectionError(f"GitHub API error: {e}")
        except Exception as e:
            raise ConnectionError(f"Failed to get pull requests: {e}")

    async def get_issues(
        self,
        repository: str,
        state: str = "open",
        labels: Optional[List[str]] = None,
        limit: Optional[int] = 50
    ) -> List[Dict[str, Any]]:
        """Get issues from a GitHub repository.
        
        Args:
            repository: Repository name (e.g., "owner/repo")
            state: Issue state ("open", "closed", "all")
            labels: Filter by labels
            limit: Maximum number of issues to return
            
        Returns:
            List of issues with metadata
            
        Raises:
            ConnectionError: If unable to connect to GitHub
            ValueError: If repository not found
        """
        try:
            repo = self._get_repository(repository)
            
            kwargs = {"state": state}
            if labels:
                kwargs["labels"] = labels

            issues = await asyncio.to_thread(
                lambda: list(repo.get_issues(**kwargs)[:limit or 50])
            )

            results = []
            for issue in issues:
                # Skip pull requests (they appear as issues in GitHub API)
                if issue.pull_request:
                    continue

                issue_labels = [label.name for label in issue.labels]
                
                issue_dict = {
                    "number": issue.number,
                    "title": issue.title,
                    "body": issue.body,
                    "state": issue.state,
                    "author": {
                        "login": issue.user.login,
                        "name": issue.user.name if issue.user.name else issue.user.login
                    },
                    "created_at": issue.created_at,
                    "updated_at": issue.updated_at,
                    "closed_at": issue.closed_at,
                    "labels": issue_labels,
                    "url": issue.html_url,
                    "comments": issue.comments,
                    "assignees": [assignee.login for assignee in issue.assignees]
                }
                results.append(issue_dict)

            return results

        except GithubException as e:
            raise ConnectionError(f"GitHub API error: {e}")
        except Exception as e:
            raise ConnectionError(f"Failed to get issues: {e}")

    async def search_issues(
        self,
        query: str,
        repositories: Optional[List[str]] = None,
        state: str = "open",
        limit: Optional[int] = 50
    ) -> List[Dict[str, Any]]:
        """Search for issues across repositories.
        
        Args:
            query: Search query for issue titles and content
            repositories: Specific repositories to search
            state: Issue state filter
            limit: Maximum number of results
            
        Returns:
            List of matching issues
            
        Raises:
            ConnectionError: If unable to connect to GitHub
        """
        try:
            target_repos = repositories or self.config["repositories"]
            
            # Build search query
            repo_filter = " OR ".join([f"repo:{repo}" for repo in target_repos])
            search_query = f"({query}) AND ({repo_filter}) AND is:issue AND state:{state}"

            issues = await asyncio.to_thread(
                lambda: list(self.github.search_issues(
                    query=search_query,
                    order="desc"
                )[:limit or 50])
            )

            results = []
            for issue in issues:
                issue_labels = [label.name for label in issue.labels]
                
                issue_dict = {
                    "number": issue.number,
                    "title": issue.title,
                    "body": issue.body[:500] if issue.body else "",  # Truncate body
                    "state": issue.state,
                    "repository": issue.repository.full_name,
                    "author": {
                        "login": issue.user.login,
                        "name": issue.user.name if issue.user.name else issue.user.login
                    },
                    "created_at": issue.created_at,
                    "updated_at": issue.updated_at,
                    "labels": issue_labels,
                    "url": issue.html_url,
                    "comments": issue.comments
                }
                results.append(issue_dict)

            return results

        except GithubException as e:
            raise ConnectionError(f"GitHub search error: {e}")
        except Exception as e:
            raise ConnectionError(f"Failed to search issues: {e}")

    async def get_error_related_issues(
        self,
        error_message: str,
        repositories: Optional[List[str]] = None,
        limit: Optional[int] = 10
    ) -> List[Dict[str, Any]]:
        """Find issues related to a specific error message.
        
        Args:
            error_message: Error message to search for
            repositories: Repositories to search
            limit: Maximum number of results
            
        Returns:
            List of related issues
            
        Raises:
            ConnectionError: If unable to connect to GitHub
        """
        # Extract key terms from error message
        import re
        
        # Remove common noise words and extract meaningful terms
        noise_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'an', 'a'}
        words = re.findall(r'\b\w+\b', error_message.lower())
        key_terms = [word for word in words if len(word) > 3 and word not in noise_words]
        
        # Build search query with key terms
        search_terms = " OR ".join(key_terms[:5])  # Use top 5 terms
        
        return await self.search_issues(
            query=search_terms,
            repositories=repositories,
            state="all",
            limit=limit
        )
