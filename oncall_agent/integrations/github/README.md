# GitHub Integration

This module provides integration with GitHub repositories for the OnCall AI Agent.

## Features

### Repository Access
- Search code across multiple repositories
- Access file contents with full metadata
- Get commit history and change information
- List pull requests and issues
- Repository information and statistics

### Code Analysis
- Stack trace analysis to find related code
- Function dependency analysis
- Similar error pattern detection
- Code complexity analysis
- Error context extraction

### Issue Management
- Search issues by content and labels
- Find error-related issues automatically
- Get issue history and comments
- Track issue lifecycle

## Configuration

### Authentication

GitHub integration requires a Personal Access Token:

1. **Create Personal Access Token**
   ```
   Go to: GitHub Settings > Developer settings > Personal access tokens > Tokens (classic)
   Generate new token with required scopes
   ```

2. **Required Scopes**
   - `repo`: Full control of private repositories
   - `public_repo`: Access to public repositories (if only using public repos)
   - `read:org`: Read org and team membership (for organization repositories)
   - `read:user`: Read user profile data

3. **Set Environment Variable**
   ```bash
   export GITHUB_TOKEN="your_token_here"
   ```

### Configuration Example

```yaml
integrations:
  code_provider:
    type: "github"
    config:
      token: "${GITHUB_TOKEN}"
      repositories:
        - "myorg/backend-service"
        - "myorg/frontend-app"
        - "myorg/shared-libraries"
      # Optional: for GitHub Enterprise
      # base_url: "https://github.mycompany.com/api/v3"
```

## Usage Examples

### Repository Provider

```python
from oncall_agent.integrations.github import GitHubRepositoryProvider

config = {
    "token": "your-github-token",
    "repositories": ["owner/repo1", "owner/repo2"]
}

repo_provider = GitHubRepositoryProvider(config)

# Search code
results = await repo_provider.search_code(
    query="def handle_error",
    file_extension="py",
    limit=10
)

# Get file content
file_content = await repo_provider.get_file_content(
    repository="owner/repo",
    file_path="src/main.py",
    branch="main"
)

# Get recent commits
commits = await repo_provider.get_recent_commits(
    repository="owner/repo",
    since=datetime.now() - timedelta(days=7),
    limit=20
)

# Search for error-related issues
error_issues = await repo_provider.get_error_related_issues(
    error_message="database connection timeout",
    limit=5
)
```

### Code Analyzer

```python
from oncall_agent.integrations.github import GitHubCodeAnalyzer

analyzer = GitHubCodeAnalyzer(repo_provider)

# Analyze stack trace
stack_trace = """
Traceback (most recent call last):
  File "app/main.py", line 45, in process_request
    result = database.connect()
psycopg2.OperationalError: could not connect to server
"""

analysis = await analyzer.analyze_stack_trace(stack_trace)
print(f"Files involved: {analysis['files_involved']}")
print(f"Recent changes: {len(analysis['recent_changes'])}")

# Find similar error patterns
patterns = await analyzer.find_similar_patterns(
    error_pattern="connection timeout",
    file_extension="py"
)

# Analyze function dependencies
function_analysis = await analyzer.analyze_function_dependencies(
    function_name="connect",
    repository="owner/repo",
    file_path="app/database.py"
)
```

## API Features

### Code Search
- **Full-text search** across repository contents
- **File extension filtering** (e.g., `.py`, `.js`, `.java`)
- **Repository-specific search** or across all configured repos
- **Context extraction** with line numbers and surrounding code
- **Match highlighting** and relevance scoring

### File Operations
- **Get file contents** with encoding detection
- **Branch-specific access** (main, develop, feature branches)
- **File metadata** (size, SHA, last modified)
- **Binary file detection** and handling

### Commit Analysis
- **Commit history** with author and date information
- **File change tracking** (additions, deletions, modifications)
- **Commit message analysis** for error-related changes
- **Author filtering** and date range queries

### Issue Integration
- **Issue search** by title, content, and labels
- **State filtering** (open, closed, all)
- **Error correlation** - find issues related to specific errors
- **Comment analysis** and resolution tracking

## Code Analysis Features

### Stack Trace Analysis
- **File extraction** from stack traces (Python, JavaScript, Java, etc.)
- **Function identification** from error traces
- **Line number mapping** to source code
- **Context extraction** around error locations
- **Recent change correlation** with error locations

### Pattern Detection
- **Similar error patterns** across the codebase
- **Error handling analysis** (try-catch blocks, logging)
- **Function complexity metrics** (cyclomatic complexity, line count)
- **Code quality indicators** (TODOs, FIXMEs, complexity warnings)

### Dependency Analysis
- **Function call mapping** within repositories
- **Cross-file dependencies** and imports
- **Caller identification** for specific functions
- **Impact analysis** for potential changes

## Error Handling

The integration handles common GitHub-specific scenarios:

- **Rate limiting**: Automatic retry with exponential backoff
- **Authentication errors**: Clear error messages for token issues
- **Repository access**: Graceful handling of private/inaccessible repos
- **API limits**: Search API has lower limits (30 requests/minute)
- **Large files**: Automatic handling of binary and large text files

## Rate Limits

GitHub API rate limits:
- **Authenticated requests**: 5,000 per hour
- **Search API**: 30 requests per minute
- **Secondary rate limits**: Dynamic based on resource usage

The integration automatically handles rate limiting with:
- Exponential backoff retry logic
- Request queuing for burst scenarios
- Rate limit header monitoring

## Limitations

1. **Search scope**: GitHub's code search has limitations on file size and repository size
2. **Private repositories**: Require appropriate token scopes
3. **Organization repositories**: May require organization membership
4. **File size limits**: Very large files may not be fully searchable
5. **Binary files**: Limited analysis capabilities for binary files

## Dependencies

```bash
pip install PyGithub
```

## GitHub Enterprise

For GitHub Enterprise installations:

```yaml
config:
  token: "${GITHUB_TOKEN}"
  base_url: "https://github.mycompany.com/api/v3"
  repositories:
    - "myorg/repo"
```

## Security Considerations

- **Token security**: Store tokens in environment variables, never in code
- **Scope limitation**: Use minimal required scopes for tokens
- **Token rotation**: Regularly rotate personal access tokens
- **Repository access**: Audit which repositories the token can access
- **Logging**: Avoid logging sensitive repository content

## References

- [GitHub API Documentation](https://docs.github.com/en/rest)
- [Personal Access Tokens](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token)
- [PyGithub Documentation](https://pygithub.readthedocs.io/)
- [GitHub Search Syntax](https://docs.github.com/en/search-github/searching-on-github/searching-code)
