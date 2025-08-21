"""Example usage of GitHub integration."""

import asyncio
from datetime import datetime, timedelta
from oncall_agent.integrations.github import GitHubRepositoryProvider, GitHubCodeAnalyzer


async def github_repository_example():
    """Example of using GitHub repository integration."""
    print("=== GitHub Repository Example ===")
    
    # Create GitHub configuration
    config = {
        "token": "your-github-token",  # Replace with your actual token
        "repositories": [
            "owner/repo1",  # Replace with your actual repositories
            "owner/repo2"
        ]
        # Uncomment for GitHub Enterprise:
        # "base_url": "https://github.mycompany.com/api/v3"
    }
    
    # Initialize GitHub repository provider
    repo_provider = GitHubRepositoryProvider(config)
    
    try:
        # Test connection
        is_healthy = await repo_provider.health_check()
        print(f"GitHub connection: {'✓ Healthy' if is_healthy else '✗ Failed'}")
        
        if not is_healthy:
            print("Please check your GitHub token and repository access")
            return
        
        # Get repository information
        print("\n--- Repository Information ---")
        for repo_name in config["repositories"][:2]:  # Show first 2 repos
            try:
                repo_info = await repo_provider.get_repository_info(repo_name)
                print(f"Repository: {repo_info['full_name']}")
                print(f"  Description: {repo_info['description'] or 'No description'}")
                print(f"  Language: {repo_info['language'] or 'Unknown'}")
                print(f"  Stars: {repo_info['stars']}, Forks: {repo_info['forks']}")
                print(f"  Last updated: {repo_info['updated_at']}")
                print(f"  Open issues: {repo_info['open_issues']}")
            except Exception as e:
                print(f"  Could not access {repo_name}: {e}")
        
        # Search for code
        print("\n--- Code Search Example ---")
        try:
            search_results = await repo_provider.search_code(
                query="function handleError",  # Search for error handling functions
                limit=5
            )
            
            print(f"Found {len(search_results)} code matches:")
            for result in search_results:
                print(f"  📄 {result['repository']}/{result['file_path']}")
                print(f"     {len(result['matches'])} matches in file")
                if result['matches']:
                    first_match = result['matches'][0]
                    print(f"     Line {first_match['line_number']}: {first_match['content'][:60]}...")
                    
        except Exception as e:
            print(f"  Code search failed: {e}")
        
        # Get recent commits
        print("\n--- Recent Commits ---")
        for repo_name in config["repositories"][:1]:  # Show commits from first repo
            try:
                commits = await repo_provider.get_recent_commits(
                    repository=repo_name,
                    limit=5
                )
                
                print(f"Recent commits in {repo_name}:")
                for commit in commits:
                    print(f"  🔄 {commit['sha'][:8]} - {commit['message'][:50]}...")
                    print(f"     By: {commit['author']['name']} on {commit['date']}")
                    print(f"     Files changed: {len(commit['files_changed'])}")
                    
            except Exception as e:
                print(f"  Could not get commits for {repo_name}: {e}")
        
        # Get pull requests
        print("\n--- Open Pull Requests ---")
        for repo_name in config["repositories"][:1]:
            try:
                prs = await repo_provider.get_pull_requests(
                    repository=repo_name,
                    state="open",
                    limit=5
                )
                
                if prs:
                    print(f"Open PRs in {repo_name}:")
                    for pr in prs:
                        print(f"  🔀 #{pr['number']}: {pr['title'][:50]}...")
                        print(f"     By: {pr['author']['login']} - {pr['state']}")
                        print(f"     +{pr['additions']} -{pr['deletions']} changes")
                else:
                    print(f"No open pull requests in {repo_name}")
                    
            except Exception as e:
                print(f"  Could not get PRs for {repo_name}: {e}")
        
        # Get issues
        print("\n--- Open Issues ---")
        for repo_name in config["repositories"][:1]:
            try:
                issues = await repo_provider.get_issues(
                    repository=repo_name,
                    state="open",
                    limit=5
                )
                
                if issues:
                    print(f"Open issues in {repo_name}:")
                    for issue in issues:
                        print(f"  🐛 #{issue['number']}: {issue['title'][:50]}...")
                        print(f"     By: {issue['author']['login']} - {len(issue['labels'])} labels")
                        print(f"     Comments: {issue['comments']}")
                else:
                    print(f"No open issues in {repo_name}")
                    
            except Exception as e:
                print(f"  Could not get issues for {repo_name}: {e}")
        
        # Search for error-related issues
        print("\n--- Error-Related Issues ---")
        try:
            error_issues = await repo_provider.get_error_related_issues(
                error_message="database connection timeout",
                limit=3
            )
            
            if error_issues:
                print("Issues related to 'database connection timeout':")
                for issue in error_issues:
                    print(f"  🔍 {issue['repository']}#{issue['number']}: {issue['title'][:50]}...")
                    print(f"     State: {issue['state']} - Labels: {', '.join(issue['labels'][:3])}")
            else:
                print("No related issues found")
                
        except Exception as e:
            print(f"  Could not search issues: {e}")
            
    except Exception as e:
        print(f"Error: {e}")


async def github_code_analyzer_example():
    """Example of using GitHub code analyzer."""
    print("\n=== GitHub Code Analyzer Example ===")
    
    # Create GitHub configuration
    config = {
        "token": "your-github-token",
        "repositories": ["owner/repo1", "owner/repo2"]
    }
    
    repo_provider = GitHubRepositoryProvider(config)
    code_analyzer = GitHubCodeAnalyzer(repo_provider)
    
    try:
        # Example stack trace analysis
        print("\n--- Stack Trace Analysis ---")
        sample_stack_trace = """
        Traceback (most recent call last):
          File "app/main.py", line 45, in process_request
            result = database.connect()
          File "app/database.py", line 23, in connect
            connection = psycopg2.connect(self.connection_string)
        psycopg2.OperationalError: could not connect to server
        """
        
        try:
            stack_analysis = await code_analyzer.analyze_stack_trace(
                stack_trace=sample_stack_trace
            )
            
            print("Stack trace analysis results:")
            print(f"  Files involved: {len(stack_analysis['files_involved'])}")
            for file_info in stack_analysis['files_involved']:
                print(f"    - {file_info['file_path']} (line {file_info.get('line_number', 'unknown')})")
            
            print(f"  Functions involved: {', '.join(stack_analysis['functions_involved'])}")
            print(f"  Related code snippets: {len(stack_analysis['related_code'])}")
            print(f"  Recent changes: {len(stack_analysis['recent_changes'])}")
            
            if stack_analysis['suggestions']:
                print("  Suggestions:")
                for suggestion in stack_analysis['suggestions'][:3]:
                    print(f"    • {suggestion}")
                    
        except Exception as e:
            print(f"  Stack trace analysis failed: {e}")
        
        # Example pattern search
        print("\n--- Similar Pattern Search ---")
        try:
            patterns = await code_analyzer.find_similar_patterns(
                error_pattern="connection timeout",
                file_extension="py"
            )
            
            if patterns:
                print(f"Found {len(patterns)} similar patterns:")
                for pattern in patterns[:3]:  # Show first 3
                    print(f"  📋 {pattern['repository']}/{pattern['file_path']}:{pattern['line_number']}")
                    print(f"     Content: {pattern['match_content'][:60]}...")
                    if pattern['context_analysis']['function_context'] != 'unknown':
                        print(f"     In function: {pattern['context_analysis']['function_context']}")
            else:
                print("No similar patterns found")
                
        except Exception as e:
            print(f"  Pattern search failed: {e}")
        
        # Example function analysis
        print("\n--- Function Dependency Analysis ---")
        try:
            # This would analyze a specific function if it exists
            repo_name = config["repositories"][0]
            
            function_analysis = await code_analyzer.analyze_function_dependencies(
                function_name="connect",
                repository=repo_name,
                file_path="app/database.py"  # Example file path
            )
            
            print(f"Function analysis for 'connect' in {repo_name}:")
            print(f"  Definition found: {bool(function_analysis['definition']['definition'])}")
            print(f"  Callers found: {len(function_analysis['callers'])}")
            
            complexity = function_analysis['complexity_analysis']
            print(f"  Complexity: {complexity.get('complexity_level', 'unknown')}")
            print(f"  Line count: {complexity.get('line_count', 0)}")
            
            if function_analysis['suggestions']:
                print("  Suggestions:")
                for suggestion in function_analysis['suggestions'][:2]:
                    print(f"    • {suggestion}")
                    
        except Exception as e:
            print(f"  Function analysis failed (this is expected if the function doesn't exist): {e}")
            
    except Exception as e:
        print(f"Error: {e}")


async def main():
    """Run all GitHub integration examples."""
    print("GitHub Integration Examples")
    print("=" * 50)
    
    await github_repository_example()
    await github_code_analyzer_example()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("\nNote: To use these examples:")
    print("1. Create a GitHub Personal Access Token:")
    print("   - Go to GitHub Settings > Developer settings > Personal access tokens")
    print("   - Generate new token with 'repo' scope")
    print("2. Set environment variable: export GITHUB_TOKEN='your_token_here'")
    print("3. Update the repository names in the config to your actual repositories")
    print("4. Install required dependency: pip install PyGithub")
    print("\nToken scopes needed:")
    print("  - repo: Full control of private repositories")
    print("  - public_repo: Access to public repositories")
    print("  - read:org: Read org membership (for org repos)")


if __name__ == "__main__":
    asyncio.run(main())
