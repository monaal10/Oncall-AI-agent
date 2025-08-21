"""GitHub code analysis utilities."""

import re
from typing import Dict, List, Optional, Any
from .repository import GitHubRepositoryProvider


class GitHubCodeAnalyzer:
    """Analyzes code from GitHub repositories to help with incident resolution.
    
    Provides utilities to analyze code structure, find related functions,
    and identify potential issues based on error messages and stack traces.
    """

    def __init__(self, repository_provider: GitHubRepositoryProvider):
        """Initialize the code analyzer.
        
        Args:
            repository_provider: GitHub repository provider instance
        """
        self.repo_provider = repository_provider

    async def analyze_stack_trace(
        self,
        stack_trace: str,
        repositories: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Analyze a stack trace to find related code and potential issues.
        
        Args:
            stack_trace: Stack trace string
            repositories: Repositories to search in
            
        Returns:
            Dictionary containing:
            - files_involved: Files mentioned in the stack trace
            - functions_involved: Functions mentioned in the stack trace
            - related_code: Code snippets from the involved files
            - recent_changes: Recent commits affecting these files
            - suggestions: Analysis suggestions
            
        Raises:
            ConnectionError: If unable to connect to GitHub
        """
        # Parse stack trace to extract file names and function names
        files_involved = self._extract_files_from_stack_trace(stack_trace)
        functions_involved = self._extract_functions_from_stack_trace(stack_trace)
        
        # Get code content for involved files
        related_code = []
        for file_info in files_involved:
            for repo in (repositories or self.repo_provider.config["repositories"]):
                try:
                    file_content = await self.repo_provider.get_file_content(
                        repository=repo,
                        file_path=file_info["file_path"]
                    )
                    
                    # Extract relevant code around the line number
                    if file_info.get("line_number"):
                        code_snippet = self._extract_code_around_line(
                            file_content["content"],
                            file_info["line_number"],
                            context_lines=10
                        )
                        
                        related_code.append({
                            "repository": repo,
                            "file_path": file_info["file_path"],
                            "line_number": file_info["line_number"],
                            "code_snippet": code_snippet,
                            "url": file_content["url"]
                        })
                    break  # Found the file, no need to check other repos
                except Exception:
                    continue  # File not found in this repo, try next
        
        # Find recent changes to involved files
        recent_changes = []
        for repo in (repositories or self.repo_provider.config["repositories"]):
            try:
                commits = await self.repo_provider.get_recent_commits(
                    repository=repo,
                    limit=20
                )
                
                for commit in commits:
                    # Check if any involved files were changed
                    involved_files = [f["file_path"] for f in files_involved]
                    changed_files = [f["filename"] for f in commit["files_changed"]]
                    
                    if any(file in changed_files for file in involved_files):
                        recent_changes.append(commit)
                        
            except Exception:
                continue
        
        # Generate suggestions based on analysis
        suggestions = self._generate_stack_trace_suggestions(
            files_involved,
            functions_involved,
            recent_changes
        )
        
        return {
            "files_involved": files_involved,
            "functions_involved": functions_involved,
            "related_code": related_code,
            "recent_changes": recent_changes[:5],  # Limit to 5 most recent
            "suggestions": suggestions
        }

    async def find_similar_patterns(
        self,
        error_pattern: str,
        repositories: Optional[List[str]] = None,
        file_extension: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Find similar error patterns in the codebase.
        
        Args:
            error_pattern: Error pattern or message to search for
            repositories: Repositories to search in
            file_extension: File extension filter
            
        Returns:
            List of similar patterns found in code
            
        Raises:
            ConnectionError: If unable to connect to GitHub
        """
        # Search for the error pattern in code
        search_results = await self.repo_provider.search_code(
            query=error_pattern,
            repositories=repositories,
            file_extension=file_extension,
            limit=20
        )
        
        # Analyze each result for context
        similar_patterns = []
        for result in search_results:
            try:
                # Get full file content for better context
                file_content = await self.repo_provider.get_file_content(
                    repository=result["repository"],
                    file_path=result["file_path"]
                )
                
                # Analyze the context around matches
                for match in result["matches"]:
                    context_analysis = self._analyze_error_context(
                        file_content["content"],
                        match["line_number"],
                        error_pattern
                    )
                    
                    similar_patterns.append({
                        "repository": result["repository"],
                        "file_path": result["file_path"],
                        "line_number": match["line_number"],
                        "match_content": match["content"],
                        "context_analysis": context_analysis,
                        "url": result["url"]
                    })
                    
            except Exception:
                continue  # Skip files that can't be analyzed
        
        return similar_patterns

    async def analyze_function_dependencies(
        self,
        function_name: str,
        repository: str,
        file_path: str
    ) -> Dict[str, Any]:
        """Analyze dependencies and callers of a specific function.
        
        Args:
            function_name: Name of the function to analyze
            repository: Repository containing the function
            file_path: File path containing the function
            
        Returns:
            Dictionary containing function analysis
            
        Raises:
            ConnectionError: If unable to connect to GitHub
        """
        try:
            # Get the file content
            file_content = await self.repo_provider.get_file_content(
                repository=repository,
                file_path=file_path
            )
            
            # Find the function definition
            function_def = self._find_function_definition(
                file_content["content"],
                function_name
            )
            
            # Find function calls within the same repository
            callers = await self.repo_provider.search_code(
                query=f"{function_name}(",
                repositories=[repository],
                limit=10
            )
            
            # Analyze function complexity and potential issues
            complexity_analysis = self._analyze_function_complexity(function_def)
            
            return {
                "function_name": function_name,
                "repository": repository,
                "file_path": file_path,
                "definition": function_def,
                "callers": callers,
                "complexity_analysis": complexity_analysis,
                "suggestions": self._generate_function_suggestions(
                    function_def,
                    complexity_analysis
                )
            }
            
        except Exception as e:
            raise ConnectionError(f"Failed to analyze function: {e}")

    def _extract_files_from_stack_trace(self, stack_trace: str) -> List[Dict[str, Any]]:
        """Extract file paths and line numbers from stack trace."""
        files = []
        
        # Common stack trace patterns
        patterns = [
            r'File "([^"]+)", line (\d+)',  # Python
            r'at ([^\s]+):(\d+):\d+',      # JavaScript/TypeScript
            r'([^\s]+):(\d+)',             # General format
            r'in ([^\s]+) on line (\d+)',  # PHP
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, stack_trace)
            for match in matches:
                file_path = match[0]
                line_number = int(match[1]) if len(match) > 1 else None
                
                files.append({
                    "file_path": file_path,
                    "line_number": line_number
                })
        
        return files

    def _extract_functions_from_stack_trace(self, stack_trace: str) -> List[str]:
        """Extract function names from stack trace."""
        functions = []
        
        # Common function patterns in stack traces
        patterns = [
            r'in (\w+)',           # Python
            r'at (\w+)',           # JavaScript
            r'(\w+)\(',            # General function calls
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, stack_trace)
            functions.extend(matches)
        
        # Remove duplicates and common words
        common_words = {'main', 'run', 'call', 'execute', 'init', 'new'}
        return list(set(func for func in functions if func not in common_words))

    def _extract_code_around_line(
        self,
        content: str,
        line_number: int,
        context_lines: int = 5
    ) -> Dict[str, Any]:
        """Extract code around a specific line number."""
        lines = content.split('\n')
        start_line = max(0, line_number - context_lines - 1)
        end_line = min(len(lines), line_number + context_lines)
        
        return {
            "start_line": start_line + 1,
            "end_line": end_line,
            "lines": lines[start_line:end_line],
            "target_line": line_number,
            "target_content": lines[line_number - 1] if line_number <= len(lines) else ""
        }

    def _analyze_error_context(
        self,
        content: str,
        line_number: int,
        error_pattern: str
    ) -> Dict[str, Any]:
        """Analyze the context around an error pattern match."""
        lines = content.split('\n')
        target_line = lines[line_number - 1] if line_number <= len(lines) else ""
        
        # Look for error handling patterns
        has_try_catch = any(
            'try:' in line or 'except' in line or 'catch' in line
            for line in lines[max(0, line_number - 10):line_number + 10]
        )
        
        # Look for logging statements
        has_logging = any(
            'log' in line.lower() or 'print' in line.lower()
            for line in lines[max(0, line_number - 5):line_number + 5]
        )
        
        # Identify the function or class this error is in
        function_context = self._find_containing_function(lines, line_number)
        
        return {
            "target_line": target_line,
            "has_error_handling": has_try_catch,
            "has_logging": has_logging,
            "function_context": function_context,
            "is_in_loop": self._is_in_loop_context(lines, line_number),
            "is_in_conditional": self._is_in_conditional_context(lines, line_number)
        }

    def _find_function_definition(self, content: str, function_name: str) -> Dict[str, Any]:
        """Find and extract function definition."""
        lines = content.split('\n')
        
        # Look for function definition patterns
        patterns = [
            rf'def {function_name}\s*\(',      # Python
            rf'function {function_name}\s*\(', # JavaScript
            rf'{function_name}\s*=\s*function', # JavaScript arrow function
            rf'def {function_name}',           # General
        ]
        
        for i, line in enumerate(lines):
            for pattern in patterns:
                if re.search(pattern, line):
                    # Extract function body (simplified)
                    function_lines = [line]
                    indent_level = len(line) - len(line.lstrip())
                    
                    for j in range(i + 1, min(i + 50, len(lines))):  # Limit to 50 lines
                        next_line = lines[j]
                        if next_line.strip() and len(next_line) - len(next_line.lstrip()) <= indent_level:
                            break
                        function_lines.append(next_line)
                    
                    return {
                        "start_line": i + 1,
                        "definition": '\n'.join(function_lines),
                        "signature": line.strip()
                    }
        
        return {"start_line": 0, "definition": "", "signature": ""}

    def _analyze_function_complexity(self, function_def: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze function complexity and potential issues."""
        if not function_def["definition"]:
            return {"complexity": "unknown", "issues": []}
        
        definition = function_def["definition"]
        lines = definition.split('\n')
        
        # Count various complexity indicators
        cyclomatic_complexity = definition.count('if ') + definition.count('for ') + \
                              definition.count('while ') + definition.count('except ') + 1
        
        line_count = len([line for line in lines if line.strip()])
        
        # Identify potential issues
        issues = []
        if cyclomatic_complexity > 10:
            issues.append("High cyclomatic complexity")
        if line_count > 50:
            issues.append("Function is very long")
        if definition.count('try:') > 3:
            issues.append("Multiple try-catch blocks")
        if 'TODO' in definition or 'FIXME' in definition:
            issues.append("Contains TODO/FIXME comments")
        
        return {
            "cyclomatic_complexity": cyclomatic_complexity,
            "line_count": line_count,
            "complexity_level": "high" if cyclomatic_complexity > 10 else "medium" if cyclomatic_complexity > 5 else "low",
            "issues": issues
        }

    def _find_containing_function(self, lines: List[str], line_number: int) -> str:
        """Find the function that contains a specific line."""
        for i in range(line_number - 1, -1, -1):
            line = lines[i]
            if re.search(r'def \w+\s*\(|function \w+\s*\(', line):
                match = re.search(r'def (\w+)|function (\w+)', line)
                if match:
                    return match.group(1) or match.group(2)
        return "unknown"

    def _is_in_loop_context(self, lines: List[str], line_number: int) -> bool:
        """Check if line is within a loop context."""
        for i in range(max(0, line_number - 10), line_number):
            if re.search(r'for |while ', lines[i]):
                return True
        return False

    def _is_in_conditional_context(self, lines: List[str], line_number: int) -> bool:
        """Check if line is within a conditional context."""
        for i in range(max(0, line_number - 5), line_number):
            if re.search(r'if |elif |else:', lines[i]):
                return True
        return False

    def _generate_stack_trace_suggestions(
        self,
        files_involved: List[Dict[str, Any]],
        functions_involved: List[str],
        recent_changes: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate suggestions based on stack trace analysis."""
        suggestions = []
        
        if recent_changes:
            suggestions.append("Check recent commits that modified the involved files")
        
        if len(functions_involved) > 5:
            suggestions.append("Consider breaking down complex function call chains")
        
        if files_involved:
            suggestions.append("Review error handling in the involved files")
            suggestions.append("Add logging statements to trace execution flow")
        
        suggestions.append("Look for similar error patterns in closed issues")
        suggestions.append("Check if this error occurs in specific environments only")
        
        return suggestions

    def _generate_function_suggestions(
        self,
        function_def: Dict[str, Any],
        complexity_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate suggestions for function improvement."""
        suggestions = []
        
        if complexity_analysis.get("complexity_level") == "high":
            suggestions.append("Consider refactoring this function to reduce complexity")
        
        if complexity_analysis.get("line_count", 0) > 50:
            suggestions.append("Break this function into smaller, focused functions")
        
        if "Multiple try-catch blocks" in complexity_analysis.get("issues", []):
            suggestions.append("Consider consolidating error handling logic")
        
        if "Contains TODO/FIXME comments" in complexity_analysis.get("issues", []):
            suggestions.append("Address TODO/FIXME comments that might be related to the issue")
        
        return suggestions
