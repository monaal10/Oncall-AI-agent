# Contributing to OnCall AI Agent

First off, thank you for considering contributing to OnCall AI Agent! 🎉

It's people like you that make OnCall AI Agent such a great tool for the DevOps community.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Style Guidelines](#style-guidelines)
- [Community](#community)

## 📜 Code of Conduct

This project and everyone participating in it is governed by the [OnCall AI Agent Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## 🤝 How Can I Contribute?

### 🐛 Reporting Bugs

Before creating bug reports, please check [existing issues](https://github.com/monaal/oncall-ai-agent/issues) as you might find that the issue has already been reported.

When creating a bug report, please include:

- **Use a clear and descriptive title**
- **Describe the exact steps to reproduce the problem**
- **Provide specific examples to demonstrate the steps**
- **Describe the behavior you observed and what behavior you expected**
- **Include screenshots if applicable**
- **Include your environment details** (OS, Python version, etc.)

### 💡 Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

- **Use a clear and descriptive title**
- **Provide a detailed description of the suggested enhancement**
- **Explain why this enhancement would be useful**
- **List some other projects where this enhancement exists, if applicable**

### 🔌 Adding Integrations

We're always looking for new integrations! Here are the most wanted:

**High Priority:**
- Azure Monitor/Log Analytics
- Google Cloud Logging/Monitoring  
- GitLab integration
- Datadog integration
- New Relic integration

**Medium Priority:**
- Bitbucket integration
- Splunk integration
- Elasticsearch/OpenSearch
- Prometheus integration

**LLM Providers:**
- Additional Ollama models
- Cohere integration
- Local model fine-tuning support

### 🧩 Creating Plugins

Plugins are a great way to extend functionality without modifying core code. See our [Plugin Development Guide](../docs/plugins/development-guide.md).

### 📖 Improving Documentation

Documentation improvements are always welcome! This includes:

- Fixing typos or grammatical errors
- Adding examples
- Improving clarity
- Adding new guides
- Translating documentation

## 🛠️ Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- Docker (optional, for containerized development)

### Setup Steps

1. **Fork the repository**

2. **Clone your fork**
   ```bash
   git clone https://github.com/YOUR_USERNAME/oncall-ai-agent.git
   cd oncall-ai-agent
   ```

3. **Run the development setup script**
   ```bash
   ./scripts/setup-dev.sh
   ```

   This script will:
   - Create a virtual environment
   - Install all dependencies
   - Set up pre-commit hooks
   - Install the package in development mode

4. **Verify your setup**
   ```bash
   # Run tests
   ./scripts/run-tests.sh
   
   # Start development server
   oncall-agent serve --config examples/configurations/minimal_config.yaml --reload
   ```

### Development Workflow

1. **Create a branch for your changes**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

2. **Make your changes**
   - Write code following our [style guidelines](#style-guidelines)
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes**
   ```bash
   # Run all tests
   ./scripts/run-tests.sh
   
   # Run specific test categories
   pytest tests/unit/
   pytest tests/integration/
   
   # Run with coverage
   pytest --cov=oncall_agent tests/
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add new integration for XYZ"
   ```

   We use [Conventional Commits](https://www.conventionalcommits.org/) format:
   - `feat:` for new features
   - `fix:` for bug fixes
   - `docs:` for documentation changes
   - `test:` for test changes
   - `refactor:` for code refactoring
   - `chore:` for maintenance tasks

## 🔄 Pull Request Process

1. **Update documentation** if you've changed APIs or added features

2. **Add tests** for any new functionality

3. **Ensure all tests pass**
   ```bash
   ./scripts/run-tests.sh
   ```

4. **Update the CHANGELOG.md** with details of changes

5. **Create the Pull Request**
   - Use a clear and descriptive title
   - Reference any related issues
   - Provide a detailed description of changes
   - Include screenshots for UI changes

6. **Respond to review feedback** promptly and respectfully

### PR Review Criteria

Your PR will be reviewed for:

- **Functionality**: Does it work as intended?
- **Code Quality**: Is it readable and maintainable?
- **Tests**: Are there adequate tests?
- **Documentation**: Is documentation updated?
- **Performance**: Does it impact performance?
- **Security**: Are there any security implications?

## 📏 Style Guidelines

### Python Code Style

We use the following tools to maintain code quality:

- **[Black](https://black.readthedocs.io/)** for code formatting
- **[isort](https://pycqa.github.io/isort/)** for import sorting
- **[flake8](https://flake8.pycqa.org/)** for linting
- **[mypy](https://mypy.readthedocs.io/)** for type checking

These are automatically run via pre-commit hooks.

### Code Organization

- **Follow the existing project structure**
- **Use clear, descriptive names** for functions, classes, and variables
- **Add docstrings** for all public functions and classes
- **Use type hints** for function parameters and return values
- **Keep functions small** and focused on a single responsibility

### Integration Guidelines

When adding new integrations:

1. **Inherit from the appropriate base class** (e.g., `LogProvider`, `LLMProvider`)
2. **Add comprehensive error handling**
3. **Include configuration validation**
4. **Add unit and integration tests**
5. **Document configuration options**
6. **Add example configurations**

### Example Integration Structure

```python
from oncall_agent.integrations.base.log_provider import LogProvider
from oncall_agent.models.config import ProviderConfig

class MyLogProvider(LogProvider):
    """Integration with MyService logging platform."""
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._validate_config()
        self._setup_client()
    
    async def fetch_logs(self, query: str, time_range: dict) -> List[dict]:
        """Fetch logs based on query and time range."""
        # Implementation here
        pass
    
    def _validate_config(self):
        """Validate provider-specific configuration."""
        required_fields = ["api_key", "endpoint"]
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required config field: {field}")
```

## 🧪 Testing Guidelines

### Test Structure

- **Unit tests**: Test individual functions/classes in isolation
- **Integration tests**: Test component interactions
- **End-to-end tests**: Test complete workflows

### Writing Tests

```python
import pytest
from unittest.mock import Mock, patch
from oncall_agent.integrations.aws.cloudwatch_logs import CloudWatchLogsProvider

class TestCloudWatchLogsProvider:
    @pytest.fixture
    def provider(self):
        config = {"region": "us-west-2", "access_key": "test", "secret_key": "test"}
        return CloudWatchLogsProvider(config)
    
    @patch("boto3.client")
    async def test_fetch_logs_success(self, mock_boto_client, provider):
        # Test implementation
        pass
    
    async def test_fetch_logs_invalid_query(self, provider):
        with pytest.raises(ValueError):
            await provider.fetch_logs("", {})
```

## 🏗️ Architecture Guidelines

### Core Principles

1. **Modularity**: Each component should be independently testable
2. **Extensibility**: Easy to add new integrations and plugins  
3. **Configuration-driven**: Behavior controlled via configuration
4. **Error handling**: Graceful degradation and clear error messages
5. **Observability**: Comprehensive logging and metrics

### Adding New Components

1. **Define abstract interfaces** in `oncall_agent/integrations/base/`
2. **Implement concrete classes** in appropriate subdirectories
3. **Register with the integration registry**
4. **Add configuration schema validation**
5. **Include comprehensive tests**

## 🌟 Recognition

Contributors will be recognized in:

- **README.md** contributors section
- **CHANGELOG.md** for significant contributions
- **Release notes** for major features
- **GitHub contributors page**

## 💬 Community

- **GitHub Discussions**: For questions and general discussion
- **Issues**: For bug reports and feature requests
- **Pull Requests**: For code contributions

## 📞 Getting Help

If you need help with contributing:

1. Check the [documentation](../docs/)
2. Search [existing issues](https://github.com/monaal/oncall-ai-agent/issues)
3. Ask in [GitHub Discussions](https://github.com/monaal/oncall-ai-agent/discussions)
4. Join our community chat (coming soon!)

## 🙏 Thank You

Your contributions make this project better for everyone. Whether you're fixing a typo, adding a feature, or helping with documentation, every contribution matters!

---

*Happy contributing!* 🚀
