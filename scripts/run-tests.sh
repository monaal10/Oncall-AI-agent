#!/bin/bash

# OnCall AI Agent Test Runner
# Runs the complete test suite with coverage reporting

set -e  # Exit on any error

echo "🧪 OnCall AI Agent Test Suite"
echo "============================="

# Check if we're in the project root
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

# Check if pytest is installed
if ! python -c "import pytest" 2>/dev/null; then
    echo "❌ Error: pytest not installed. Install with: pip install pytest pytest-asyncio pytest-cov pytest-mock"
    exit 1
fi

# Create test reports directory
mkdir -p test-reports

echo ""
echo "📋 Running unit tests..."
echo "----------------------"

# Run unit tests with coverage
python -m pytest tests/unit/ \
    --cov=oncall_agent \
    --cov-report=term-missing \
    --cov-report=html:test-reports/coverage-html \
    --cov-report=xml:test-reports/coverage.xml \
    --junit-xml=test-reports/junit.xml \
    -v \
    --tb=short

# Check if unit tests passed
if [ $? -eq 0 ]; then
    echo "✅ Unit tests passed!"
else
    echo "❌ Unit tests failed!"
    exit 1
fi

echo ""
echo "📊 Test Coverage Summary"
echo "----------------------"
echo "📁 HTML coverage report: test-reports/coverage-html/index.html"
echo "📄 XML coverage report: test-reports/coverage.xml"
echo "📋 JUnit report: test-reports/junit.xml"

echo ""
echo "🎯 Test Categories Covered:"
echo "• Core components (setup, runtime, factory)"
echo "• Integration registry and validation"
echo "• AWS CloudWatch integrations"
echo "• GitHub repository integration"
echo "• LLM providers (OpenAI, Anthropic, Ollama, HuggingFace)"
echo "• Runbook integrations (Markdown, unified)"
echo "• Base provider classes"
echo "• Configuration validation"

echo ""
echo "✨ All tests completed successfully!"
echo ""
echo "💡 To run specific test categories:"
echo "   pytest tests/unit/test_core/                    # Core components"
echo "   pytest tests/unit/test_integrations/            # All integrations"
echo "   pytest tests/unit/test_utils/                   # Utilities"
echo ""
echo "💡 To run tests with different verbosity:"
echo "   pytest tests/unit/ -v                          # Verbose output"
echo "   pytest tests/unit/ -vv                         # Very verbose output"
echo "   pytest tests/unit/ -q                          # Quiet output"
echo ""
echo "💡 To run tests for a specific file:"
echo "   pytest tests/unit/test_core/test_setup_manager.py -v"
