# LLM Provider Integration

This module provides comprehensive integration with Large Language Models (LLMs) through LangChain for the OnCall AI Agent, designed specifically for LangGraph workflows.

## Features

### Supported Providers
- **OpenAI**: GPT-4, GPT-3.5-turbo, and other OpenAI models
- **Anthropic**: Claude-3 (Opus, Sonnet, Haiku) and Claude-2 models
- **Ollama**: Local models (Llama2, CodeLlama, Mistral, etc.)

### LangGraph Integration
- **Native LangChain compatibility** for seamless LangGraph workflows
- **Unified interface** across all providers
- **Streaming support** for real-time responses
- **Fallback mechanisms** for high availability
- **State management** for complex workflows

### Specialized Functions
- **Incident resolution generation** with structured output
- **Log analysis** with pattern detection and severity assessment
- **Code analysis** with fix suggestions and best practices
- **Context-aware processing** with intelligent truncation

## Architecture

### Provider Hierarchy
```
LLMProvider (Abstract Base)
├── OpenAIProvider (LangChain ChatOpenAI/OpenAI)
├── AnthropicProvider (LangChain ChatAnthropic)
├── OllamaProvider (LangChain Ollama/ChatOllama)
└── LLMManager (Orchestrates multiple providers)
```

### LangGraph Integration Flow
```
LangGraph Workflow → LLMManager → LangChain Model → Provider API
```

## Configuration

### Unified Manager (Recommended for Production)

```yaml
integrations:
  llm_provider:
    type: "llm_manager"
    config:
      # Primary provider
      primary_provider:
        type: "openai"
        config:
          api_key: "${OPENAI_API_KEY}"
          model: "gpt-4"
          max_tokens: 2000
          temperature: 0.1
      
      # Fallback providers
      fallback_providers:
        - type: "anthropic"
          config:
            api_key: "${ANTHROPIC_API_KEY}"
            model: "claude-3-sonnet-20240229"
        
        - type: "ollama"
          config:
            model: "llama2"
            base_url: "http://localhost:11434"
```

### Individual Providers

```yaml
# OpenAI only
llm_provider:
  type: "openai"
  config:
    api_key: "${OPENAI_API_KEY}"
    model: "gpt-4"
    max_tokens: 2000
    temperature: 0.1

# Anthropic only  
llm_provider:
  type: "anthropic"
  config:
    api_key: "${ANTHROPIC_API_KEY}"
    model: "claude-3-sonnet-20240229"

# Ollama only (local)
llm_provider:
  type: "ollama"
  config:
    model: "llama2"
    base_url: "http://localhost:11434"
```

## Usage Examples

### LangGraph Integration (Primary Use Case)

```python
from oncall_agent.integrations.llm import LLMManager
from langgraph.graph import StateGraph, END

# Initialize LLM manager
llm_manager = LLMManager(config)

# Get LangChain model for LangGraph
model = llm_manager.get_langchain_model()

# Use in LangGraph workflow
def create_incident_workflow():
    workflow = StateGraph(IncidentState)
    
    # Add node that uses the LLM
    async def resolution_node(state):
        messages = [HumanMessage(content=state['incident_description'])]
        response = await model.ainvoke(messages)
        state['resolution'] = response.content
        return state
    
    workflow.add_node("resolve", resolution_node)
    workflow.set_entry_point("resolve")
    workflow.add_edge("resolve", END)
    
    return workflow.compile()

# Execute workflow
workflow = create_incident_workflow()
result = await workflow.ainvoke(initial_state)
```

### Direct Provider Usage

```python
from oncall_agent.integrations.llm import OpenAIProvider

provider = OpenAIProvider(config)

# Generate incident resolution
resolution = await provider.generate_resolution(incident_context)

# Analyze logs
log_analysis = await provider.analyze_logs(log_entries, context="API errors")

# Analyze code
code_analysis = await provider.analyze_code_context(code_snippets, error_message)

# Stream response
async for chunk in provider.stream_response("Analyze this error..."):
    print(chunk, end="")
```

### Manager with Fallbacks

```python
llm_manager = LLMManager(config_with_fallbacks)

# Automatic fallback if primary provider fails
resolution = await llm_manager.generate_resolution(incident_context)

# Health check all providers
health = await llm_manager.health_check()
```

## Core Functions

### Primary Function: `generate_resolution()`
**Main function for incident resolution**
- **Input**: Incident context with logs, metrics, code, runbook guidance
- **Output**: Structured resolution with steps, code changes, root cause analysis
- **Use**: Primary function for AI agent incident resolution

### Analysis Functions
- **`analyze_logs()`**: Pattern detection and severity assessment from log data
- **`analyze_code_context()`**: Code issue identification and fix suggestions
- **`stream_response()`**: Real-time response streaming for better UX

### LangGraph Integration
- **`get_langchain_model()`**: Get model for direct LangGraph usage
- **`create_langgraph_workflow()`**: Pre-built workflow template
- **Native LangChain compatibility**: Works with all LangGraph features

## Model Capabilities

### OpenAI Models
| Model | Context Window | Best For | Streaming | Functions |
|-------|---------------|----------|-----------|-----------|
| **GPT-4** | 8K-128K | Complex reasoning, code analysis | ✅ | ✅ |
| **GPT-3.5-turbo** | 4K-16K | Fast responses, cost-effective | ✅ | ✅ |

### Anthropic Models  
| Model | Context Window | Best For | Streaming | Functions |
|-------|---------------|----------|-----------|-----------|
| **Claude-3 Opus** | 200K | Complex analysis, large context | ✅ | ❌ |
| **Claude-3 Sonnet** | 200K | Balanced performance | ✅ | ❌ |
| **Claude-3 Haiku** | 200K | Fast responses | ✅ | ❌ |

### Ollama Models (Local)
| Model | Context Window | Best For | Streaming | Functions |
|-------|---------------|----------|-----------|-----------|
| **Llama2** | 4K | General purpose, privacy | ✅ | ❌ |
| **CodeLlama** | 16K | Code analysis, programming | ✅ | ❌ |
| **Mistral** | 8K | Efficient, multilingual | ✅ | ❌ |

## LangGraph Workflow Example

```python
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage

# Define state
class IncidentState:
    incident_description: str
    log_analysis: Optional[Dict] = None
    code_analysis: Optional[Dict] = None
    resolution: Optional[Dict] = None

# Create workflow
def create_incident_resolution_workflow(llm_manager):
    workflow = StateGraph(IncidentState)
    
    # Get the LangChain model
    model = llm_manager.get_langchain_model()
    
    # Define nodes
    async def analyze_logs_node(state):
        if state.get("log_data"):
            analysis = await llm_manager.analyze_logs(state["log_data"])
            state["log_analysis"] = analysis
        return state
    
    async def analyze_code_node(state):
        if state.get("code_context"):
            analysis = await llm_manager.analyze_code_context(
                state["code_context"], 
                state["incident_description"]
            )
            state["code_analysis"] = analysis
        return state
    
    async def generate_resolution_node(state):
        # Combine all context
        full_context = {
            "incident_description": state["incident_description"],
            "log_analysis": state.get("log_analysis", {}),
            "code_analysis": state.get("code_analysis", {}),
            "runbook_guidance": state.get("runbook_guidance", "")
        }
        
        resolution = await llm_manager.generate_resolution(full_context)
        state["resolution"] = resolution
        return state
    
    # Build workflow
    workflow.add_node("analyze_logs", analyze_logs_node)
    workflow.add_node("analyze_code", analyze_code_node)
    workflow.add_node("generate_resolution", generate_resolution_node)
    
    # Define flow
    workflow.set_entry_point("analyze_logs")
    workflow.add_edge("analyze_logs", "analyze_code")
    workflow.add_edge("analyze_code", "generate_resolution")
    workflow.add_edge("generate_resolution", END)
    
    return workflow.compile()
```

## Response Formats

### Resolution Response
```python
{
    "resolution_summary": "Brief description of the issue and fix",
    "detailed_steps": "Step-by-step resolution instructions",
    "code_changes": "Specific code changes needed",
    "root_cause_analysis": "Analysis of what caused the issue",
    "confidence_score": 0.85,  # 0-1 confidence level
    "reasoning": "Full explanation of the reasoning",
    "additional_recommendations": "Prevention and improvement suggestions"
}
```

### Log Analysis Response
```python
{
    "error_patterns": ["Connection timeout", "Pool exhaustion"],
    "severity_assessment": "High",
    "timeline": "Error started at 14:30, peaked at 14:35",
    "affected_components": ["database", "auth-service"],
    "suggested_queries": ["Check connection pool metrics", "Review database logs"]
}
```

### Code Analysis Response
```python
{
    "potential_issues": ["Hard-coded timeout value", "No connection retry logic"],
    "suggested_fixes": "Add connection pooling and retry mechanism",
    "best_practices": ["Use connection pooling", "Implement circuit breaker"],
    "testing_recommendations": ["Test with connection failures", "Load testing"]
}
```

## Error Handling

### Provider-Specific Errors
- **OpenAI**: API key validation, rate limiting, model availability
- **Anthropic**: Authentication, quota limits, model access
- **Ollama**: Server connectivity, model availability, local resource limits

### Fallback Strategy
1. **Primary provider** attempts first
2. **Fallback providers** tried in order
3. **Error aggregation** provides detailed failure information
4. **Graceful degradation** with reduced functionality if needed

### Rate Limiting
- **Automatic retry** with exponential backoff
- **Provider-specific limits** respected
- **Fallback activation** on rate limit errors
- **Request queuing** for burst scenarios

## Performance Considerations

### Context Window Management
- **Automatic truncation** to fit model limits
- **Priority-based content** (keep most important data)
- **Token estimation** for efficient usage
- **Context optimization** per provider

### Streaming Benefits
- **Real-time feedback** for long-running analyses
- **Better user experience** with progressive responses
- **Lower perceived latency** for complex queries
- **Cancellation support** for interrupted requests

### Local vs Cloud Models
- **Local models (Ollama)**: Privacy, no API costs, potential latency
- **Cloud models (OpenAI/Anthropic)**: Better performance, API costs, internet dependency
- **Hybrid approach**: Local for development, cloud for production

## Dependencies

### Required
```bash
pip install langchain langchain-core langgraph
```

### Provider-Specific
```bash
# OpenAI
pip install langchain-openai

# Anthropic
pip install langchain-anthropic

# Ollama (local models)
pip install langchain-community

# All providers
pip install langchain-openai langchain-anthropic langchain-community
```

## Security Considerations

### API Key Management
- **Environment variables** for API keys (never in code)
- **Key rotation** support with configuration updates
- **Scope limitation** using least-privilege principles
- **Audit logging** for API usage

### Data Privacy
- **Local models** for sensitive data (Ollama)
- **Data retention policies** awareness for cloud providers
- **Content filtering** to remove sensitive information
- **Audit trails** for all LLM interactions

### Network Security
- **HTTPS enforcement** for all API communications
- **Timeout configuration** to prevent hanging requests
- **Rate limiting** to prevent abuse
- **Error handling** to avoid information leakage

## References

- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Anthropic API Documentation](https://docs.anthropic.com/)
- [Ollama Documentation](https://ollama.ai/)
