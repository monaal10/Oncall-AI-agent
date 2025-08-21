"""Example usage of LLM integrations with LangGraph."""

import asyncio
from datetime import datetime, timedelta
from oncall_agent.integrations.llm import (
    LLMManager,
    OpenAIProvider,
    AnthropicProvider,
    OllamaProvider
)


async def individual_provider_examples():
    """Examples of using individual LLM providers."""
    print("=== Individual LLM Provider Examples ===")
    
    # Sample incident context
    sample_context = {
        "incident_description": "Database connection timeout in user authentication service",
        "log_data": [
            {
                "timestamp": datetime.now(),
                "level": "ERROR",
                "message": "psycopg2.OperationalError: could not connect to server: Connection timed out",
                "source": "auth-service"
            },
            {
                "timestamp": datetime.now() - timedelta(minutes=1),
                "level": "WARN", 
                "message": "Connection pool exhausted, waiting for available connection",
                "source": "auth-service"
            }
        ],
        "metric_data": [
            {"name": "database_connections", "value": 95, "unit": "count"},
            {"name": "response_time", "value": 5000, "unit": "ms"}
        ],
        "code_context": [
            {
                "file_path": "auth/database.py",
                "content": "def connect():\n    return psycopg2.connect(DATABASE_URL, connect_timeout=30)",
                "language": "python"
            }
        ],
        "runbook_guidance": "Database Connection Issues: Check connection pool settings and database server status."
    }
    
    # OpenAI Provider Example
    print("\n--- OpenAI Provider ---")
    try:
        openai_config = {
            "api_key": "your-openai-api-key",  # Replace with actual key
            "model": "gpt-4",
            "max_tokens": 1500,
            "temperature": 0.1
        }
        
        openai_provider = OpenAIProvider(openai_config)
        
        # Test health
        health = await openai_provider.health_check()
        print(f"Health: {'✓ Healthy' if health['healthy'] else '✗ Failed'}")
        if health['healthy']:
            print(f"  Model: {health['model_info']['model_name']}")
            print(f"  Latency: {health['latency']:.0f}ms")
        
        # Generate resolution
        if health['healthy']:
            print("\n  Generating resolution...")
            resolution = await openai_provider.generate_resolution(sample_context)
            print(f"  Summary: {resolution['resolution_summary'][:80]}...")
            print(f"  Confidence: {resolution['confidence_score']:.2f}")
        
    except Exception as e:
        print(f"  OpenAI provider failed: {e}")
    
    # Anthropic Provider Example
    print("\n--- Anthropic Provider ---")
    try:
        anthropic_config = {
            "api_key": "your-anthropic-api-key",  # Replace with actual key
            "model": "claude-3-sonnet-20240229",
            "max_tokens": 1500,
            "temperature": 0.1
        }
        
        anthropic_provider = AnthropicProvider(anthropic_config)
        
        # Test health
        health = await anthropic_provider.health_check()
        print(f"Health: {'✓ Healthy' if health['healthy'] else '✗ Failed'}")
        if health['healthy']:
            print(f"  Model: {health['model_info']['model_name']}")
            print(f"  Context window: {health['model_info']['context_window']} tokens")
        
        # Analyze logs
        if health['healthy']:
            print("\n  Analyzing logs...")
            log_analysis = await anthropic_provider.analyze_logs(
                sample_context["log_data"],
                context=sample_context["incident_description"]
            )
            print(f"  Severity: {log_analysis['severity_assessment']}")
            print(f"  Error patterns: {len(log_analysis['error_patterns'])}")
        
    except Exception as e:
        print(f"  Anthropic provider failed: {e}")
    
    # Ollama Provider Example
    print("\n--- Ollama Provider (Local) ---")
    try:
        ollama_config = {
            "model": "llama2",
            "base_url": "http://localhost:11434",
            "temperature": 0.1,
            "timeout": 120
        }
        
        ollama_provider = OllamaProvider(ollama_config)
        
        # Check model availability
        availability = await ollama_provider.check_model_availability()
        print(f"Model available: {'✓ Yes' if availability['available'] else '✗ No'}")
        if availability['available']:
            print(f"  Model: {availability['model']}")
            print(f"  Server: {availability['base_url']}")
        else:
            print(f"  Error: {availability['error']}")
        
        # Analyze code context
        if availability['available']:
            print("\n  Analyzing code...")
            code_analysis = await ollama_provider.analyze_code_context(
                sample_context["code_context"],
                error_message="Database connection timeout"
            )
            print(f"  Issues found: {len(code_analysis['potential_issues'])}")
            print(f"  Fixes suggested: {bool(code_analysis['suggested_fixes'])}")
        
    except Exception as e:
        print(f"  Ollama provider failed: {e}")


async def llm_manager_example():
    """Example of using unified LLM manager."""
    print("\n=== LLM Manager Example ===")
    
    # Create LLM manager configuration with fallbacks
    config = {
        "primary_provider": {
            "type": "openai",
            "config": {
                "api_key": "your-openai-api-key",
                "model": "gpt-4",
                "max_tokens": 2000,
                "temperature": 0.1
            }
        },
        "fallback_providers": [
            {
                "type": "anthropic",
                "config": {
                    "api_key": "your-anthropic-api-key",
                    "model": "claude-3-sonnet-20240229",
                    "max_tokens": 2000,
                    "temperature": 0.1
                }
            },
            {
                "type": "ollama",
                "config": {
                    "model": "llama2",
                    "base_url": "http://localhost:11434",
                    "temperature": 0.1
                }
            }
        ]
    }
    
    try:
        llm_manager = LLMManager(config)
        
        # Test all providers
        print("\n--- Provider Health Check ---")
        health_status = await llm_manager.health_check()
        
        primary_health = health_status["primary_provider"]
        print(f"Primary provider: {'✓ Healthy' if primary_health['healthy'] else '✗ Failed'}")
        if primary_health['healthy']:
            print(f"  Model: {primary_health['model_info']['model_name']}")
            print(f"  Latency: {primary_health['latency']:.0f}ms")
        
        for i, fallback_health in enumerate(health_status["fallback_providers"]):
            print(f"Fallback {i+1}: {'✓ Healthy' if fallback_health['healthy'] else '✗ Failed'}")
        
        # Get provider information
        print("\n--- Provider Information ---")
        provider_info = llm_manager.get_provider_info()
        
        primary_info = provider_info["primary_provider"]
        print(f"Primary: {primary_info['provider']} ({primary_info['model_name']})")
        print(f"  Context window: {primary_info['context_window']} tokens")
        print(f"  Supports streaming: {primary_info['supports_streaming']}")
        
        for fallback_info in provider_info["fallback_providers"]:
            print(f"Fallback: {fallback_info['provider']} ({fallback_info['model_name']})")
        
        # Generate resolution with fallback support
        print("\n--- Resolution Generation with Fallbacks ---")
        incident_context = {
            "incident_description": "API service returning 500 errors after deployment",
            "log_data": [
                {
                    "timestamp": datetime.now(),
                    "level": "ERROR",
                    "message": "Internal server error in user authentication",
                    "source": "api-gateway"
                }
            ],
            "metric_data": [
                {"name": "error_rate", "value": 25.5, "unit": "percent"},
                {"name": "response_time", "value": 2500, "unit": "ms"}
            ],
            "code_context": [],
            "runbook_guidance": "API Error Response: Check recent deployments and database connectivity."
        }
        
        try:
            resolution = await llm_manager.generate_resolution(incident_context)
            print("✓ Resolution generated successfully")
            print(f"  Summary: {resolution['resolution_summary'][:100]}...")
            print(f"  Confidence: {resolution['confidence_score']:.2f}")
            print(f"  Has code changes: {bool(resolution['code_changes'])}")
            
        except Exception as e:
            print(f"✗ Resolution generation failed: {e}")
        
        # Stream response example
        print("\n--- Streaming Response Example ---")
        try:
            print("Streaming response: ", end="", flush=True)
            async for chunk in llm_manager.stream_response(
                "Briefly explain what causes database connection timeouts."
            ):
                print(chunk, end="", flush=True)
            print("\n✓ Streaming completed")
            
        except Exception as e:
            print(f"\n✗ Streaming failed: {e}")
            
    except Exception as e:
        print(f"LLM Manager initialization failed: {e}")


async def langgraph_integration_example():
    """Example of LangGraph integration."""
    print("\n=== LangGraph Integration Example ===")
    
    try:
        # Create LLM manager
        config = {
            "primary_provider": {
                "type": "openai",
                "config": {
                    "api_key": "your-openai-api-key",
                    "model": "gpt-4",
                    "max_tokens": 1500
                }
            }
        }
        
        llm_manager = LLMManager(config)
        
        # Get LangChain model for LangGraph
        print("\n--- LangChain Model for LangGraph ---")
        langchain_model = llm_manager.get_langchain_model()
        print(f"Model type: {type(langchain_model).__name__}")
        print(f"Model class: {langchain_model.__class__.__module__}.{langchain_model.__class__.__name__}")
        
        # Create basic LangGraph workflow
        print("\n--- Creating LangGraph Workflow ---")
        try:
            workflow = llm_manager.create_langgraph_workflow()
            print("✓ LangGraph workflow created successfully")
            print(f"  Workflow type: {type(workflow).__name__}")
            
            # Example workflow execution
            print("\n--- Workflow Execution Example ---")
            initial_state = {
                "incident_description": "Database connection errors in production",
                "log_data": [
                    {
                        "timestamp": datetime.now(),
                        "level": "ERROR", 
                        "message": "Connection timeout after 30 seconds",
                        "source": "database-service"
                    }
                ],
                "metric_data": [
                    {"name": "db_connections", "value": 100, "unit": "count"}
                ],
                "code_context": [
                    {
                        "file_path": "db/connection.py",
                        "content": "def connect(): return psycopg2.connect(url, timeout=30)",
                        "language": "python"
                    }
                ],
                "runbook_guidance": "Check database server status and connection pool settings.",
                "analysis_steps": []
            }
            
            # Note: This would actually execute the workflow if credentials were valid
            print("  Workflow state prepared:")
            print(f"    Incident: {initial_state['incident_description']}")
            print(f"    Log entries: {len(initial_state['log_data'])}")
            print(f"    Metrics: {len(initial_state['metric_data'])}")
            print(f"    Code snippets: {len(initial_state['code_context'])}")
            print("  → This state would be processed through the LangGraph workflow")
            
        except Exception as e:
            print(f"✗ LangGraph workflow creation failed: {e}")
        
        # Direct LangChain model usage
        print("\n--- Direct LangChain Model Usage ---")
        try:
            from langchain_core.messages import HumanMessage
            
            # This shows how the model can be used directly in LangGraph
            test_message = HumanMessage(content="What are common causes of database connection timeouts?")
            
            print("✓ LangChain model ready for LangGraph integration")
            print("  Message type:", type(test_message).__name__)
            print("  Model supports async:", hasattr(langchain_model, 'ainvoke'))
            print("  Model supports streaming:", hasattr(langchain_model, 'astream'))
            
            # Example of what LangGraph would do
            print("\n  Example LangGraph node function:")
            print("""
            async def resolution_node(state):
                model = llm_manager.get_langchain_model()
                messages = [HumanMessage(content=state['incident_description'])]
                response = await model.ainvoke(messages)
                state['resolution'] = response.content
                return state
            """)
            
        except Exception as e:
            print(f"✗ Direct model usage failed: {e}")
            
    except Exception as e:
        print(f"LangGraph integration failed: {e}")


async def streaming_example():
    """Example of streaming responses."""
    print("\n=== Streaming Response Example ===")
    
    try:
        # Create a simple provider for streaming
        config = {
            "api_key": "your-openai-api-key",  # Replace with actual key
            "model": "gpt-3.5-turbo",
            "temperature": 0.1
        }
        
        provider = OpenAIProvider(config)
        
        print("Streaming analysis of incident...")
        print("Response: ", end="", flush=True)
        
        prompt = """
        Analyze this incident and provide a brief resolution:
        
        Incident: API gateway returning 503 errors
        Logs: "Upstream server connection failed"
        Metrics: Error rate at 15%, response time 3000ms
        
        Provide a concise analysis and resolution steps.
        """
        
        full_response = ""
        async for chunk in provider.stream_response(prompt):
            print(chunk, end="", flush=True)
            full_response += chunk
        
        print(f"\n\n✓ Streaming completed ({len(full_response)} characters)")
        
    except Exception as e:
        print(f"✗ Streaming example failed: {e}")


async def main():
    """Run all LLM integration examples."""
    print("LLM Integration Examples (LangChain + LangGraph)")
    print("=" * 60)
    
    await individual_provider_examples()
    await llm_manager_example()
    await langgraph_integration_example()
    await streaming_example()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("\nSetup Instructions:")
    print("\n1. Install dependencies:")
    print("   pip install langchain langchain-openai langchain-anthropic langchain-community langgraph")
    print("\n2. Set up API keys:")
    print("   export OPENAI_API_KEY='your-openai-key'")
    print("   export ANTHROPIC_API_KEY='your-anthropic-key'")
    print("\n3. For Ollama (local models):")
    print("   - Install Ollama: https://ollama.ai/")
    print("   - Pull a model: ollama pull llama2")
    print("   - Start server: ollama serve")
    print("\n4. Update the API keys in the examples")
    print("\nLangGraph Integration:")
    print("  - Use llm_manager.get_langchain_model() in your LangGraph workflows")
    print("  - All providers return LangChain-compatible models")
    print("  - Built-in fallback and retry logic")
    print("  - Supports streaming for real-time responses")
    print("\nKey Functions for AI Agent:")
    print("  - generate_resolution(): Main incident resolution function")
    print("  - analyze_logs(): Log pattern analysis")
    print("  - analyze_code_context(): Code issue analysis")
    print("  - stream_response(): Real-time response streaming")


if __name__ == "__main__":
    asyncio.run(main())
