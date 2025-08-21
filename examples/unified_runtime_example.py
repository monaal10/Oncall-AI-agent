"""Example of unified runtime interface for AI agent integration."""

import asyncio
from datetime import datetime, timedelta
from oncall_agent.core.setup_manager import SetupManager
from oncall_agent.utils.config_validator import ConfigValidator


async def setup_example():
    """Example of setting up the unified runtime interface."""
    print("=== OnCall AI Agent Setup Example ===")
    
    # Example user configuration (minimal - only what user has)
    user_config = {
        # AWS credentials (user has AWS access)
        "aws": {
            "region": "us-west-2"
            # access_key_id and secret_access_key from environment or IAM role
        },
        
        # GitHub access (required)
        "github": {
            # token from GITHUB_TOKEN environment variable
            "repositories": [
                "myorg/backend-api",
                "myorg/frontend-app",
                "myorg/shared-utils"
            ]
        },
        
        # OpenAI access (user has OpenAI API key)
        "openai": {
            # api_key from OPENAI_API_KEY environment variable
            "model": "gpt-4",
            "max_tokens": 2000,
            "temperature": 0.1
        },
        
        # Optional: Runbooks directory
        "runbooks": {
            "directory": "/path/to/company/runbooks"
        },
        
        # User preferences
        "preferences": {
            "preferred_cloud_provider": "aws",
            "preferred_llm_provider": "openai"
        }
    }
    
    # Validate configuration first
    print("\n🔍 Validating configuration...")
    validator = ConfigValidator()
    validation_result = validator.validate_config(user_config)
    
    if not validation_result["valid"]:
        print("❌ Configuration validation failed:")
        for error in validation_result["errors"]:
            print(f"  • {error}")
        
        print("\n💡 Suggestions:")
        for suggestion in validation_result["suggestions"]:
            print(f"  • {suggestion}")
        
        return None
    
    print("✅ Configuration valid!")
    
    # Setup the system
    setup_manager = SetupManager()
    
    try:
        # This creates the unified runtime interface
        runtime_interface = await setup_manager.setup_from_config(user_config)
        
        print("\n🎉 Setup completed successfully!")
        return runtime_interface
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return None


async def runtime_interface_example(runtime_interface):
    """Example of using the unified runtime interface."""
    print("\n=== Runtime Interface Usage Example ===")
    
    if not runtime_interface:
        print("❌ No runtime interface available")
        return
    
    # These are the unified functions that the AI agent will use
    print("\n📋 Available runtime functions:")
    runtime_functions = runtime_interface.get_runtime_functions()
    for func_name in runtime_functions.keys():
        print(f"  • {func_name}")
    
    # Example 1: Get logs (works with any cloud provider)
    print("\n📊 Testing unified logs function...")
    try:
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=1)
        
        # This function works the same regardless of cloud provider (AWS/Azure/GCP)
        logs = await runtime_interface.get_logs(
            query="ERROR",
            time_range=(start_time, end_time),
            service_name="auth",
            limit=5
        )
        
        print(f"✓ Retrieved {len(logs)} log entries")
        for log in logs[:3]:  # Show first 3
            print(f"  [{log['timestamp']}] {log['level']}: {log['message'][:60]}...")
            
    except Exception as e:
        print(f"✗ Logs function failed: {e}")
    
    # Example 2: Get code context (always GitHub)
    print("\n💻 Testing unified code function...")
    try:
        # This function provides standardized code context
        code_context = await runtime_interface.get_code_context(
            error_info={
                "message": "database connection timeout",
                "stack_trace": "psycopg2.OperationalError: could not connect to server"
            }
        )
        
        print(f"✓ Retrieved {len(code_context)} code context items")
        for code in code_context[:3]:  # Show first 3
            print(f"  📄 {code['repository']}/{code['file_path']} (relevance: {code['relevance']:.2f})")
            
    except Exception as e:
        print(f"✗ Code function failed: {e}")
    
    # Example 3: Get LLM response (works with any LLM provider)
    print("\n🤖 Testing unified LLM function...")
    try:
        # This function works the same regardless of LLM provider (OpenAI/Anthropic/Ollama)
        incident_context = {
            "incident_description": "Database connection errors in production auth service",
            "log_data": [
                {
                    "timestamp": datetime.now(),
                    "level": "ERROR",
                    "message": "psycopg2.OperationalError: could not connect to server",
                    "source": "auth-service"
                }
            ],
            "metric_data": [
                {"metric_name": "database_connections", "value": 95, "unit": "count"},
                {"metric_name": "error_rate", "value": 15.5, "unit": "percent"}
            ],
            "code_context": [
                {
                    "repository": "myorg/backend-api",
                    "file_path": "auth/database.py",
                    "content": "def connect(): return psycopg2.connect(DATABASE_URL, timeout=30)"
                }
            ],
            "runbook_guidance": "Database Issues: Check connection pool and server status"
        }
        
        resolution = await runtime_interface.get_llm_response(
            context=incident_context,
            response_type="resolution"
        )
        
        print(f"✓ Generated resolution (confidence: {resolution.get('confidence_score', 0):.2f})")
        print(f"  Summary: {resolution.get('resolution_summary', '')[:80]}...")
        print(f"  Steps provided: {bool(resolution.get('detailed_steps'))}")
        print(f"  Code changes: {bool(resolution.get('code_changes'))}")
        
    except Exception as e:
        print(f"✗ LLM function failed: {e}")
    
    # Example 4: Get metrics (optional - only if configured)
    if runtime_interface.get_metrics:
        print("\n📈 Testing unified metrics function...")
        try:
            metrics = await runtime_interface.get_metrics(
                resource_info={
                    "namespace": "auth-service",
                    "dimensions": {"environment": "production"}
                },
                time_range=(start_time, end_time),
                metric_names=["CPUUtilization", "DatabaseConnections"]
            )
            
            print(f"✓ Retrieved {len(metrics)} metric data points")
            for metric in metrics[:3]:  # Show first 3
                print(f"  📊 {metric['metric_name']}: {metric['value']} {metric['unit']}")
                
        except Exception as e:
            print(f"✗ Metrics function failed: {e}")
    else:
        print("\n📈 Metrics function: not configured")
    
    # Example 5: Get runbook guidance (optional - only if configured)
    if runtime_interface.get_runbook_guidance:
        print("\n📖 Testing unified runbook function...")
        try:
            guidance = await runtime_interface.get_runbook_guidance(
                error_context="database connection timeout in production",
                limit=2
            )
            
            print(f"✓ Retrieved runbook guidance ({len(guidance)} characters)")
            print(f"  Preview: {guidance[:100]}...")
            
        except Exception as e:
            print(f"✗ Runbook function failed: {e}")
    else:
        print("\n📖 Runbook function: not configured")


async def ai_agent_integration_example(runtime_interface):
    """Example showing how an AI agent would use the runtime interface."""
    print("\n=== AI Agent Integration Example ===")
    
    if not runtime_interface:
        print("❌ No runtime interface available")
        return
    
    print("🤖 Simulating AI agent incident resolution workflow...")
    
    # This is how an AI agent or LangGraph workflow would use the unified interface
    async def resolve_incident(incident_description: str) -> Dict[str, Any]:
        """AI agent incident resolution using unified runtime functions."""
        
        print(f"\n🔍 Analyzing incident: {incident_description}")
        
        # Step 1: Get logs (cloud-agnostic)
        print("  📊 Fetching relevant logs...")
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=2)
        
        logs = await runtime_interface.get_logs(
            query="ERROR",
            time_range=(start_time, end_time),
            service_name="auth",
            limit=10
        )
        print(f"    Found {len(logs)} log entries")
        
        # Step 2: Get code context
        print("  💻 Analyzing code context...")
        code_context = await runtime_interface.get_code_context(
            error_info={"message": incident_description}
        )
        print(f"    Found {len(code_context)} relevant code items")
        
        # Step 3: Get metrics (if available)
        metrics = []
        if runtime_interface.get_metrics:
            print("  📈 Fetching metrics...")
            metrics = await runtime_interface.get_metrics(
                resource_info={"namespace": "auth-service"},
                time_range=(start_time, end_time)
            )
            print(f"    Found {len(metrics)} metric data points")
        
        # Step 4: Get runbook guidance (if available)
        runbook_guidance = ""
        if runtime_interface.get_runbook_guidance:
            print("  📖 Getting runbook guidance...")
            runbook_guidance = await runtime_interface.get_runbook_guidance(
                error_context=incident_description
            )
            print(f"    Retrieved {len(runbook_guidance)} characters of guidance")
        
        # Step 5: Generate resolution using LLM
        print("  🧠 Generating AI resolution...")
        context = {
            "incident_description": incident_description,
            "log_data": logs,
            "metric_data": metrics,
            "code_context": code_context,
            "runbook_guidance": runbook_guidance
        }
        
        resolution = await runtime_interface.get_llm_response(
            context=context,
            response_type="resolution"
        )
        
        print(f"    Generated resolution (confidence: {resolution.get('confidence_score', 0):.2f})")
        
        return {
            "incident": incident_description,
            "analysis": {
                "logs_found": len(logs),
                "code_items": len(code_context),
                "metrics_found": len(metrics),
                "runbook_guidance_length": len(runbook_guidance)
            },
            "resolution": resolution
        }
    
    # Test the AI agent workflow
    test_incident = "Authentication service experiencing database connection timeouts"
    
    try:
        result = await resolve_incident(test_incident)
        
        print(f"\n✅ AI Agent Resolution Complete!")
        print(f"📋 Analysis Summary:")
        analysis = result["analysis"]
        print(f"  • Logs analyzed: {analysis['logs_found']}")
        print(f"  • Code items reviewed: {analysis['code_items']}")
        print(f"  • Metrics data points: {analysis['metrics_found']}")
        print(f"  • Runbook guidance: {analysis['runbook_guidance_length']} chars")
        
        resolution = result["resolution"]
        print(f"\n🎯 Resolution Generated:")
        print(f"  • Summary: {resolution.get('resolution_summary', '')[:100]}...")
        print(f"  • Confidence: {resolution.get('confidence_score', 0):.2f}")
        print(f"  • Has detailed steps: {bool(resolution.get('detailed_steps'))}")
        print(f"  • Has code changes: {bool(resolution.get('code_changes'))}")
        
    except Exception as e:
        print(f"❌ AI agent workflow failed: {e}")


async def langgraph_integration_example(runtime_interface):
    """Example showing LangGraph integration."""
    print("\n=== LangGraph Integration Example ===")
    
    if not runtime_interface:
        print("❌ No runtime interface available")
        return
    
    try:
        # Get the LangChain model for LangGraph
        langchain_model = runtime_interface.get_langchain_model()
        
        print("🔗 LangGraph Integration Ready:")
        print(f"  • Model type: {type(langchain_model).__name__}")
        print(f"  • Model class: {langchain_model.__class__.__module__}")
        print(f"  • Supports async: {hasattr(langchain_model, 'ainvoke')}")
        print(f"  • Supports streaming: {hasattr(langchain_model, 'astream')}")
        
        # Example of how this would be used in LangGraph
        print(f"\n📝 LangGraph Usage Pattern:")
        print("""
        # In your LangGraph workflow:
        from langgraph.graph import StateGraph, END
        from langchain_core.messages import HumanMessage
        
        # Get the model from runtime interface
        model = runtime_interface.get_langchain_model()
        
        # Use in LangGraph nodes
        async def analysis_node(state):
            # Get context using unified functions
            logs = await runtime_interface.get_logs(...)
            code = await runtime_interface.get_code_context(...)
            
            # Use LangChain model directly
            messages = [HumanMessage(content=state['incident'])]
            response = await model.ainvoke(messages)
            
            state['analysis'] = response.content
            return state
        
        # Build workflow
        workflow = StateGraph(State)
        workflow.add_node("analyze", analysis_node)
        workflow.set_entry_point("analyze")
        workflow.add_edge("analyze", END)
        
        # Execute
        app = workflow.compile()
        result = await app.ainvoke(initial_state)
        """)
        
        # Show the actual model info
        model_info = runtime_interface._llm_provider.get_model_info()
        print(f"\n🤖 Model Information:")
        print(f"  • Provider: {model_info['provider']}")
        print(f"  • Model: {model_info['model_name']}")
        print(f"  • Context window: {model_info['context_window']} tokens")
        print(f"  • Supports streaming: {model_info['supports_streaming']}")
        
    except Exception as e:
        print(f"❌ LangGraph integration example failed: {e}")


async def main():
    """Run the complete unified runtime example."""
    print("OnCall AI Agent - Unified Runtime Interface")
    print("=" * 60)
    
    # Step 1: Setup the system
    runtime_interface = await setup_example()
    
    if runtime_interface:
        # Step 2: Test the runtime interface
        await runtime_interface_example(runtime_interface)
        
        # Step 3: Show AI agent integration
        await ai_agent_integration_example(runtime_interface)
        
        # Step 4: Show LangGraph integration
        await langgraph_integration_example(runtime_interface)
        
        # Final summary
        print(f"\n" + "=" * 60)
        print("🎯 Key Takeaways:")
        print("1. ✅ Setup automatically detects and configures available integrations")
        print("2. ✅ Runtime interface provides unified functions regardless of cloud provider")
        print("3. ✅ AI agent code doesn't need to know which providers are used")
        print("4. ✅ LangGraph workflows get direct access to LangChain models")
        print("5. ✅ Fallback support for high availability")
        
        print(f"\n🚀 Runtime Functions Ready for AI Agent:")
        functions = runtime_interface.get_runtime_functions()
        for func_name in functions.keys():
            print(f"  • {func_name}() - Standardized interface")
        
        print(f"\n🔗 LangGraph Model Ready:")
        print(f"  • runtime_interface.get_langchain_model() - Direct LangChain model access")
        
    else:
        print("\n❌ Setup failed - check configuration and credentials")
        
        # Show sample configuration
        print("\n💡 Sample configuration:")
        validator = ConfigValidator()
        sample_config = validator.generate_sample_config(["aws", "github", "openai"])
        print(sample_config[:500] + "...")


if __name__ == "__main__":
    print("Note: This example requires actual credentials to run successfully.")
    print("Update the configuration with your actual credentials and paths.\n")
    
    asyncio.run(main())
