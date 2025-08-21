"""Example usage of runbook integrations."""

import asyncio
from datetime import datetime, timedelta
from oncall_agent.integrations.runbooks import (
    UnifiedRunbookProvider,
    PDFRunbookProvider,
    MarkdownRunbookProvider,
    DocxRunbookProvider,
    WebRunbookProvider,
    RunbookType
)


async def unified_runbook_example():
    """Example of using unified runbook provider."""
    print("=== Unified Runbook Provider Example ===")
    
    # Create unified configuration
    config = {
        "providers": {
            "pdf_runbooks": {
                "type": "pdf",
                "config": {
                    "runbook_directory": "/path/to/pdf/runbooks",
                    "recursive": True,
                    "cache_enabled": True
                }
            },
            "markdown_runbooks": {
                "type": "markdown", 
                "config": {
                    "runbook_directory": "/path/to/markdown/runbooks",
                    "recursive": True,
                    "file_extensions": [".md", ".markdown", ".txt"]
                }
            },
            "web_runbooks": {
                "type": "web_link",
                "config": {
                    "base_urls": [
                        "https://docs.example.com/runbooks",
                        "https://wiki.example.com/operations"
                    ],
                    "timeout": 30,
                    "max_pages": 20
                }
            }
        }
    }
    
    # Initialize unified provider
    runbook_provider = UnifiedRunbookProvider(config)
    
    try:
        # Test connection to all providers
        print("\n--- Provider Health Check ---")
        health_status = await runbook_provider.health_check()
        for provider_name, is_healthy in health_status.items():
            status = "✓ Healthy" if is_healthy else "✗ Failed"
            print(f"  {provider_name}: {status}")
        
        # List all available runbooks
        print("\n--- Available Runbooks ---")
        all_runbooks = await runbook_provider.list_runbooks()
        
        if all_runbooks:
            print(f"Found {len(all_runbooks)} total runbooks:")
            
            # Group by type
            by_type = {}
            for runbook in all_runbooks:
                runbook_type = runbook['type']
                if runbook_type not in by_type:
                    by_type[runbook_type] = []
                by_type[runbook_type].append(runbook)
            
            for runbook_type, runbooks in by_type.items():
                print(f"  {runbook_type.upper()}: {len(runbooks)} runbooks")
                for runbook in runbooks[:3]:  # Show first 3 of each type
                    print(f"    - {runbook['title']}")
                if len(runbooks) > 3:
                    print(f"    ... and {len(runbooks) - 3} more")
        else:
            print("No runbooks found (check your configuration paths)")
        
        # Search across all runbooks
        print("\n--- Search Example ---")
        search_query = "database connection error"
        search_results = await runbook_provider.search_runbooks(
            query=search_query,
            limit=5
        )
        
        if search_results:
            print(f"Search results for '{search_query}':")
            for result in search_results:
                print(f"  📖 {result['title']} ({result['type']})")
                print(f"     Relevance: {result['relevance_score']:.2f}")
                print(f"     Excerpt: {result['excerpt'][:80]}...")
                print(f"     Source: {result['provider']}")
        else:
            print(f"No results found for '{search_query}'")
        
        # Find relevant runbooks for specific error
        print("\n--- Error Context Analysis ---")
        error_context = """
        Application Error: Database connection timeout
        Service: user-authentication-service
        Error: psycopg2.OperationalError: could not connect to server: Connection timed out
        """
        
        try:
            relevant_runbooks = await runbook_provider.find_relevant_runbooks(
                error_context=error_context,
                limit=3
            )
            
            if relevant_runbooks:
                print("Most relevant runbooks for this error:")
                for runbook in relevant_runbooks:
                    print(f"  🎯 {runbook['title']} (relevance: {runbook['relevance_score']:.2f})")
                    print(f"     Type: {runbook['type']} | Provider: {runbook['provider']}")
                    print(f"     Excerpt: {runbook['excerpt'][:60]}...")
            else:
                print("No relevant runbooks found for this error")
                
        except Exception as e:
            print(f"  Error context analysis failed: {e}")
        
        # Get comprehensive context
        print("\n--- Comprehensive Runbook Context ---")
        try:
            comprehensive_context = await runbook_provider.get_comprehensive_runbook_context(
                error_context=error_context,
                limit=2
            )
            
            print("Comprehensive analysis:")
            print(f"  Summary: {comprehensive_context['summary']}")
            print(f"  Relevant sections: {len(comprehensive_context['sections'])}")
            print(f"  Combined text length: {len(comprehensive_context['combined_text'])} characters")
            
            if comprehensive_context['sections']:
                print("  Top relevant sections:")
                for section in comprehensive_context['sections'][:3]:
                    print(f"    • {section['title']} (from {section.get('source_runbook', 'unknown')})")
                    print(f"      Relevance: {section.get('relevance_score', 0):.2f}")
                    
        except Exception as e:
            print(f"  Comprehensive context analysis failed: {e}")
        
        # Example: Get specific runbook content
        print("\n--- Get Specific Runbook Content ---")
        if all_runbooks:
            first_runbook = all_runbooks[0]
            try:
                content = await runbook_provider.get_runbook_text(
                    runbook_id=first_runbook['id'],
                    provider_name=first_runbook['provider']
                )
                
                print(f"Content from '{first_runbook['title']}':")
                print(f"  Length: {len(content)} characters")
                print(f"  Preview: {content[:200]}...")
                
                # Get sections
                provider = runbook_provider.providers[first_runbook['provider']]
                sections = await provider.get_runbook_sections(first_runbook['id'])
                print(f"  Sections: {len(sections)}")
                for section in sections[:3]:
                    print(f"    - {section['title']} (level {section['level']})")
                    
            except Exception as e:
                print(f"  Could not get content: {e}")
                
    except Exception as e:
        print(f"Error: {e}")


async def individual_provider_examples():
    """Examples of using individual runbook providers."""
    print("\n=== Individual Provider Examples ===")
    
    # PDF Provider Example
    print("\n--- PDF Provider ---")
    try:
        pdf_config = {
            "runbook_directory": "/path/to/pdf/runbooks",
            "recursive": True,
            "cache_enabled": True
        }
        
        pdf_provider = PDFRunbookProvider(pdf_config)
        
        # List PDF runbooks
        pdf_runbooks = await pdf_provider.list_runbooks()
        print(f"Found {len(pdf_runbooks)} PDF runbooks")
        
        if pdf_runbooks:
            first_pdf = pdf_runbooks[0]
            print(f"  Example: {first_pdf['title']}")
            print(f"    Pages: {first_pdf.get('pages', 'unknown')}")
            print(f"    Size: {first_pdf['size']} bytes")
            
            # Get content
            content = await pdf_provider.get_runbook_text(first_pdf['id'])
            print(f"    Content length: {len(content)} characters")
            
    except Exception as e:
        print(f"  PDF provider failed: {e}")
    
    # Markdown Provider Example
    print("\n--- Markdown Provider ---")
    try:
        md_config = {
            "runbook_directory": "/path/to/markdown/runbooks",
            "recursive": True,
            "file_extensions": [".md", ".markdown"]
        }
        
        md_provider = MarkdownRunbookProvider(md_config)
        
        # Search markdown runbooks
        md_results = await md_provider.search_runbooks(
            query="troubleshooting",
            limit=3
        )
        
        print(f"Found {len(md_results)} Markdown runbooks matching 'troubleshooting'")
        for result in md_results:
            print(f"  📝 {result['title']}")
            print(f"     Relevance: {result['relevance_score']:.2f}")
            
            # Get sections
            sections = await md_provider.get_runbook_sections(result['id'])
            print(f"     Sections: {len(sections)}")
            
    except Exception as e:
        print(f"  Markdown provider failed: {e}")
    
    # Web Provider Example
    print("\n--- Web Provider ---")
    try:
        web_config = {
            "base_urls": ["https://docs.example.com"],
            "timeout": 30,
            "max_pages": 10,
            "cache_ttl": 1800
        }
        
        web_provider = WebRunbookProvider(web_config)
        
        # Search web runbooks
        web_results = await web_provider.search_runbooks(
            query="error handling",
            limit=3
        )
        
        print(f"Found {len(web_results)} web runbooks matching 'error handling'")
        for result in web_results:
            print(f"  🌐 {result['title']}")
            print(f"     URL: {result['url']}")
            print(f"     Relevance: {result['relevance_score']:.2f}")
        
        # Close the web client
        await web_provider.close()
        
    except Exception as e:
        print(f"  Web provider failed: {e}")


async def runbook_runtime_example():
    """Example of runtime runbook text extraction."""
    print("\n=== Runtime Runbook Text Extraction ===")
    
    # This demonstrates the main use case: getting runbook text at runtime
    config = {
        "providers": {
            "local_runbooks": {
                "type": "markdown",
                "config": {
                    "runbook_directory": "/path/to/runbooks",
                    "recursive": True
                }
            }
        }
    }
    
    runbook_provider = UnifiedRunbookProvider(config)
    
    try:
        # Simulate an incident requiring runbook guidance
        incident_description = "Database connection failures in production"
        
        print(f"Incident: {incident_description}")
        print("\n--- Finding Relevant Runbooks ---")
        
        # Step 1: Find relevant runbooks
        relevant_runbooks = await runbook_provider.find_relevant_runbooks(
            error_context=incident_description,
            limit=3
        )
        
        if not relevant_runbooks:
            print("No relevant runbooks found")
            return
        
        print(f"Found {len(relevant_runbooks)} relevant runbooks")
        
        # Step 2: Get the actual runbook text for the AI agent
        for runbook in relevant_runbooks:
            print(f"\n--- Extracting: {runbook['title']} ---")
            
            try:
                # This is the key function: get runbook text at runtime
                runbook_text = await runbook_provider.get_runbook_text(
                    runbook_id=runbook['id']
                )
                
                print(f"✓ Successfully extracted {len(runbook_text)} characters")
                print(f"  Preview: {runbook_text[:150]}...")
                
                # Get structured sections for better context
                provider = runbook_provider._detect_provider(runbook['id'])
                if provider:
                    sections = await provider.get_runbook_sections(runbook['id'])
                    print(f"  Sections: {len(sections)}")
                    
                    # Find most relevant section
                    relevant_sections = await provider.find_relevant_sections(
                        runbook['id'],
                        incident_description
                    )
                    
                    if relevant_sections:
                        top_section = relevant_sections[0]
                        print(f"  Most relevant section: '{top_section['title']}'")
                        print(f"    Relevance: {top_section.get('relevance_score', 0):.2f}")
                        print(f"    Content preview: {top_section['content'][:100]}...")
                
                # This runbook_text would be passed to the AI agent
                print(f"  → This text would be provided to the AI agent for analysis")
                
            except Exception as e:
                print(f"✗ Failed to extract text: {e}")
        
        # Step 3: Get comprehensive context for AI agent
        print(f"\n--- Comprehensive Context for AI Agent ---")
        
        comprehensive_context = await runbook_provider.get_comprehensive_runbook_context(
            error_context=incident_description,
            limit=2
        )
        
        print("Context prepared for AI agent:")
        print(f"  Combined text length: {len(comprehensive_context['combined_text'])} characters")
        print(f"  Number of relevant sections: {len(comprehensive_context['sections'])}")
        print(f"  Summary: {comprehensive_context['summary']}")
        
        # The combined_text would be the main input to the AI agent
        ai_input_text = comprehensive_context['combined_text']
        print(f"\n  AI Agent Input Preview:")
        print(f"  {ai_input_text[:300]}...")
        print(f"  [... {len(ai_input_text) - 300} more characters ...]")
        
    except Exception as e:
        print(f"Error: {e}")


async def main():
    """Run all runbook integration examples."""
    print("Runbook Integration Examples")
    print("=" * 60)
    
    await unified_runbook_example()
    await individual_provider_examples()
    await runbook_runtime_example()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("\nSetup Instructions:")
    print("1. Create runbook directories:")
    print("   mkdir -p /path/to/pdf/runbooks")
    print("   mkdir -p /path/to/markdown/runbooks")
    print("   mkdir -p /path/to/docx/runbooks")
    print("\n2. Add some sample runbooks to test:")
    print("   - PDF files: troubleshooting guides, procedures")
    print("   - Markdown files: documentation, guides")
    print("   - Word documents: formal procedures")
    print("   - Web links: online documentation")
    print("\n3. Install optional dependencies:")
    print("   pip install PyPDF2 pdfplumber python-docx httpx beautifulsoup4")
    print("\n4. Update the file paths in the examples to match your setup")
    print("\nKey Runtime Function:")
    print("  runbook_text = await provider.get_runbook_text(runbook_id)")
    print("  # This text is then provided to the AI agent for analysis")


if __name__ == "__main__":
    asyncio.run(main())
