# Runbook Integration

This module provides comprehensive integration with various runbook sources for the OnCall AI Agent.

## Features

### Supported Runbook Types
- **PDF Files**: Extract text from PDF documents with OCR support
- **Markdown Files**: Parse Markdown with section structure and formatting
- **Word Documents**: Extract content from DOCX files with metadata
- **Web Links**: Crawl and extract content from documentation websites

### Unified Interface
- **Single provider** that can handle multiple runbook types simultaneously
- **Intelligent search** across all configured runbook sources
- **Relevance scoring** to find the most applicable guidance
- **Section extraction** for structured runbook content
- **Runtime text extraction** for AI agent processing

## Architecture

### Provider Hierarchy
```
RunbookProvider (Abstract Base)
├── PDFRunbookProvider
├── MarkdownRunbookProvider  
├── DocxRunbookProvider
├── WebRunbookProvider
└── UnifiedRunbookProvider (orchestrates all types)
```

### Key Functions
All providers implement these core functions:
- `get_runbook_text(runbook_id)` - **Main runtime function** for AI agent
- `search_runbooks(query)` - Find runbooks by content
- `list_runbooks()` - Get all available runbooks
- `get_runbook_sections(runbook_id)` - Get structured sections
- `find_relevant_sections(runbook_id, error_context)` - Find relevant sections

## Configuration

### Unified Provider (Recommended)

```yaml
integrations:
  runbook_provider:
    type: "unified"
    config:
      providers:
        pdf_runbooks:
          type: "pdf"
          config:
            runbook_directory: "/path/to/pdf/runbooks"
            recursive: true
            cache_enabled: true
            
        markdown_runbooks:
          type: "markdown"
          config:
            runbook_directory: "/path/to/markdown/runbooks"
            recursive: true
            file_extensions: [".md", ".markdown", ".txt"]
            
        web_runbooks:
          type: "web_link"
          config:
            base_urls: ["https://docs.company.com/runbooks"]
            timeout: 30
            max_pages: 50
```

### Individual Providers

```yaml
# PDF only
runbook_provider:
  type: "pdf"
  config:
    runbook_directory: "/path/to/runbooks"
    recursive: true

# Markdown only  
runbook_provider:
  type: "markdown"
  config:
    runbook_directory: "/path/to/runbooks"
    file_extensions: [".md", ".markdown"]

# Web only
runbook_provider:
  type: "web_link"
  config:
    base_urls: ["https://docs.example.com"]
```

## Usage Examples

### Runtime Text Extraction (Main Use Case)

```python
from oncall_agent.integrations.runbooks import UnifiedRunbookProvider

# Initialize provider
provider = UnifiedRunbookProvider(config)

# This is the key function for AI agent integration
async def get_runbook_guidance(incident_description: str) -> str:
    """Get runbook text for AI agent processing."""
    
    # Find relevant runbooks
    relevant_runbooks = await provider.find_relevant_runbooks(
        error_context=incident_description,
        limit=3
    )
    
    if not relevant_runbooks:
        return "No relevant runbooks found."
    
    # Get the actual runbook text
    all_text = []
    for runbook in relevant_runbooks:
        try:
            # This is the main runtime function
            text = await provider.get_runbook_text(runbook['id'])
            all_text.append(f"--- {runbook['title']} ---\n{text}")
        except Exception as e:
            print(f"Could not load {runbook['title']}: {e}")
    
    return "\n\n".join(all_text)

# Usage in incident resolution
incident = "Database connection timeout in production"
runbook_guidance = await get_runbook_guidance(incident)
# runbook_guidance now contains all relevant runbook text for the AI agent
```

### Individual Provider Usage

```python
# PDF runbooks
pdf_provider = PDFRunbookProvider(pdf_config)
pdf_text = await pdf_provider.get_runbook_text("database-troubleshooting.pdf")

# Markdown runbooks
md_provider = MarkdownRunbookProvider(md_config)
md_text = await md_provider.get_runbook_text("api-errors.md")

# Web runbooks
web_provider = WebRunbookProvider(web_config)
web_text = await web_provider.get_runbook_text("https://docs.company.com/db-issues")
```

### Search and Analysis

```python
# Search across all runbook types
results = await provider.search_runbooks(
    query="database connection error",
    limit=5
)

# Get comprehensive context
context = await provider.get_comprehensive_runbook_context(
    error_context="PostgreSQL connection timeout",
    limit=3
)

# The context['combined_text'] contains all relevant runbook content
ai_input = context['combined_text']
```

## Runbook Content Structure

### PDF Files
- **Extracted text** with page markers
- **Metadata** (title, author, creation date)
- **Section detection** based on text patterns
- **OCR support** for scanned documents (optional)

### Markdown Files
- **Full Markdown parsing** with formatting preservation
- **Header-based sections** (H1-H6)
- **Code block detection** and language identification
- **Link and image metadata**
- **Table content extraction**

### Word Documents
- **Paragraph and table extraction**
- **Style and formatting information**
- **Document metadata** (author, title, comments)
- **Header/footer content** (optional)
- **Section structure** based on heading styles

### Web Content
- **HTML parsing** with content extraction
- **Link discovery** and crawling
- **Metadata extraction** (title, description, Open Graph)
- **Content caching** with TTL
- **Multi-page site support**

## Content Processing

### Text Extraction
- **Encoding detection** and proper Unicode handling
- **Formatting preservation** or stripping (configurable)
- **Section boundaries** identified automatically
- **Metadata enrichment** with source information

### Search and Relevance
- **Keyword extraction** from search queries
- **Context-aware matching** with surrounding text
- **Relevance scoring** based on frequency and context
- **Cross-reference detection** between runbooks

### Caching
- **Text caching** for expensive operations (PDF extraction)
- **Web content caching** with TTL
- **File modification detection** for cache invalidation
- **Configurable cache settings** per provider

## Error Handling

### File-based Providers (PDF, Markdown, DOCX)
- **File not found** errors with clear messages
- **Permission errors** when accessing directories
- **Encoding errors** with fallback strategies
- **Corrupted file handling** with graceful skipping

### Web-based Providers
- **Network timeouts** with retry logic
- **HTTP errors** (404, 403, etc.) with appropriate handling
- **Content parsing errors** with fallback text extraction
- **Rate limiting** awareness and respect

### Provider Coordination
- **Individual provider failures** don't affect others
- **Graceful degradation** when some providers are unavailable
- **Error aggregation** across multiple sources
- **Health monitoring** for all configured providers

## Dependencies

### Required
```bash
# Core dependencies (included in base requirements)
pip install aiofiles httpx
```

### Optional (install as needed)
```bash
# PDF support
pip install PyPDF2 pdfplumber

# Word document support  
pip install python-docx

# Web content support
pip install httpx beautifulsoup4

# All runbook formats
pip install PyPDF2 pdfplumber python-docx httpx beautifulsoup4
```

## Best Practices

### Runbook Organization
```
runbooks/
├── infrastructure/
│   ├── database/
│   │   ├── connection-issues.md
│   │   ├── performance-tuning.pdf
│   │   └── backup-procedures.docx
│   ├── networking/
│   └── monitoring/
├── applications/
│   ├── api-services/
│   ├── frontend/
│   └── batch-jobs/
└── incident-response/
    ├── escalation-procedures.pdf
    ├── communication-templates.md
    └── post-mortem-process.docx
```

### Content Guidelines
- **Clear titles** and section headers
- **Error symptoms** and diagnostic steps
- **Step-by-step procedures** with commands
- **Prerequisites** and assumptions
- **Related links** and references
- **Regular updates** and version control

### Performance Optimization
- **Enable caching** for frequently accessed runbooks
- **Limit web crawling** to prevent excessive requests
- **Use specific directories** rather than scanning entire filesystems
- **Configure appropriate timeouts** for web requests

## Integration with AI Agent

The runbook integration is designed to provide context to the AI agent:

```python
# Main runtime workflow
async def get_incident_guidance(incident_description: str) -> str:
    # 1. Find relevant runbooks
    relevant_runbooks = await provider.find_relevant_runbooks(incident_description)
    
    # 2. Extract text content
    guidance_text = ""
    for runbook in relevant_runbooks:
        text = await provider.get_runbook_text(runbook['id'])  # Key function
        guidance_text += f"\n--- {runbook['title']} ---\n{text}"
    
    # 3. Return to AI agent for processing
    return guidance_text
```

## Limitations

1. **PDF OCR**: OCR support requires additional dependencies and processing time
2. **Web crawling**: Limited by site structure and robots.txt policies
3. **File formats**: Some complex document formats may not extract perfectly
4. **Large files**: Very large runbooks may impact performance
5. **Dynamic content**: Web pages with JavaScript-generated content may not be fully captured

## References

- [PyPDF2 Documentation](https://pypdf2.readthedocs.io/)
- [python-docx Documentation](https://python-docx.readthedocs.io/)
- [Beautiful Soup Documentation](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
- [HTTPX Documentation](https://www.python-httpx.org/)
