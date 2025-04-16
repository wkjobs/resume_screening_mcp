# Resume Screening MCP Service

A service for screening resumes using Large Language Models. This service evaluates resumes against specific requirements using semantic understanding rather than simple keyword matching.

## Installation

### Using UV (recommended)

```bash
uv pip install .
```

### Using pip

```bash
pip install .
```

## Configuration

The application uses environment variables for configuration. Create a `.env` file in the root directory with the following variables:

```
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini
```

## Usage

### Command Line Interface

```bash
# Process resumes in a specific directory
resume-screening --resume-dir=./my_resumes --report-dir=./reports

# Or using environment variables
RESUME_DIR=./my_resumes REPORT_DIR=./reports resume-screening
```

### As MCP Service

```python
from fastmcp import FastMCPClient

client = FastMCPClient()
response = await client.call("resume_screening", {
    "prompt": "筛选精通Python且有金融机构从业经验的简历",
    "resume_dir": "./resumes",
    "report_dir": "./reports"
})
print(response)
```

## Directory Structure

- `resume_screening_mcp/` - Main package directory
  - `__init__.py` - Package initialization
  - `core.py` - Core resume processing functionality
  - `cli.py` - Command line interface
  - `server.py` - MCP server implementation

## Development

```bash
# Install development dependencies
uv pip install -e ".[dev]"

# Run tests
pytest
```

## License

MIT 