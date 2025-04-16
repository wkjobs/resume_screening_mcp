"""Main entry point for resume-screening-mcp package."""

import sys
from resume_screening_mcp.server import run_server
from resume_screening_mcp.cli import main as cli_main

if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        # Remove the "cli" argument and pass the rest to the CLI
        sys.argv.pop(1)
        sys.exit(cli_main())
    else:
        # Default to running the server
        run_server() 