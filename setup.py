"""Setup script for resume-screening-mcp package."""

from setuptools import setup, find_packages

# Use the more modern pyproject.toml for configuration,
# this is just for compatibility
setup(
    name="resume_screening_mcp",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "resume-screening=resume_screening_mcp.cli:main",
            "resume-screening-server=resume_screening_mcp.server:run_server",
        ],
    },
) 