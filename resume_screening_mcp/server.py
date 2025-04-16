"""MCP server implementation for resume screening."""

import os
import logging
import json
from typing import Dict, Any
from pathlib import Path
from dotenv import load_dotenv
from fastmcp import FastMCP

from resume_screening_mcp.cli import setup_logging
from resume_screening_mcp.core import (
    get_resume_list,
    summarize_resume,
    evaluate_resume_requirements
)

# Load environment variables
load_dotenv()

# Configure logger
logger = logging.getLogger(__name__)

async def process_resumes_mcp(request) -> Dict[str, Any]:
    """MCP Server tool to process resumes"""
    try:
        logger.info(f"Received request: {request}")
        
        # Get the user's original prompt
        user_prompt = ""
        
        # 1. First try to get prompt from the request object itself
        if hasattr(request, 'prompt') and request.prompt:
            user_prompt = request.prompt
            logger.info(f"Got user prompt from request.prompt: {user_prompt}")
        elif hasattr(request, 'text') and request.text:
            user_prompt = request.text
            logger.info(f"Got user prompt from request.text: {user_prompt}")
        elif hasattr(request, 'content') and request.content:
            user_prompt = request.content
            logger.info(f"Got user prompt from request.content: {user_prompt}")
        elif hasattr(request, 'message') and request.message:
            user_prompt = request.message
            logger.info(f"Got user prompt from request.message: {user_prompt}")
        
        # 2. Extract args parameter
        if hasattr(request, 'args'):
            # If request has args attribute, use it directly
            args = request.args or {}
        elif isinstance(request, dict):
            # If request is a dictionary, use it directly
            args = request
        elif isinstance(request, str):
            # If request is a string, it might be the original prompt or JSON
            logger.info(f"Request is a string: {request}")
            
            # If we don't have a prompt yet, use this string as prompt
            if not user_prompt:
                user_prompt = request
                logger.info(f"Using request string as prompt: {user_prompt}")
            
            # Try to parse as JSON
            try:
                args = json.loads(request)
                if not isinstance(args, dict):
                    args = {}
            except Exception as e:
                logger.warning(f"Could not parse request string as JSON: {e}")
                args = {}
        else:
            # Other cases, use empty dict
            logger.warning(f"Unknown request type: {type(request)}")
            args = {}
        
        logger.info(f"Parsed parameters: {args}")
        
        # 3. If we still don't have a prompt, try to get from args from various possible fields
        if not user_prompt:
            # Try to get prompt from multiple possible fields in args
            for field in ['prompt', 'query', 'text', 'content', 'message', 'user_query', 'instruction']:
                if field in args and args[field] and isinstance(args[field], str):
                    user_prompt = args[field]
                    logger.info(f"Got user prompt from args['{field}']: {user_prompt}")
                    break
        
        # Log the final extracted prompt
        if user_prompt:
            logger.info(f"Successfully extracted user prompt: {user_prompt}")
        else:
            logger.warning("Could not extract user prompt")
            # If we completely failed to extract a prompt, use default text
            user_prompt = "筛选精通Python且有金融机构从业经验的简历"
            logger.info(f"Using default prompt: {user_prompt}")
        
        # Extract screening requirements - using the entire prompt directly
        requirements = user_prompt.strip()
        
        # Get other parameters, using default values as fallback
        # Get defaults from environment variables
        default_resume_dir = os.getenv("RESUME_DIR", "resumes")
        default_report_dir = os.getenv("REPORT_DIR", "reports")
        
        resume_dir = args.get("resume_dir", default_resume_dir)
        report_dir = args.get("report_dir", default_report_dir)
        
        # Log prompt and directory info
        logger.info(f"Will use these screening requirements: {requirements}")
        logger.info(f"Using resume directory: {resume_dir}")
        logger.info(f"Using report directory: {report_dir}")
        
        # Convert relative paths to absolute paths
        resume_abs_path = Path(resume_dir).absolute()
        report_abs_path = Path(report_dir).absolute()
        
        logger.info(f"Resume directory absolute path: {resume_abs_path}")
        logger.info(f"Report directory absolute path: {report_abs_path}")
        
        # Create report directory if it doesn't exist
        report_path = Path(report_dir)
        if not report_path.exists():
            logger.info(f"Creating report directory: {report_path.absolute()}")
            report_path.mkdir(parents=True, exist_ok=True)
        
        # Check if resume directory exists
        resume_dir_path = Path(resume_dir)
        if not resume_dir_path.exists():
            logger.error(f"Resume directory does not exist: {resume_dir_path.absolute()}")
            return {
                "status": "error",
                "message": f"Resume directory '{resume_dir}' does not exist",
                "results": []
            }
        
        # Get resume list
        logger.info(f"Starting to get resume list...")
        resume_files = get_resume_list(resume_dir)
        logger.info(f"Got resume list length: {len(resume_files)}")
        
        if not resume_files:
            return {
                "status": "error",
                "message": f"No resume files found in '{resume_dir}'",
                "results": []
            }
        
        # Step 1: First summarize each resume
        logger.info("===== SUMMARIZING RESUMES =====")
        summaries = []
        for resume_file in resume_files:
            logger.info(f"Summarizing resume: {resume_file}")
            summary = summarize_resume(resume_file)
            summaries.append(summary)
            
            # Print a preview of the summary to console
            logger.info(f"\n===== RESUME SUMMARY: {summary.get('file_name')} =====")
            if "error" in summary:
                logger.error(f"Error: {summary.get('error')}")
            else:
                for key, value in summary.items():
                    if key not in ["file_name", "file_path", "parsing_error"]:
                        if isinstance(value, str) and len(value) > 200:
                            logger.info(f"{key}: {value[:200]}...")
                        else:
                            logger.info(f"{key}: {value}")
            logger.info("=" * 50)
        
        # Save summaries to file
        if summaries:
            summary_file = report_path / f"resume_summaries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(summaries, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved resume summaries to: {summary_file}")
        
        # Step 2: Process each resume for matching requirements
        logger.info("\n===== MATCHING RESUMES TO REQUIREMENTS =====")
        results = []
        for resume_file in resume_files:
            logger.info(f"Processing resume: {resume_file}")
            evaluation = evaluate_resume_requirements(resume_file, requirements)
            results.append(evaluation)
        
        # Generate report
        if results:
            output_file = report_path / f"resume_screening_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Convert to pandas DataFrame
            import pandas as pd
            df = pd.DataFrame(results)
            
            # Add original query and extracted requirements to DataFrame
            df["original_query"] = user_prompt
            df["extracted_requirements"] = requirements
            
            # Save to Excel
            excel_file = f"{output_file}.xlsx"
            df.to_excel(excel_file, index=False)
            
            # Generate text report
            text_report = f"""Resume Screening Report (AI Semantic Matching)
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

原始查询: {user_prompt}
提取的筛选要求: {requirements}

Summary:
- Total resumes processed: {len(results)}
- Total matches found: {len([r for r in results if r.get('matched', False)])}

Matched Resumes:
"""
            
            for result in results:
                if result.get('matched', False):
                    # Find the corresponding summary
                    resume_summary = next((s for s in summaries if s.get('file_path') == result.get('file_path')), None)
                    
                    text_report += f"\nFile: {result['file_name']}"
                    
                    text_report += f"\nMatched Requirements: {result.get('matched_requirements', requirements)}"
                    
                    # Add summary information if available
                    if resume_summary and 'error' not in resume_summary:
                        text_report += "\n\nResume Summary:"
                        if 'education' in resume_summary:
                            text_report += f"\nEducation: {resume_summary.get('education', 'Not specified')}"
                        if 'work_experience' in resume_summary:
                            text_report += f"\nWork Experience: {resume_summary.get('work_experience', 'Not specified')}"
                        if 'skills' in resume_summary:
                            text_report += f"\nSkills: {resume_summary.get('skills', 'Not specified')}"
                        if 'achievements' in resume_summary:
                            text_report += f"\nAchievements: {resume_summary.get('achievements', 'Not specified')}"
                        if 'overall_summary' in resume_summary:
                            text_report += f"\nOverall: {resume_summary.get('overall_summary', '')}"
                    
                    text_report += f"\n\nReasoning: {result.get('reasoning', 'Not provided')[:200]}...\n"
                    text_report += "\n" + "-"*50 + "\n"
            
            # Save text report
            text_file = f"{output_file}.txt"
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(text_report)
            
            logger.info(f"Processing complete! Found {len([r for r in results if r.get('matched', False)])} matching resumes.")
            logger.info(f"Excel report saved to: {excel_file}")
            logger.info(f"Text report saved to: {text_file}")
            
            return {
                "status": "success",
                "message": f"Processing complete! Found {len([r for r in results if r.get('matched', False)])} matching resumes.",
                "excel_report": str(excel_file),
                "text_report": str(text_file),
                "results": results,
                "summaries": summaries
            }
        else:
            return {
                "status": "warning",
                "message": "No results generated.",
                "results": []
            }
    except Exception as e:
        logger.error(f"Error in process_resumes_mcp: {e}")
        return {
            "status": "error",
            "message": str(e),
            "results": []
        }

def create_app() -> FastMCP:
    """Create and configure the MCP server application."""
    # Setup logging
    setup_logging()
    
    # Initialize the MCP Server
    app = FastMCP()
    
    # Register the tool
    @app.tool("resume_screening", 
             description="Process resumes using LLM to summarize and match against requirements")
    async def resume_screening_tool(request) -> Dict[str, Any]:
        logger.info(f"Tool was called, received request: {type(request)}")
        # Check request type and handle appropriately
        try:
            return await process_resumes_mcp(request)
        except Exception as e:
            logger.error(f"Tool call error: {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"Error processing request: {str(e)}",
                "results": []
            }
    
    return app

def run_server():
    """Run the MCP server."""
    # Get defaults from environment variables
    default_resume_dir = os.getenv("RESUME_DIR", "resumes")
    default_report_dir = os.getenv("REPORT_DIR", "reports")
    
    app = create_app()
    
    # Print banner
    logger.info("========================")
    logger.info("Resume Screening MCP Server Starting")
    logger.info(f"Resume Directory: {default_resume_dir}")
    logger.info(f"Report Directory: {default_report_dir}")
    logger.info("========================")
    
    # Run the server
    app.run()

if __name__ == "__main__":
    run_server()