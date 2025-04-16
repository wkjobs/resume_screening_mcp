"""Command-line interface for resume screening."""

import os
import argparse
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
from dotenv import load_dotenv

from resume_screening_mcp.core import (
    get_resume_list,
    summarize_resume,
    evaluate_resume_requirements
)

# Load environment variables from .env file
load_dotenv()

def setup_logging(log_dir: str = "logs") -> str:
    """Setup logging configuration and return log file path."""
    # Create log directory
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(exist_ok=True)
    
    # Create log file with timestamp
    log_file = log_dir_path / f"resume_screening_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Set log level from environment variable or default to INFO
    log_level_str = os.getenv("LOG_LEVEL", "INFO")
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    # Print log file path
    print(f"Log file path: {log_file.absolute()}")
    
    return str(log_file)

def process_resumes(resume_dir: str, report_dir: str, requirements: str) -> Dict[str, Any]:
    """Process resumes against requirements and generate reports."""
    logger = logging.getLogger(__name__)
    
    try:
        # Create absolute paths
        resume_dir_path = Path(resume_dir)
        report_dir_path = Path(report_dir)
        
        logger.info(f"Using resume directory: {resume_dir_path.absolute()}")
        logger.info(f"Using report directory: {report_dir_path.absolute()}")
        logger.info(f"Using requirements: {requirements}")
        
        # Ensure report directory exists
        report_dir_path.mkdir(exist_ok=True, parents=True)
        
        # Check if resume directory exists
        if not resume_dir_path.exists():
            logger.error(f"Resume directory does not exist: {resume_dir_path.absolute()}")
            return {
                "status": "error",
                "message": f"Resume directory '{resume_dir}' does not exist",
                "results": []
            }
        
        # Get resume list
        logger.info("Getting resume list...")
        resume_files = get_resume_list(resume_dir)
        logger.info(f"Found {len(resume_files)} resume files")
        
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
            summary_file = report_dir_path / f"resume_summaries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
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
            output_file = report_dir_path / f"resume_screening_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Convert to pandas DataFrame
            df = pd.DataFrame(results)
            
            # Add original query and extracted requirements to DataFrame
            df["original_query"] = requirements
            df["extracted_requirements"] = requirements
            
            # Save to Excel
            excel_file = f"{output_file}.xlsx"
            df.to_excel(excel_file, index=False)
            
            # Generate text report
            text_report = f"""Resume Screening Report (AI Semantic Matching)
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

原始查询: {requirements}
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
        logger.error(f"Error in process_resumes: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e),
            "results": []
        }

def main():
    """Main entry point for the command line interface."""
    # Get defaults from environment variables
    default_resume_dir = os.getenv("RESUME_DIR", "resumes")
    default_report_dir = os.getenv("REPORT_DIR", "reports")
    
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Resume Screening Tool')
    parser.add_argument('--resume-dir', type=str, default=default_resume_dir, 
                        help=f'Directory containing resume files (default: {default_resume_dir})')
    parser.add_argument('--report-dir', type=str, default=default_report_dir, 
                        help=f'Directory for saving reports (default: {default_report_dir})')
    parser.add_argument('--requirements', type=str, 
                        default="筛选精通Python且有金融机构从业经验的简历",
                        help='Requirements for screening resumes')
    parser.add_argument('--log-dir', type=str, default="logs", 
                        help='Directory for storing log files')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    log_file = setup_logging(args.log_dir)
    
    # Print banner
    print("==================================================")
    print("         Resume Screening Tool                    ")
    print("==================================================")
    print(f"Resume Directory: {args.resume_dir}")
    print(f"Report Directory: {args.report_dir}")
    print(f"Requirements: {args.requirements}")
    print(f"Log File: {log_file}")
    print("==================================================")
    
    # Process resumes
    result = process_resumes(
        resume_dir=args.resume_dir,
        report_dir=args.report_dir,
        requirements=args.requirements
    )
    
    # Print result
    if result["status"] == "success":
        print("\nProcessing complete!")
        print(f"Found {len([r for r in result['results'] if r.get('matched', False)])} matching resumes.")
        print(f"Excel report: {result.get('excel_report')}")
        print(f"Text report: {result.get('text_report')}")
    else:
        print(f"\nError: {result.get('message')}")
    
    return 0

if __name__ == "__main__":
    main() 