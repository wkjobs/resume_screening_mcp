"""Core functionality for resume screening."""

import os
import json
import logging
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime
import openai
import warnings
import re

# Filter PDF warnings
warnings.filterwarnings("ignore", category=UserWarning, 
                       message=".*Advanced encoding /GBK-EUC-H not implemented yet.*")

# Configure logger
logger = logging.getLogger(__name__)

class ResumeProcessor:
    """Class for processing resume files."""
    
    def __init__(self, resume_dir: str, keywords: List[str]):
        self.resume_dir = Path(resume_dir)
        self.keywords = keywords
        self.results = []

    def process_directory(self) -> List[Dict[str, Any]]:
        """Process all resumes in the directory"""
        all_files = list(self.resume_dir.glob("**/*"))
        resume_files = [f for f in all_files if f.suffix.lower() in ['.pdf', '.docx', '.doc']]
        
        for resume_file in resume_files:
            content = self.extract_text(resume_file)
            if content:
                match_result = self.match_keywords(content, resume_file)
                if match_result["matched"]:
                    self.results.append(match_result)
        
        return self.results

    def extract_text(self, file_path: Path) -> str:
        """Extract text from PDF or DOCX files"""
        if file_path.suffix.lower() == '.pdf':
            return self.extract_from_pdf(file_path)
        elif file_path.suffix.lower() in ['.docx', '.doc']:
            return self.extract_from_docx(file_path)
        return ""

    def extract_from_pdf(self, file_path: Path) -> str:
        """Extract text from PDF file"""
        try:
            from PyPDF2 import PdfReader
            
            # Locally ignore PDF warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, message=".*Advanced encoding.*")
                warnings.filterwarnings("ignore", category=UserWarning, message=".*not implemented yet.*")
                
                # Try to read file with PyPDF2
                reader = PdfReader(str(file_path))
                text = ""
                for page in reader.pages:
                    text += page.extract_text() or ""
                
                # If extracted text is empty, try alternative method
                if not text.strip():
                    logger.warning(f"PyPDF2 couldn't extract text from {file_path}, trying pdfplumber")
                    try:
                        import pdfplumber
                        with pdfplumber.open(file_path) as pdf:
                            for page in pdf.pages:
                                text += page.extract_text() or ""
                    except ImportError:
                        logger.warning("pdfplumber not installed, can't use alternative method")
                    except Exception as e:
                        logger.warning(f"Error using pdfplumber: {e}")
                
                return text
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {e}")
            return ""

    def extract_from_docx(self, file_path: Path) -> str:
        """Extract text from DOCX file"""
        try:
            from docx import Document
            doc = Document(file_path)
            return " ".join([paragraph.text for paragraph in doc.paragraphs])
        except Exception as e:
            logger.error(f"Error processing DOCX {file_path}: {e}")
            return ""

    def match_keywords(self, content: str, file_path: Path) -> Dict[str, Any]:
        """Match keywords in the content"""
        content_lower = content.lower()
        matches = []
        
        for keyword in self.keywords:
            if keyword.lower() in content_lower:
                matches.append(keyword)
        
        return {
            "file_name": file_path.name,
            "file_path": str(file_path),
            "matched": len(matches) > 0,
            "matched_keywords": matches,
            "match_count": len(matches)
        }


# Helper functions
def get_resume_list(resume_dir: str) -> List[str]:
    """Get a list of resume files in the specified directory"""
    try:
        resume_dir_path = Path(resume_dir)
        
        if not resume_dir_path.exists():
            logger.error(f"Directory does not exist: {resume_dir_path.absolute()}")
            return []
        
        all_files = list(resume_dir_path.glob("**/*"))
        resume_files = [str(f) for f in all_files if f.suffix.lower() in ['.pdf', '.docx', '.doc']]
        
        if not resume_files:
            logger.error(f"No PDF or DOCX resume files found in '{resume_dir}'")
            return []
        
        return resume_files
    except Exception as e:
        logger.error(f"Error listing resumes: {str(e)}")
        return []


def get_resume_content(file_path: str) -> str:
    """Extract content from a resume file"""
    try:
        processor = ResumeProcessor("", [])  # Empty parameters as we just need extraction methods
        
        path = Path(file_path)
        
        if not path.exists():
            logger.error(f"File does not exist: {path.absolute()}")
            return ""
        
        content = processor.extract_text(path)
        if not content:
            logger.error(f"Could not extract text from '{file_path}'")
            return ""
        
        # Return a snippet if content is too long
        content_length = len(content)
        
        if content_length > 5000:
            return content[:5000] + "... (content truncated)"
        return content
    except Exception as e:
        logger.error(f"Error extracting content: {str(e)}")
        return ""


def query_llm(system_message: str, user_message: str, 
              model: str = None, temperature: float = 0.1) -> str:
    """Send a direct query to the LLM using the configured client"""
    try:
        # Use environment variable for model if not specified
        if model is None:
            model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        
        # Create default OpenAI client (will use OPENAI_API_KEY from env)
        client = openai.OpenAI()
        
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error querying LLM: {e}")
        return f"Error: {str(e)}"


def summarize_resume(file_path: str) -> Dict[str, Any]:
    """Summarize a resume using direct LLM call"""
    try:
        content = get_resume_content(file_path)
        if not content:
            return {
                "file_name": Path(file_path).name,
                "file_path": file_path,
                "error": "Could not extract content from file"
            }
        
        # Create system message
        system_message = """You are an expert HR assistant that helps with resume analysis.
You can summarize resumes to highlight key information by Chinese like education, experience, skills, and achievements."""
        
        # Create user message
        user_message = f"""
Please provide a concise summary of this resume:
{content}

Your summary should include:
1. Candidate's education background
2. Work experience (years, companies, roles)
3. Key skills and expertise
4. Any notable achievements or certifications

Format your response as a JSON:
{{
    "education": "...",
    "work_experience": "...",
    "skills": "...",
    "achievements": "...",
    "overall_summary": "..."
}}
"""
        
        # Get direct response from LLM
        response_text = query_llm(system_message, user_message)
        
        # Try to extract JSON from the response
        try:
            json_match = re.search(r'({[\s\S]*})', response_text)
            
            if json_match:
                summary = json.loads(json_match.group(1))
            else:
                # Create a basic summary if JSON parsing fails
                summary = {
                    "overall_summary": response_text,
                    "parsing_error": "Could not parse structured summary"
                }
            
            # Add file info
            summary["file_name"] = Path(file_path).name
            summary["file_path"] = file_path
            
            return summary
        except Exception as e:
            return {
                "file_name": Path(file_path).name, 
                "file_path": file_path,
                "overall_summary": response_text,
                "parsing_error": str(e)
            }
            
    except Exception as e:
        return {
            "file_name": Path(file_path).name,
            "file_path": file_path, 
            "error": str(e)
        }


def evaluate_resume_requirements(file_path: str, requirements: str) -> Dict[str, Any]:
    """Evaluate a resume against requirements using direct LLM call"""
    try:
        content = get_resume_content(file_path)
        if not content:
            return {
                "file_name": Path(file_path).name,
                "file_path": file_path,
                "matched": False,
                "error": "Could not extract content from file",
                "matched_requirements": requirements
            }
        
        # Log the requirements being evaluated
        logger.info(f"Evaluating resume {file_path} against requirements: {requirements}")
        
        # Create clearer system prompt
        system_message = """You are an expert HR assistant specializing in resume screening and evaluation.
Your task is to determine if a resume matches the EXACT requirements specified by the user.
IMPORTANT INSTRUCTIONS:
1. You must ONLY evaluate based on the requirements provided, not any default or assumed requirements.
2. Do NOT use "精通python" or "金融机构从业经验" as default requirements unless they are explicitly mentioned.
3. Focus on semantic understanding rather than exact keyword matching.
4. A resume is considered a match ONLY if it meets the specified requirements.
5. Be specific in your reasoning and provide clear explanations.
6. Answer in Chinese for better communication with the user."""
        
        # Create more direct user prompt
        user_message = f"""
我需要你评估这份简历是否满足以下筛选需求:

{requirements}

以下是简历内容:
{content}

请根据语义理解而非简单关键词匹配来评估。例如，如果要求是"有金融行业IT产品经理经验"，应查找简历中提到的银行、保险、证券等金融机构工作经历，以及IT项目或产品管理职位。

请按以下JSON格式提供评估结果:
{{
    "matched": true/false,  // 是否满足需求
    "reasoning": "详细的推理过程，解释为什么简历满足或不满足要求"
}}

请确保严格按照提供的需求进行评估，不要添加任何未提及的要求。
"""
        
        # Get LLM response
        logger.info("Sending evaluation request to LLM...")
        response_text = query_llm(system_message, user_message)
        
        # Extract JSON
        try:
            json_match = re.search(r'({[\s\S]*})', response_text)
            
            if json_match:
                # Parse JSON response
                evaluation = json.loads(json_match.group(1))
                
                # Add file info and requirements
                evaluation["file_name"] = Path(file_path).name
                evaluation["file_path"] = file_path
                evaluation["matched_requirements"] = requirements
                
                # Log evaluation result
                logger.info(f"Resume evaluation result: match={evaluation.get('matched')}, requirements={requirements}")
                return evaluation
            else:
                # Cannot extract JSON, do manual parsing
                logger.warning(f"Cannot extract JSON from LLM response, attempting manual parsing")
                matched = "match" in response_text.lower() and not "not match" in response_text.lower()
                evaluation = {
                    "matched": matched,
                    "matched_requirements": requirements,
                    "reasoning": response_text
                }
                
                # Add file info
                evaluation["file_name"] = Path(file_path).name
                evaluation["file_path"] = file_path
                
                return evaluation
        except Exception as e:
            logger.error(f"Error parsing LLM response for {file_path}: {e}", exc_info=True)
            logger.error(f"Original response: {response_text}")
            
            return {
                "file_name": Path(file_path).name,
                "file_path": file_path,
                "matched": False,
                "error": f"Error parsing response: {e}",
                "raw_response": response_text,
                "matched_requirements": requirements
            }
            
    except Exception as e:
        logger.error(f"Error evaluating resume {file_path}: {e}", exc_info=True)
        return {
            "file_name": Path(file_path).name,
            "file_path": file_path,
            "matched": False,
            "error": str(e),
            "matched_requirements": requirements
        } 