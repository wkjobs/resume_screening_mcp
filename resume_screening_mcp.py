import os
from typing import List, Dict, Any
from pathlib import Path
import json
from datetime import datetime
import asyncio
import openai
from fastmcp import FastMCP
import argparse  # 导入argparse模块
import logging  # 导入logging模块
import warnings  # 导入warnings模块
import re  # 导入正则表达式模块

# 过滤PDF警告
warnings.filterwarnings("ignore", category=UserWarning, message=".*Advanced encoding /GBK-EUC-H not implemented yet.*")

# 配置日志记录
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)  # 创建日志目录
log_file = LOG_DIR / f"resume_screening_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# 配置日志格式和级别
logging.basicConfig(
    level=logging.INFO,  # 将日志级别从DEBUG改为INFO，屏蔽所有DEBUG日志
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()  # 同时输出到控制台
    ]
)
logger = logging.getLogger(__name__)

# 打印日志文件路径
print(f"日志文件路径: {log_file.absolute()}")

# 添加命令行参数解析
parser = argparse.ArgumentParser(description='Resume Screening MCP Server')
parser.add_argument('--resume-dir', type=str, default='resumes', help='Directory containing resume files')
parser.add_argument('--report-dir', type=str, default='.', help='Directory for saving reports')
args = parser.parse_args()

# 设置默认目录（可以被MCP请求中的参数覆盖）
DEFAULT_RESUME_DIR = args.resume_dir
DEFAULT_REPORT_DIR = args.report_dir

# Azure OpenAI configuration
AZURE_API_KEY = "7x96wF5ZdlZLxV57rJrKBqSSNffzsC4B5dMLd8KzHrAbkVdSXtbSJQQJ99AKACHYHv6XJ3w3AAABACOGtSC0"
AZURE_ENDPOINT = "https://ouyeelf-gpt-test1.openai.azure.com/"
AZURE_DEPLOYMENT = "gpt-4o-mini"

# Configure OpenAI client
client = openai.AzureOpenAI(
    api_key=AZURE_API_KEY,
    api_version="2023-05-15",
    azure_endpoint=AZURE_ENDPOINT
)

class ResumeProcessor:
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
            import warnings
            
            # 局部忽略PDF警告
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, message=".*Advanced encoding.*")
                warnings.filterwarnings("ignore", category=UserWarning, message=".*not implemented yet.*")
                
                # 尝试使用PyPDF2读取文件
                reader = PdfReader(str(file_path))
                text = ""
                for page in reader.pages:
                    text += page.extract_text() or ""
                
                # 如果提取的文本为空，尝试使用备选方法
                if not text.strip():
                    logger.warning(f"PyPDF2无法提取文本从 {file_path}，尝试使用pdfplumber")
                    try:
                        import pdfplumber
                        with pdfplumber.open(file_path) as pdf:
                            for page in pdf.pages:
                                text += page.extract_text() or ""
                    except ImportError:
                        logger.warning("pdfplumber未安装，无法使用备选方法")
                    except Exception as e:
                        logger.warning(f"使用pdfplumber时出错: {e}")
                
                return text
        except Exception as e:
            logger.error(f"处理PDF {file_path} 时出错: {e}")
            return ""

    def extract_from_docx(self, file_path: Path) -> str:
        """Extract text from DOCX file"""
        try:
            from docx import Document
            doc = Document(file_path)
            return " ".join([paragraph.text for paragraph in doc.paragraphs])
        except Exception as e:
            print(f"Error processing DOCX {file_path}: {e}")
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
        #logger.debug(f"扫描简历目录: {resume_dir_path.absolute()}")
        
        if not resume_dir_path.exists():
            logger.error(f"目录不存在: {resume_dir_path.absolute()}")
            return []
        
        all_files = list(resume_dir_path.glob("**/*"))
        #logger.debug(f"目录中找到的所有文件数: {len(all_files)}")
        #for file in all_files:
            #logger.debug(f"找到文件: {file.absolute()}")
        
        resume_files = [str(f) for f in all_files if f.suffix.lower() in ['.pdf', '.docx', '.doc']]
        
        #logger.debug(f"找到的简历文件数: {len(resume_files)}")
        #for resume in resume_files:
            #logger.debug(f"简历文件: {resume}")
        
        if not resume_files:
            logger.error(f"在 '{resume_dir}' 中没有找到PDF或DOCX简历文件")
            return []
        
        return resume_files
    except Exception as e:
        logger.error(f"列出简历时出错: {str(e)}")
        return []

def get_resume_content(file_path: str) -> str:
    """Extract content from a resume file"""
    try:
        #logger.debug(f"尝试提取简历内容: {file_path}")
        processor = ResumeProcessor("", [])  # Empty parameters as we just need extraction methods
        
        path = Path(file_path)
        #logger.debug(f"简历路径: {path.absolute()}")
        
        if not path.exists():
            logger.error(f"文件不存在: {path.absolute()}")
            return ""
        
        content = processor.extract_text(path)
        if not content:
            logger.error(f"无法从 '{file_path}' 提取文本")
            return ""
        
        # Return a snippet if content is too long
        content_length = len(content)
        #logger.debug(f"提取到的内容长度: {content_length} 字符")
        
        if content_length > 5000:
            return content[:5000] + "... (content truncated)"
        return content
    except Exception as e:
        logger.error(f"提取内容时出错: {str(e)}")
        return ""

def query_llm(system_message: str, user_message: str) -> str:
    """Send a direct query to the LLM"""
    try:
        response = client.chat.completions.create(
            model=AZURE_DEPLOYMENT,
            temperature=0.1,
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
            import re
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
        
        # 记录评估的要求
        logger.info(f"评估简历 {file_path} 是否满足要求: {requirements}")
        
        # 创建更清晰明确的系统提示
        system_message = """You are an expert HR assistant specializing in resume screening and evaluation.
Your task is to determine if a resume matches the EXACT requirements specified by the user.
IMPORTANT INSTRUCTIONS:
1. You must ONLY evaluate based on the requirements provided, not any default or assumed requirements.
2. Do NOT use "精通python" or "金融机构从业经验" as default requirements unless they are explicitly mentioned.
3. Focus on semantic understanding rather than exact keyword matching.
4. A resume is considered a match ONLY if it meets the specified requirements.
5. Be specific in your reasoning and provide clear explanations.
6. Answer in Chinese for better communication with the user."""
        
        # 创建更直接的用户提示
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
        
        # 获取LLM回应
        logger.info("向LLM发送评估请求...")
        response_text = query_llm(system_message, user_message)
        #logger.debug(f"LLM返回的原始响应: {response_text[:200]}...")
        
        # 提取JSON
        try:
            import re
            json_match = re.search(r'({[\s\S]*})', response_text)
            
            if json_match:
                # 解析JSON响应
                evaluation = json.loads(json_match.group(1))
                
                # 添加文件信息和需求
                evaluation["file_name"] = Path(file_path).name
                evaluation["file_path"] = file_path
                evaluation["matched_requirements"] = requirements
                
                # 记录评估结果
                logger.info(f"简历评估结果: 匹配={evaluation.get('matched')}, 需求={requirements}")
                return evaluation
            else:
                # 无法提取JSON，进行手动解析
                logger.warning(f"无法从LLM响应中提取JSON，尝试手动解析")
                matched = "match" in response_text.lower() and not "not match" in response_text.lower()
                evaluation = {
                    "matched": matched,
                    "matched_requirements": requirements,
                    "reasoning": response_text
                }
                
                # 添加文件信息
                evaluation["file_name"] = Path(file_path).name
                evaluation["file_path"] = file_path
                
                return evaluation
        except Exception as e:
            logger.error(f"解析LLM响应时出错 {file_path}: {e}", exc_info=True)
            logger.error(f"原始响应: {response_text}")
            
            return {
                "file_name": Path(file_path).name,
                "file_path": file_path,
                "matched": False,
                "error": f"解析响应出错: {e}",
                "raw_response": response_text,
                "matched_requirements": requirements
            }
            
    except Exception as e:
        logger.error(f"简历评估出错 {file_path}: {e}", exc_info=True)
        return {
            "file_name": Path(file_path).name,
            "file_path": file_path,
            "matched": False,
            "error": str(e),
            "matched_requirements": requirements
        }

async def process_resumes_mcp(request) -> Dict[str, Any]:
    """MCP Server tool to process resumes"""
    try:
        logger.info(f"Received request: {request}")
        
        # 获取用户的原始提示词
        user_prompt = ""
        
        # 1. 首先尝试从request对象本身获取提示词
        if hasattr(request, 'prompt') and request.prompt:
            user_prompt = request.prompt
            logger.info(f"从request.prompt获取到用户提示词: {user_prompt}")
        elif hasattr(request, 'text') and request.text:
            user_prompt = request.text
            logger.info(f"从request.text获取到用户提示词: {user_prompt}")
        elif hasattr(request, 'content') and request.content:
            user_prompt = request.content
            logger.info(f"从request.content获取到用户提示词: {user_prompt}")
        elif hasattr(request, 'message') and request.message:
            user_prompt = request.message
            logger.info(f"从request.message获取到用户提示词: {user_prompt}")
        
        # 2. 提取args参数
        if hasattr(request, 'args'):
            # 如果request有args属性，直接使用
            args = request.args or {}
        elif isinstance(request, dict):
            # 如果request是一个字典，直接使用
            args = request
        elif isinstance(request, str):
            # 如果request是字符串，可能是原始提示词或JSON
            logger.info(f"Request is a string: {request}")
            
            # 如果还没有提示词，使用这个字符串作为提示词
            if not user_prompt:
                user_prompt = request
                logger.info(f"将request字符串作为提示词: {user_prompt}")
            
            # 尝试解析为JSON
            try:
                args = json.loads(request)
                if not isinstance(args, dict):
                    args = {}
            except Exception as e:
                logger.warning(f"无法解析请求字符串为JSON: {e}")
                args = {}
        else:
            # 其他情况，使用空字典
            logger.warning(f"未知的请求类型: {type(request)}")
            args = {}
        
        logger.info(f"解析后的参数: {args}")
        
        # 3. 如果还没有提示词，尝试从args中的各种可能字段获取
        if not user_prompt:
            # 尝试从args中的多个可能字段获取提示词
            for field in ['prompt', 'query', 'text', 'content', 'message', 'user_query', 'instruction']:
                if field in args and args[field] and isinstance(args[field], str):
                    user_prompt = args[field]
                    logger.info(f"从args['{field}']获取到用户提示词: {user_prompt}")
                    break
        
        # 记录最终提取到的提示词
        if user_prompt:
            logger.info(f"成功提取到用户提示词: {user_prompt}")
        else:
            logger.warning("未能提取到用户提示词")
            # 如果完全没有提取到提示词，使用默认文本
            user_prompt = "筛选精通Python且有金融机构从业经验的简历"
            logger.info(f"使用默认提示词: {user_prompt}")
        
        # 提取筛选需求 - 这里不再拆分为数组，而是直接使用整个提示词
        requirements = user_prompt.strip()
        
        # 获取其他参数，使用默认值作为后备
        resume_dir = args.get("resume_dir", DEFAULT_RESUME_DIR)
        report_dir = args.get("report_dir", DEFAULT_REPORT_DIR)
        
        # 记录提示词和目录信息
        logger.info(f"将使用以下筛选需求: {requirements}")
        logger.info(f"使用简历目录: {resume_dir}")
        logger.info(f"使用报告目录: {report_dir}")
        
        # 将相对路径转换为绝对路径
        resume_abs_path = Path(resume_dir).absolute()
        report_abs_path = Path(report_dir).absolute()
        
        logger.info(f"简历目录绝对路径: {resume_abs_path}")
        logger.info(f"报告目录绝对路径: {report_abs_path}")
        
        # Create report directory if it doesn't exist
        report_path = Path(report_dir)
        if not report_path.exists():
            logger.info(f"创建报告目录: {report_path.absolute()}")
            report_path.mkdir(parents=True, exist_ok=True)
        
        # 检查简历目录是否存在
        resume_dir_path = Path(resume_dir)
        if not resume_dir_path.exists():
            logger.error(f"简历目录不存在: {resume_dir_path.absolute()}")
            return {
                "status": "error",
                "message": f"简历目录 '{resume_dir}' 不存在",
                "results": []
            }
        
        # Get resume list
        logger.info(f"开始获取简历列表...")
        resume_files = get_resume_list(resume_dir)
        logger.info(f"获取的简历列表长度: {len(resume_files)}")
        
        if not resume_files:
            return {
                "status": "error",
                "message": f"在 '{resume_dir}' 中未找到简历文件",
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
            
            # 添加原始查询和提取的要求到DataFrame
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

# Initialize the MCP Server
app = FastMCP()

# Register the tool
@app.tool("resume_screening", 
         description="Process resumes using LLM to summarize and match against requirements")
async def resume_screening_tool(request) -> Dict[str, Any]:
    logger.info(f"工具被调用，收到请求: {type(request)}")
    # 检查request类型并适当处理
    try:
        return await process_resumes_mcp(request)
    except Exception as e:
        logger.error(f"工具调用错误: {e}", exc_info=True)
        return {
            "status": "error",
            "message": f"处理请求时出错: {str(e)}",
            "results": []
        }

# Run the server
if __name__ == "__main__":
    logger.info("========================")
    logger.info("简历筛选MCP服务器启动")
    logger.info(f"简历目录: {DEFAULT_RESUME_DIR}")
    logger.info(f"报告目录: {DEFAULT_REPORT_DIR}")
    logger.info(f"日志文件: {log_file.absolute()}")
    logger.info("========================")
    app.run() 