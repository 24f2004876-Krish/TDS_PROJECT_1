# /// script
# requires-python = ">=3.8"
# dependencies = [
#   "fastapi[standard]",
#   "uvicorn",
#   "requests",
#   "openai>=1.0.0",
#   "pydantic",
#   "httpx>=0.24.0",
#   "python-dotenv"
# ]
# ///

import requests
import os
import base64
import time
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize clients - will be done lazily to avoid initialization errors
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_USERNAME = os.getenv("GITHUB_USERNAME", "24f2004876-Krish")
SECRET = os.getenv("SECRET")

def get_openai_client():
    """Lazy initialization of OpenAI client"""
    from openai import OpenAI
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Pydantic models for request validation
class TaskRequest(BaseModel):
    email: str
    secret: str
    task: str
    round: int
    nonce: str
    brief: str
    checks: List[str]
    evaluation_url: str
    attachments: Optional[List[Dict[str, str]]] = []

class EvaluationResponse(BaseModel):
    email: str
    task: str
    round: int
    nonce: str
    repo_url: str
    commit_sha: str
    pages_url: str

app = FastAPI()

def validate_secret(secret: str) -> bool:
    """Validate the secret against environment variable"""
    return secret == SECRET

def create_github_repo(repo_name: str) -> Dict:
    """Create a new GitHub repository with MIT license"""
    payload = {
        "name": repo_name,
        "private": False,
        "auto_init": True,
        "license_template": "mit"
    }
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }
    
    response = requests.post(
        "https://api.github.com/user/repos",
        headers=headers,
        json=payload
    )
    
    if response.status_code == 201:
        return response.json()
    elif response.status_code == 422:
        # Repo might already exist, try to get it
        return get_repo_info(repo_name)
    else:
        raise Exception(f"Failed to create repo: {response.status_code}, {response.text}")

def get_repo_info(repo_name: str) -> Dict:
    """Get information about an existing repository"""
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json"
    }
    
    response = requests.get(
        f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo_name}",
        headers=headers
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to get repo info: {response.status_code}, {response.text}")

def enable_github_pages(repo_name: str):
    """Enable GitHub Pages for the repository"""
    # Wait a bit for repo to be fully initialized
    time.sleep(2)
    
    payload = {
        "build_type": "legacy",
        "source": {
            "branch": "main",
            "path": "/"
        }
    }
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }
    
    response = requests.post(
        f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo_name}/pages",
        headers=headers,
        json=payload
    )
    
    if response.status_code not in [201, 409]:  # 409 means already exists
        raise Exception(f"Failed to enable GitHub Pages: {response.status_code}, {response.text}")

def get_file_content(repo_name: str, file_path: str) -> Optional[str]:
    """Get the content of an existing file from the repository"""
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json"
    }
    
    response = requests.get(
        f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo_name}/contents/{file_path}",
        headers=headers
    )
    
    if response.status_code == 200:
        content_b64 = response.json().get("content")
        if content_b64:
            return base64.b64decode(content_b64).decode('utf-8')
    return None

def get_file_sha(repo_name: str, file_path: str) -> Optional[str]:
    """Get the SHA of an existing file in the repository"""
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json"
    }
    
    response = requests.get(
        f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo_name}/contents/{file_path}",
        headers=headers
    )
    
    if response.status_code == 200:
        return response.json().get("sha")
    return None

def get_sha_of_latest_commit(repo_name: str, branch: str = "main") -> str:
    """Get SHA of the latest commit on specified branch"""
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json"
    }
    
    response = requests.get(
        f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo_name}/commits/{branch}",
        headers=headers
    )
    
    if response.status_code != 200:
        raise Exception(f"Failed to get latest commit: {response.status_code}, {response.text}")
    return response.json().get("sha")


def get_file_sha(repo_name: str, file_path: str) -> Optional[str]:
    """Get the SHA of an existing file in the repository"""
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json"
    }
    
    response = requests.get(
        f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo_name}/contents/{file_path}",
        headers=headers
    )
    
    if response.status_code == 200:
        return response.json().get("sha")
    return None

def push_files_to_repo(repo_name: str, files: List[Dict], is_update: bool = False):
    """Push files to the GitHub repository"""
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }
    
    for file in files:
        file_name = file.get("name")
        file_content = file.get("content")
        
        # Encode content to base64
        if isinstance(file_content, bytes):
            file_content_b64 = base64.b64encode(file_content).decode('utf-8')
        else:
            file_content_b64 = base64.b64encode(file_content.encode('utf-8')).decode('utf-8')
        
        payload = {
            "message": f"{'Update' if is_update else 'Add'} {file_name}",
            "content": file_content_b64
        }
        
        # Always check if file exists and get its SHA (even for new repos with auto_init)
        existing_sha = get_file_sha(repo_name, file_name)
        if existing_sha:
            payload["sha"] = existing_sha
        
        response = requests.put(
            f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo_name}/contents/{file_name}",
            headers=headers,
            json=payload
        )
        
        if response.status_code not in [200, 201]:
            raise Exception(f"Failed to push file {file_name}: {response.status_code}, {response.text}")
        
        time.sleep(0.5)  # Rate limiting

def generate_readme(brief: str, checks: List[str], code_explanation: str) -> str:
    """Generate a professional README using OpenAI"""
    client = get_openai_client()
    
    prompt = f"""Generate a professional README.md for a GitHub project with the following details:

Project Brief: {brief}

Evaluation Criteria:
{chr(10).join(f"- {check}" for check in checks)}

Code Explanation: {code_explanation}

Create a comprehensive README with these sections:
1. Project Title (derived from brief)
2. Overview/Description
3. Features
4. Setup Instructions
5. Usage Guide
6. Code Explanation
7. License (MIT)

Make it professional, clear, and well-formatted in Markdown."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a technical writer creating professional README files."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=1500
    )
    
    return response.choices[0].message.content

def clean_generated_code(code: str) -> str:
    """Clean up generated code by removing markdown code blocks and extra content"""
    code = code.strip()
    
    # Remove markdown code blocks
    if code.startswith("```"):
        lines = code.split('\n')
        # Find the first line that's not a code fence
        start_idx = 1
        for i, line in enumerate(lines):
            if i > 0 and not line.strip().startswith("```"):
                start_idx = i
                break
        
        # Find the last code fence
        end_idx = len(lines) - 1
        for i in range(len(lines) - 1, 0, -1):
            if lines[i].strip().startswith("```"):
                end_idx = i
                break
        
        code = '\n'.join(lines[start_idx:end_idx])
    
    # Remove any leading/trailing whitespace
    code = code.strip()
    
    return code

def generate_app_code(brief: str, checks: List[str], attachments: List[Dict], round_num: int = 1, existing_code: Optional[str] = None) -> str:
    """Generate HTML/CSS/JS app code using OpenAI"""
    client = get_openai_client()
    
    attachment_info = ""
    if attachments:
        attachment_info = "Attachments provided:\n"
        for att in attachments:
            attachment_info += f"- {att.get('name')}: {att.get('url')[:100]}...\n"
    
    if round_num == 1:
        prompt = f"""Create a complete, single-page HTML application for the following brief:

{brief}

Evaluation Criteria:
{chr(10).join(f"- {check}" for check in checks)}

{attachment_info}

Requirements:
1. Create a SINGLE HTML file (index.html) with embedded CSS and JavaScript
2. The app should be production-ready and fully functional
3. Handle query parameters (e.g., ?url=...)
4. Include proper error handling
5. Make it visually appealing with modern CSS
6. Add comments explaining key parts
7. Use vanilla JavaScript (no external dependencies unless absolutely necessary)
8. If using external libraries, use CDN links

Generate ONLY the complete HTML code, starting with <!DOCTYPE html> and nothing else. Do not include any explanations or markdown."""
    else:
        prompt = f"""You are given an existing HTML application. Modify it based on the following new requirements:

EXISTING CODE:
{existing_code}

NEW REQUIREMENTS:
{brief}

Evaluation Criteria:
{chr(10).join(f"- {check}" for check in checks)}

{attachment_info}

Instructions:
1. Keep all existing functionality intact
2. Add/modify features based on the new requirements
3. Maintain the same structure (single HTML file with embedded CSS/JS)
4. Ensure the changes integrate smoothly with existing code
5. Keep it production-ready and fully functional

Generate ONLY the complete updated HTML code, starting with <!DOCTYPE html> and nothing else. Do not include any explanations or markdown."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert web developer. Generate ONLY HTML code without any explanations, markdown formatting, or additional text."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=4000
    )
    
    html_code = response.choices[0].message.content.strip()
    return clean_generated_code(html_code)

def generate_code_explanation(html_code: str) -> str:
    """Generate explanation of the code"""
    client = get_openai_client()
    
    prompt = f"""Briefly explain how this code works in 2-3 paragraphs:

{html_code[:2000]}

Focus on:
1. Main functionality
2. Key technical approaches
3. How it meets the requirements"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a technical writer explaining code clearly and concisely."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=500
    )
    
    return response.choices[0].message.content

def notify_evaluation_api(evaluation_url: str, data: EvaluationResponse, max_retries: int = 5):
    """Send evaluation response to the instructor's API with exponential backoff"""
    headers = {"Content-Type": "application/json"}
    
    # Check if it's a placeholder URL
    if "example.com" in evaluation_url.lower():
        print(f"⚠️  Skipping notification to placeholder URL: {evaluation_url}")
        print(f"📋 Evaluation data that would be sent:")
        print(f"   {data.model_dump()}")
        return {"status": "skipped", "reason": "placeholder_url"}
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                evaluation_url,
                headers=headers,
                json=data.model_dump(),
                timeout=30
            )
            
            if response.status_code == 200:
                print(f"✅ Successfully notified evaluation API")
                return response.json()
            else:
                print(f"⚠️  Evaluation API returned {response.status_code}: {response.text[:200]}")
        except Exception as e:
            print(f"⚠️  Attempt {attempt + 1} failed: {str(e)}")
        
        if attempt < max_retries - 1:
            delay = 2 ** attempt  # Exponential backoff: 1, 2, 4, 8 seconds
            time.sleep(delay)
    
    # Don't raise exception for placeholder URLs or 403 errors
    print(f"⚠️  Could not notify evaluation API after {max_retries} attempts")
    print(f"📋 Final evaluation data:")
    print(f"   {data.model_dump()}")
    return {"status": "failed", "attempts": max_retries}

def process_round1(data: TaskRequest) -> EvaluationResponse:
    """Process round 1: Create and deploy the initial app"""
    repo_name = f"{data.task}"
    
    # Generate app code using LLM
    print("Generating app code with LLM...")
    html_code = generate_app_code(data.brief, data.checks, data.attachments, round_num=1, existing_code=None)
    
    # Validate that we have proper HTML
    if not html_code.strip().startswith('<!DOCTYPE') and not html_code.strip().startswith('<html'):
        print("⚠️  Generated code doesn't start with HTML tags, cleaning up...")
        html_code = clean_generated_code(html_code)
    
    # Generate code explanation
    print("Generating code explanation...")
    code_explanation = generate_code_explanation(html_code)
    
    # Generate README
    print("Generating README...")
    readme_content = generate_readme(data.brief, data.checks, code_explanation)
    
    # Prepare files to push
    files = [
        {"name": "index.html", "content": html_code},
        {"name": "README.md", "content": readme_content}
    ]
    
    # Create repository
    print(f"Creating GitHub repository: {repo_name}")
    repo_info = create_github_repo(repo_name)
    
    # Wait for repo to be fully initialized
    time.sleep(3)
    
    # Enable GitHub Pages
    print("Enabling GitHub Pages...")
    enable_github_pages(repo_name)
    
    # Push files (will automatically handle existing files from auto_init)
    print("Pushing files to repository...")
    push_files_to_repo(repo_name, files, is_update=False)
    
    # Get latest commit SHA
    time.sleep(2)  # Wait for files to be committed
    commit_sha = get_sha_of_latest_commit(repo_name)
    
    # Prepare response
    pages_url = f"https://{GITHUB_USERNAME}.github.io/{repo_name}/"
    repo_url = f"https://github.com/{GITHUB_USERNAME}/{repo_name}"
    
    evaluation_response = EvaluationResponse(
        email=data.email,
        task=data.task,
        round=data.round,
        nonce=data.nonce,
        repo_url=repo_url,
        commit_sha=commit_sha,
        pages_url=pages_url
    )
    
    # Notify evaluation API
    print("Notifying evaluation API...")
    notify_evaluation_api(data.evaluation_url, evaluation_response)
    
    return evaluation_response

def process_round2(data: TaskRequest) -> EvaluationResponse:
    """Process round 2: Update the app based on feedback"""
    repo_name = f"{data.task}"
    
    # Fetch existing index.html from the repository
    print("Fetching existing code from repository...")
    existing_html = get_file_content(repo_name, "index.html")
    
    if not existing_html:
        raise Exception("Could not fetch existing index.html from repository")
    
    print(f"Found existing code ({len(existing_html)} characters)")
    
    # Generate updated app code using LLM with existing code as context
    print("Generating updated app code with LLM...")
    html_code = generate_app_code(data.brief, data.checks, data.attachments, round_num=2, existing_code=existing_html)
    
    # Validate that we have proper HTML
    if not html_code.strip().startswith('<!DOCTYPE') and not html_code.strip().startswith('<html'):
        print("⚠️  Generated code doesn't start with HTML tags, cleaning up...")
        html_code = clean_generated_code(html_code)
    
    # Generate code explanation
    print("Generating updated code explanation...")
    code_explanation = generate_code_explanation(html_code)
    
    # Generate updated README
    print("Generating updated README...")
    readme_content = generate_readme(data.brief, data.checks, code_explanation)
    
    # Prepare files to push
    files = [
        {"name": "index.html", "content": html_code},
        {"name": "README.md", "content": readme_content}
    ]
    
    # Push updated files
    print("Pushing updated files to repository...")
    push_files_to_repo(repo_name, files, is_update=True)
    
    # Get latest commit SHA
    time.sleep(2)
    commit_sha = get_sha_of_latest_commit(repo_name)
    
    # Prepare response
    pages_url = f"https://{GITHUB_USERNAME}.github.io/{repo_name}/"
    repo_url = f"https://github.com/{GITHUB_USERNAME}/{repo_name}"
    
    evaluation_response = EvaluationResponse(
        email=data.email,
        task=data.task,
        round=data.round,
        nonce=data.nonce,
        repo_url=repo_url,
        commit_sha=commit_sha,
        pages_url=pages_url
    )
    
    # Notify evaluation API
    print("Notifying evaluation API...")
    notify_evaluation_api(data.evaluation_url, evaluation_response)
    
    return evaluation_response

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "message": "FastAPI App for Automated Project Submission"}

@app.post("/handle_task")
async def handle_task(data: TaskRequest):
    """Main endpoint to handle task requests"""
    
    # Validate secret
    if not validate_secret(data.secret):
        raise HTTPException(status_code=401, detail="Invalid secret")
    
    try:
        if data.round == 1:
            print(f"\n{'='*60}")
            print(f"🚀 Processing Round 1 for task: {data.task}")
            print(f"{'='*60}\n")
            result = process_round1(data)
            print(f"\n{'='*60}")
            print(f"✅ Round 1 completed successfully!")
            print(f"📦 Repository: {result.repo_url}")
            print(f"🌐 GitHub Pages: {result.pages_url}")
            print(f"📝 Commit SHA: {result.commit_sha}")
            print(f"{'='*60}\n")
            return {
                "status": "success",
                "message": "Round 1 completed successfully",
                "data": result.model_dump()
            }
        elif data.round == 2:
            print(f"\n{'='*60}")
            print(f"🔄 Processing Round 2 for task: {data.task}")
            print(f"{'='*60}\n")
            result = process_round2(data)
            print(f"\n{'='*60}")
            print(f"✅ Round 2 completed successfully!")
            print(f"📦 Repository: {result.repo_url}")
            print(f"🌐 GitHub Pages: {result.pages_url}")
            print(f"📝 Commit SHA: {result.commit_sha}")
            print(f"{'='*60}\n")
            return {
                "status": "success",
                "message": "Round 2 completed successfully",
                "data": result.model_dump()
            }
        else:
            raise HTTPException(status_code=400, detail="Invalid round number")
    
    except Exception as e:
        print(f"\n❌ Error processing task: {str(e)}\n")
        raise HTTPException(status_code=500, detail=f"Error processing task: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    # Validate required environment variables
    required_vars = ["OPENAI_API_KEY", "GITHUB_TOKEN", "SECRET"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
        print("\nPlease set the following environment variables:")
        print("- OPENAI_API_KEY: Your OpenAI API key")
        print("- GITHUB_TOKEN: Your GitHub Personal Access Token")
        print("- SECRET: Your secret key for validation")
        print("- GITHUB_USERNAME: Your GitHub username (optional, defaults to '24f2004876-Krish')")
        exit(1)
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)