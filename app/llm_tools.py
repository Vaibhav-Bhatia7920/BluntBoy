import os
from typing import Optional
from langchain_core.tools import tool
from app.sandbox import run_command  # Day 1 Docker sandbox execution instance

# ==========================================
# 1. DIRECTORY STRUCTURE TOOL
# ==========================================
@tool
def run_tree(path: str = ".", depth: int = 3) -> str:
    """
    Explore the folder hierarchy of the repository.
    Use this first to understand sub-package layouts and folder structure.
    
    Args:
        path: Relative path to inspect (defaults to root '.')
        depth: Maximum directory recursion depth (defaults to 3)
    """
    # Exclude bloated directories and virtualenvs to save tokens
    ignore_pattern = "node_modules|.git|__pycache__|.venv|*.egg-info|dist|build|.pytest_cache"
    cmd = f"tree -L {depth} -I '{ignore_pattern}' {path}"
    
    result = run_command(cmd)
    print(result)
    if result["exit_code"] != 0:
        return f"Error executing tree: {result['stderr']}"
    return result["stdout"]


# ==========================================
# 2. FILE CONTENT SEARCH TOOL (ripgrep)
# ==========================================
@tool
def run_ripgrep(pattern: str, path: str = ".", context_lines: int = 2, max_count: int = 25) -> str:
    """
    Search inside file contents for symbols, function definitions, variables, or error strings.
    
    Args:
        pattern: The exact string or regex to search for (e.g., 'def verify_token', 'JWTError')
        path: Subdirectory or specific file to search within (defaults to root '.')
        context_lines: Number of lines above and below each match to return (default: 2)
        max_count: Maximum matching lines per file to prevent prompt flooding (default: 25)
    """
    # -i: case insensitive, -C: context lines, -n: line numbers, --max-count: limit matches
    cmd = f"rg -i -n '{pattern}' -C {context_lines} --max-count {max_count} --color=never {path}"
    
    result = run_command(cmd)
    
    if result["exit_code"] == 1:
        return f"No matches found for pattern: '{pattern}'"
    elif result["exit_code"] != 0:
        return f"Error executing ripgrep: {result['stderr']}"
        
    # Truncate overall output to prevent overflowing LLM context window
    lines = result["stdout"].splitlines()
    if len(lines) > 80:
        return "\n".join(lines[:80]) + f"\n\n... [Truncated: {len(lines) - 80} additional lines omitted. Narrow your search path or query.]"
        
    return result["stdout"]


# ==========================================
# 3. FILENAME & GLOB SEARCH TOOL (find)
# ==========================================
@tool
def find_files(pattern: str, path: str = ".") -> str:
    """
    Locate files by exact filename or wildcard pattern without reading file contents.
    Use this when searching for configs, models, tests, or specific file names.
    
    Args:
        pattern: Filename or glob pattern (e.g., '*.py', 'test_auth.py', 'docker-compose*')
        path: Directory to search within (defaults to '.')
    """
    cmd = (
        f"find {path} -type f -name '{pattern}' "
        f"-not -path '*/.*' -not -path '*/node_modules/*' -not -path '*/__pycache__*' "
        f"| head -n 30"
    )
    
    result = run_command(cmd)
    if result["exit_code"] != 0:
        return f"Error finding files: {result['stderr']}"
        
    output = result["stdout"].strip()
    return output if output else f"No files matching '{pattern}' were found."


# ==========================================
# 4. TARGETED LINE-RANGE READER (sed)
# ==========================================
@tool
def read_file_snippet(file_path: str, start_line: int = 1, end_line: int = 60) -> str:
    """
    Read an exact range of lines from a specific file. 
    Never reads entire 1000-line files—only inspects relevant snippets.
    
    Args:
        file_path: Relative path to the file (e.g., 'packages/core/auth.py')
        start_line: 1-indexed starting line number (default: 1)
        end_line: 1-indexed ending line number (default: 60)
    """
    # Guardrail: Limit maximum read window to 100 lines at a time
    if end_line - start_line > 100:
        end_line = start_line + 100

    # Uses sed to extract line slice efficiently inside the container
    cmd = f"sed -n '{start_line},{end_line}p' {file_path}"
    
    result = run_command(cmd)
    if result["exit_code"] != 0:
        return f"Error reading file {file_path}: {result['stderr']}"
        
    output = result["stdout"]
    if not output.strip():
        return f"File '{file_path}' is empty or lines {start_line}-{end_line} do not exist."
        
    # Format with line numbers for clean LLM reference
    numbered_lines = []
    for idx, line in enumerate(output.splitlines(), start=start_line):
        numbered_lines.append(f"{idx:4d} | {line}")
        
    return "\n".join(numbered_lines)


# List of all available navigation tools for model binding
NAVIGATOR_TOOLS = [run_tree, run_ripgrep, find_files, read_file_snippet]