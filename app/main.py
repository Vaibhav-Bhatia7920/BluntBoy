from fastapi import FastAPI
from app.sandbox import run_command
from app.state import MonorepoState
app = FastAPI()

@app.get("/health")
def health_check():
    command = "docker exec -i monorepo-sandbox echo 'Hello from the sandbox!'"
    stdout, stderr = run_command(command)
    if stderr:
        return {"status": "error", "message": stderr}
    return {"status": "success", "message": stdout.strip()}

@app.post("/agent/run")
def get_response(issue_title: str, issue_description: str):
    state = MonorepoState(
        issue_title=issue_title,
        issue_description=issue_description,

        project_root="",
        target_packages=[],
        filesystem_map="",     
        search_results={},
        relevant_files=[],

        proposed_plan="",
        file_contents={},
        diffs_to_apply=[],

        test_command="",
        test_stdout="",
        test_stderr="",
        
        is_resolved=False,
        iteration_count=0
    )
    return {"status": "success", "state": state}

