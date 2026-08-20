from pydantic import BaseModel, Field
from typing import List, Dict, Optional


# class SearchResultItem(BaseModel):
#     file_path: str   # This replaces your dictionary key
#     output: str


# class RepoNavigatorResponse(BaseModel):

#     target_packages: List[str]       # Affected sub-packages
#     filesystem_map: str              # Visual structure via `tree`
#     search_results: List[SearchResultItem]  # Outputs from ripgrep
#     relevant_files: List[str]        # Resolved absolute/relative file paths


class SearchResultItem(BaseModel):
    file: str = Field(description="Relative file path where the match was found")
    line: Optional[int] = Field(default=None, description="Line number of the match")
    content: str = Field(description="Matching code snippet or symbol declaration")


class RepoNavigatorResponse(BaseModel):
    target_packages: List[str] = Field(
        description="List of affected sub-packages or modules (e.g., ['packages/core', 'apps/api'])"
    )
    filesystem_map: str = Field(
        description="Truncated visual directory structure from `tree` execution"
    )
    search_results: List[SearchResultItem] = Field(
        description="Structured key findings and matches obtained via `ripgrep`"
    )
    relevant_files: List[str] = Field(
        description="Target candidate file paths relative to repo root that need inspection or editing"
    )


class DiffBlock(BaseModel):
    file: str = Field(
        description="Relative file path of the file to modify (e.g., 'packages/core/auth.py')"
    )
    search: str = Field(
        description="Exact code block currently in the file to find and replace. Must match whitespace, indentation, and newlines exactly."
    )
    replace: str = Field(
        description="Exact replacement code block to substitute in place of the search block."
    )


class PlannerCoderResponse(BaseModel):
    proposed_plan: str = Field(
        description="Step-by-step technical plan explaining why this fix works and what lines are changing."
    )
    test_command: str = Field(
        description="Targeted test command to run in the sandbox to verify only this fix (e.g., 'pytest packages/core/tests/test_auth.py')"
    )
    diffs_to_apply: List[DiffBlock] = Field(
        description="List of deterministic search-and-replace blocks to apply to the repository."
    )


class PatcherNodeResponse(BaseModel):
    proposed_plan: str = Field(
        description="Execution-phase plan describing the exact diffs that should be applied next."
    )
    diffs_to_apply: List[DiffBlock] = Field(
        description="Execution-phase diffs to apply or hand off to the next step."
    )