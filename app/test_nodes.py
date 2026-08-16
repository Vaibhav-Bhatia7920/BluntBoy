# test_nodes.py
import json
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
import os
from app.state import MonorepoState
from app.state import repo_navigator_node
from app.state import planner_node
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file
api_key = os.getenv("OPENAI_API_KEY")
console = Console()

def run_test():
    console.print(Panel("[bold blue]⚡ Testing BLUNT Pipeline: Node 1 -> Node 2[/bold blue]", expand=False))

    # 1. Initialize State with synthetic bug
    initial_state: MonorepoState = {
        "issue_title": "Fix TypeError: decode_token crashes when token is None",
        "issue_description": (
            "When passing None to decode_token in packages/core/auth.py, "
            "it raises an AttributeError because it directly calls token.split('.'). "
            "It should safely return an empty dict when token is None or invalid."
        ),
        "config": {
            "llm_model": "gpt-5.6-luna",  # Or "gpt-4o", "claude-3-5-sonnet", etc.
            "api_key": api_key,
            "max_search_turns": 3,
            "tree_depth": 3,
            "context_lines": 2,
            "rg_max_count": 25
        },
        "project_root": "/workspace",
        "target_packages": [],
        "filesystem_map": "",
        "search_results": [],
        "relevant_files": [],
        "proposed_plan": "",
        "file_contents": {},
        "diffs_to_apply": [],
        "test_command": "",
        "test_stdout": "",
        "test_stderr": "",
        "is_resolved": False,
        "iteration_count": 0
    }

    # ==========================================
    # STAGE 1: RUN REPO NAVIGATOR (Node 1)
    # ==========================================
    console.print("\n[bold cyan]▶ [1/2] Executing RepoNavigator Node...[/bold cyan]")
    state_after_nav = repo_navigator_node(initial_state)

    nav_table = Table(title="RepoNavigator Findings", border_style="cyan")
    nav_table.add_column("Key", style="bold yellow")
    nav_table.add_column("Value", style="white")
    
    nav_table.add_row("Target Packages", str(state_after_nav.get("target_packages")))
    nav_table.add_row("Relevant Files", str(state_after_nav.get("relevant_files")))
    nav_table.add_row("Search Hits", str(len(state_after_nav.get("search_results", []))))
    console.print(nav_table)

    # Basic Assertion for Node 1
    assert len(state_after_nav.get("relevant_files", [])) > 0, "❌ RepoNavigator failed to find candidate files!"
    assert any("auth.py" in f for f in state_after_nav["relevant_files"]), "❌ RepoNavigator missed 'auth.py'!"
    console.print("[bold green]✓ Node 1 Passed: Repository mapped and target file isolated.[/bold green]")

    # ==========================================
    # STAGE 2: RUN PLANNER & CODER (Node 2)
    # ==========================================
    console.print("\n[bold magenta]▶ [2/2] Executing Planner & Execution Node...[/bold magenta]")
    state_after_plan = planner_node(state_after_nav)

    # Display Proposed Plan & Test Command
    console.print(Panel(state_after_plan.get("proposed_plan", "No plan"), title="[bold green]Proposed Step-by-Step Fix[/bold green]"))
    console.print(f"[bold yellow]Verification Test Command:[/bold yellow] [cyan]{state_after_plan.get('test_command')}[/cyan]")

    # Display Generated Diffs
    diffs = state_after_plan.get("diffs_to_apply", [])
    console.print(f"\n[bold green]Generated Surgical Diffs ({len(diffs)} block(s)):[/bold green]")
    
    for idx, diff in enumerate(diffs, 1):
        console.print(f"\n[dim]--- Patch #{idx} on [bold]{diff['file']}[/bold] ---[/dim]")
        diff_view = f"<<<<<<< SEARCH (Current Code)\n{diff['search']}\n=======\n{diff['replace']}\n>>>>>>> REPLACE"
        console.print(Syntax(diff_view, "diff", theme="monokai", line_numbers=True))

    # Basic Assertions for Node 2
    assert len(diffs) > 0, "❌ Planner failed to generate search/replace diffs!"
    assert "search" in diffs[0] and "replace" in diffs[0], "❌ Malformed DiffBlock schema!"
    
    # Check if the search block actually exists in the read file
    target_file = diffs[0]["file"]
    original_code = state_after_plan["file_contents"].get(target_file, "")
    assert diffs[0]["search"] in original_code, (
        f"❌ Hallucinated Diff! Search block does not exist verbatim in {target_file}.\n"
        f"Search block:\n{repr(diffs[0]['search'])}\n"
    )

    console.print("\n[bold green]✨ SUCCESS: Nodes 1 & 2 executed with 100% exact string-matching compatibility![/bold green]")

if __name__ == "__main__":
    run_test()