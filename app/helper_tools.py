import re
from typing import List
from app.template import SearchResultItem

def parse_ripgrep_output(raw_output: str) -> List[SearchResultItem]:
    """Parses raw ripgrep stdout into structured SearchResultItem models."""
    items: List[SearchResultItem] = []
    
    for line in raw_output.splitlines():
        # Match standard ripgrep format: "file_path:line_number:content"
        match = re.match(r"^([^:]+):(\d+):(.*)$", line.strip())
        if match:
            items.append(
                SearchResultItem(
                    file=match.group(1),
                    line=int(match.group(2)),
                    content=match.group(3).strip()
                )
            )
        elif line.strip() and not line.startswith("..."):
            # Fallback for non-standard lines or context separators
            items.append(
                SearchResultItem(
                    file="unknown",
                    line=None,
                    content=line.strip()
                )
            )
            
    return items