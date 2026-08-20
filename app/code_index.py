import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tree_sitter import Language, Parser

try:
    from tree_sitter_python import language as python_language
except ImportError:  # pragma: no cover - dependency resolution fallback
    python_language = None

try:
    from tree_sitter_javascript import language as javascript_language
except ImportError:  # pragma: no cover - dependency resolution fallback
    javascript_language = None


@dataclass
class SymbolEntry:
    path: str
    kind: str
    name: str
    signature: str
    imports: List[str]
    line: int


def _repo_root(project_root: str) -> Path:
    root = Path(project_root).expanduser()
    return root if root.is_absolute() else Path.cwd() / root


def _db_path(project_root: str) -> Path:
    return _repo_root(project_root) / ".monorepo_agent" / "symbol_map.sqlite3"


def _ensure_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS files (
            path TEXT PRIMARY KEY,
            digest TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS symbols (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL,
            kind TEXT NOT NULL,
            name TEXT NOT NULL,
            signature TEXT NOT NULL,
            imports TEXT NOT NULL,
            line INTEGER NOT NULL,
            FOREIGN KEY(path) REFERENCES files(path) ON DELETE CASCADE
        )
        """
    )
    connection.execute("CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name)")
    connection.execute("CREATE INDEX IF NOT EXISTS idx_symbols_path ON symbols(path)")


def _digest(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8", errors="ignore")).hexdigest()


def _iter_source_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_dir():
            continue
        if any(part in {".git", ".venv", "__pycache__", "node_modules", "dist", "build"} for part in path.parts):
            continue
        if path.suffix.lower() not in {".py", ".js", ".jsx", ".ts", ".tsx"}:
            continue
        yield path


def _language_for_path(path: Path) -> Optional[Any]:
    suffix = path.suffix.lower()
    if suffix == ".py":
        return python_language() if python_language is not None else None
    if suffix in {".js", ".jsx", ".ts", ".tsx"}:
        return javascript_language() if javascript_language is not None else None
    return None


def _parser_for_language(language: Any) -> Parser:
    parser = Parser()
    if hasattr(parser, "set_language"):
        parser.set_language(language)
    else:
        parser.language = Language(language)
    return parser


def _parser_for_path(path: Path) -> Optional[Parser]:
    language = _language_for_path(path)
    if language is None:
        return None
    return _parser_for_language(language)


def _node_text(source: bytes, node: Any) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")


def _line_number(node: Any) -> int:
    return int(node.start_point[0]) + 1


def _signature_from_parameters(source: bytes, parameters_node: Optional[Any]) -> str:
    if parameters_node is None:
        return "()"

    parts: List[str] = []
    for child in parameters_node.named_children:
        if child.type == "identifier":
            parts.append(_node_text(source, child))
        elif child.type == "typed_parameter":
            name_node = child.child_by_field_name("name")
            if name_node is not None:
                parts.append(_node_text(source, name_node))
        elif child.type == "default_parameter":
            name_node = child.child_by_field_name("name")
            if name_node is not None:
                parts.append(_node_text(source, name_node))
        elif child.type == "list_splat_pattern":
            parts.append("*args")
        elif child.type == "dictionary_splat_pattern":
            parts.append("**kwargs")
    return f"({', '.join(parts)})"


def _imports_from_tree(source: bytes, root_node: Any) -> List[str]:
    imports: List[str] = []
    for child in root_node.named_children:
        if child.type == "import_statement":
            for named_child in child.named_children:
                if named_child.type == "dotted_name":
                    imports.append(_node_text(source, named_child))
                elif named_child.type == "aliased_import":
                    name_node = named_child.child_by_field_name("name")
                    alias_node = named_child.child_by_field_name("alias")
                    if name_node is not None:
                        item = _node_text(source, name_node)
                        if alias_node is not None:
                            item = f"{item} as {_node_text(source, alias_node)}"
                        imports.append(item)
        elif child.type == "import_from_statement":
            module_node = child.child_by_field_name("module_name")
            module_name = _node_text(source, module_node) if module_node is not None else ""
            for named_child in child.named_children:
                if named_child.type == "dotted_name":
                    continue
                if named_child.type == "aliased_import":
                    name_node = named_child.child_by_field_name("name")
                    alias_node = named_child.child_by_field_name("alias")
                    if name_node is not None:
                        item = _node_text(source, name_node)
                        if alias_node is not None:
                            item = f"{item} as {_node_text(source, alias_node)}"
                        imports.append(f"{module_name}:{item}" if module_name else item)
                elif named_child.type == "dotted_name" and module_name:
                    imports.append(f"{module_name}:{_node_text(source, named_child)}")
    return imports


def _generic_symbols_from_tree(path: Path, source: bytes, root_node: Any) -> List[SymbolEntry]:
    entries: List[SymbolEntry] = []
    for child in root_node.named_children:
        if child.type in {"function_definition", "function_declaration", "method_definition"}:
            name_node = child.child_by_field_name("name")
            if name_node is not None:
                entries.append(
                    SymbolEntry(
                        path=str(path),
                        kind="function",
                        name=_node_text(source, name_node),
                        signature="",
                        imports=[],
                        line=_line_number(child),
                    )
                )
        elif child.type in {"class_definition", "class_declaration"}:
            name_node = child.child_by_field_name("name")
            if name_node is not None:
                entries.append(
                    SymbolEntry(
                        path=str(path),
                        kind="class",
                        name=_node_text(source, name_node),
                        signature="",
                        imports=[],
                        line=_line_number(child),
                    )
                )
    return entries


def _symbols_from_tree(path: Path, source: bytes, root_node: Any) -> List[SymbolEntry]:
    imports = _imports_from_tree(source, root_node)
    entries: List[SymbolEntry] = []

    def visit(node: Any, parent_class: Optional[str] = None) -> None:
        for child in node.named_children:
            if child.type == "function_definition":
                name_node = child.child_by_field_name("name")
                parameters_node = child.child_by_field_name("parameters")
                if name_node is not None:
                    entries.append(
                        SymbolEntry(
                            path=str(path),
                            kind="method" if parent_class else "function",
                            name=_node_text(source, name_node),
                            signature=_signature_from_parameters(source, parameters_node),
                            imports=imports,
                            line=_line_number(child),
                        )
                    )
            elif child.type == "class_definition":
                name_node = child.child_by_field_name("name")
                body_node = child.child_by_field_name("body")
                if name_node is not None:
                    entries.append(
                        SymbolEntry(
                            path=str(path),
                            kind="class",
                            name=_node_text(source, name_node),
                            signature="",
                            imports=imports,
                            line=_line_number(child),
                        )
                    )
                if body_node is not None:
                    visit(body_node, _node_text(source, name_node) if name_node is not None else parent_class)

    visit(root_node)
    return entries


def build_symbol_map(project_root: str) -> Dict[str, Any]:
    root = _repo_root(project_root)
    db_path = _db_path(project_root)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(db_path)
    try:
        _ensure_schema(connection)
        connection.execute("DELETE FROM symbols")
        connection.execute("DELETE FROM files")

        symbol_entries: List[SymbolEntry] = []
        file_count = 0
        for path in _iter_source_files(root):
            source_text = path.read_text(encoding="utf-8", errors="ignore")
            source = source_text.encode("utf-8", errors="ignore")
            relative_path = str(path.relative_to(root))
            connection.execute(
                "INSERT OR REPLACE INTO files(path, digest) VALUES (?, ?)",
                (relative_path, _digest(source_text)),
            )
            parser = _parser_for_path(path)
            if parser is None:
                continue
            tree = parser.parse(source)
            if path.suffix.lower() == ".py":
                symbol_entries.extend(_symbols_from_tree(Path(relative_path), source, tree.root_node))
            else:
                symbol_entries.extend(_generic_symbols_from_tree(Path(relative_path), source, tree.root_node))
            file_count += 1

        for entry in symbol_entries:
            connection.execute(
                "INSERT INTO symbols(path, kind, name, signature, imports, line) VALUES (?, ?, ?, ?, ?, ?)",
                (entry.path, entry.kind, entry.name, entry.signature, json.dumps(entry.imports), entry.line),
            )

        connection.commit()
        return {
            "db_path": str(db_path),
            "file_count": file_count,
            "symbol_count": len(symbol_entries),
        }
    finally:
        connection.close()


def load_symbol_map(project_root: str, limit: int = 200) -> str:
    db_path = _db_path(project_root)
    if not db_path.exists():
        return ""

    connection = sqlite3.connect(db_path)
    try:
        _ensure_schema(connection)
        rows = connection.execute(
            "SELECT path, kind, name, signature, imports, line FROM symbols ORDER BY path, line LIMIT ?",
            (limit,),
        ).fetchall()
    finally:
        connection.close()

    lines = []
    for path, kind, name, signature, imports, line in rows:
        import_list = ", ".join(json.loads(imports)[:5])
        lines.append(f"{path}:{line} {kind} {name}{signature} imports=[{import_list}]")
    return "\n".join(lines)