"""Guard the dependency rule (design doc §2): core/ may import only the standard library
and voice_agent.core itself. Everything else (adapters, providers, storage, notify, app,
and any third-party package) is rejected, including imports inside functions.

The files are parsed, not executed, so a bad import is caught even if it never runs.
"""

import ast
import sys
from pathlib import Path

import pytest

CORE_DIR = Path(__file__).resolve().parents[2] / "src" / "voice_agent" / "core"
ALLOWED_PREFIX = "voice_agent.core"


def is_allowed(module: str) -> bool:
    if module == ALLOWED_PREFIX or module.startswith(ALLOWED_PREFIX + "."):
        return True
    top = module.split(".")[0]
    return top in sys.stdlib_module_names or top == "__future__"


def imports_in(path: Path) -> list[tuple[int, str]]:
    """(line, module) for every import in the file. Relative imports are resolved
    against the file's package so `from ..providers import x` is seen for what it is."""
    folder_parts = path.relative_to(CORE_DIR.parent).parts[:-1]  # ("core",) or ("core", "sub")
    package = "voice_agent." + ".".join(folder_parts)
    found = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
        if isinstance(node, ast.Import):
            found += [(node.lineno, alias.name) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative: climb (level - 1) packages up from this file's package
                parts = package.split(".")
                base = parts[: len(parts) - (node.level - 1)]
                module = ".".join(base + ([node.module] if node.module else []))
            else:
                module = node.module or ""
            found.append((node.lineno, module))
    return found


def violations(core_dir: Path = CORE_DIR) -> list[str]:
    bad = []
    for path in sorted(core_dir.rglob("*.py")):
        for line, module in imports_in(path):
            if not is_allowed(module):
                rel = path.relative_to(core_dir.parent.parent)
                bad.append(f"{rel}:{line} imports {module}")
    return bad


def test_core_imports_only_stdlib_and_itself():
    bad = violations()
    assert not bad, "core/ must not import outside itself or the stdlib:\n" + "\n".join(bad)


@pytest.mark.parametrize(
    ("module", "allowed"),
    [
        ("asyncio", True),
        ("collections.abc", True),
        ("voice_agent.core.interfaces", True),
        ("voice_agent.providers.anthropic_llm", False),
        ("voice_agent.app.config", False),
        ("voice_agent.coreish", False),  # prefix lookalike
        ("anthropic", False),
        ("pydantic", False),
    ],
)
def test_is_allowed(module, allowed):
    assert is_allowed(module) is allowed
