"""Builds the system prompt for every model call in a session.

Three blocks, always in this order (design §8.2):

1. Instructions: persona, rules, speaking style, tool use. From prompts/instructions.md,
   filled with the owner's config once at startup.
2. Knowledge base: every file in kb/, joined in a fixed order. The cache breakpoint goes here.
3. This call: today's date, who's calling, the last call's summary. Changes every session.

Blocks 1 and 2 must be byte-for-byte the same on every call, or the prompt cache misses and
every turn gets slower and dearer. So they're built once at startup from files only, and
anything that changes (the date, the caller, how stale the status is) goes in block 3.

Knowledge base files may start with a small front-matter block:

    ---
    featured: yes
    link: github.com/someone/project
    ---

which is turned into plain text the model can use (links are rewritten the way they're
said aloud). Unknown keys are rejected, so a typo fails at startup.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from voice_agent.core.greeting import usable_name
from voice_agent.core.interfaces import SystemBlock

INSTRUCTIONS_PATH = Path(__file__).parent / "prompts" / "instructions.md"

# Files every knowledge base needs, in prompt order. Projects go after experience.
CORE_FILES = ("profile.md", "status.md", "experience.md", "skills.md", "faq.md", "boundaries.md")
PROJECTS_AFTER = "experience.md"
PROJECTS_DIR = "projects"

FRONT_MATTER_KEYS = {
    "status.md": {"updated"},
    PROJECTS_DIR: {"featured", "link"},
}
_PLACEHOLDER = re.compile(r"\{(\w+)\}")


class PromptError(RuntimeError):
    """The instructions or knowledge base can't be built. Raised at startup."""


@dataclass(frozen=True, slots=True)
class KnowledgeBase:
    text: str  # block 2, exactly as sent
    status_updated: date | None  # from status.md, for the staleness note in block 3
    featured: tuple[str, ...]  # featured project titles, in prompt order
    links: dict[str, str]  # project title -> link as written (for v1 texts)


@dataclass(frozen=True, slots=True)
class CallContext:
    """What block 3 says about this session."""

    today: date
    caller_name: str | None  # as saved; dropped if it doesn't look like a real name
    anonymous: bool
    last_summary: str | None  # already limited to recent calls by the session
    last_call_date: date | None = None


@dataclass(frozen=True, slots=True)
class Prompt:
    """Blocks 1 and 2, built once at startup. `system()` adds block 3 per session."""

    instructions: str
    kb: KnowledgeBase
    status_stale_days: int

    def system(self, call: CallContext) -> list[SystemBlock]:
        return [
            SystemBlock(self.instructions),
            SystemBlock(self.kb.text, cache=True),
            SystemBlock(call_context(call, self.kb.status_updated, self.status_stale_days)),
        ]


# ---------------------------------------------------------------------------
# Block 1: instructions
# ---------------------------------------------------------------------------


def load_instructions(values: dict[str, str], path: Path = INSTRUCTIONS_PATH) -> str:
    """Read the instructions template and fill in its {placeholders}.

    Every placeholder in the file must have a value and every value must be used, so a
    renamed setting or a typo fails at startup instead of reaching the model as "{tone}".
    """
    template = path.read_text(encoding="utf-8")
    used = set(_PLACEHOLDER.findall(template))
    missing, unused = used - values.keys(), values.keys() - used
    if missing or unused:
        raise PromptError(
            f"{path.name}: no value for {sorted(missing)}; unused values {sorted(unused)}"
        )
    return _PLACEHOLDER.sub(lambda m: values[m.group(1)], template).strip()


# ---------------------------------------------------------------------------
# Block 2: knowledge base
# ---------------------------------------------------------------------------


def load_knowledge_base(directory: Path, featured_limit: int) -> KnowledgeBase:
    """Join the knowledge base files into one block, in a fixed order.

    Order: profile, status, experience, the projects (featured first, then by file name),
    skills, faq, boundaries. File order never depends on the file system, because a
    reordering would silently break the prompt cache.
    """
    if not directory.is_dir():
        raise PromptError(f"knowledge base folder not found: {directory}")
    missing = [name for name in CORE_FILES if not (directory / name).is_file()]
    if missing:
        raise PromptError(f"knowledge base is missing: {', '.join(missing)}")
    extra = sorted(
        p.name for p in directory.glob("*.md") if p.name not in CORE_FILES + ("README.md",)
    )
    if extra:  # it would be silently left out of the prompt
        raise PromptError(
            f"knowledge base has files the prompt doesn't load: {', '.join(extra)}. "
            "Move them into one of the core files or projects/."
        )

    sections: list[str] = []
    status_updated: date | None = None
    featured: list[str] = []
    links: dict[str, str] = {}

    for name in CORE_FILES:
        meta, body = _read(directory / name, FRONT_MATTER_KEYS.get(name, set()))
        if name == "status.md" and "updated" in meta:
            status_updated = _parse_date(meta["updated"], name)
            body = f"{body}\n\n(Status last updated {_spoken_date(status_updated)}.)"
        sections.append(body)
        if name == PROJECTS_AFTER:
            sections.extend(_projects(directory / PROJECTS_DIR, featured, links))

    if featured:
        lead = featured[:featured_limit]
        sections.insert(
            0,
            "# Featured projects\nWhen asked about my work, lead with these, one line "
            f"each: {'; '.join(lead)}.",
        )
    text = "<knowledge_base>\n" + "\n\n".join(sections) + "\n</knowledge_base>"
    return KnowledgeBase(text, status_updated, tuple(featured[:featured_limit]), links)


def _projects(folder: Path, featured: list[str], links: dict[str, str]) -> list[str]:
    if not folder.is_dir():
        return []
    loaded = []
    for path in sorted(folder.glob("*.md")):
        meta, body = _read(path, FRONT_MATTER_KEYS[PROJECTS_DIR])
        title = _title(body, path)
        is_featured = _parse_flag(meta.get("featured", "no"), path.name)
        if "link" in meta:
            links[title] = meta["link"]
            body = f"{body}\n\nLink, said aloud: {spoken_link(meta['link'])}."
        loaded.append((not is_featured, path.name, title, body))
    loaded.sort()  # featured first, then by file name
    featured.extend(title for not_featured, _, title, _ in loaded if not not_featured)
    return [body for *_, body in loaded]


def _read(path: Path, allowed_keys: set[str]) -> tuple[dict[str, str], str]:
    """Split a file into its front matter (key: value lines) and its body."""
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n").strip()
    if not text:
        raise PromptError(f"{path.name} is empty")
    meta: dict[str, str] = {}
    if text.startswith("---\n"):
        end = text.find("\n---", 4)
        if end == -1:
            raise PromptError(f"{path.name}: front matter has no closing ---")
        for line in text[4:end].splitlines():
            if not line.strip():
                continue
            key, sep, value = line.partition(":")
            key = key.strip()
            if not sep or key not in allowed_keys:
                raise PromptError(
                    f"{path.name}: unexpected front matter {line.strip()!r} "
                    f"(allowed: {sorted(allowed_keys) or 'none'})"
                )
            meta[key] = value.strip()
        text = text[end + 4 :].strip()
    if not text:
        raise PromptError(f"{path.name} has no content")
    return meta, text


def _title(body: str, path: Path) -> str:
    first = body.splitlines()[0]
    if not first.startswith("# "):
        raise PromptError(f"{path.name} must start with a '# Title' line")
    return first[2:].strip()


def _parse_flag(value: str, name: str) -> bool:
    if value.lower() in ("yes", "true"):
        return True
    if value.lower() in ("no", "false"):
        return False
    raise PromptError(f"{name}: featured must be yes or no, not {value!r}")


def _parse_date(value: str, name: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as e:
        raise PromptError(f"{name}: updated must be a date like 2026-11-15, not {value!r}") from e


def spoken_link(link: str) -> str:
    """github.com/some-one/x_y -> 'github dot com slash some dash one slash x underscore y'.

    The model reads links aloud, and a hyphen or slash it skips sends the caller to the
    wrong address.
    """
    link = re.sub(r"^https?://", "", link.strip()).removeprefix("www.").rstrip("/")
    words = {".": " dot ", "/": " slash ", "-": " dash ", "_": " underscore "}
    return re.sub(r"[./_-]", lambda m: words[m.group()], link)


def _spoken_date(d: date) -> str:
    return f"{d:%B} {d.day}, {d.year}"


# ---------------------------------------------------------------------------
# Block 3: this call
# ---------------------------------------------------------------------------


def call_context(call: CallContext, status_updated: date | None, status_stale_days: int) -> str:
    lines = ["<this_call>", f"Today is {call.today:%A}, {_spoken_date(call.today)}."]

    name = usable_name(call.caller_name)
    if call.anonymous:
        lines.append(
            "The caller's number is withheld, so I can't recognise them or call them back. "
            "Their name is unknown unless they say it."
        )
    elif name:
        lines.append(f"The caller is {name}, from an earlier call.")
    else:
        lines.append("The caller's name is unknown.")

    if call.last_summary:
        when = f" on {_spoken_date(call.last_call_date)}" if call.last_call_date else ""
        lines.append(
            f"Summary of our last call{when}, for context only (anything from it, like a "
            f"message, was already dealt with): {call.last_summary}"
        )

    if status_updated and (call.today - status_updated).days > status_stale_days:
        lines.append(
            f"My status was last updated on {_spoken_date(status_updated)}. When talking "
            f'about what I\'m doing now, say "as of {_spoken_date(status_updated)}".'
        )

    lines.append("</this_call>")
    return "\n".join(lines)
