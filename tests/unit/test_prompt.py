import shutil
from datetime import date
from pathlib import Path

import pytest

from voice_agent.core.prompt import (
    CallContext,
    Prompt,
    PromptError,
    call_context,
    load_instructions,
    load_knowledge_base,
    spoken_link,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_KB = REPO_ROOT / "kb.example"
VALUES = {
    "owner_name": "Alex",
    "tone": "friendly, concise",
    "do_not_discuss": "salary, address",
    "max_sentences": "3",
}
TODAY = date(2026, 11, 15)


@pytest.fixture
def kb_dir(tmp_path: Path) -> Path:
    """A copy of the example knowledge base that a test can change."""
    target = tmp_path / "kb"
    shutil.copytree(EXAMPLE_KB, target)
    return target


def call(**overrides) -> CallContext:
    base = {"today": TODAY, "caller_name": None, "anonymous": False, "last_summary": None}
    return CallContext(**{**base, **overrides})


# -- instructions --------------------------------------------------------------


def test_repo_instructions_fill_every_placeholder():
    text = load_instructions(VALUES)
    assert "Alex's AI" in text and "at most 3 sentences" in text
    assert "{" not in text


def test_instructions_reject_missing_or_unused_values(tmp_path):
    template = tmp_path / "i.md"
    template.write_text("Hi {owner_name}, tone {tone}.")
    with pytest.raises(PromptError, match="no value for \\['tone'\\]"):
        load_instructions({"owner_name": "A"}, template)
    with pytest.raises(PromptError, match="unused values \\['extra'\\]"):
        load_instructions({"owner_name": "A", "tone": "t", "extra": "x"}, template)


# -- knowledge base ------------------------------------------------------------


def test_example_kb_loads_in_fixed_order():
    kb = load_knowledge_base(EXAMPLE_KB, featured_limit=3)
    t = kb.text
    order = [
        "# Featured projects",
        "# Profile",
        "# Status",
        "# Experience",
        "# The pretend widget",  # featured project first
        "# An old made-up project",
        "# Skills",
        "# Likely questions",
        "# Boundaries",
    ]
    positions = [t.index(heading) for heading in order]
    assert positions == sorted(positions)
    assert t.startswith("<knowledge_base>") and t.endswith("</knowledge_base>")
    assert "Example knowledge base (fake)" not in t  # README.md isn't loaded


def test_front_matter_becomes_plain_text():
    kb = load_knowledge_base(EXAMPLE_KB, featured_limit=3)
    assert "---" not in kb.text and "featured:" not in kb.text
    assert "Link, said aloud: example dot com slash fake dash widget." in kb.text
    assert "(Status last updated January 1, 2026.)" in kb.text
    assert kb.status_updated == date(2026, 1, 1)
    assert kb.featured == ("The pretend widget",)
    assert kb.links == {"The pretend widget": "https://example.com/fake-widget"}


def test_same_files_give_identical_text(kb_dir):
    # The cache only hits if block 2 is byte-for-byte the same every time.
    assert load_knowledge_base(kb_dir, 3).text == load_knowledge_base(kb_dir, 3).text


def test_featured_limit(kb_dir):
    (kb_dir / "projects" / "another.md").write_text("---\nfeatured: yes\n---\n\n# Another\n\nText.")
    kb = load_knowledge_base(kb_dir, featured_limit=1)
    assert kb.featured == ("Another",)  # featured ones sort by file name
    assert "lead with these, one line each: Another." in kb.text


def test_no_projects_folder_is_fine(kb_dir):
    shutil.rmtree(kb_dir / "projects")
    kb = load_knowledge_base(kb_dir, 3)
    assert kb.featured == () and "# Featured projects" not in kb.text


@pytest.mark.parametrize(
    ("change", "error"),
    [
        (lambda d: (d / "faq.md").unlink(), "missing: faq.md"),
        (lambda d: (d / "writing.md").write_text("# Writing"), "doesn't load: writing.md"),
        (lambda d: (d / "skills.md").write_text("  \n"), "skills.md is empty"),
        (lambda d: (d / "skills.md").write_text("---\nfeatured: yes\n---\n# S"), "unexpected"),
        (lambda d: (d / "status.md").write_text("---\nupdatd: 2026-01-01\n---\n# S"), "updatd"),
        (lambda d: (d / "status.md").write_text("---\nupdated: soon\n---\n# S"), "a date like"),
        (lambda d: (d / "status.md").write_text("---\nupdated: 2026-01-01\n# S"), "closing ---"),
        (lambda d: (d / "projects" / "x.md").write_text("No title"), "'# Title'"),
        (
            lambda d: (d / "projects" / "x.md").write_text("---\nfeatured: maybe\n---\n# X"),
            "yes or no",
        ),
    ],
)
def test_bad_knowledge_base_fails_at_startup(kb_dir, change, error):
    change(kb_dir)
    with pytest.raises(PromptError, match=error):
        load_knowledge_base(kb_dir, 3)


def test_missing_folder(tmp_path):
    with pytest.raises(PromptError, match="folder not found"):
        load_knowledge_base(tmp_path / "nope", 3)


@pytest.mark.parametrize(
    ("link", "spoken"),
    [
        (
            "github.com/thanoojrao/voice-agent",
            "github dot com slash thanoojrao slash voice dash agent",
        ),
        ("https://www.example.com/a_b/", "example dot com slash a underscore b"),
    ],
)
def test_spoken_link(link, spoken):
    assert spoken_link(link) == spoken


# -- block 3: this call --------------------------------------------------------


def test_known_caller():
    text = call_context(call(caller_name="Sam"), None, 14)
    assert "Today is Sunday, November 15, 2026." in text
    assert "The caller is Sam, from an earlier call." in text


def test_mangled_name_is_treated_as_unknown():
    text = call_context(call(caller_name="um it's sam from acme"), None, 14)
    assert "The caller's name is unknown." in text and "acme" not in text


def test_anonymous_caller():
    text = call_context(call(caller_name="Sam", anonymous=True), None, 14)
    assert "number is withheld" in text and "Sam" not in text


def test_last_summary():
    text = call_context(
        call(
            caller_name="Sam",
            last_summary="Asked about the voice agent.",
            last_call_date=date(2026, 11, 1),
        ),
        None,
        14,
    )
    assert "Summary of our last call on November 1, 2026: Asked about the voice agent." in text


def test_stale_status_note():
    updated = date(2026, 10, 1)
    assert 'say "as of October 1, 2026"' in call_context(call(), updated, 14)
    assert "as of" not in call_context(call(), date(2026, 11, 10), 14)


# -- the whole prompt ----------------------------------------------------------


def test_system_blocks_order_and_cache_breakpoint():
    prompt = Prompt(
        instructions=load_instructions(VALUES),
        kb=load_knowledge_base(EXAMPLE_KB, 3),
        status_stale_days=14,
    )
    first, second = prompt.system(call(caller_name="Sam")), prompt.system(call(caller_name="Priya"))
    assert [b.cache for b in first] == [False, True, False]
    assert first[0].text.startswith("# Who I am")
    assert first[1].text.startswith("<knowledge_base>")
    assert first[2].text.startswith("<this_call>")
    # Only block 3 differs between callers, so the cached prefix is shared.
    assert first[:2] == second[:2] and first[2] != second[2]
