"""Personal email addresses must never reach the public repo.

The real addresses come from places git never sees (.env and the private kb/), so this test
names nothing itself, and a failure lists only file names. It skips where neither exists,
such as a fresh clone.
"""

import re
import subprocess
from pathlib import Path

import pytest

from voice_agent.app.config import Settings

ROOT = Path(__file__).resolve().parents[2]
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
# The knowledge base may write an address for speech: "jane at example dot com".
SPOKEN_EMAIL = re.compile(r"([\w.+-]+) at ([\w-]+) dot (\w+)", re.IGNORECASE)
MIN_NAME = 6  # shorter parts before the @ ("sam") are too common to check on their own


def private_addresses() -> set[str]:
    settings = Settings(_env_file=ROOT / ".env")  # type: ignore[call-arg]
    found = {v.lower() for v in (settings.owner_email, settings.gmail_address) if v}
    kb = ROOT / "kb"
    if kb.is_dir():
        for path in kb.rglob("*.md"):
            text = path.read_text(encoding="utf-8")
            found |= {m.lower() for m in EMAIL.findall(text)}
            found |= {f"{a}@{b}.{c}".lower() for a, b, c in SPOKEN_EMAIL.findall(text)}
    return found


def needles(addresses: set[str]) -> set[str]:
    """Each address, and the part before its @ when that's distinctive enough."""
    out = set(addresses)
    out |= {a.split("@")[0] for a in addresses if len(a.split("@")[0]) >= MIN_NAME}
    return out


def tracked_files() -> list[Path]:
    listing = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True
    ).stdout.decode()
    return [ROOT / name for name in listing.split("\0") if name]


def test_no_private_email_in_tracked_files():
    look_for = needles(private_addresses())
    if not look_for:
        pytest.skip("no private address to check against (no .env values or kb/)")
    leaks = set()
    for path in tracked_files():
        try:
            text = path.read_text(encoding="utf-8").lower()
        except (UnicodeDecodeError, OSError):
            continue  # binary or missing
        if any(needle in text for needle in look_for):
            leaks.add(str(path.relative_to(ROOT)))
    assert not leaks, f"a private email address appears in: {sorted(leaks)}"


def test_needles_include_the_name_before_the_at():
    assert needles({"jane.doe@example.com", "sam@x.io"}) == {
        "jane.doe@example.com",
        "jane.doe",
        "sam@x.io",
    }
