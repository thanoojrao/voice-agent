"""Apply pending SQL migrations from migrations/NNNN_name.sql, in order.

Run it:   uv run python -m voice_agent.storage.migrate [db_path]
The app also calls apply_migrations() at startup, and tests call it on a fresh database.

Uses the stdlib sqlite3 driver on purpose: migrations run once, before any async work.
"""

import re
import sqlite3
import sys
from pathlib import Path

MIGRATIONS_DIR = Path(__file__).resolve().parents[3] / "migrations"
_NAME = re.compile(r"^(\d{4})_[a-z0-9_]+\.sql$")


def discover(migrations_dir: Path = MIGRATIONS_DIR) -> list[tuple[int, Path]]:
    """Return (version, path) pairs sorted by version. Rejects bad names and duplicate numbers."""
    if not migrations_dir.is_dir():
        raise RuntimeError(f"Migrations folder not found: {migrations_dir}")
    found: dict[int, Path] = {}
    for path in migrations_dir.glob("*.sql"):
        m = _NAME.match(path.name)
        if not m:
            raise RuntimeError(f"Bad migration file name: {path.name} (expected NNNN_name.sql)")
        version = int(m.group(1))
        if version in found:
            raise RuntimeError(
                f"Two migrations numbered {version:04d}: {found[version].name}, {path.name}"
            )
        found[version] = path
    return sorted(found.items())


def current_version(conn: sqlite3.Connection) -> int:
    conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
    row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
    return row[0] or 0


def apply_migrations(db_path: Path | str, migrations_dir: Path = MIGRATIONS_DIR) -> list[int]:
    """Apply every migration newer than the database's version. Returns the versions applied.

    Each migration and its version bump run in one transaction, so a failing file leaves
    the database exactly as it was before that file.
    """
    if str(db_path) != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, isolation_level=None)  # we manage transactions ourselves
    try:
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA journal_mode=WAL")
        have = current_version(conn)
        applied = []
        for version, path in discover(migrations_dir):
            if version <= have:
                continue
            sql = path.read_text(encoding="utf-8")
            try:
                bump = f"INSERT INTO schema_version (version) VALUES ({version});"
                conn.executescript(f"BEGIN;\n{sql}\n;{bump}\nCOMMIT;")
            except sqlite3.Error as exc:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise RuntimeError(f"Migration {path.name} failed: {exc}") from exc
            applied.append(version)
        return applied
    finally:
        conn.close()


def main() -> None:
    db_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/agent.db")
    applied = apply_migrations(db_path)
    if applied:
        print(f"{db_path}: applied {', '.join(f'{v:04d}' for v in applied)}")
    else:
        print(f"{db_path}: already up to date")


if __name__ == "__main__":
    main()
