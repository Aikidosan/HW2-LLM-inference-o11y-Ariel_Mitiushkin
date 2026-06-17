"""Schema-rendering helper (provided complete).

Loads the schema directly from sqlite and renders quoted CREATE TABLE
text suitable for prompt context. Identifiers are always double-quoted
so reserved-word table/column names (e.g. `order`) don't break either
the PRAGMA introspection here or the SQL the model emits later.
"""
from __future__ import annotations

import sqlite3
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_DIR = ROOT / "data" / "bird"


def db_path(db_id: str) -> Path:
    return DB_DIR / f"{db_id}.sqlite"


def _q(ident: str) -> str:
    """Double-quote a SQL identifier, escaping any embedded quotes."""
    return '"' + ident.replace('"', '""') + '"'


# Number of distinct example values to show per column. Sample values are the
# single biggest accuracy lever for BIRD-style text-to-SQL: they let the model
# (a) pick the right column when several are plausible and (b) match string
# literals *exactly* (e.g. "Australian Grand Prix" vs "Australian GP"), which is
# the most common silent-wrong-answer cause on this benchmark.
SAMPLE_VALUES_PER_COLUMN = 3
# Cap each rendered value so a single wide TEXT/BLOB cell can't blow up the
# prompt (and the prefix-cache budget). Values longer than this are truncated.
MAX_SAMPLE_VALUE_LEN = 40


def _sample_values(conn: sqlite3.Connection, table: str, col: str) -> str:
    """Return a short ", "-joined preview of distinct non-null values, or "".

    Best-effort: any error (odd type, huge BLOB, etc.) yields no hint rather
    than failing schema rendering.
    """
    try:
        rows = conn.execute(
            f"SELECT DISTINCT {_q(col)} FROM {_q(table)} "
            f"WHERE {_q(col)} IS NOT NULL LIMIT {SAMPLE_VALUES_PER_COLUMN}"
        ).fetchall()
    except sqlite3.Error:
        return ""
    vals: list[str] = []
    for (v,) in rows:
        s = str(v).replace("\n", " ").strip()
        if len(s) > MAX_SAMPLE_VALUE_LEN:
            s = s[:MAX_SAMPLE_VALUE_LEN] + "…"
        vals.append(s)
    return ", ".join(vals)


@lru_cache(maxsize=32)
def render_schema(db_id: str) -> str:
    path = db_path(db_id)
    if not path.exists():
        raise FileNotFoundError(f"DB {db_id} not found at {path}. Did you run scripts/load_data.py?")

    parts: list[str] = [f"-- Database: {db_id}"]
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name"
            )
        ]
        for t in tables:
            parts.append(f"\nCREATE TABLE {_q(t)} (")
            col_lines: list[str] = []
            for _cid, name, ctype, notnull, _dflt, pk in conn.execute(f"PRAGMA table_info({_q(t)})"):
                line = f"  {_q(name)} {ctype}"
                if pk:
                    line += " PRIMARY KEY"
                if notnull and not pk:
                    line += " NOT NULL"
                examples = _sample_values(conn, t, name)
                if examples:
                    line += f"  -- e.g. {examples}"
                col_lines.append(line)
            for fk in conn.execute(f"PRAGMA foreign_key_list({_q(t)})"):
                # (id, seq, ref_table, from, to, on_update, on_delete, match)
                from_col, ref_table, to_col = fk[3], fk[2], fk[4]
                # SQLite leaves `to` NULL when the FK references the parent's PK
                # implicitly (no column named). Render REFERENCES "table" in that
                # case instead of crashing on _q(None).
                ref = f"{_q(ref_table)}({_q(to_col)})" if to_col is not None else _q(ref_table)
                col_lines.append(f"  FOREIGN KEY ({_q(from_col)}) REFERENCES {ref}")
            parts.append(",\n".join(col_lines))
            parts.append(");")
    return "\n".join(parts)


def available_dbs() -> list[str]:
    if not DB_DIR.exists():
        return []
    return sorted(p.stem for p in DB_DIR.glob("*.sqlite"))
