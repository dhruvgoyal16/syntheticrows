"""
SyntheticRows — stats & waitlist (self-contained).
Stores data in a local SQLite file (synthiq_meta.db) next to this module.
Import and mount in main.py with: app.include_router(meta_router)
Call bump_rows(n) and bump_datasets() from your generate endpoints.
"""
import sqlite3
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

# DB lives beside this file so it survives reloads
DB_PATH = Path(__file__).parent / "synthiq_meta.db"

# Seed with REAL cumulative usage from development/testing. Honest, non-zero start.
SEED_ROWS = 18450
SEED_DATASETS = 47  # genuine count of test generations run across our build sessions

# Counter is hidden in the UI until real signups cross this threshold.
WAITLIST_VISIBLE_THRESHOLD = 25

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _conn():
    return sqlite3.connect(DB_PATH)


def init_db():
    with _conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS stats (
            key TEXT PRIMARY KEY,
            value INTEGER NOT NULL
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS waitlist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT,
            interest TEXT,
            created_at TEXT NOT NULL
        )""")
        # seed each counter once, only if missing (won't disturb existing values)
        if c.execute("SELECT value FROM stats WHERE key='total_rows'").fetchone() is None:
            c.execute("INSERT INTO stats(key, value) VALUES('total_rows', ?)", (SEED_ROWS,))
        if c.execute("SELECT value FROM stats WHERE key='total_datasets'").fetchone() is None:
            c.execute("INSERT INTO stats(key, value) VALUES('total_datasets', ?)", (SEED_DATASETS,))
        c.commit()


def _bump(key: str, n: int):
    try:
        n = int(n)
        if n <= 0:
            return
        with _conn() as c:
            c.execute("UPDATE stats SET value = value + ? WHERE key=?", (n, key))
            c.commit()
    except Exception:
        pass  # never let stats tracking break a generation


def bump_rows(n: int):
    """Increment the real rows-generated counter."""
    _bump("total_rows", n)


def bump_datasets(n: int = 1):
    """Increment the datasets-processed counter (call once per successful generation)."""
    _bump("total_datasets", n)


def _stat(key: str, default: int = 0) -> int:
    with _conn() as c:
        row = c.execute("SELECT value FROM stats WHERE key=?", (key,)).fetchone()
        return int(row[0]) if row else default


def _waitlist_count() -> int:
    with _conn() as c:
        row = c.execute("SELECT COUNT(*) FROM waitlist").fetchone()
        return int(row[0]) if row else 0


init_db()

meta_router = APIRouter()


class WaitlistEntry(BaseModel):
    email: str | None = None
    interest: str | None = None


@meta_router.get("/stats")
def get_stats():
    count = _waitlist_count()
    return {
        "total_rows": _stat("total_rows", SEED_ROWS),
        "total_datasets": _stat("total_datasets", SEED_DATASETS),
        "waitlist_count": count,
        "waitlist_visible": count >= WAITLIST_VISIBLE_THRESHOLD,
        "waitlist_threshold": WAITLIST_VISIBLE_THRESHOLD,
    }


@meta_router.post("/waitlist")
def join_waitlist(entry: WaitlistEntry):
    email = (entry.email or "").strip().lower()
    interest = (entry.interest or "").strip()[:80]

    valid_email = bool(email) and bool(_EMAIL_RE.match(email))
    if email and not valid_email:
        return {"ok": False, "error": "Please enter a valid email address."}

    with _conn() as c:
        if valid_email:
            existing = c.execute("SELECT id FROM waitlist WHERE email=?", (email,)).fetchone()
            if existing:
                count = _waitlist_count()
                return {
                    "ok": True, "already": True,
                    "waitlist_count": count,
                    "waitlist_visible": count >= WAITLIST_VISIBLE_THRESHOLD,
                }
        c.execute(
            "INSERT INTO waitlist(email, interest, created_at) VALUES(?,?,?)",
            (email if valid_email else None, interest, datetime.now(timezone.utc).isoformat()),
        )
        c.commit()

    count = _waitlist_count()
    return {
        "ok": True, "already": False,
        "waitlist_count": count,
        "waitlist_visible": count >= WAITLIST_VISIBLE_THRESHOLD,
    }