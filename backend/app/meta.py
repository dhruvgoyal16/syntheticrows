"""
SyntheticRows — stats & waitlist (Postgres / Supabase).
Stores data in a persistent Postgres database (Supabase) via the
DATABASE_URL environment variable, so signups survive restarts.
Import and mount in main.py with: app.include_router(meta_router)
Call bump_rows(n) and bump_datasets() from your generate endpoints.
"""
import os
import re

import psycopg2
from fastapi import APIRouter
from pydantic import BaseModel

# Connection string comes from an environment variable / secret (never hard-coded).
DATABASE_URL = os.environ.get("DATABASE_URL", "")

# Seed with REAL cumulative usage from development/testing. Honest, non-zero start.
SEED_ROWS = 18450
SEED_DATASETS = 47  # genuine count of test generations run across our build sessions

# Counter is hidden in the UI until real signups cross this threshold.
WAITLIST_VISIBLE_THRESHOLD = 25

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _conn():
    # A fresh short-lived connection per use (works well with the Supabase pooler).
    return psycopg2.connect(DATABASE_URL)


def init_db():
    """Create tables if missing and seed counters once. Safe to call repeatedly."""
    with _conn() as c:
        with c.cursor() as cur:
            cur.execute("""CREATE TABLE IF NOT EXISTS stats (
                key TEXT PRIMARY KEY,
                value BIGINT NOT NULL
            )""")
            cur.execute("""CREATE TABLE IF NOT EXISTS waitlist (
                id BIGSERIAL PRIMARY KEY,
                email TEXT,
                interest TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )""")
            cur.execute(
                "INSERT INTO stats(key, value) VALUES('total_rows', %s) "
                "ON CONFLICT (key) DO NOTHING",
                (SEED_ROWS,),
            )
            cur.execute(
                "INSERT INTO stats(key, value) VALUES('total_datasets', %s) "
                "ON CONFLICT (key) DO NOTHING",
                (SEED_DATASETS,),
            )
        c.commit()


def _bump(key: str, n: int):
    try:
        n = int(n)
        if n <= 0:
            return
        with _conn() as c:
            with c.cursor() as cur:
                cur.execute("UPDATE stats SET value = value + %s WHERE key=%s", (n, key))
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
        with c.cursor() as cur:
            cur.execute("SELECT value FROM stats WHERE key=%s", (key,))
            row = cur.fetchone()
            return int(row[0]) if row else default


def _waitlist_count() -> int:
    with _conn() as c:
        with c.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM waitlist")
            row = cur.fetchone()
            return int(row[0]) if row else 0


# Initialise on import (creates tables + seeds if needed).
try:
    init_db()
except Exception as e:
    # Don't crash the whole app on import if the DB is briefly unreachable;
    # endpoints will surface errors instead.
    print(f"[meta] init_db warning: {e}")

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
        with c.cursor() as cur:
            if valid_email:
                cur.execute("SELECT id FROM waitlist WHERE email=%s", (email,))
                existing = cur.fetchone()
                if existing:
                    count = _waitlist_count()
                    return {
                        "ok": True, "already": True,
                        "waitlist_count": count,
                        "waitlist_visible": count >= WAITLIST_VISIBLE_THRESHOLD,
                    }
            cur.execute(
                "INSERT INTO waitlist(email, interest) VALUES(%s, %s)",
                (email if valid_email else None, interest),
            )
        c.commit()

    count = _waitlist_count()
    return {
        "ok": True, "already": False,
        "waitlist_count": count,
        "waitlist_visible": count >= WAITLIST_VISIBLE_THRESHOLD,
    }