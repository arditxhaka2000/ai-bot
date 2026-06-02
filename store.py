"""
store.py — SQLite persistence so the bot remembers across restarts.

Tables:
  • messages — per-user chat turns, used to rebuild LLM context after a restart
  • turns    — audit log of every (in -> out) with source/lang (powers /stats)
  • state    — per-user state: awaiting slot, sticky language, handoff flag
  • leads    — actionable requests captured for the team (bookings, pickups,
               address changes, returns, damage claims, escalations)

Connections are opened per call (simple and thread-safe under gunicorn). The
DB path is read from config at call time so tests can point it at a temp file.
"""

import os
import sqlite3
from datetime import datetime, timezone

import config


def _now():
    return datetime.now(timezone.utc).isoformat()


def _conn():
    path = config.DB_PATH
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init():
    """Create tables/indexes if they don't exist. Safe to call repeatedly."""
    with _conn() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                ts      TEXT, sender TEXT, channel TEXT,
                role    TEXT, content TEXT
            );
            CREATE INDEX IF NOT EXISTS ix_messages_sender ON messages(sender, id);

            CREATE TABLE IF NOT EXISTS turns (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                ts      TEXT, sender TEXT, channel TEXT, lang TEXT,
                source  TEXT, msg_in TEXT, msg_out TEXT
            );

            CREATE TABLE IF NOT EXISTS state (
                sender   TEXT PRIMARY KEY,
                awaiting TEXT, lang TEXT, handoff INTEGER DEFAULT 0,
                updated  TEXT
            );

            CREATE TABLE IF NOT EXISTS leads (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                ts      TEXT, sender TEXT, channel TEXT,
                kind    TEXT, text TEXT, status TEXT DEFAULT 'new'
            );
            """
        )


# --- chat history -------------------------------------------------------------

def add_message(sender, role, content, channel="messenger"):
    with _conn() as c:
        c.execute(
            "INSERT INTO messages (ts, sender, channel, role, content) "
            "VALUES (?,?,?,?,?)",
            (_now(), sender, channel, role, content),
        )


def get_history(sender, limit_messages):
    """Return the last `limit_messages` turns as [{role, content}], oldest first."""
    with _conn() as c:
        rows = c.execute(
            "SELECT role, content FROM messages WHERE sender=? "
            "ORDER BY id DESC LIMIT ?",
            (sender, limit_messages),
        ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


# --- audit --------------------------------------------------------------------

def log_turn(sender, channel, lang, source, msg_in, msg_out):
    with _conn() as c:
        c.execute(
            "INSERT INTO turns (ts, sender, channel, lang, source, msg_in, msg_out) "
            "VALUES (?,?,?,?,?,?,?)",
            (_now(), sender, channel, lang, source, msg_in, msg_out),
        )


def stats():
    with _conn() as c:
        total = c.execute("SELECT COUNT(*) AS n FROM turns").fetchone()["n"]
        rows = c.execute(
            "SELECT source, COUNT(*) AS n FROM turns GROUP BY source"
        ).fetchall()
        leads = c.execute(
            "SELECT COUNT(*) AS n FROM leads WHERE status='new'"
        ).fetchone()["n"]
        handoffs = c.execute(
            "SELECT COUNT(*) AS n FROM state WHERE handoff=1"
        ).fetchone()["n"]
    by_source = {}
    for r in rows:
        key = (r["source"] or "?").split(":")[0]
        by_source[key] = by_source.get(key, 0) + r["n"]
    return {"messages_logged": total, "by_source": by_source,
            "open_leads": leads, "active_handoffs": handoffs}


def recent_turns(limit=50):
    with _conn() as c:
        rows = c.execute(
            "SELECT ts, sender, channel, lang, source, msg_in, msg_out "
            "FROM turns ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


# --- per-user state -----------------------------------------------------------

def get_state(sender):
    with _conn() as c:
        row = c.execute(
            "SELECT awaiting, lang, handoff FROM state WHERE sender=?", (sender,)
        ).fetchone()
    if not row:
        return {"awaiting": None, "lang": None, "handoff": False}
    return {"awaiting": row["awaiting"], "lang": row["lang"],
            "handoff": bool(row["handoff"])}


def set_state(sender, **fields):
    """Upsert any of: awaiting, lang, handoff."""
    cur = get_state(sender)
    cur.update({k: v for k, v in fields.items() if k in cur})
    with _conn() as c:
        c.execute(
            "INSERT INTO state (sender, awaiting, lang, handoff, updated) "
            "VALUES (?,?,?,?,?) "
            "ON CONFLICT(sender) DO UPDATE SET "
            "awaiting=excluded.awaiting, lang=excluded.lang, "
            "handoff=excluded.handoff, updated=excluded.updated",
            (sender, cur["awaiting"], cur["lang"], int(bool(cur["handoff"])),
             _now()),
        )


# --- leads (the team's worklist) ----------------------------------------------

def add_lead(sender, channel, kind, text):
    """Record an actionable request. Skips if an open lead of the same kind
    already exists for this sender (avoids duplicates). Returns True if added."""
    with _conn() as c:
        existing = c.execute(
            "SELECT 1 FROM leads WHERE sender=? AND kind=? AND status='new' "
            "LIMIT 1", (sender, kind)
        ).fetchone()
        if existing:
            return False
        c.execute(
            "INSERT INTO leads (ts, sender, channel, kind, text, status) "
            "VALUES (?,?,?,?,?, 'new')",
            (_now(), sender, channel, kind, text),
        )
    return True


def recent_leads(limit=50, status=None):
    query = "SELECT id, ts, sender, channel, kind, text, status FROM leads"
    params = []
    if status:
        query += " WHERE status=?"
        params.append(status)
    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    with _conn() as c:
        rows = c.execute(query, params).fetchall()
    return [dict(r) for r in rows]


def active_handoffs():
    with _conn() as c:
        rows = c.execute(
            "SELECT sender, lang, updated FROM state WHERE handoff=1 "
            "ORDER BY updated DESC"
        ).fetchall()
    return [dict(r) for r in rows]


# --- maintenance --------------------------------------------------------------

def reset_sender(sender):
    """Forget one user (history, state, leads). Used by responder.reset()."""
    with _conn() as c:
        c.execute("DELETE FROM messages WHERE sender=?", (sender,))
        c.execute("DELETE FROM state WHERE sender=?", (sender,))
        c.execute("DELETE FROM leads WHERE sender=?", (sender,))
