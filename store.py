"""
store.py — persistence so the bot remembers across restarts.

Dual backend, chosen automatically:
  • Turso (libSQL) over HTTP  — when TURSO_DATABASE_URL + TURSO_AUTH_TOKEN are
    set. Free, cloud-hosted, persists on Render. libSQL is a SQLite fork, so the
    SQL below is unchanged.
  • Local SQLite file         — otherwise (great for local dev and tests).

Everything goes through one `query()` helper, so the rest of the bot doesn't
care which backend is live. Turso calls are made resilient: a transient cloud
error degrades gracefully (reads return empty, writes are skipped + logged)
rather than breaking a customer's reply.

Tables:
  messages — per-user chat turns, used to rebuild LLM context after a restart
  turns    — audit log of every (in -> out) with source/lang (powers /stats)
  state    — per-user state: awaiting slot, sticky language, handoff flag
  leads    — actionable requests captured for the team
"""

import os
import sqlite3
from datetime import datetime, timezone

import requests

import config


def _now():
    return datetime.now(timezone.utc).isoformat()


def _use_turso():
    return bool(config.TURSO_DATABASE_URL and config.TURSO_AUTH_TOKEN)


def backend():
    return "turso" if _use_turso() else "sqlite"


# --- unified query interface --------------------------------------------------

def query(sql, params=(), fetch=False):
    """Run one statement. Returns list[dict] when fetch=True, else None."""
    if _use_turso():
        return _turso_query(sql, params, fetch)
    return _sqlite_query(sql, params, fetch)


# --- SQLite backend -----------------------------------------------------------

def _sqlite_conn():
    path = config.DB_PATH
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _sqlite_query(sql, params, fetch):
    with _sqlite_conn() as c:
        cur = c.execute(sql, tuple(params))
        if fetch:
            return [dict(r) for r in cur.fetchall()]
    return None


# --- Turso (libSQL) HTTP backend ----------------------------------------------

def _turso_endpoint():
    # libsql://name-org.turso.io  ->  https://name-org.turso.io/v2/pipeline
    base = config.TURSO_DATABASE_URL.replace("libsql://", "https://").rstrip("/")
    return f"{base}/v2/pipeline"


def _to_arg(v):
    if v is None:
        return {"type": "null"}
    if isinstance(v, bool):
        return {"type": "integer", "value": str(int(v))}
    if isinstance(v, int):
        return {"type": "integer", "value": str(v)}
    if isinstance(v, float):
        return {"type": "float", "value": str(v)}
    return {"type": "text", "value": str(v)}


def _from_cell(cell):
    t = cell.get("type")
    if t == "null":
        return None
    val = cell.get("value")
    if t == "integer":
        return int(val)
    if t == "float":
        return float(val)
    return val


def _turso_query(sql, params, fetch):
    payload = {
        "requests": [
            {"type": "execute",
             "stmt": {"sql": sql, "args": [_to_arg(p) for p in params]}},
            {"type": "close"},
        ]
    }
    try:
        resp = requests.post(
            _turso_endpoint(),
            headers={"Authorization": f"Bearer {config.TURSO_AUTH_TOKEN}"},
            json=payload,
            timeout=15,
        )
        if resp.status_code != 200:
            print(f"[store] turso HTTP {resp.status_code}: {resp.text[:200]}")
            return [] if fetch else None
        data = resp.json()
        result = data["results"][0]
        if result.get("type") == "error":
            print(f"[store] turso error: {result.get('error')}")
            return [] if fetch else None
        if not fetch:
            return None
        res = result["response"]["result"]
        cols = [c["name"] for c in res["cols"]]
        return [dict(zip(cols, (_from_cell(c) for c in row)))
                for row in res["rows"]]
    except (requests.RequestException, KeyError, ValueError) as e:
        print(f"[store] turso request failed: {e}")
        return [] if fetch else None


# --- schema -------------------------------------------------------------------

_SCHEMA = [
    """CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT, sender TEXT, channel TEXT, role TEXT, content TEXT)""",
    "CREATE INDEX IF NOT EXISTS ix_messages_sender ON messages(sender, id)",
    """CREATE TABLE IF NOT EXISTS turns (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT, sender TEXT, channel TEXT, lang TEXT,
        source TEXT, msg_in TEXT, msg_out TEXT)""",
    """CREATE TABLE IF NOT EXISTS state (
        sender TEXT PRIMARY KEY,
        awaiting TEXT, lang TEXT, handoff INTEGER DEFAULT 0, updated TEXT)""",
    """CREATE TABLE IF NOT EXISTS leads (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT, sender TEXT, channel TEXT,
        kind TEXT, text TEXT, status TEXT DEFAULT 'new')""",
]


def init():
    """Create tables/indexes if missing. Safe to call repeatedly."""
    for stmt in _SCHEMA:
        query(stmt)


# --- chat history -------------------------------------------------------------

def add_message(sender, role, content, channel="messenger"):
    query("INSERT INTO messages (ts, sender, channel, role, content) "
          "VALUES (?,?,?,?,?)", (_now(), sender, channel, role, content))


def get_history(sender, limit_messages):
    """Last `limit_messages` turns as [{role, content}], oldest first."""
    rows = query(
        "SELECT role, content FROM messages WHERE sender=? "
        "ORDER BY id DESC LIMIT ?", (sender, limit_messages), fetch=True)
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


# --- audit --------------------------------------------------------------------

def log_turn(sender, channel, lang, source, msg_in, msg_out):
    query("INSERT INTO turns (ts, sender, channel, lang, source, msg_in, msg_out)"
          " VALUES (?,?,?,?,?,?,?)",
          (_now(), sender, channel, lang, source, msg_in, msg_out))


def stats():
    total_rows = query("SELECT COUNT(*) AS n FROM turns", fetch=True)
    total = total_rows[0]["n"] if total_rows else 0
    rows = query("SELECT source, COUNT(*) AS n FROM turns GROUP BY source",
                 fetch=True)
    leads_rows = query("SELECT COUNT(*) AS n FROM leads WHERE status='new'",
                       fetch=True)
    handoff_rows = query("SELECT COUNT(*) AS n FROM state WHERE handoff=1",
                         fetch=True)
    by_source = {}
    for r in rows:
        key = (r["source"] or "?").split(":")[0]
        by_source[key] = by_source.get(key, 0) + r["n"]
    return {
        "backend": backend(),
        "messages_logged": total,
        "by_source": by_source,
        "open_leads": leads_rows[0]["n"] if leads_rows else 0,
        "active_handoffs": handoff_rows[0]["n"] if handoff_rows else 0,
    }


def recent_turns(limit=50):
    return query("SELECT ts, sender, channel, lang, source, msg_in, msg_out "
                 "FROM turns ORDER BY id DESC LIMIT ?", (limit,), fetch=True)


# --- per-user state -----------------------------------------------------------

def get_state(sender):
    rows = query("SELECT awaiting, lang, handoff FROM state WHERE sender=?",
                 (sender,), fetch=True)
    if not rows:
        return {"awaiting": None, "lang": None, "handoff": False}
    r = rows[0]
    return {"awaiting": r["awaiting"], "lang": r["lang"],
            "handoff": bool(r["handoff"])}


def set_state(sender, **fields):
    """Upsert any of: awaiting, lang, handoff."""
    cur = get_state(sender)
    cur.update({k: v for k, v in fields.items() if k in cur})
    query(
        "INSERT INTO state (sender, awaiting, lang, handoff, updated) "
        "VALUES (?,?,?,?,?) "
        "ON CONFLICT(sender) DO UPDATE SET "
        "awaiting=excluded.awaiting, lang=excluded.lang, "
        "handoff=excluded.handoff, updated=excluded.updated",
        (sender, cur["awaiting"], cur["lang"], int(bool(cur["handoff"])), _now()))


# --- leads (the team's worklist) ----------------------------------------------

def add_lead(sender, channel, kind, text):
    """Record an actionable request. Skips if an open lead of the same kind
    already exists for this sender. Returns True if added."""
    existing = query("SELECT 1 AS x FROM leads WHERE sender=? AND kind=? "
                     "AND status='new' LIMIT 1", (sender, kind), fetch=True)
    if existing:
        return False
    query("INSERT INTO leads (ts, sender, channel, kind, text, status) "
          "VALUES (?,?,?,?,?, 'new')", (_now(), sender, channel, kind, text))
    return True


def recent_leads(limit=50, status=None):
    if status:
        return query("SELECT id, ts, sender, channel, kind, text, status "
                     "FROM leads WHERE status=? ORDER BY id DESC LIMIT ?",
                     (status, limit), fetch=True)
    return query("SELECT id, ts, sender, channel, kind, text, status "
                 "FROM leads ORDER BY id DESC LIMIT ?", (limit,), fetch=True)


def active_handoffs():
    return query("SELECT sender, lang, updated FROM state WHERE handoff=1 "
                 "ORDER BY updated DESC", fetch=True)


# --- maintenance --------------------------------------------------------------

def reset_sender(sender):
    """Forget one user (history, state, leads)."""
    query("DELETE FROM messages WHERE sender=?", (sender,))
    query("DELETE FROM state WHERE sender=?", (sender,))
    query("DELETE FROM leads WHERE sender=?", (sender,))
