"""
database/schema.py
──────────────────
Creates all tables and seeds dummy data on first run.
Idempotent – safe to call every startup.
"""

import os
from database.db import DatabaseManager


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT NOT NULL,
    department TEXT NOT NULL,
    email      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS leave_balance (
    user_id          INTEGER PRIMARY KEY,
    total_leaves     INTEGER NOT NULL DEFAULT 20,
    used_leaves      INTEGER NOT NULL DEFAULT 0,
    remaining_leaves INTEGER NOT NULL DEFAULT 20,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS leave_requests (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL,
    start_date TEXT NOT NULL,
    end_date   TEXT NOT NULL,
    reason     TEXT NOT NULL,
    status     TEXT NOT NULL DEFAULT 'Pending',
    created_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS hr_requests (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      INTEGER NOT NULL,
    request_text TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS conversation_history (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    user_id    INTEGER NOT NULL,
    role       TEXT NOT NULL,  -- 'user' | 'assistant'
    content    TEXT NOT NULL,
    agent      TEXT,           -- which agent produced this response
    intent     TEXT,           -- detected intent (for assistant turns)
    timestamp  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id   TEXT PRIMARY KEY,
    user_id      INTEGER NOT NULL,
    active_agent TEXT,          -- agent currently handling multi-turn dialogue
    agent_state  TEXT,          -- JSON blob for slot-filling state
    updated_at   TEXT NOT NULL
);
"""

# Dummy users to seed for first run
SEED_USERS = [
    (1, "Alice Johnson",  "Engineering",   "alice@acmecorp.com"),
    (2, "Bob Martinez",   "Marketing",     "bob@acmecorp.com"),
    (3, "Carol Williams", "HR",            "carol@acmecorp.com"),
]

SEED_BALANCES = [
    # (user_id, total, used, remaining)
    (1, 20, 5, 15),
    (2, 20, 8, 12),
    (3, 20, 2, 18),
]


def initialize_database(db: DatabaseManager, active_user_id: int) -> None:
    """
    Create all tables and seed data if this is a fresh database.
    Ensures the active_user_id from config always exists.
    """
    # Create tables
    for statement in SCHEMA_SQL.strip().split(";"):
        stmt = statement.strip()
        if stmt:
            db.execute(stmt)

    # Seed dummy users only if the table is empty
    existing = db.fetch_one("SELECT COUNT(*) AS cnt FROM users")
    if existing and existing["cnt"] == 0:
        for uid, name, dept, email in SEED_USERS:
            db.execute(
                "INSERT OR IGNORE INTO users (id, name, department, email) VALUES (?, ?, ?, ?)",
                (uid, name, dept, email),
            )
        for uid, total, used, remaining in SEED_BALANCES:
            db.execute(
                """INSERT OR IGNORE INTO leave_balance
                   (user_id, total_leaves, used_leaves, remaining_leaves)
                   VALUES (?, ?, ?, ?)""",
                (uid, total, used, remaining),
            )

    # Always ensure the active user exists (idempotent)
    user = db.fetch_one("SELECT id FROM users WHERE id = ?", (active_user_id,))
    if not user:
        db.execute(
            "INSERT OR IGNORE INTO users (id, name, department, email) VALUES (?, ?, ?, ?)",
            (active_user_id, f"Demo User {active_user_id}", "General", "demo@acmecorp.com"),
        )
        db.execute(
            """INSERT OR IGNORE INTO leave_balance
               (user_id, total_leaves, used_leaves, remaining_leaves)
               VALUES (?, 20, 0, 20)""",
            (active_user_id,),
        )
