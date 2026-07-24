import sqlite3
import os
import time

from .config import STATS_DB_PATH

DB_PATH = STATS_DB_PATH

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stats (
            key TEXT PRIMARY KEY,
            value INTEGER DEFAULT 0
        )
    ''')
    for key in ['messages_safe', 'messages_ban', 'bans_confirmed', 'false_positives', 'false_negatives', 'accurate_classifications']:
        cursor.execute('INSERT OR IGNORE INTO stats (key, value) VALUES (?, 0)', (key,))
    conn.commit()
    conn.close()
    init_groups_table()
    init_review_decisions_table()
    init_strikes_table()


def init_review_decisions_table():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS review_decisions (
            review_key TEXT PRIMARY KEY,
            decision TEXT NOT NULL,
            decided_by INTEGER,
            decided_at REAL NOT NULL
        )
    ''')
    conn.commit()
    conn.close()


def review_key(guild_id: int, message_id: int, user_id: int) -> str:
    return f"{guild_id}:{message_id}:{user_id}"


def claim_review_decision(
    guild_id: int,
    message_id: int,
    user_id: int,
    decision: str,
    decided_by: int,
) -> str | None:
    """
    Atomically claim the first admin decision for a review event.

    Returns None if this caller won the claim, otherwise the existing decision.
    """
    key = review_key(guild_id, message_id, user_id)
    now = time.time()
    # isolation_level=None so we can use an explicit IMMEDIATE transaction.
    conn = sqlite3.connect(DB_PATH, timeout=30, isolation_level=None)
    try:
        cursor = conn.cursor()
        cursor.execute("BEGIN IMMEDIATE")
        try:
            cursor.execute(
                "SELECT decision FROM review_decisions WHERE review_key = ?",
                (key,),
            )
            row = cursor.fetchone()
            if row:
                conn.execute("ROLLBACK")
                return row[0]
            cursor.execute(
                """
                INSERT INTO review_decisions (review_key, decision, decided_by, decided_at)
                VALUES (?, ?, ?, ?)
                """,
                (key, decision, decided_by, now),
            )
            conn.execute("COMMIT")
            return None
        except Exception:
            conn.execute("ROLLBACK")
            raise
    finally:
        conn.close()


def get_review_decision(
    guild_id: int,
    message_id: int,
    user_id: int,
) -> str | None:
    key = review_key(guild_id, message_id, user_id)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT decision FROM review_decisions WHERE review_key = ?",
        (key,),
    )
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None


def init_strikes_table():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS strikes (
            scope_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            ts REAL NOT NULL
        )
    ''')
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_strikes_scope_user_ts
        ON strikes (scope_id, user_id, ts)
    ''')
    conn.commit()
    conn.close()

def get_stat(key: str) -> int:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT value FROM stats WHERE key = ?', (key,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else 0

def increment_stat(key: str) -> int:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('UPDATE stats SET value = value + 1 WHERE key = ?', (key,))
    cursor.execute('SELECT value FROM stats WHERE key = ?', (key,))
    new_value = cursor.fetchone()[0]
    conn.commit()
    conn.close()
    return new_value

def init_groups_table():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS groups (
            chat_id INTEGER PRIMARY KEY,
            member_count INTEGER DEFAULT 0,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def add_group(chat_id: int, member_count: int) -> bool:
    """Add or refresh a group. Returns True if it was a new group."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT chat_id FROM groups WHERE chat_id = ?', (chat_id,))
    exists = cursor.fetchone()
    if not exists:
        cursor.execute('INSERT INTO groups (chat_id, member_count) VALUES (?, ?)', (chat_id, member_count))
    else:
        cursor.execute('''
            UPDATE groups SET member_count = ?, last_updated = CURRENT_TIMESTAMP
            WHERE chat_id = ?
        ''', (member_count, chat_id))
    conn.commit()
    conn.close()
    return not exists

def update_group_member_count(chat_id: int, member_count: int):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE groups SET member_count = ?, last_updated = CURRENT_TIMESTAMP 
        WHERE chat_id = ?
    ''', (member_count, chat_id))
    conn.commit()
    conn.close()

def get_groups_count() -> int:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM groups')
    count = cursor.fetchone()[0]
    conn.close()
    return count

def get_total_members() -> int:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT SUM(member_count) FROM groups')
    total = cursor.fetchone()[0]
    conn.close()
    return total or 0

def get_all_group_ids() -> list:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT chat_id FROM groups')
    ids = [row[0] for row in cursor.fetchall()]
    conn.close()
    return ids

def decrement_stat(key: str) -> int:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('UPDATE stats SET value = MAX(0, value - 1) WHERE key = ?', (key,))
    cursor.execute('SELECT value FROM stats WHERE key = ?', (key,))
    new_value = cursor.fetchone()[0]
    conn.commit()
    conn.close()
    return new_value