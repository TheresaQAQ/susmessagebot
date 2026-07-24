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
    init_review_evidence_table()
    init_auto_ban_table()
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


def get_review_decision_owner(
    guild_id: int,
    message_id: int,
    user_id: int,
) -> tuple[str, int] | None:
    """Return (decision, decided_by) when a claim exists."""
    key = review_key(guild_id, message_id, user_id)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT decision, decided_by FROM review_decisions WHERE review_key = ?",
        (key,),
    )
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    return row[0], int(row[1])


def release_review_decision(
    guild_id: int,
    message_id: int,
    user_id: int,
    decision: str,
    decided_by: int,
) -> bool:
    """
    Release a claim so a failed review action can be retried.

    Only deletes the row when decision and decided_by still match the claim.
    """
    key = review_key(guild_id, message_id, user_id)
    conn = sqlite3.connect(DB_PATH, timeout=30, isolation_level=None)
    try:
        cursor = conn.cursor()
        cursor.execute("BEGIN IMMEDIATE")
        try:
            cursor.execute(
                """
                DELETE FROM review_decisions
                WHERE review_key = ? AND decision = ? AND decided_by = ?
                """,
                (key, decision, decided_by),
            )
            deleted = cursor.rowcount > 0
            conn.execute("COMMIT")
            return deleted
        except Exception:
            conn.execute("ROLLBACK")
            raise
    finally:
        conn.close()


def init_review_evidence_table():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS review_evidence (
            review_key TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            reason TEXT NOT NULL DEFAULT '',
            created_at REAL NOT NULL
        )
    ''')
    conn.commit()
    conn.close()


def store_review_evidence(
    guild_id: int,
    message_id: int,
    user_id: int,
    content: str,
    reason: str = "",
) -> None:
    key = review_key(guild_id, message_id, user_id)
    now = time.time()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO review_evidence (review_key, content, reason, created_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(review_key) DO UPDATE SET
            content = excluded.content,
            reason = excluded.reason,
            created_at = excluded.created_at
        """,
        (key, content, reason, now),
    )
    conn.commit()
    conn.close()


def get_review_evidence(
    guild_id: int,
    message_id: int,
    user_id: int,
) -> str | None:
    key = review_key(guild_id, message_id, user_id)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT content FROM review_evidence WHERE review_key = ?",
        (key,),
    )
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None


def get_review_reason(
    guild_id: int,
    message_id: int,
    user_id: int,
) -> str | None:
    key = review_key(guild_id, message_id, user_id)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT reason FROM review_evidence WHERE review_key = ?",
        (key,),
    )
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None


def init_auto_ban_table():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS auto_bans (
            scope_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            message_id INTEGER NOT NULL,
            created_at REAL NOT NULL,
            PRIMARY KEY (scope_id, user_id)
        )
    ''')
    conn.commit()
    conn.close()


def record_auto_ban(scope_id: int, user_id: int, message_id: int) -> None:
    """Remember which message triggered an automatic ban for false-alarm reversal."""
    now = time.time()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO auto_bans (scope_id, user_id, message_id, created_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(scope_id, user_id) DO UPDATE SET
            message_id = excluded.message_id,
            created_at = excluded.created_at
        """,
        (scope_id, user_id, message_id, now),
    )
    conn.commit()
    conn.close()


def clear_auto_ban(scope_id: int, user_id: int) -> None:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "DELETE FROM auto_bans WHERE scope_id = ? AND user_id = ?",
        (scope_id, user_id),
    )
    conn.commit()
    conn.close()


def take_reversible_auto_ban(
    scope_id: int,
    user_id: int,
    message_id: int,
) -> bool:
    """
    Consume an auto-ban record if it was triggered by this message.

    Returns True only when a false-alarm for this message should unban.
    """
    conn = sqlite3.connect(DB_PATH, timeout=30, isolation_level=None)
    try:
        cursor = conn.cursor()
        cursor.execute("BEGIN IMMEDIATE")
        try:
            cursor.execute(
                """
                DELETE FROM auto_bans
                WHERE scope_id = ? AND user_id = ? AND message_id = ?
                """,
                (scope_id, user_id, message_id),
            )
            deleted = cursor.rowcount > 0
            conn.execute("COMMIT")
            return deleted
        except Exception:
            conn.execute("ROLLBACK")
            raise
    finally:
        conn.close()


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


def remove_group(chat_id: int) -> bool:
    """Remove a group row. Returns True if a row was deleted."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM groups WHERE chat_id = ?', (chat_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted

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