import sqlite3
import os

from config import STATS_DB_PATH

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
    """Add a new group. Returns True if it was a new group, False if already exists."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT chat_id FROM groups WHERE chat_id = ?', (chat_id,))
    exists = cursor.fetchone()
    if not exists:
        cursor.execute('INSERT INTO groups (chat_id, member_count) VALUES (?, ?)', (chat_id, member_count))
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