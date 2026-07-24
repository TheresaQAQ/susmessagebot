"""Short-window strike tracking for auto-ban after repeated BAN triggers."""
from __future__ import annotations

import sqlite3
import time

from .config import STATS_DB_PATH

# 3 BAN classifications within this window → auto-ban (no admin confirm).
STRIKE_WINDOW_SECONDS = 600
STRIKE_THRESHOLD = 3


def _fmt_duration(seconds: int) -> str:
    seconds = max(0, int(seconds))
    mins, secs = divmod(seconds, 60)
    if mins and secs:
        return f"{mins}m{secs}s"
    if mins:
        return f"{mins}m"
    return f"{secs}s"


def remove_notice_text(scope_id: int, user_id: int) -> str:
    """English-only delete notice with strike remaining + time until oldest strike expires."""
    count = strikes.count(scope_id, user_id)
    remaining = max(0, strikes.threshold - count)
    reset_in = strikes.seconds_until_oldest_expires(scope_id, user_id)
    reset_str = _fmt_duration(reset_in)
    window_min = strikes.window_seconds // 60
    return (
        "Your message was removed for violating community rules.\n"
        f"Repeated triggers within a {window_min}-minute sliding window lead to an auto-ban. "
        f"Status {count}/{strikes.threshold}; {remaining} more trigger(s) until ban. "
        f"Oldest strike expires in about {reset_str}."
    )


def ban_notice_text(
    appeal_discord_user_id: str = "",
    *,
    automatic: bool = False,
) -> str:
    """English-only ban notice; optionally includes a Discord user ID to add for appeals."""
    if automatic:
        text = (
            "You were automatically banned for triggering risk control multiple times "
            "in a short period."
        )
    else:
        text = (
            "You were banned after a moderator confirmed a community-rule violation."
        )
    uid = (appeal_discord_user_id or "").strip()
    if uid:
        text += (
            f"\n\nTo appeal, add this Discord user and send feedback: `{uid}`"
        )
    return text


class StrikeTracker:
    def __init__(
        self,
        window_seconds: int = STRIKE_WINDOW_SECONDS,
        threshold: int = STRIKE_THRESHOLD,
        db_path: str | None = None,
    ):
        self.window_seconds = window_seconds
        self.threshold = threshold
        self.db_path = db_path or STATS_DB_PATH
        self._ensure_table()

    def _connect(self) -> sqlite3.Connection:
        # isolation_level=None so callers can use explicit IMMEDIATE transactions.
        return sqlite3.connect(self.db_path, timeout=30, isolation_level=None)

    def _ensure_table(self) -> None:
        conn = self._connect()
        try:
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
        finally:
            conn.close()

    def _prune(self, conn: sqlite3.Connection, scope_id: int, user_id: int, now: float) -> None:
        cutoff = now - self.window_seconds
        conn.execute(
            "DELETE FROM strikes WHERE scope_id = ? AND user_id = ? AND ts <= ?",
            (scope_id, user_id, cutoff),
        )

    def record(self, scope_id: int, user_id: int) -> bool:
        """
        Record a BAN trigger. Returns True if the user should be auto-banned
        (threshold reached within the window).
        """
        now = time.time()
        conn = self._connect()
        try:
            cursor = conn.cursor()
            cursor.execute("BEGIN IMMEDIATE")
            try:
                self._prune(conn, scope_id, user_id, now)
                cursor.execute(
                    "INSERT INTO strikes (scope_id, user_id, ts) VALUES (?, ?, ?)",
                    (scope_id, user_id, now),
                )
                cursor.execute(
                    "SELECT COUNT(*) FROM strikes WHERE scope_id = ? AND user_id = ?",
                    (scope_id, user_id),
                )
                count = cursor.fetchone()[0]
                conn.execute("COMMIT")
                return count >= self.threshold
            except Exception:
                conn.execute("ROLLBACK")
                raise
        finally:
            conn.close()

    def clear(self, scope_id: int, user_id: int) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "DELETE FROM strikes WHERE scope_id = ? AND user_id = ?",
                (scope_id, user_id),
            )
        finally:
            conn.close()

    def count(self, scope_id: int, user_id: int) -> int:
        now = time.time()
        conn = self._connect()
        try:
            cursor = conn.cursor()
            cursor.execute("BEGIN IMMEDIATE")
            try:
                self._prune(conn, scope_id, user_id, now)
                cursor.execute(
                    "SELECT COUNT(*) FROM strikes WHERE scope_id = ? AND user_id = ?",
                    (scope_id, user_id),
                )
                count = cursor.fetchone()[0]
                conn.execute("COMMIT")
                return count
            except Exception:
                conn.execute("ROLLBACK")
                raise
        finally:
            conn.close()

    def seconds_until_oldest_expires(self, scope_id: int, user_id: int) -> int:
        """Seconds until the oldest strike in the window drops off (0 if none)."""
        now = time.time()
        conn = self._connect()
        try:
            cursor = conn.cursor()
            cursor.execute("BEGIN IMMEDIATE")
            try:
                self._prune(conn, scope_id, user_id, now)
                cursor.execute(
                    """
                    SELECT MIN(ts) FROM strikes
                    WHERE scope_id = ? AND user_id = ?
                    """,
                    (scope_id, user_id),
                )
                row = cursor.fetchone()
                conn.execute("COMMIT")
                if not row or row[0] is None:
                    return 0
                return max(0, int(row[0] + self.window_seconds - now))
            except Exception:
                conn.execute("ROLLBACK")
                raise
        finally:
            conn.close()


strikes = StrikeTracker()
