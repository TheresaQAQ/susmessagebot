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
    """Trilingual delete notice with strike remaining + time until oldest strike expires."""
    count = strikes.count(scope_id, user_id)
    remaining = max(0, strikes.threshold - count)
    reset_in = strikes.seconds_until_oldest_expires(scope_id, user_id)
    reset_str = _fmt_duration(reset_in)
    window_min = strikes.window_seconds // 60
    return (
        "⚠️ 你的消息因违反社区规则已被删除。\n"
        f"短时间内（{window_min} 分钟滑动窗口）多次触发将被直接封禁。"
        f"当前 {count}/{strikes.threshold}，再触发 {remaining} 次将封禁；"
        f"约 {reset_str} 后最早一次记录过期。\n\n"
        "Your message was removed for violating community rules.\n"
        f"Repeated triggers within a {window_min}-minute sliding window lead to an auto-ban. "
        f"Status {count}/{strikes.threshold}; {remaining} more trigger(s) until ban. "
        f"Oldest strike expires in about {reset_str}.\n\n"
        "Ваше сообщение удалено за нарушение правил сообщества.\n"
        f"Повторные срабатывания в скользящем окне {window_min} мин. приводят к автобану. "
        f"Сейчас {count}/{strikes.threshold}; до бана осталось {remaining}. "
        f"Самая ранняя запись истечёт примерно через {reset_str}."
    )


def ban_notice_text(appeal_discord_user_id: str = "") -> str:
    """Trilingual ban notice; optionally includes a Discord user ID to add for appeals."""
    text = (
        "🚫 你因短时间内多次触发风控，已被机器人自动封禁。\n"
        "You were automatically banned for triggering risk control multiple times "
        "in a short period.\n"
        "Вы были автоматически заблокированы за многократное срабатывание системы "
        "защиты за короткое время."
    )
    uid = (appeal_discord_user_id or "").strip()
    if uid:
        text += (
            f"\n\n"
            f"如需申请解封，请添加该 Discord 用户并提交反馈：`{uid}`\n"
            f"To appeal, add this Discord user and send feedback: `{uid}`\n"
            f"Для апелляции добавьте этого пользователя Discord и напишите: `{uid}`"
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
