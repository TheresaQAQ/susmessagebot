"""Short-window strike tracking for auto-ban after repeated BAN triggers."""
from __future__ import annotations

from collections import defaultdict, deque
import time

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
    ):
        self.window_seconds = window_seconds
        self.threshold = threshold
        self._strikes: dict[tuple[int, int], deque[float]] = defaultdict(deque)

    def _prune(self, key: tuple[int, int], now: float) -> None:
        q = self._strikes[key]
        while q and now - q[0] > self.window_seconds:
            q.popleft()
        if not q and key in self._strikes:
            del self._strikes[key]

    def record(self, scope_id: int, user_id: int) -> bool:
        """
        Record a BAN trigger. Returns True if the user should be auto-banned
        (threshold reached within the window).
        """
        key = (scope_id, user_id)
        now = time.time()
        self._prune(key, now)
        self._strikes[key].append(now)
        return len(self._strikes[key]) >= self.threshold

    def clear(self, scope_id: int, user_id: int) -> None:
        self._strikes.pop((scope_id, user_id), None)

    def count(self, scope_id: int, user_id: int) -> int:
        key = (scope_id, user_id)
        self._prune(key, time.time())
        return len(self._strikes.get(key, ()))

    def seconds_until_oldest_expires(self, scope_id: int, user_id: int) -> int:
        """Seconds until the oldest strike in the window drops off (0 if none)."""
        key = (scope_id, user_id)
        now = time.time()
        self._prune(key, now)
        q = self._strikes.get(key)
        if not q:
            return 0
        return max(0, int(q[0] + self.window_seconds - now))


strikes = StrikeTracker()