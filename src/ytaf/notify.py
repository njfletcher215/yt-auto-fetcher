"""Alert delivery, over ntfy.

Notification failure is never allowed to fail the pipeline -- a dead ntfy
server should not stop tonight's downloads. `send` returns False and logs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from . import http
from .config import NotifyConfig

PRIORITY_MIN = 1
PRIORITY_DEFAULT = 3
PRIORITY_HIGH = 4
PRIORITY_URGENT = 5


@dataclass(frozen=True)
class Alert:
    title: str
    message: str
    priority: int = PRIORITY_DEFAULT
    tags: tuple[str, ...] = ()
    check: str = ""


class Notifier(Protocol):
    def send(self, alert: Alert) -> bool: ...


@dataclass
class NullNotifier:
    """Used when notifications are disabled. Records what it swallowed so
    tests and `--dry-run` can see what would have gone out."""

    sent: list[Alert] = field(default_factory=list)

    def send(self, alert: Alert) -> bool:
        self.sent.append(alert)
        return False


class NtfyNotifier:
    def __init__(self, config: NotifyConfig) -> None:
        self.config = config

    @property
    def endpoint(self) -> str:
        """`<ntfy_url>/<topic>`, with no doubled slash."""
        return f"{self.config.ntfy_url.rstrip('/')}/{self.config.topic}"

    def headers(self, alert: Alert) -> dict[str, str]:
        """ntfy's X-Title / X-Priority / X-Tags."""
        headers = {
            "X-Title": alert.title.encode("utf-8").decode("latin-1"),
            "X-Priority": str(alert.priority),
        }
        if alert.tags:
            headers["X-Tags"] = ",".join(alert.tags)
        return headers

    def send(self, alert: Alert) -> bool:
        try:
            response = http.post(
                self.endpoint,
                body=alert.message.encode("utf-8"),
                headers=self.headers(alert),
            )
        except Exception:  # noqa: BLE001 - a dead ntfy box must not fail the run
            return False
        return response.ok


def build_notifier(config: NotifyConfig) -> Notifier:
    """NtfyNotifier when configured, NullNotifier otherwise."""
    if config.configured:
        return NtfyNotifier(config)
    return NullNotifier()
