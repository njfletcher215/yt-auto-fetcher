"""Phase 14 alert delivery over ntfy.

One rule dominates: a notification failure must never fail the pipeline. If
the ntfy box is down, tonight's downloads still happen.
"""

from __future__ import annotations

import pytest

from ytaf import notify
from ytaf.config import NotifyConfig
from ytaf.notify import Alert, NtfyNotifier, NullNotifier


class TestEndpoint:
    def test_joins_url_and_topic(self, notify_config):
        assert NtfyNotifier(notify_config).endpoint == "https://ntfy.test/yt-test"

    def test_does_not_double_the_slash(self):
        cfg = NotifyConfig(enabled=True, ntfy_url="https://ntfy.test/", topic="t")
        assert NtfyNotifier(cfg).endpoint == "https://ntfy.test/t"


class TestHeaders:
    def test_sets_the_title(self, notify_config):
        headers = NtfyNotifier(notify_config).headers(Alert("A Title", "body"))
        assert headers["X-Title"] == "A Title"

    def test_sets_the_priority(self, notify_config):
        headers = NtfyNotifier(notify_config).headers(
            Alert("t", "b", priority=notify.PRIORITY_URGENT)
        )
        assert headers["X-Priority"] == "5"

    def test_joins_tags_with_commas(self, notify_config):
        headers = NtfyNotifier(notify_config).headers(
            Alert("t", "b", tags=("warning", "cd"))
        )
        assert headers["X-Tags"] == "warning,cd"

    def test_omits_tags_when_empty(self, notify_config):
        assert "X-Tags" not in NtfyNotifier(notify_config).headers(Alert("t", "b"))

    def test_encodes_a_non_ascii_title(self, notify_config):
        # HTTP headers are latin-1; a raw unicode title raises at send time.
        headers = NtfyNotifier(notify_config).headers(Alert("模型降級", "body"))
        headers["X-Title"].encode("latin-1")


class TestSend:
    def test_posts_to_the_topic(self, notify_config, fake_http):
        fake_http.route("/yt-test", status=200, json_body={"id": "1"})
        assert NtfyNotifier(notify_config).send(Alert("t", "b")) is True
        assert fake_http.called("/yt-test", "POST")

    def test_sends_the_message_as_the_body(self, notify_config, fake_http):
        fake_http.route("/yt-test", status=200, json_body={})
        NtfyNotifier(notify_config).send(Alert("t", "the message"))
        _, _, kwargs = fake_http.calls[0]
        body = kwargs.get("body") or kwargs.get("data") or b""
        if isinstance(body, bytes):
            body = body.decode()
        assert "the message" in body

    def test_a_server_error_returns_false_without_raising(self, notify_config, fake_http):
        fake_http.route("/yt-test", status=500, body="boom")
        assert NtfyNotifier(notify_config).send(Alert("t", "b")) is False

    def test_a_transport_failure_returns_false_without_raising(
        self, notify_config, monkeypatch
    ):
        from ytaf import http as http_mod
        from ytaf.errors import YtafError

        def boom(*args, **kwargs):
            raise YtafError("connection refused")

        monkeypatch.setattr(http_mod, "post", boom)
        assert NtfyNotifier(notify_config).send(Alert("t", "b")) is False

    def test_an_unexpected_exception_still_returns_false(self, notify_config, monkeypatch):
        from ytaf import http as http_mod

        def boom(*args, **kwargs):
            raise RuntimeError("something nobody predicted")

        monkeypatch.setattr(http_mod, "post", boom)
        assert NtfyNotifier(notify_config).send(Alert("t", "b")) is False


class TestNullNotifier:
    def test_records_what_it_swallowed(self):
        n = NullNotifier()
        n.send(Alert("t", "b"))
        assert len(n.sent) == 1

    def test_reports_not_sent(self):
        assert NullNotifier().send(Alert("t", "b")) is False

    def test_makes_no_http_calls(self, fake_http):
        NullNotifier().send(Alert("t", "b"))
        assert fake_http.calls == []


class TestBuildNotifier:
    def test_builds_ntfy_when_configured(self, notify_config):
        assert isinstance(notify.build_notifier(notify_config), NtfyNotifier)

    @pytest.mark.parametrize(
        "cfg",
        [
            NotifyConfig(enabled=False, ntfy_url="https://ntfy.test", topic="t"),
            NotifyConfig(enabled=True, ntfy_url="", topic="t"),
            NotifyConfig(enabled=True, ntfy_url="https://ntfy.test", topic=""),
            NotifyConfig(),
        ],
    )
    def test_falls_back_to_null(self, cfg):
        assert isinstance(notify.build_notifier(cfg), NullNotifier)
