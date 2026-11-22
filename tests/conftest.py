"""Shared fixtures and test doubles.

Nothing in this suite touches the network, YouTube, Jellyfin or the real
filesystem outside tmp_path. There are exactly three seams:

  * `ytaf.http.get` / `ytaf.http.post`   -- RSS, Jellyfin, ntfy
  * `ytaf.ytdlp.run`                     -- every yt-dlp invocation
  * `config.paths`                       -- all filesystem writes

Implementation contract that follows from this: modules must call
`http.get(...)` and `ytdlp.run(...)` through the module, not import the
functions by name, or monkeypatching cannot reach them. `test_seams.py`
enforces that.
"""

from __future__ import annotations

import json
import random
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from ytaf import http, ytdlp
from ytaf.config import Config, JellyfinConfig, NotifyConfig
from ytaf.ledger import Candidate, Ledger

FIXTURES = Path(__file__).parent / "fixtures"


class _NoSpawn:
    """Stands in for `ytaf.ytdlp`'s subprocess module.

    Proxies everything (TimeoutExpired, PIPE, ...) to the real module but
    refuses to actually spawn. A test can still set `.run` on it, which
    shadows this method the way monkeypatching the real module would.
    """

    def __init__(self, real) -> None:
        self._real = real

    def __getattr__(self, name: str):
        return getattr(self._real, name)

    def run(self, args, *rest, **kwargs):
        raise AssertionError(
            f"test tried to spawn {list(args)!r}; request the `fake_ytdlp` fixture"
        )


@pytest.fixture(autouse=True)
def _no_real_io(request, monkeypatch):
    """Fail loudly rather than reaching the real world.

    Autouse, so a test that forgets `fake_http` or `fake_ytdlp` gets a clear
    error instead of hitting YouTube from a test run once these functions are
    actually implemented.

    The guard sits at the lowest level of each seam: `ytaf.http`'s two entry
    points, and `ytaf.ytdlp`'s own reference to the subprocess module. It
    rebinds that reference rather than setting `subprocess.run`, because the
    latter is the one shared module object and would also break the two tests
    that spawn a Python interpreter on purpose.

    `test_ytdlp.py` can still exercise the real `ytdlp.run` by patching
    `ytdlp.subprocess.run`, which lands on the shim. The `fake_*` fixtures
    override this too. Tests marked `network` opt out entirely.
    """
    if request.node.get_closest_marker("network"):
        return

    def no_http(url, *args, **kwargs):
        raise AssertionError(
            f"test tried to reach {url!r}; request the `fake_http` fixture"
        )

    monkeypatch.setattr(http, "get", no_http)
    monkeypatch.setattr(http, "post", no_http)
    monkeypatch.setattr(ytdlp, "subprocess", _NoSpawn(ytdlp.subprocess))

# A valid-shaped YouTube id: 11 chars of the base64url alphabet.
VIDEO_IDS = [
    "aaaaaaaaaa1",
    "bbbbbbbbbb2",
    "cccccccccc3",
    "dddddddddd4",
    "eeeeeeeeee5",
    "ffffffffff6",
    "gggggggggg7",
    "hhhhhhhhhh8",
    "iiiiiiiiii9",
    "jjjjjjjjj10",
    "kkkkkkkkk11",
    "lllllllll12",
    "mmmmmmmmm13",
    "nnnnnnnnn14",
    "ooooooooo15",
]
CHANNEL_IDS = [
    "UCaaaaaaaaaaaaaaaaaaaaaa",
    "UCbbbbbbbbbbbbbbbbbbbbbb",
    "UCcccccccccccccccccccccc",
    "UCdddddddddddddddddddddd",
]


# --- paths and config --------------------------------------------------------
@pytest.fixture
def root(tmp_path: Path) -> Path:
    """Stand-in for /mnt/raid1/youtube. Every test gets a fresh one."""
    return tmp_path / "youtube"


@pytest.fixture
def config(root: Path) -> Config:
    """Default config rooted in tmp_path, with the directory tree created."""
    cfg = Config.for_root(root)
    cfg.paths.ensure()
    return cfg


@pytest.fixture
def jellyfin_config() -> JellyfinConfig:
    return JellyfinConfig(
        base_url="https://jellyfin.test",
        api_key="TESTKEY",
        user_id="user-1",
        library_id="lib-1",
    )


@pytest.fixture
def notify_config() -> NotifyConfig:
    return NotifyConfig(enabled=True, ntfy_url="https://ntfy.test", topic="yt-test")


@pytest.fixture
def rng() -> random.Random:
    """Seeded, so every "random" assertion in the suite is reproducible."""
    return random.Random(20260914)


# --- ledger ------------------------------------------------------------------
@pytest.fixture
def ledger(config: Config) -> Ledger:
    led = Ledger.open(config.paths.ledger)
    yield led
    led.close()


@pytest.fixture
def raw_conn(config: Config) -> sqlite3.Connection:
    """A connection opened without going through Ledger, for schema asserts."""
    conn = sqlite3.connect(config.paths.ledger)
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


# --- candidate factory -------------------------------------------------------
@pytest.fixture
def make_candidate():
    """Build a Candidate with sane metadata; override anything by keyword.

    Metadata keys match yt-dlp's, since that is what the ledger stores.
    """

    counter = {"n": 0}

    def _make(
        video_id: str | None = None,
        *,
        title: str | None = None,
        description: str = "A description.",
        channel: str = "Test Channel",
        channel_id: str | None = None,
        duration: int | None = 900,
        view_count: int | None = 10_000,
        upload_date: str = "20260914",
        is_live: bool | None = None,
        like_count: int | None = 500,
        channel_follower_count: int | None = 100_000,
        seen_at: str = "2026-09-14T04:00:00+00:00",
        source: str = "rss",
        metadata: dict[str, Any] | None = None,
        **columns: Any,
    ) -> Candidate:
        n = counter["n"]
        counter["n"] = n + 1
        vid = video_id if video_id is not None else VIDEO_IDS[n % len(VIDEO_IDS)]
        meta: dict[str, Any] = {
            "id": vid,
            "title": title if title is not None else f"Video {vid}",
            "description": description,
            "channel": channel,
            "channel_id": channel_id or CHANNEL_IDS[0],
            "uploader": channel,
            "duration": duration,
            "view_count": view_count,
            "upload_date": upload_date,
            "like_count": like_count,
            "channel_follower_count": channel_follower_count,
        }
        if is_live is not None:
            meta["is_live"] = is_live
        meta = {k: v for k, v in meta.items() if v is not None}
        if metadata is not None:
            meta.update(metadata)
        return Candidate(
            video_id=vid, seen_at=seen_at, source=source, metadata=meta, **columns
        )

    return _make


@pytest.fixture
def populated_ledger(ledger: Ledger, make_candidate):
    """Twelve candidates across four channels with assorted scores/labels.

    Shaped so selection, training, retention and monitor checks all have
    something non-trivial to chew on.
    """
    for i in range(12):
        ledger.upsert(
            make_candidate(
                VIDEO_IDS[i],
                channel=f"Channel {i % 4}",
                channel_id=CHANNEL_IDS[i % 4],
                score=None if i >= 10 else 1.0 - (i * 0.05),
                label=2 if i < 2 else None,
            )
        )
    return ledger


# --- HTTP double -------------------------------------------------------------
@dataclass
class FakeHttp:
    """Routes URL substrings to canned responses and records every call.

    Match is "first route whose key is a substring of the URL", so tests can
    key on a path fragment and ignore query ordering.
    """

    routes: dict[str, http.Response] = field(default_factory=dict)
    calls: list[tuple[str, str, dict[str, Any]]] = field(default_factory=list)
    default: http.Response | None = None

    def route(
        self,
        key: str,
        *,
        status: int = 200,
        body: bytes | str = b"",
        json_body: Any = None,
        headers: dict[str, str] | None = None,
    ) -> FakeHttp:
        if json_body is not None:
            body = json.dumps(json_body).encode()
            headers = {"Content-Type": "application/json", **(headers or {})}
        if isinstance(body, str):
            body = body.encode()
        self.routes[key] = http.Response(
            status=status, body=body, headers=headers or {}, url=key
        )
        return self

    def _resolve(self, url: str) -> http.Response:
        for key, response in self.routes.items():
            if key in url:
                return response
        if self.default is not None:
            return self.default
        raise AssertionError(f"FakeHttp: no route matches {url!r}")

    def get(self, url: str, **kwargs: Any) -> http.Response:
        self.calls.append(("GET", url, kwargs))
        return self._resolve(url)

    def post(self, url: str, **kwargs: Any) -> http.Response:
        self.calls.append(("POST", url, kwargs))
        return self._resolve(url)

    # --- assertions used across several test modules ------------------------
    def urls(self, method: str | None = None) -> list[str]:
        return [u for m, u, _ in self.calls if method is None or m == method]

    def called(self, fragment: str, method: str | None = None) -> bool:
        return any(fragment in u for u in self.urls(method))

    def call_count(self, fragment: str = "") -> int:
        return sum(1 for u in self.urls() if fragment in u)


@pytest.fixture
def fake_http(monkeypatch: pytest.MonkeyPatch) -> FakeHttp:
    fake = FakeHttp()
    monkeypatch.setattr(http, "get", fake.get)
    monkeypatch.setattr(http, "post", fake.post)
    return fake


# --- yt-dlp double -----------------------------------------------------------
@dataclass
class FakeYtdlp:
    """Replaces `ytaf.ytdlp.run`, recording argument lists.

    `handler` lets a test decide the result per invocation; otherwise every
    call succeeds with `stdout`. `written` can describe files the fake should
    create, so the download stage has something to find.
    """

    stdout: str = ""
    returncode: int = 0
    stderr: str = ""
    calls: list[list[str]] = field(default_factory=list)
    handler: Any = None
    created: list[Path] = field(default_factory=list)

    def run(self, args, **kwargs: Any):
        args = list(args)
        self.calls.append(args)
        if self.handler is not None:
            return self.handler(args, **kwargs)
        return ytdlp.YtdlpResult(
            args=tuple(args),
            returncode=self.returncode,
            stdout=self.stdout,
            stderr=self.stderr,
        )

    @property
    def last_args(self) -> list[str]:
        assert self.calls, "yt-dlp was never invoked"
        return self.calls[-1]

    def flat(self) -> list[str]:
        return [arg for call in self.calls for arg in call]

    def invoked_with(self, fragment: str) -> bool:
        return any(fragment in arg for arg in self.flat())


@pytest.fixture
def fake_ytdlp(monkeypatch: pytest.MonkeyPatch) -> FakeYtdlp:
    fake = FakeYtdlp()
    monkeypatch.setattr(ytdlp, "run", fake.run)
    monkeypatch.setattr(ytdlp, "executable", lambda: "/usr/bin/yt-dlp")
    monkeypatch.setattr(ytdlp, "version", lambda: "2026.08.19")
    return fake


# --- fixture data ------------------------------------------------------------
@pytest.fixture
def fixture_text():
    def _read(name: str) -> str:
        return (FIXTURES / name).read_text(encoding="utf-8")

    return _read


@pytest.fixture
def fixture_json(fixture_text):
    def _read(name: str) -> Any:
        return json.loads(fixture_text(name))

    return _read


@pytest.fixture
def takeout_dir(tmp_path: Path, fixture_text) -> Path:
    """An unpacked Takeout tree in the real nested layout."""
    base = tmp_path / "Takeout" / "YouTube and YouTube Music"
    (base / "history").mkdir(parents=True)
    (base / "subscriptions").mkdir(parents=True)
    (base / "history" / "watch-history.json").write_text(
        fixture_text("watch-history.json"), encoding="utf-8"
    )
    (base / "history" / "like-history.json").write_text(
        fixture_text("like-history.json"), encoding="utf-8"
    )
    (base / "subscriptions" / "subscriptions.csv").write_text(
        fixture_text("subscriptions.csv"), encoding="utf-8"
    )
    return tmp_path / "Takeout"


@pytest.fixture
def embedder():
    """The deterministic embedder. No model download, no torch."""
    from ytaf.embed import HashingEmbedder

    return HashingEmbedder(dim=64, seed=7)


@pytest.fixture
def library_video(config: Config):
    """Create a fake downloaded video plus sidecars on disk.

    Lays the file out the way the output template does -- channel folder,
    date-prefixed filename, `[video_id]` suffix. Returns a callable so a test
    can make several.
    """

    def _make(
        video_id: str = "aaaaaaaaaa1",
        *,
        channel: str = "Test Channel",
        upload_date: str = "20260914",
        title: str = "A Video",
        size: int = 1024,
        root: Path | None = None,
        with_info_json: bool = True,
        with_thumbnail: bool = True,
    ) -> Path:
        channel_dir = (root or config.paths.library) / channel
        channel_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{upload_date} - {title} [{video_id}]"
        video = channel_dir / f"{stem}.mp4"
        video.write_bytes(b"\0" * size)
        if with_info_json:
            (channel_dir / f"{stem}.info.json").write_text(
                json.dumps(
                    {
                        "id": video_id,
                        "title": title,
                        "description": "A description.",
                        "channel": channel,
                        "upload_date": upload_date,
                        "duration": 900,
                        "view_count": 10_000,
                    }
                ),
                encoding="utf-8",
            )
        if with_thumbnail:
            (channel_dir / f"{stem}.jpg").write_bytes(b"\xff\xd8\xff")
        return video

    return _make


@pytest.fixture
def permanent_library(tmp_path: Path) -> Path:
    """An existing Jellyfin library outside anything retention can reach."""
    target = tmp_path / "permanent"
    target.mkdir()
    return target
