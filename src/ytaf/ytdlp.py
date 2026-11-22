"""yt-dlp invocation: argument construction and subprocess handling.

Argument lists are built by pure functions so the test suite can assert the
exact flags without running yt-dlp. Two invariants are enforced here rather
than left to review:

  * no authentication flag ever reaches yt-dlp (guide: discovery and download
    are anonymous, "with no cookies");
  * video ids are always passed after `--`, so an id beginning with `-` is
    never read as an option.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from . import spec
from .config import Config
from .errors import YtafError, YtdlpError

EXECUTABLE = "yt-dlp"
# Honoured by `executable()`, so a test or a Pi with yt-dlp somewhere odd can
# redirect without patching code.
EXECUTABLE_ENV_VAR = "YTAF_YTDLP"


@dataclass(frozen=True)
class YtdlpResult:
    args: tuple[str, ...]
    returncode: int
    stdout: str = ""
    stderr: str = ""

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def executable() -> str:
    """Resolve the yt-dlp binary, honouring $YTAF_YTDLP. Raises if absent."""
    override = os.environ.get(EXECUTABLE_ENV_VAR)
    if override:
        return override
    found = shutil.which(EXECUTABLE)
    if not found:
        raise YtafError(
            f"{EXECUTABLE} not found on PATH; set ${EXECUTABLE_ENV_VAR}"
        )
    return found


def assert_no_auth_flags(args: Sequence[str]) -> None:
    """Raise YtafError if any argument is (or starts with) a forbidden flag.

    Catches both `--cookies FILE` and `--cookies=FILE`.
    """
    for arg in args:
        for flag in spec.FORBIDDEN_YTDLP_FLAGS:
            if arg == flag or arg.startswith(flag + "="):
                raise YtafError(f"forbidden authentication flag: {arg!r}")


def run(
    args: Sequence[str],
    *,
    timeout: float | None = None,
    check: bool = True,
    cwd: str | os.PathLike[str] | None = None,
) -> YtdlpResult:
    """Run yt-dlp with `args`.

    Validates the arguments first, then executes. `check=True` raises
    YtdlpError on a non-zero exit, carrying stderr for the monitor.
    """
    assert_no_auth_flags(args)
    full_args = [executable(), *args]
    try:
        completed = subprocess.run(
            full_args, capture_output=True, text=True, timeout=timeout, cwd=cwd
        )
    except subprocess.TimeoutExpired as exc:
        raise YtdlpError(f"yt-dlp timed out: {exc}") from exc

    result = YtdlpResult(
        args=tuple(full_args),
        returncode=completed.returncode,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
    )
    if check and not result.ok:
        raise YtdlpError(
            f"yt-dlp exited {result.returncode}",
            returncode=result.returncode,
            stderr=result.stderr,
        )
    return result


def version() -> str:
    """`yt-dlp --version` output, stripped."""
    return run(["--version"]).stdout.strip()


def update() -> YtdlpResult:
    """`yt-dlp -U`. Phase 14's first response to broken extraction."""
    return run(["-U"], check=False)


# --- argument builders -------------------------------------------------------
def search_query(keyword: str, limit: int) -> str:
    """`ytsearchdateN:keyword` -- date-ordered, so recent uploads only."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    return f"{spec.SEARCH_PREFIX}{limit}:{keyword}"


def build_search_args(
    keyword: str,
    *,
    limit: int = 40,
    match_filters: str | None = spec.MATCH_FILTERS,
    sleep_requests: float | None = None,
) -> list[str]:
    """`yt-dlp --flat-playlist -J "ytsearchdate40:keyword"`.

    Flat search results already carry duration and view count, so these
    candidates skip the enrichment pass that RSS entries need.
    """
    args = ["--flat-playlist", "-J"]
    if match_filters:
        args += ["--match-filters", match_filters]
    if sleep_requests is not None:
        args += ["--sleep-requests", str(sleep_requests)]
    args += ["--", search_query(keyword, limit)]
    return args


def build_metadata_args(
    video_ids: Iterable[str],
    *,
    sleep_requests: float | None = None,
    match_filters: str | None = None,
) -> list[str]:
    """Metadata-only fetch (`-J --no-download`) to enrich RSS entries.

    RSS gives no duration and no live status, so the prefilter cannot judge a
    feed entry until this has run. Fetches JSON only -- no media, and one
    request per video.
    """
    ids = list(video_ids)
    if not ids:
        raise ValueError("video_ids must not be empty")
    args = ["-J", "--skip-download"]
    if match_filters:
        args += ["--match-filters", match_filters]
    if sleep_requests is not None:
        args += ["--sleep-requests", str(sleep_requests)]
    args += ["--", *ids]
    return args


def build_download_args(
    video_id: str,
    config: Config,
    *,
    output_template: str | None = None,
) -> list[str]:
    """The download invocation.

    Must contain, exactly: the configured format selector,
    `--merge-output-format mp4`, `--write-info-json`, `--write-thumbnail`,
    `--convert-thumbnails jpg`, `--embed-chapters`, `--sponsorblock-mark
    sponsor`, `--sleep-requests <n>`, `-o <library>/<template>`, then `--`,
    then the video id.

    Deliberately no `--download-archive`: the ledger is the record of what we
    hold, and an archive file would refuse to re-fetch the very videos the
    repeat arm exists to offer a second time.
    """
    if not video_id:
        raise ValueError("video_id must not be empty")
    template = output_template or str(config.paths.library / spec.OUTPUT_TEMPLATE)
    return [
        "-f", config.download.format_selector,
        "--merge-output-format", config.download.merge_output_format,
        "--write-info-json",
        "--write-thumbnail",
        "--convert-thumbnails", "jpg",
        "--embed-chapters",
        "--sponsorblock-mark", "sponsor",
        "--sleep-requests", str(config.discovery.sleep_requests),
        "-o", template,
        "--", video_id,
    ]


def build_list_formats_args(video_id: str) -> list[str]:
    """`yt-dlp --list-formats <id>` -- for debugging a selector fallthrough."""
    return ["--list-formats", "--", video_id]


# --- parsed invocations ------------------------------------------------------
def _entries_from_doc(doc: Any) -> list[dict[str, Any]]:
    if isinstance(doc, dict) and "entries" in doc:
        return [e for e in doc["entries"] if e is not None]
    if isinstance(doc, list):
        return [e for e in doc if e is not None]
    return [doc]


def parse_json_output(stdout: str) -> list[dict[str, Any]]:
    """Parse `-J` output: one JSON document, or JSON-lines from `-j`.

    Accepts a playlist wrapper (`{"entries": [...]}`) and unwraps it.
    """
    text = stdout.strip()
    if not text:
        return []

    try:
        doc = json.loads(text)
    except json.JSONDecodeError:
        doc = None
    if doc is not None:
        return _entries_from_doc(doc)

    entries: list[dict[str, Any]] = []
    any_parsed = False
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        any_parsed = True
        if value is not None:
            entries.append(value)

    if not any_parsed:
        raise YtdlpError(f"could not parse yt-dlp output: {text[:200]!r}")
    return entries


def search(
    keyword: str,
    *,
    limit: int = 40,
    match_filters: str | None = spec.MATCH_FILTERS,
    sleep_requests: float | None = None,
) -> list[dict[str, Any]]:
    """Run a search and return the flat-playlist entries."""
    args = build_search_args(
        keyword, limit=limit, match_filters=match_filters, sleep_requests=sleep_requests
    )
    result = run(args, check=False)
    return parse_json_output(result.stdout)


def fetch_metadata(
    video_ids: Iterable[str],
    *,
    sleep_requests: float | None = None,
) -> dict[str, dict[str, Any]]:
    """Full metadata for each id, keyed by video id.

    Ids yt-dlp could not resolve (deleted, private, geo-blocked) are simply
    absent from the result; one bad id does not fail the batch.
    """
    ids = list(video_ids)
    if not ids:
        return {}
    args = build_metadata_args(ids, sleep_requests=sleep_requests)
    result = run(args, check=False)
    entries = parse_json_output(result.stdout)
    out: dict[str, dict[str, Any]] = {}
    for entry in entries:
        video_id = entry.get("id")
        if video_id:
            out[video_id] = entry
    return out


def download(video_id: str, config: Config, *, dry_run: bool = False) -> YtdlpResult:
    """Download one video. `dry_run` builds and validates args without running."""
    args = build_download_args(video_id, config)
    assert_no_auth_flags(args)
    if dry_run:
        return YtdlpResult(args=tuple(args), returncode=0)
    return run(args)
