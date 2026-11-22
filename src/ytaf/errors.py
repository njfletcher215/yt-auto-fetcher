"""Exception hierarchy. Every failure the pipeline raises on purpose."""

from __future__ import annotations


class YtafError(Exception):
    """Base class for every error this package raises deliberately."""


class ConfigError(YtafError):
    """Configuration missing, unparseable, or in violation of a hard constraint."""


class LedgerError(YtafError):
    """The SQLite ledger is missing, corrupt, or was handed invalid data."""


class MetadataError(YtafError):
    """Metadata we were handed cannot be used.

    Malformed Takeout JSON, an unreadable .info.json, a video with no upload
    date (and therefore no derivable episode number). Distinct from a bad
    argument from our own code, which stays a plain ValueError.
    """


class DiscoveryError(YtafError):
    """A discovery source could not be reached or parsed."""


class YtdlpError(YtafError):
    """yt-dlp exited non-zero.

    Carries enough context for the Phase 14 "extraction broken" check to tell
    a single bad video apart from YouTube having broken the extractor.
    """

    def __init__(
        self,
        message: str,
        *,
        video_id: str | None = None,
        returncode: int | None = None,
        stderr: str = "",
    ) -> None:
        super().__init__(message)
        self.video_id = video_id
        self.returncode = returncode
        self.stderr = stderr


class DownloadError(YtafError):
    """A download leg failed for a reason other than yt-dlp's exit status."""


class JellyfinError(YtafError):
    """The Jellyfin API was unreachable or returned an unusable response."""


class ModelError(YtafError):
    """A model could not be trained, loaded, or applied."""


class UnsafePathError(YtafError):
    """A write or delete was aimed outside the configured library root."""
