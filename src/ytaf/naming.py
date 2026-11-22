"""Library paths: the Python mirror of the yt-dlp output template.

Jellyfin serves this as a Home Videos library, so there are no season or
episode numbers to derive -- the channel folder groups, the filename titles,
and the date prefix sorts chronologically within a channel.

The `[video_id]` suffix is functional. A homevideos library does not read the
NFO uniqueid a TV library would, so recovering a ledger row from a Jellyfin
item means reading the id back out of the path.
"""

from __future__ import annotations

import os
import re
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import spec
from .config import Config
from .errors import ConfigError, MetadataError, UnsafePathError

# Characters replaced in a path component.
UNSAFE_CHARS = '/\\:*?"<>|\n\r\t'
REPLACEMENT = "_"
SIDECAR_SUFFIXES = (".info.json", ".jpg", ".webp", ".png", ".nfo")


@dataclass(frozen=True)
class VideoNaming:
    channel: str
    upload_date: str
    title: str
    video_id: str
    ext: str = "mp4"

    @property
    def filename(self) -> str:
        """`"<upload_date> - <title> [<video_id>].<ext>"`."""
        return f"{self.upload_date} - {self.title} [{self.video_id}].{self.ext}"

    @property
    def relative_path(self) -> Path:
        """`<channel>/<filename>`."""
        return Path(self.channel) / self.filename


def sanitize_component(text: str, *, max_length: int = 150) -> str:
    """Make one path component safe.

    Replaces the unsafe set, collapses whitespace, strips leading dots and
    trailing dots/spaces (which some filesystems and SMB shares reject),
    truncates, and never returns empty -- an untitled video becomes "_"
    rather than producing a path with an empty segment.
    """
    out = "".join(REPLACEMENT if ch in UNSAFE_CHARS else ch for ch in text)
    out = re.sub(r" +", " ", out)
    out = out.strip()
    out = out.lstrip(".")
    out = out.rstrip(". ")
    out = out[:max_length]
    out = out.rstrip(". ")
    return out or REPLACEMENT


def naming_for(metadata: dict[str, Any], *, ext: str = "mp4") -> VideoNaming:
    """Derive naming from yt-dlp metadata.

    Requires an id. A missing upload date is tolerated (it becomes
    "00000000") because, unlike an episode number, the date prefix is only
    for sorting -- nothing collides and nothing is mis-indexed.
    """
    video_id = metadata.get("id")
    if not video_id:
        raise MetadataError("metadata is missing an id")
    channel = sanitize_component(
        str(metadata.get("channel") or metadata.get("uploader") or "")
    )
    title = sanitize_component(str(metadata.get("title") or ""))
    upload_date = str(metadata.get("upload_date") or "00000000")
    return VideoNaming(
        channel=channel, upload_date=upload_date, title=title, video_id=video_id, ext=ext
    )


def ensure_within(root: str | os.PathLike[str], path: str | os.PathLike[str]) -> Path:
    """Resolve `path` and raise UnsafePathError if it leaves `root`.

    A channel named `../../etc` must not be able to aim a write or a delete
    outside the library. Resolves symlinks, and compares path components
    rather than string prefixes so `/library-backup` is not treated as living
    inside `/library`.
    """
    root_resolved = Path(root).resolve()
    path_resolved = Path(path).resolve()
    try:
        path_resolved.relative_to(root_resolved)
    except ValueError:
        raise UnsafePathError(f"{path} escapes {root}") from None
    return path_resolved


def library_path(config: Config, metadata: dict[str, Any], *, ext: str = "mp4") -> Path:
    """Absolute path under `config.paths.library`, guarded against escape."""
    naming_obj = naming_for(metadata, ext=ext)
    candidate = config.paths.library / naming_obj.relative_path
    return ensure_within(config.paths.library, candidate)


def output_template(config: Config) -> str:
    """The `-o` argument: library root joined to `spec.OUTPUT_TEMPLATE`."""
    return str(config.paths.library / spec.OUTPUT_TEMPLATE)


def video_id_from_path(path: str | os.PathLike[str]) -> str | None:
    """Read the `[video_id]` suffix back out of a filename."""
    name = os.fspath(path)
    matches = re.findall(spec.VIDEO_ID_IN_PATH, name)
    return matches[-1] if matches else None


def sidecars(video_path: str | os.PathLike[str]) -> list[Path]:
    """Existing sidecar files for a video (info.json, thumbnail)."""
    video_path = Path(video_path)
    base = video_path.stem
    parent = video_path.parent
    found = []
    for suffix in SIDECAR_SUFFIXES:
        candidate = parent / f"{base}{suffix}"
        if candidate.exists() and candidate != video_path:
            found.append(candidate)
    return found


def find_downloaded(
    config: Config, metadata: dict[str, Any], *, extensions: Iterable[str] = ("mp4", "mkv")
) -> Path | None:
    """Locate the file yt-dlp just wrote, tolerating a container fallback.

    Falls back to a glob on `[video_id]` if the expected name is absent --
    yt-dlp's own sanitization of a pathological title may not match ours
    character for character, but the id always will.
    """
    extensions = tuple(extensions)
    naming_obj = naming_for(metadata)
    channel_dir = config.paths.library / naming_obj.channel
    base = f"{naming_obj.upload_date} - {naming_obj.title} [{naming_obj.video_id}]"

    for ext in extensions:
        candidate = channel_dir / f"{base}.{ext}"
        if candidate.exists():
            return candidate

    if channel_dir.is_dir():
        suffix_options = tuple(f"[{naming_obj.video_id}].{ext}" for ext in extensions)
        for candidate in sorted(channel_dir.iterdir()):
            if candidate.name.endswith(suffix_options):
                return candidate

    return None


def move_to_permanent(
    config: Config, video_path: str | os.PathLike[str]
) -> Path:
    """Move a video and its sidecars to the permanent library.

    Preserves the channel subdirectory. Raises ConfigError if
    `paths.permanent_library` is unset, and UnsafePathError if the source is
    not inside the managed library. Returns the new path.
    """
    if config.paths.permanent_library is None:
        raise ConfigError("paths.permanent_library is not configured")

    source = ensure_within(config.paths.library, video_path)
    sidecar_paths = sidecars(source)

    target_dir = Path(config.paths.permanent_library) / source.parent.name
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / source.name

    shutil.move(str(source), str(target))
    for sidecar in sidecar_paths:
        shutil.move(str(sidecar), str(target_dir / sidecar.name))

    return target
