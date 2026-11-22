"""Configuration: dataclasses, TOML loading, and hard-constraint validation.

The dataclasses here are plain data and are implemented. Everything that reads
a file or validates (`find_config_file`, `load_config`, `config_from_dict`,
`validate`) is a stub -- that is the logic the test suite pins.
"""

from __future__ import annotations

import os
import re
import tomllib
import typing
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

from . import spec
from .errors import ConfigError

DEFAULT_ROOT = "~/yt-auto-fetcher-data"

CONFIG_ENV_VAR = "YTAF_CONFIG"
SYSTEM_CONFIG_PATH = Path("/etc/yt-auto-fetcher/config.toml")
LOCAL_CONFIG_NAME = "config.toml"

STRATEGIES = frozenset({"rss", "search", "both"})
MODEL_KINDS = frozenset({"centroid", "classifier"})
EMBEDDERS = frozenset({"hashing", "sentence-transformer"})


@dataclass(frozen=True)
class Paths:
    """Every filesystem location the pipeline touches.

    Derived from `root` by default so a scratch run and a real run differ by
    exactly one setting. `permanent_library` is the exception: it is somewhere
    else entirely, outside anything retention is allowed to delete.
    """

    root: Path
    library: Path
    state: Path
    logs: Path
    ledger: Path
    model_dir: Path
    permanent_library: Path | None = None

    @classmethod
    def under(
        cls,
        root: str | os.PathLike[str],
        *,
        permanent_library: str | os.PathLike[str] | None = None,
    ) -> Paths:
        """Standard layout under `root`."""
        root = Path(root).expanduser()
        state = root / "state"
        return cls(
            root=root,
            library=root / "library",
            state=state,
            logs=root / "logs",
            ledger=state / "ledger.db",
            model_dir=state / "model",
            permanent_library=(
                Path(permanent_library).expanduser() if permanent_library else None
            ),
        )

    @property
    def centroid_path(self) -> Path:
        return self.model_dir / "centroid.pkl"

    @property
    def classifier_path(self) -> Path:
        return self.model_dir / "classifier.pkl"

    def ensure(self) -> None:
        """Create the managed directory tree. Safe to call repeatedly.

        Never creates `permanent_library`: that is an existing Jellyfin
        library, and silently making an empty directory at a mistyped path
        would send saved videos somewhere nobody looks.
        """
        for d in (self.root, self.library, self.state, self.logs, self.model_dir):
            d.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class DiscoveryConfig:
    strategy: str = "both"
    channel_ids: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    search_results_per_keyword: int = 40
    sleep_requests: int = 2

    @property
    def uses_rss(self) -> bool:
        return self.strategy in ("rss", "both")

    @property
    def uses_search(self) -> bool:
        return self.strategy in ("search", "both")


@dataclass(frozen=True)
class PrefilterConfig:
    min_duration: int = 300
    max_duration: int = 7200
    min_view_count: int = 500
    allow_live: bool = False


@dataclass(frozen=True)
class SelectionConfig:
    """The daily 5 + 2 + 3.

    `n_model_repeat` re-offers videos that were downloaded before, never
    watched, and have since been reaped. Eligibility is gated on *recency*,
    not on how many times they were offered: three ignored offers two years
    ago says much less than one ignored offer last week.
    """

    n_model_new: int = spec.DEFAULT_N_MODEL_NEW
    n_model_repeat: int = spec.DEFAULT_N_MODEL_REPEAT
    n_random: int = spec.DEFAULT_N_RANDOM
    # A re-offer candidate is ineligible until this long after its last offer.
    repeat_cooldown_days: int = 90
    # Hard stop: after this many ignored offers, stop re-offering entirely.
    # 0 disables the cap and leaves recency as the only gate.
    max_offers: int = 4

    @property
    def total(self) -> int:
        return self.n_model_new + self.n_model_repeat + self.n_random

    @property
    def n_model(self) -> int:
        return self.n_model_new + self.n_model_repeat


@dataclass(frozen=True)
class ModelConfig:
    kind: str = "centroid"
    embedder: str = "hashing"
    sentence_transformer: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384


@dataclass(frozen=True)
class FeedbackConfig:
    """How Jellyfin playback becomes a training label.

    The bands are not a simple threshold because the signal is not monotonic:
    abandoning a video a third of the way in is the *strongest* negative,
    while bailing in the first few seconds or never opening it is weak.
    """

    watch_threshold: float = spec.WATCH_THRESHOLD
    partial_floor: float = spec.PARTIAL_FLOOR
    # Sync `label` from Jellyfin's heart (UserData.IsFavorite) on each run.
    sync_favorites: bool = True
    # How far back to look for playback activity.
    lookback_hours: int = 48


@dataclass(frozen=True)
class DownloadConfig:
    format_selector: str = spec.FORMAT_SELECTOR
    merge_output_format: str = spec.MERGE_OUTPUT_FORMAT


@dataclass(frozen=True)
class JellyfinConfig:
    base_url: str = ""
    api_key: str = ""
    user_id: str = ""
    library_id: str = ""
    verify_tls: bool = True

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.api_key)


@dataclass(frozen=True)
class RetentionConfig:
    """Everything older than `max_age_days` goes, with no exceptions.

    Likes and unwatched videos are not spared -- a like is a taste signal, not
    a keep-forever instruction. The only way a video survives is `ytaf save`,
    which moves it out of the managed library entirely.
    """

    max_age_days: int = 14
    log_retention_days: int = 90
    min_free_gb: int = 20


@dataclass(frozen=True)
class NotifyConfig:
    enabled: bool = False
    ntfy_url: str = ""
    topic: str = "yt-auto-fetcher"

    @property
    def configured(self) -> bool:
        return bool(self.enabled and self.ntfy_url and self.topic)


@dataclass(frozen=True)
class Config:
    paths: Paths
    discovery: DiscoveryConfig = field(default_factory=DiscoveryConfig)
    prefilter: PrefilterConfig = field(default_factory=PrefilterConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)
    download: DownloadConfig = field(default_factory=DownloadConfig)
    jellyfin: JellyfinConfig = field(default_factory=JellyfinConfig)
    retention: RetentionConfig = field(default_factory=RetentionConfig)
    notify: NotifyConfig = field(default_factory=NotifyConfig)
    source_path: Path | None = None

    @classmethod
    def for_root(cls, root: str | os.PathLike[str], **overrides: object) -> Config:
        """All-defaults config rooted at `root`. Used by tests and `ytaf init`."""
        return cls(paths=Paths.under(root), **overrides)  # type: ignore[arg-type]

    def with_(self, **overrides: object) -> Config:
        return replace(self, **overrides)  # type: ignore[arg-type]


def find_config_file(explicit: str | os.PathLike[str] | None = None) -> Path | None:
    """Resolve which config file to load.

    Order: `explicit`, then $YTAF_CONFIG, then ./config.toml, then
    /etc/yt-auto-fetcher/config.toml. Returns None when nothing exists.
    An `explicit` path that does not exist is an error, not a fallthrough.
    """
    if explicit is not None:
        path = Path(explicit)
        if not path.is_file():
            raise ConfigError(f"config file not found: {path}")
        return path

    env_value = os.environ.get(CONFIG_ENV_VAR)
    if env_value:
        path = Path(env_value)
        if not path.is_file():
            raise ConfigError(
                f"${CONFIG_ENV_VAR} points at a missing config file: {path}"
            )
        return path

    local = Path.cwd() / LOCAL_CONFIG_NAME
    if local.is_file():
        return local

    if SYSTEM_CONFIG_PATH.is_file():
        return SYSTEM_CONFIG_PATH

    return None


_SECTION_TYPES: dict[str, type] = {
    "discovery": DiscoveryConfig,
    "prefilter": PrefilterConfig,
    "selection": SelectionConfig,
    "model": ModelConfig,
    "feedback": FeedbackConfig,
    "download": DownloadConfig,
    "jellyfin": JellyfinConfig,
    "retention": RetentionConfig,
    "notify": NotifyConfig,
}
_KNOWN_SECTIONS = frozenset({"paths", *_SECTION_TYPES})
_PATHS_KEYS = frozenset(
    {"root", "library", "state", "logs", "ledger", "model_dir", "permanent_library"}
)


def _coerce_value(section: str, key: str, hint: object, value: object) -> object:
    origin = typing.get_origin(hint)
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ConfigError(
                f"[{section}].{key} must be a list, got {type(value).__name__}"
            )
        args = typing.get_args(hint)
        item_type = args[0] if args else str
        for item in value:
            if not isinstance(item, item_type):
                raise ConfigError(
                    f"[{section}].{key} items must be {item_type.__name__}"
                )
        return tuple(value)
    if hint is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(
                f"[{section}].{key} must be a number, got {type(value).__name__}"
            )
        return float(value)
    if hint is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(
                f"[{section}].{key} must be an integer, got {type(value).__name__}"
            )
        return value
    if hint is bool:
        if not isinstance(value, bool):
            raise ConfigError(
                f"[{section}].{key} must be a boolean, got {type(value).__name__}"
            )
        return value
    if hint is str:
        if not isinstance(value, str):
            raise ConfigError(
                f"[{section}].{key} must be a string, got {type(value).__name__}"
            )
        return value
    return value


def _build_section(data: Mapping[str, object], section_name: str, cls: type) -> object:
    section_data = data.get(section_name) or {}
    if not isinstance(section_data, Mapping):
        raise ConfigError(f"[{section_name}] must be a table")
    hints = typing.get_type_hints(cls)
    known_fields = {f.name for f in fields(cls)}
    kwargs: dict[str, object] = {}
    for key, value in section_data.items():
        if key not in known_fields:
            raise ConfigError(f"unknown key '{key}' in [{section_name}]")
        kwargs[key] = _coerce_value(section_name, key, hints[key], value)
    try:
        return cls(**kwargs)
    except TypeError as exc:
        raise ConfigError(f"[{section_name}]: {exc}") from exc


def _build_paths(data: Mapping[str, object]) -> Paths:
    section = data.get("paths") or {}
    if not isinstance(section, Mapping):
        raise ConfigError("[paths] must be a table")
    for key in section:
        if key not in _PATHS_KEYS:
            raise ConfigError(f"unknown key '{key}' in [paths]")

    root = section.get("root", DEFAULT_ROOT)
    if not isinstance(root, str):
        raise ConfigError("[paths].root must be a string")

    permanent_library = section.get("permanent_library")
    if permanent_library is not None and not isinstance(permanent_library, str):
        raise ConfigError("[paths].permanent_library must be a string")

    paths = Paths.under(root, permanent_library=permanent_library)

    overrides: dict[str, Path] = {}
    for key in ("library", "state", "logs", "ledger", "model_dir"):
        if key in section:
            value = section[key]
            if not isinstance(value, str):
                raise ConfigError(f"[paths].{key} must be a string")
            overrides[key] = Path(value).expanduser()
    if overrides:
        paths = replace(paths, **overrides)
    return paths


def config_from_dict(data: Mapping[str, object], *, source_path: Path | None = None) -> Config:
    """Build a validated Config from parsed TOML.

    Unknown sections and unknown keys are errors -- a typo in a config key
    should not silently leave a default in place on a box you only touch
    monthly.
    """
    if not isinstance(data, Mapping):
        raise ConfigError("config must be a table")

    unknown_sections = set(data) - _KNOWN_SECTIONS
    if unknown_sections:
        raise ConfigError(f"unknown section(s): {', '.join(sorted(unknown_sections))}")

    cfg = Config(
        paths=_build_paths(data),
        **{name: _build_section(data, name, cls) for name, cls in _SECTION_TYPES.items()},
        source_path=source_path,
    )
    validate(cfg)
    return cfg


def load_config(explicit: str | os.PathLike[str] | None = None) -> Config:
    """Find, parse and validate the config. Falls back to defaults if absent."""
    path = find_config_file(explicit)
    if path is None:
        return config_from_dict({})

    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"failed to parse {path}: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"failed to read {path}: {exc}") from exc

    return config_from_dict(data, source_path=path)


def _is_safe_format_selector(selector: str) -> bool:
    if "acodec^=mp4a" not in selector:
        return False
    if not re.search(r"vcodec\^=(avc1|h264)", selector):
        return False
    match = re.search(r"height<=(\d+)", selector)
    if not match:
        return False
    return int(match.group(1)) <= spec.MAX_HEIGHT


def validate(config: Config) -> None:
    """Raise ConfigError if a hard constraint is violated.

    Hard constraints:
      * `n_random >= 1`. The random arm is the control group; the
        model-vs-chance comparison is not computable without it.
      * `n_model_new >= 1`, `n_model_repeat >= 0`.
      * the format selector must cap height at <= 1080 and ask for avc1/h264
        plus an mp4a audio track; merge format must be mp4.
      * prefilter bounds coherent (min < max, both >= 0).
      * `0 < partial_floor < watch_threshold < 1`.
      * `repeat_cooldown_days >= 0`, `max_offers >= 0`.
      * `retention.max_age_days >= 1`.
      * strategy, model kind and embedder must be known values.
      * a strategy using search needs a keyword; one using rss needs a
        channel id.
    """
    sel = config.selection
    if sel.n_model_new <= 0:
        raise ConfigError("selection.n_model_new must be >= 1")
    if sel.n_model_repeat < 0:
        raise ConfigError("selection.n_model_repeat must be >= 0")
    if sel.n_random <= 0:
        raise ConfigError(
            "selection.n_random must be >= 1: it is the random control group the "
            "exploration/model-vs-chance check is computed from"
        )
    if sel.repeat_cooldown_days < 0:
        raise ConfigError("selection.repeat_cooldown_days must be >= 0")
    if sel.max_offers < 0:
        raise ConfigError("selection.max_offers must be >= 0")

    dl = config.download
    if not _is_safe_format_selector(dl.format_selector):
        raise ConfigError(
            "download.format_selector must cap height at <= "
            f"{spec.MAX_HEIGHT} and request avc1/h264 video with mp4a audio: "
            f"{dl.format_selector!r}"
        )
    if dl.merge_output_format != "mp4":
        raise ConfigError("download.merge_output_format must be 'mp4'")

    pf = config.prefilter
    if pf.min_duration < 0 or pf.max_duration < 0:
        raise ConfigError("prefilter duration bounds must be >= 0")
    if pf.min_duration >= pf.max_duration:
        raise ConfigError("prefilter.min_duration must be less than max_duration")

    fb = config.feedback
    if not (0 < fb.partial_floor < fb.watch_threshold < 1):
        raise ConfigError(
            "feedback bands must satisfy 0 < partial_floor < watch_threshold < 1"
        )

    ret = config.retention
    if ret.max_age_days <= 0:
        raise ConfigError("retention.max_age_days must be >= 1")

    disc = config.discovery
    if disc.strategy not in STRATEGIES:
        raise ConfigError(f"discovery.strategy must be one of {sorted(STRATEGIES)}")
    # Only the exclusive strategies are required to be self-sufficient: "rss"
    # alone with no channels, or "search" alone with no keywords, produces
    # nothing at all. "both" is allowed to lean on just one source -- the
    # shipped example ships with channel_ids empty until a Takeout import
    # fills it in, relying on keywords alone in the meantime.
    if disc.strategy == "rss" and not disc.channel_ids:
        raise ConfigError(
            "discovery.strategy uses rss but no channel_ids are configured"
        )
    if disc.strategy == "search" and not disc.keywords:
        raise ConfigError(
            "discovery.strategy uses search but no keywords are configured"
        )

    model = config.model
    if model.kind not in MODEL_KINDS:
        raise ConfigError(f"model.kind must be one of {sorted(MODEL_KINDS)}")
    if model.embedder not in EMBEDDERS:
        raise ConfigError(f"model.embedder must be one of {sorted(EMBEDDERS)}")
