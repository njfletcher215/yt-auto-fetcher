"""Command line entry points.

One `ytaf` multiplexer plus the per-stage entry points a scheduler calls.

Exit codes:
    0  success
    1  the stage ran and failed
    2  bad invocation or bad config
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from . import discover, download, feedback, jellyfin, monitor, naming, retention, score
from . import spec, takeout, training, ytdlp
from .config import Config, Paths, load_config
from .errors import ConfigError, LedgerError
from .ledger import Ledger

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_USAGE = 2

# Stages a scheduler fires.
STAGE_COMMANDS = (
    "discover",
    "score",
    "download",
    "feedback",
    "retrain",
    "retain",
    "monitor",
)
# Things a person runs by hand.
MANUAL_COMMANDS = (
    "init",
    "import-takeout",
    "status",
    "like",
    "dislike",
    "clear",
    "save",
    "verify-format",
)
COMMANDS = STAGE_COMMANDS + MANUAL_COMMANDS

_SENTIMENT_COMMANDS = ("like", "dislike", "clear", "save")


def build_parser() -> argparse.ArgumentParser:
    """The `ytaf` parser.

    Global flags: `--config PATH`, `--root PATH` (override paths.root),
    `--dry-run`, `--verbose`, `--json`.

    `like`, `dislike`, `clear` and `save` each take one or more video ids.
    """
    parser = argparse.ArgumentParser(prog="ytaf")
    parser.add_argument("--config", default=None, help="path to config.toml")
    parser.add_argument("--root", default=None, help="override paths.root")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--json", action="store_true")

    subparsers = parser.add_subparsers(dest="command")

    for name in STAGE_COMMANDS:
        subparsers.add_parser(name)
    subparsers.add_parser("init")
    subparsers.add_parser("status")

    import_p = subparsers.add_parser("import-takeout")
    import_p.add_argument("directory")

    for name in _SENTIMENT_COMMANDS:
        sentiment_p = subparsers.add_parser(name)
        sentiment_p.add_argument("video_ids", nargs="+")

    verify_p = subparsers.add_parser("verify-format")
    verify_p.add_argument("video_id")

    return parser


def _eprint(message: object) -> None:
    print(f"error: {message}", file=sys.stderr)


def _resolve_config(args: argparse.Namespace) -> Config:
    config = load_config(args.config)
    if args.root:
        config = config.with_(
            paths=Paths.under(args.root, permanent_library=config.paths.permanent_library)
        )
    return config


def _jellyfin_item_id(client: jellyfin.JellyfinClient, video_id: str) -> str | None:
    for item in client.played_items(limit=1000):
        if jellyfin.video_id_from_item(item) == video_id:
            return item.item_id
    return None


def _sync_favorite(config: Config, video_id: str, favorite: bool) -> None:
    """Best-effort heart/un-heart. A down or unconfigured Jellyfin is not an
    error -- the ledger write is what matters."""
    client = jellyfin.build_client(config.jellyfin)
    if client is None:
        return
    try:
        item_id = _jellyfin_item_id(client, video_id)
        if item_id is not None:
            client.set_favorite(item_id, favorite)
    except Exception:  # noqa: BLE001 - the ledger write must not be lost
        pass


# --- dispatch -----------------------------------------------------------------
def _stage_discover(args: argparse.Namespace) -> int:
    report = discover.run(args.resolved_config, dry_run=args.dry_run)
    return EXIT_OK if report.ok else EXIT_FAILURE


def _stage_score(args: argparse.Namespace) -> int:
    score.run(args.resolved_config, dry_run=args.dry_run)
    return EXIT_OK


def _stage_download(args: argparse.Namespace) -> int:
    report = download.run(args.resolved_config, dry_run=args.dry_run)
    return EXIT_FAILURE if report.total_failure else EXIT_OK


def _stage_feedback(args: argparse.Namespace) -> int:
    feedback.run(args.resolved_config)
    return EXIT_OK


def _stage_retrain(args: argparse.Namespace) -> int:
    training.retrain(args.resolved_config)
    return EXIT_OK


def _stage_retain(args: argparse.Namespace) -> int:
    retention.run(args.resolved_config, dry_run=args.dry_run)
    return EXIT_OK


def _stage_monitor(args: argparse.Namespace) -> int:
    result = monitor.run(args.resolved_config)
    return EXIT_OK if result.ok else EXIT_FAILURE


def cmd_init(args: argparse.Namespace) -> int:
    config = args.resolved_config
    config.paths.ensure()
    Ledger.open(config.paths.ledger).close()
    return EXIT_OK


def cmd_import_takeout(args: argparse.Namespace) -> int:
    config = args.resolved_config
    with Ledger.open(config.paths.ledger) as led:
        report = takeout.import_takeout(led, args.directory)
    print(f"discovered channel ids: {', '.join(report.channel_ids)}")
    print(
        f"imported {report.rows_inserted} rows "
        f"({report.watch_entries} watch, {report.like_entries} like)"
    )
    if report.skipped:
        print(f"skipped (not found): {', '.join(report.skipped)}")
    return EXIT_OK


def cmd_status(args: argparse.Namespace) -> int:
    """Ledger counts by state, model age, disk free, last discovery, and the
    model-vs-random-vs-repeat watch-percent comparison -- one screen."""
    config = args.resolved_config
    with Ledger.open(config.paths.ledger) as led:
        stats = led.selection_stats()
        data = {
            "ledger": {
                "total": led.count(),
                "downloaded": led.count(downloaded=1),
                "saved": led.count(saved=1),
                "liked": led.count(label=spec.LABEL_LIKE),
                "disliked": led.count(label=spec.LABEL_DISLIKE),
            },
            "selection": {
                "model": {"n": stats.model_n, "mean_watch_pct": stats.model_mean_watch_pct},
                "repeat": {"n": stats.repeat_n, "mean_watch_pct": stats.repeat_mean_watch_pct},
                "random": {"n": stats.random_n, "mean_watch_pct": stats.random_mean_watch_pct},
            },
            "disk_free_gb": round(retention.free_gb(config.paths.library), 1),
            "model": {
                "centroid_exists": config.paths.centroid_path.exists(),
                "classifier_exists": config.paths.classifier_path.exists(),
            },
        }

    if args.json:
        print(json.dumps(data))
    else:
        led_info = data["ledger"]
        print(
            f"Ledger: {led_info['total']} candidates, {led_info['downloaded']} downloaded, "
            f"{led_info['saved']} saved, {led_info['liked']} liked, "
            f"{led_info['disliked']} disliked"
        )
        print(
            f"Selection: model={stats.model_n} repeat={stats.repeat_n} random={stats.random_n}"
        )
        print(f"Disk free: {data['disk_free_gb']} GB")
    return EXIT_OK


# --- sentiment --------------------------------------------------------------
def cmd_like(args: argparse.Namespace) -> int:
    """Set `label = LABEL_LIKE` and heart the item in Jellyfin.

    The heart is the same state Streamyfin shows, so a like made here and a
    like made on the phone are not two competing sources of truth. A
    Jellyfin that is down or unconfigured is not an error -- the ledger write
    is what matters, and the next feedback run reconciles.
    """
    config = args.resolved_config
    if args.dry_run:
        return EXIT_OK
    with Ledger.open(config.paths.ledger) as led:
        for video_id in args.video_ids:
            led.set_label(video_id, spec.LABEL_LIKE)
            _sync_favorite(config, video_id, True)
    return EXIT_OK


def cmd_dislike(args: argparse.Namespace) -> int:
    """Set `label = LABEL_DISLIKE`, and un-heart in Jellyfin if hearted.

    No Jellyfin client exposes a dislike control, so this state lives only in
    the ledger. That is also why the feedback stage never reads an un-hearted
    item as a cleared dislike.
    """
    config = args.resolved_config
    if args.dry_run:
        return EXIT_OK
    with Ledger.open(config.paths.ledger) as led:
        for video_id in args.video_ids:
            led.set_label(video_id, spec.LABEL_DISLIKE)
            _sync_favorite(config, video_id, False)
    return EXIT_OK


def cmd_clear(args: argparse.Namespace) -> int:
    """Clear sentiment back to neutral, and un-heart in Jellyfin.

    Returns `label` to None, or to LABEL_HISTORY if the row came from the
    Takeout watch-history import -- un-liking something should not invent a
    weak positive for a video that was never watched.

    This is the only way out of a dislike.
    """
    config = args.resolved_config
    if args.dry_run:
        return EXIT_OK
    with Ledger.open(config.paths.ledger) as led:
        for video_id in args.video_ids:
            row = led.get(video_id)
            if row is None:
                raise LedgerError(f"unknown video_id: {video_id}")
            new_label = spec.LABEL_HISTORY if row.source == "history" else None
            led.set_label(video_id, new_label)
            _sync_favorite(config, video_id, False)
    return EXIT_OK


def cmd_save(args: argparse.Namespace) -> int:
    """Move a video to the permanent library, out of retention's reach.

    Fails with EXIT_USAGE if `paths.permanent_library` is unset.
    """
    config = args.resolved_config
    with Ledger.open(config.paths.ledger) as led:
        for video_id in args.video_ids:
            retention.save(led, config, video_id, dry_run=args.dry_run)
    return EXIT_OK


# --- other ------------------------------------------------------------------
def _find_library_file(config: Config, video_id: str) -> "Path | None":
    from pathlib import Path  # local: keeps the module-level namespace lean

    library = config.paths.library
    if not library.is_dir():
        return None
    for path in sorted(library.rglob("*")):
        if path.suffix in (".mp4", ".mkv") and naming.video_id_from_path(path.name) == video_id:
            return path
    return None


def _is_valid_format(info: dict) -> bool:
    vcodec = str(info.get("vcodec") or "")
    acodec = str(info.get("acodec") or "")
    height = info.get("height") or 0
    video_ok = vcodec.startswith("avc1") or vcodec.startswith("h264")
    audio_ok = acodec.startswith("mp4a")
    height_ok = height <= spec.MAX_HEIGHT
    return video_ok and audio_ok and height_ok


def cmd_verify_format(args: argparse.Namespace) -> int:
    """Download one video with the real selector and assert the result is
    H.264/AAC at or below 1080p.

    Touches the network, so it is never part of any automated stage -- hence
    no `main_verify_format`.
    """
    config = args.resolved_config
    video_id = args.video_id

    ytdlp.download(video_id, config)

    path = _find_library_file(config, video_id)
    if path is None:
        _eprint(f"no downloaded file found for {video_id}")
        return EXIT_FAILURE

    info_path = next((s for s in naming.sidecars(path) if s.name.endswith(".info.json")), None)
    if info_path is None:
        _eprint(f"no .info.json sidecar found for {video_id}")
        return EXIT_FAILURE

    info = json.loads(info_path.read_text(encoding="utf-8"))
    if not _is_valid_format(info):
        _eprint(
            f"{video_id} is not H.264/AAC at or below {spec.MAX_HEIGHT}p: "
            f"vcodec={info.get('vcodec')!r} acodec={info.get('acodec')!r} "
            f"height={info.get('height')!r}"
        )
        return EXIT_FAILURE
    return EXIT_OK


_DISPATCH = {
    "discover": _stage_discover,
    "score": _stage_score,
    "download": _stage_download,
    "feedback": _stage_feedback,
    "retrain": _stage_retrain,
    "retain": _stage_retain,
    "monitor": _stage_monitor,
    "init": cmd_init,
    "import-takeout": cmd_import_takeout,
    "status": cmd_status,
    "like": cmd_like,
    "dislike": cmd_dislike,
    "clear": cmd_clear,
    "save": cmd_save,
    "verify-format": cmd_verify_format,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else EXIT_USAGE

    if args.command is None:
        return EXIT_USAGE

    try:
        args.resolved_config = _resolve_config(args)
    except ConfigError as exc:
        _eprint(exc)
        return EXIT_USAGE

    handler = _DISPATCH[args.command]
    try:
        return handler(args)
    except ConfigError as exc:
        _eprint(exc)
        return EXIT_USAGE
    except Exception as exc:  # noqa: BLE001 - a stage failure is EXIT_FAILURE, not a crash
        _eprint(exc)
        return EXIT_FAILURE


# --- per-stage entry points -------------------------------------------------
def _run_stage_entry_point(command: str, argv: Sequence[str] | None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    return main([command, *argv])


def main_discover(argv: Sequence[str] | None = None) -> int:
    return _run_stage_entry_point("discover", argv)


def main_score(argv: Sequence[str] | None = None) -> int:
    return _run_stage_entry_point("score", argv)


def main_download(argv: Sequence[str] | None = None) -> int:
    return _run_stage_entry_point("download", argv)


def main_feedback(argv: Sequence[str] | None = None) -> int:
    return _run_stage_entry_point("feedback", argv)


def main_retrain(argv: Sequence[str] | None = None) -> int:
    return _run_stage_entry_point("retrain", argv)


def main_retain(argv: Sequence[str] | None = None) -> int:
    return _run_stage_entry_point("retain", argv)


def main_monitor(argv: Sequence[str] | None = None) -> int:
    return _run_stage_entry_point("monitor", argv)


if __name__ == "__main__":
    sys.exit(main())
