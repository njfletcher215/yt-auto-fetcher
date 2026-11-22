# yt-auto-fetcher

A self-hosted pipeline that discovers ~10 YouTube videos a day matched to your
taste, downloads them without transcoding, and lays them out for Jellyfin to
serve.

```
discover → score → download → (watch) → feedback → retrain
                                                  ↘ retain
```

## The daily ten

| Arm | Count | What it is |
|---|---|---|
`model` | 5 | Top-scoring brand new candidates. The system's actual output. |
`repeat` | 2 | Downloaded before, never watched, since reaped. |
`random` | 3 | Drawn at random. At least one random video is required, or the suggestion quality will degrade. |

## Retention

Everything in the managed library older than 14 days is deleted.

`ytaf save` moves a video into `paths.permanent_library`,
where it cannot be purged.

## Layout

| Path | What it is |
|---|---|
| `src/ytaf/spec.py` | Constants pinned exactly: format selector, output template, ledger schema, playback bands |
| `src/ytaf/config.py` | TOML config, path derivation, hard-constraint validation |
| `src/ytaf/ledger.py` | The SQLite ledger — one row per video, plus the pool queries |
| `src/ytaf/takeout.py` | One-time Google Takeout import |
| `src/ytaf/rss.py`, `prefilter.py`, `discover.py` | Discovery |
| `src/ytaf/embed.py`, `models.py`, `training.py` | Embeddings, centroid, classifier, signal derivation |
| `src/ytaf/select.py`, `score.py` | Scoring and the 5+2+3 selection |
| `src/ytaf/naming.py`, `download.py` | Download and library layout |
| `src/ytaf/jellyfin.py`, `feedback.py` | Playback and favorite sync |
| `src/ytaf/retention.py` | The 14-day sweep and `save` |
| `src/ytaf/notify.py`, `monitor.py` | ntfy alerts and the failure table |
| `src/ytaf/http.py`, `ytdlp.py` | The only two I/O seams in the package |
| `src/ytaf/cli.py` | `ytaf <command>` plus per-stage entry points |

## Setup

```bash
uv venv --python 3.13 .venv && uv pip install --python .venv/bin/python -e ".[dev]"
```

## Running the tests

```bash
.venv/bin/python -m pytest
```

Nothing in the suite touches the network, YouTube, Jellyfin, or any path
outside `tmp_path` — an autouse fixture fails any test that tries. Two tests
are marked `slow` (they load a real sentence transformer) and skip unless the
optional `embed` extra is installed:

```bash
.venv/bin/python -m pytest -m "not slow"
```

## Configuration

Copy `config.example.toml` to `config.toml` and set `paths.root`. Config is
found at `$YTAF_CONFIG`, then `./config.toml`, then
`/etc/yt-auto-fetcher/config.toml`.

Detailed explainations of all config options can be found in the example config.

