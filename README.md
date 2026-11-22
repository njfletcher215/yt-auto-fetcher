# yt-auto-fetcher

A self-hosted pipeline that discovers ~10 YouTube videos a day matched to your
taste, downloads them without transcoding, and lays them out for Jellyfin to
serve. Derived from
[`~/obsidian/task-management/guides/temp/youtube-discovery-pipeline.md`](file:///home/nat/obsidian/task-management/guides/temp/youtube-discovery-pipeline.md),
with the deviations listed below.

```
discover → score → download → (watch) → feedback → retrain
                                                  ↘ retain
```

## Status

Test suite written, implementation not started. Every module under `src/ytaf/`
is a signature and a docstring that raises `NotImplementedError`; the tests
define the contract.

Deployment (systemd units, RAID paths, Jellyfin and Streamyfin setup) is
deliberately out of scope until the pipeline works locally.

## The daily ten

| Arm | Count | What it is |
|---|---|---|
`model` | 5 | Top-scoring brand new candidates. The system's actual output. |
`repeat` | 2 | Downloaded before, never watched, since reaped. A second chance, gated on *how long ago* we last offered it rather than how many times — three ignored offers two years ago says much less than one ignored offer last week. |
`random` | 3 | Drawn at random. Mandatory: it is the control group the model-vs-chance check is computed against, which is why `n_random = 0` is a config error rather than a warning. |

If the repeat pool is empty its two slots go unused rather than being
backfilled, so the arms stay comparable week to week.

## Signals

`label` is sentiment and moves freely in any direction — like, dislike,
neither, and back. `source` is provenance and never changes. A rediscovery by
RSS touches neither.

Precedence when turning a row into a training label, strongest first:

1. **saved** — `ytaf save`, the most deliberate act available
2. **liked / disliked** — explicit, via `ytaf like` or Jellyfin's heart
3. **playback bands** — see below
4. **watch history** — present in the Takeout import, nothing more

An explicit like outranks a measured watch-percent on purpose: a like is
deliberate, whereas a video watched to the end might just mean autoplay ran
while you were asleep.

The playback bands are **not** monotonic:

| Watched | Signal | Why |
|---|---|---|
≥ 60% | positive | counts as watched; rewatches add weight with diminishing returns |
15–60% | **strongest negative** | it got a fair hearing and lost |
< 15%, opened | weak negative | you misread the thumbnail |
never opened | weak negative | offered and ignored |

## Retention

Everything in the managed library older than 14 days is deleted. No exceptions
— not likes, not unwatched videos. The ledger row and all its signal survive;
only bytes go.

`ytaf save` is the sole reprieve: it moves a video into
`paths.permanent_library`, outside anything retention can reach. Nothing in
the retention module can address a path outside `paths.library`, so saved
videos are out of range by construction rather than by a flag check.

## Deviations from the guide

- **Home Videos, not TV Shows.** No season/episode numbering, so no `SxxExx`
  and no `MMDD` episode collisions when a channel posts twice in a day.
  Layout is `channel/YYYYMMDD - Title [videoId].mp4`.
- **No NFO generation.** It existed to feed a TV library's episode resolver
  and to stop TheTVDB mangling channel names; Home Videos has neither problem.
  Jellyfin's NFO readers may not cover generic `Video` items at all — worth
  confirming when you first point Jellyfin at the library. If you want
  descriptions showing up there, this comes back.
- **The `[videoId]` filename suffix is functional.** Without an NFO
  `uniqueid`, it is how the feedback stage maps a Jellyfin item back to a
  ledger row.
- **No `--download-archive`.** The ledger knows strictly more (held, reaped,
  watched, offer count) and an archive file would refuse to re-fetch exactly
  the videos the repeat arm exists to offer again. `Ledger.held_ids` is the
  skip test and it covers saved videos.
- **RSS entries are enriched before filtering.** The feed carries no duration
  and no live status — three of the prefilter's four clauses — so new ids go
  through one metadata-only `yt-dlp -J` call first. JSON only, no media:
  roughly one request per new upload per day.
- **Four extra ledger columns** past the guide's twelve: `play_count`,
  `offer_count`, `last_offered_at`, `saved`. `Ledger.initialize` migrates an
  older ledger by adding them.
- **Retention is 14 days flat**, replacing the guide's 30 days + >90% watched
  + keep-anything-liked.

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

Settings that validation refuses rather than accepting quietly:

- a format selector that does not cap height at 1080p and request avc1/h264 +
  mp4a — the whole design depends on iOS direct play
- `selection.n_random = 0` — no control group means no degradation check
- `selection.n_model_new = 0`
- feedback bands outside `0 < partial_floor < watch_threshold < 1`
- `retention.max_age_days = 0`

## Loose ends

- **Takeout likes path.** The guide names `history/like-history.json`; real
  exports have also shipped `playlists/Liked videos.csv`. Both are supported
  because we cannot tell which a current export produces without waiting on
  one. Once you run the real import, delete the unused path and its test.
- **Jellyfin NFO support for Home Videos** — see the deviations above.
- **A dislike has no UI.** `ytaf dislike` is the only way in, and `ytaf clear`
  the only way out. The feedback stage therefore never reads an un-hearted
  item as a cleared dislike. A custom frontend would fix this properly.
