"""yt-auto-fetcher: personal YouTube discovery, scoring, download and serving.

Stages, each a systemd oneshot on natpi:

    discover -> score -> download -> (watch) -> feedback -> retrain

See the guide at
~/obsidian/task-management/guides/temp/youtube-discovery-pipeline.md.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
