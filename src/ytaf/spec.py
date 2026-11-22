"""Constants the design pins exactly.

These are not tunables. They encode hard constraints -- the H.264/AAC format
selector that makes iOS direct play possible, the ledger schema, the output
template the feedback stage depends on for mapping Jellyfin items back to
ledger rows. The test suite asserts the rest of the codebase routes through
this module rather than restating these strings, so a regression here shows up
as a test failure instead of a silent transcode.
"""

from __future__ import annotations

# --- H.264 at 1080p, never transcode ----------------------------------------
# YouTube only serves H.264 up to 1080p; above that you are back in VP9/AV1,
# which iOS AVFoundation will not direct play and the Pi cannot transcode.
FORMAT_SELECTOR = "bv*[vcodec^=avc1][height<=1080]+ba[acodec^=mp4a]/b[ext=mp4]"
MERGE_OUTPUT_FORMAT = "mp4"
MAX_HEIGHT = 1080

# Fallback when avc1 falls through; same constraints, different codec spelling.
FORMAT_SELECTOR_H264_FALLBACK = (
    "bv*[vcodec^=h264][height<=1080]+ba[acodec^=mp4a]/b[ext=mp4]"
)

# --- Output template --------------------------------------------------------
# Relative to the library root. Jellyfin serves this as a *Home Videos*
# library, so there are no season/episode numbers to derive and no TheTVDB to
# fight: the folder groups by channel and the filename is the title.
#
# The trailing `[id]` is load-bearing, not decoration -- it is how the
# feedback stage recovers a video_id from a Jellyfin item path, since a
# homevideos library does not read the NFO uniqueid a TV library would.
OUTPUT_TEMPLATE = "%(channel)s/%(upload_date)s - %(title)s [%(id)s].%(ext)s"
# Matches the `[id]` suffix above when reading a path back.
VIDEO_ID_IN_PATH = r"\[([A-Za-z0-9_-]{11})\]"

# --- Discovery --------------------------------------------------------------
RSS_FEED_URL = "https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
# ytsearchdate, not ytsearch: recent uploads only, and the result set actually
# changes from day to day.
SEARCH_PREFIX = "ytsearchdate"
MATCH_FILTERS = "duration > 300 & duration < 7200 & !is_live & view_count > 500"

# --- Daily selection --------------------------------------------------------
# 5 brand new model picks + 2 re-offers of videos downloaded before but never
# watched + 3 random. The random arm is mandatory (see config.validate): it is
# the control group the model-vs-chance check is measured against, and with
# zero of them that measurement is not merely noisy, it is impossible.
DEFAULT_N_MODEL_NEW = 5
DEFAULT_N_MODEL_REPEAT = 2
DEFAULT_N_RANDOM = 3

MODEL = "model"
REPEAT = "repeat"
RANDOM = "random"
SELECTIONS = frozenset({MODEL, REPEAT, RANDOM})

# --- Labels -----------------------------------------------------------------
# Sentiment, freely mutable in any direction by an explicit user action.
# Distinct from `source`, which records only how a row entered the ledger.
LABEL_DISLIKE = -1
LABEL_HISTORY = 1  # present in Takeout watch history; weakly positive
LABEL_LIKE = 2
LABELS = frozenset({LABEL_DISLIKE, LABEL_HISTORY, LABEL_LIKE})

SOURCES = frozenset({"history", "like", "rss", "search"})

# --- Playback bands --------------------------------------------------------
# The negative signal is strongest in the middle, not at zero: abandoning a
# video a third of the way in means it got a fair hearing and lost. Quitting
# in the first few seconds, or never opening it, says far less.
WATCH_THRESHOLD = 0.60  # at or above this, it counts as watched
PARTIAL_FLOOR = 0.15  # [PARTIAL_FLOOR, WATCH_THRESHOLD) is the strong negative

# --- Ledger -----------------------------------------------------------------
LEDGER_SCHEMA = """
CREATE TABLE IF NOT EXISTS candidates (
    video_id        TEXT PRIMARY KEY,
    seen_at         TEXT NOT NULL,
    source          TEXT,      -- history | like | rss | search (provenance only)
    metadata        TEXT,      -- full yt-dlp info JSON
    score           REAL,
    selected        INTEGER DEFAULT 0,
    selection       TEXT,      -- 'model' | 'repeat' | 'random'
    downloaded      INTEGER DEFAULT 0,
    filepath        TEXT,
    watched         INTEGER,   -- backfilled from Jellyfin
    watch_pct       REAL,
    label           INTEGER,   -- -1 dislike | 1 history | 2 like | NULL none
    play_count      INTEGER DEFAULT 0,   -- Jellyfin PlayCount; rewatches
    offer_count     INTEGER DEFAULT 0,   -- times downloaded and offered
    last_offered_at TEXT,                -- recency gate for re-offers
    saved           INTEGER DEFAULT 0    -- moved to the permanent library
);
"""

LEDGER_COLUMNS = (
    "video_id",
    "seen_at",
    "source",
    "metadata",
    "score",
    "selected",
    "selection",
    "downloaded",
    "filepath",
    "watched",
    "watch_pct",
    "label",
    "play_count",
    "offer_count",
    "last_offered_at",
    "saved",
)

# --- Anonymity --------------------------------------------------------------
# Discovery and download are anonymous. A cookie flag reaching yt-dlp would
# put the account at risk for no benefit at 10 videos/day.
FORBIDDEN_YTDLP_FLAGS = frozenset(
    {
        "--cookies",
        "--cookies-from-browser",
        "--username",
        "--password",
        "--netrc",
        "--netrc-location",
        "--video-password",
        "--ap-username",
        "--ap-password",
    }
)
