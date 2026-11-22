"""yt-dlp argument construction and invocation.

Argument lists are the most consequential strings in this project: the format
selector decides whether the Pi serves or melts, and an accidental cookie flag
puts the account at risk for no gain. Both are asserted exactly.
"""

from __future__ import annotations

import json

import pytest

from ytaf import spec, ytdlp
from ytaf.errors import YtafError, YtdlpError


def value_after(args, flag):
    """The argument following `flag`. Fails the test if `flag` is absent."""
    assert flag in args, f"{flag} not in {args}"
    index = args.index(flag)
    assert index + 1 < len(args), f"{flag} has no value"
    return args[index + 1]


class TestNoAuthFlags:
    """The guide is explicit: discovery and download are anonymous."""

    def test_accepts_a_clean_argument_list(self):
        ytdlp.assert_no_auth_flags(["-f", "bv*", "--", "aircAruvnKk"])

    @pytest.mark.parametrize("flag", sorted(spec.FORBIDDEN_YTDLP_FLAGS))
    def test_rejects_each_forbidden_flag(self, flag):
        with pytest.raises(YtafError):
            ytdlp.assert_no_auth_flags(["-f", "bv*", flag, "value"])

    @pytest.mark.parametrize(
        "arg",
        [
            "--cookies=/home/nat/cookies.txt",
            "--cookies-from-browser=firefox",
            "--username=nat",
            "--password=hunter2",
        ],
    )
    def test_rejects_the_equals_form(self, arg):
        with pytest.raises(YtafError):
            ytdlp.assert_no_auth_flags([arg])

    def test_does_not_false_positive_on_a_lookalike_value(self):
        # A video titled "--cookies" is not a flag; only the flag position is.
        ytdlp.assert_no_auth_flags(["-o", "my --cookies video.mp4", "--", "aircAruvnKk"])

    def test_run_validates_before_executing(self, monkeypatch):
        called = []
        monkeypatch.setattr(
            ytdlp.subprocess, "run", lambda *a, **k: called.append(a) or None
        )
        with pytest.raises(YtafError):
            ytdlp.run(["--cookies", "/tmp/c.txt"])
        assert called == []


class TestSearchArgs:
    def test_uses_ytsearchdate_not_ytsearch(self):
        query = ytdlp.search_query("computerphile", 40)
        assert query.startswith("ytsearchdate40:")
        assert not query.startswith("ytsearch:")

    def test_search_query_embeds_the_limit(self):
        assert ytdlp.search_query("cats", 10) == "ytsearchdate10:cats"

    def test_search_query_preserves_a_multiword_keyword(self):
        assert ytdlp.search_query("neural networks", 5) == "ytsearchdate5:neural networks"

    @pytest.mark.parametrize("limit", [0, -1])
    def test_search_query_rejects_a_nonpositive_limit(self, limit):
        with pytest.raises(ValueError):
            ytdlp.search_query("cats", limit)

    def test_args_request_a_flat_playlist(self):
        args = ytdlp.build_search_args("computerphile")
        assert "--flat-playlist" in args

    def test_args_request_json(self):
        args = ytdlp.build_search_args("computerphile")
        assert "-J" in args or "--dump-single-json" in args

    def test_args_never_download(self):
        args = ytdlp.build_search_args("computerphile")
        assert "--skip-download" in args or "-J" in args
        assert "-o" not in args

    def test_args_carry_the_match_filter(self):
        args = ytdlp.build_search_args("computerphile")
        assert value_after(args, "--match-filters") == spec.MATCH_FILTERS

    def test_match_filter_can_be_omitted(self):
        args = ytdlp.build_search_args("computerphile", match_filters=None)
        assert "--match-filters" not in args

    def test_args_carry_the_sleep_interval(self):
        args = ytdlp.build_search_args("computerphile", sleep_requests=2)
        assert value_after(args, "--sleep-requests") == "2"

    def test_query_is_the_last_argument(self):
        args = ytdlp.build_search_args("computerphile", limit=40)
        assert args[-1] == "ytsearchdate40:computerphile"

    def test_query_is_separated_by_a_double_dash(self):
        # A keyword beginning with a dash would otherwise be read as a flag.
        args = ytdlp.build_search_args("-weird-keyword")
        assert "--" in args
        assert args.index("--") < args.index("ytsearchdate40:-weird-keyword")

    def test_carries_no_auth_flags(self):
        ytdlp.assert_no_auth_flags(ytdlp.build_search_args("computerphile"))


class TestMetadataArgs:
    def test_requests_json_without_downloading(self):
        args = ytdlp.build_metadata_args(["aircAruvnKk"])
        assert "--skip-download" in args or "-J" in args or "-j" in args
        assert "-o" not in args

    def test_is_not_flat_so_duration_comes_back(self):
        # --flat-playlist is what makes search cheap, and also what makes it
        # return no duration. The enrichment pass must not inherit it.
        args = ytdlp.build_metadata_args(["aircAruvnKk"])
        assert "--flat-playlist" not in args

    def test_passes_every_id_after_a_double_dash(self):
        args = ytdlp.build_metadata_args(["aircAruvnKk", "WUvTyaaNkzM"])
        assert args[-2:] == ["aircAruvnKk", "WUvTyaaNkzM"]
        assert args[-3] == "--"

    def test_carries_the_sleep_interval(self):
        args = ytdlp.build_metadata_args(["aircAruvnKk"], sleep_requests=2)
        assert value_after(args, "--sleep-requests") == "2"

    def test_rejects_an_empty_id_list(self):
        with pytest.raises(ValueError):
            ytdlp.build_metadata_args([])

    def test_carries_no_auth_flags(self):
        ytdlp.assert_no_auth_flags(ytdlp.build_metadata_args(["aircAruvnKk"]))


class TestDownloadArgs:
    @pytest.fixture
    def args(self, config):
        return ytdlp.build_download_args("aircAruvnKk", config)

    def test_uses_the_configured_format_selector(self, args):
        assert value_after(args, "-f") == spec.FORMAT_SELECTOR

    def test_honours_a_config_override_of_the_selector(self, config):
        from ytaf.config import DownloadConfig

        cfg = config.with_(
            download=DownloadConfig(format_selector="bv*[vcodec^=h264][height<=720]+ba")
        )
        args = ytdlp.build_download_args("aircAruvnKk", cfg)
        assert value_after(args, "-f") == "bv*[vcodec^=h264][height<=720]+ba"

    def test_merges_to_mp4(self, args):
        assert value_after(args, "--merge-output-format") == "mp4"

    def test_does_not_use_a_download_archive(self, args):
        # The ledger is the record of what we hold. An archive file only
        # knows "ever downloaded" and would refuse to re-fetch exactly the
        # videos the repeat arm exists to offer a second time.
        assert "--download-archive" not in args

    def test_writes_the_info_json(self, args):
        # Our own pipeline reads this for naming and ledger metadata.
        assert "--write-info-json" in args

    def test_writes_and_converts_the_thumbnail(self, args):
        assert "--write-thumbnail" in args
        assert value_after(args, "--convert-thumbnails") == "jpg"

    def test_embeds_chapters(self, args):
        assert "--embed-chapters" in args

    def test_marks_sponsor_segments(self, args):
        assert value_after(args, "--sponsorblock-mark") == "sponsor"

    def test_does_not_remove_sponsor_segments(self, args):
        # --sponsorblock-remove re-encodes, which is the one thing this whole
        # design exists to avoid.
        assert "--sponsorblock-remove" not in args

    def test_sleeps_between_requests(self, args):
        assert value_after(args, "--sleep-requests") == "2"

    def test_sleep_interval_is_configurable(self, config):
        from ytaf.config import DiscoveryConfig

        cfg = config.with_(discovery=DiscoveryConfig(sleep_requests=5))
        args = ytdlp.build_download_args("aircAruvnKk", cfg)
        assert value_after(args, "--sleep-requests") == "5"

    def test_output_template_is_rooted_in_the_library(self, config, args):
        template = value_after(args, "-o")
        assert template.startswith(str(config.paths.library))

    def test_output_template_matches_the_spec(self, config, args):
        template = value_after(args, "-o")
        assert template == str(config.paths.library / spec.OUTPUT_TEMPLATE)

    def test_output_template_can_be_overridden(self, config):
        args = ytdlp.build_download_args(
            "aircAruvnKk", config, output_template="/tmp/x/%(id)s.%(ext)s"
        )
        assert value_after(args, "-o") == "/tmp/x/%(id)s.%(ext)s"

    def test_video_id_is_last(self, args):
        assert args[-1] == "aircAruvnKk"

    def test_video_id_follows_a_double_dash(self, args):
        assert args[-2] == "--"

    def test_a_video_id_starting_with_a_dash_is_safe(self, config):
        args = ytdlp.build_download_args("-dashStart1", config)
        assert args[-2:] == ["--", "-dashStart1"]

    def test_passes_a_bare_id_not_a_url(self, args):
        # The guide passes ids. A URL would also work, but ids keep the
        # archive file and the ledger keyed identically.
        assert not args[-1].startswith("http")

    def test_carries_no_auth_flags(self, args):
        ytdlp.assert_no_auth_flags(args)

    def test_does_not_request_a_rate_limit_bypass(self, args):
        for forbidden in ("--no-check-certificates", "--legacy-server-connect"):
            assert forbidden not in args

    def test_rejects_an_empty_video_id(self, config):
        with pytest.raises(ValueError):
            ytdlp.build_download_args("", config)


class TestListFormatsArgs:
    def test_builds_the_debugging_command(self):
        args = ytdlp.build_list_formats_args("aircAruvnKk")
        assert "--list-formats" in args or "-F" in args
        assert args[-2:] == ["--", "aircAruvnKk"]


class TestParseJsonOutput:
    def test_parses_a_single_document(self):
        out = ytdlp.parse_json_output(json.dumps({"id": "a", "title": "T"}))
        assert out == [{"id": "a", "title": "T"}]

    def test_unwraps_a_playlist(self, fixture_text):
        out = ytdlp.parse_json_output(fixture_text("search-flat.json"))
        assert len(out) == 5
        assert out[0]["id"] == "searchHit01"

    def test_parses_json_lines(self):
        text = '{"id": "a"}\n{"id": "b"}\n'
        assert [e["id"] for e in ytdlp.parse_json_output(text)] == ["a", "b"]

    def test_ignores_blank_lines(self):
        assert len(ytdlp.parse_json_output('\n{"id": "a"}\n\n')) == 1

    def test_ignores_warning_lines_on_stdout(self):
        # yt-dlp occasionally interleaves a WARNING line into stdout.
        text = 'WARNING: something\n{"id": "a"}\n'
        assert [e["id"] for e in ytdlp.parse_json_output(text)] == ["a"]

    def test_empty_output_is_an_empty_list(self):
        assert ytdlp.parse_json_output("") == []

    def test_unparseable_output_raises(self):
        with pytest.raises(YtdlpError):
            ytdlp.parse_json_output("not json at all, not even close")

    def test_a_playlist_with_no_entries_is_empty(self):
        assert ytdlp.parse_json_output('{"_type": "playlist", "entries": []}') == []

    def test_drops_null_entries(self):
        # yt-dlp emits null for an entry it could not extract.
        text = '{"_type": "playlist", "entries": [{"id": "a"}, null]}'
        assert len(ytdlp.parse_json_output(text)) == 1


class TestRun:
    def test_returns_a_result_on_success(self, monkeypatch):
        class Completed:
            returncode = 0
            stdout = "out"
            stderr = ""

        monkeypatch.setattr(ytdlp.subprocess, "run", lambda *a, **k: Completed())
        monkeypatch.setattr(ytdlp, "executable", lambda: "/usr/bin/yt-dlp")
        result = ytdlp.run(["--version"])
        assert result.ok
        assert result.stdout == "out"

    def test_raises_on_nonzero_when_checking(self, monkeypatch):
        class Completed:
            returncode = 1
            stdout = ""
            stderr = "ERROR: Video unavailable"

        monkeypatch.setattr(ytdlp.subprocess, "run", lambda *a, **k: Completed())
        monkeypatch.setattr(ytdlp, "executable", lambda: "/usr/bin/yt-dlp")
        with pytest.raises(YtdlpError) as excinfo:
            ytdlp.run(["--", "aircAruvnKk"])
        assert "Video unavailable" in excinfo.value.stderr

    def test_returns_a_failed_result_when_not_checking(self, monkeypatch):
        class Completed:
            returncode = 1
            stdout = ""
            stderr = "ERROR"

        monkeypatch.setattr(ytdlp.subprocess, "run", lambda *a, **k: Completed())
        monkeypatch.setattr(ytdlp, "executable", lambda: "/usr/bin/yt-dlp")
        result = ytdlp.run(["--", "aircAruvnKk"], check=False)
        assert not result.ok
        assert result.returncode == 1

    def test_puts_the_executable_first(self, monkeypatch):
        seen = {}

        class Completed:
            returncode = 0
            stdout = ""
            stderr = ""

        def fake_run(args, **kwargs):
            seen["args"] = args
            return Completed()

        monkeypatch.setattr(ytdlp.subprocess, "run", fake_run)
        monkeypatch.setattr(ytdlp, "executable", lambda: "/usr/bin/yt-dlp")
        ytdlp.run(["--version"])
        assert seen["args"][0] == "/usr/bin/yt-dlp"

    def test_does_not_use_a_shell(self, monkeypatch):
        seen = {}

        class Completed:
            returncode = 0
            stdout = ""
            stderr = ""

        def fake_run(args, **kwargs):
            seen.update(kwargs)
            return Completed()

        monkeypatch.setattr(ytdlp.subprocess, "run", fake_run)
        monkeypatch.setattr(ytdlp, "executable", lambda: "/usr/bin/yt-dlp")
        ytdlp.run(["--version"])
        assert seen.get("shell") in (None, False)

    def test_a_timeout_becomes_a_ytdlp_error(self, monkeypatch):
        import subprocess as sp

        def fake_run(*a, **k):
            raise sp.TimeoutExpired(cmd="yt-dlp", timeout=1)

        monkeypatch.setattr(ytdlp.subprocess, "run", fake_run)
        monkeypatch.setattr(ytdlp, "executable", lambda: "/usr/bin/yt-dlp")
        with pytest.raises(YtdlpError):
            ytdlp.run(["--version"], timeout=1)

    def test_a_missing_binary_is_a_clear_error(self, monkeypatch):
        monkeypatch.setattr(ytdlp.shutil, "which", lambda _: None)
        monkeypatch.delenv("YTAF_YTDLP", raising=False)
        with pytest.raises(YtafError):
            ytdlp.executable()

    def test_honours_the_env_override(self, monkeypatch, tmp_path):
        binary = tmp_path / "yt-dlp"
        binary.write_text("#!/bin/sh\n")
        binary.chmod(0o755)
        monkeypatch.setenv("YTAF_YTDLP", str(binary))
        assert ytdlp.executable() == str(binary)


class TestHighLevelCalls:
    def test_search_parses_entries(self, fake_ytdlp, fixture_text):
        fake_ytdlp.stdout = fixture_text("search-flat.json")
        entries = ytdlp.search("computerphile")
        assert [e["id"] for e in entries] == [
            "searchHit01",
            "searchHit02",
            "searchHit03",
            "searchHit04",
            "searchHit05",
        ]

    def test_search_passes_the_limit_through(self, fake_ytdlp, fixture_text):
        fake_ytdlp.stdout = fixture_text("search-flat.json")
        ytdlp.search("computerphile", limit=12)
        assert "ytsearchdate12:computerphile" in fake_ytdlp.last_args

    def test_fetch_metadata_keys_by_video_id(self, fake_ytdlp, fixture_text):
        fake_ytdlp.stdout = fixture_text("video.info.json")
        out = ytdlp.fetch_metadata(["aircAruvnKk"])
        assert set(out) == {"aircAruvnKk"}
        assert out["aircAruvnKk"]["duration"] == 1140

    def test_fetch_metadata_omits_ids_it_could_not_resolve(self, fake_ytdlp, fixture_text):
        # A deleted or geo-blocked video must not fail the batch.
        fake_ytdlp.stdout = fixture_text("video.info.json")
        out = ytdlp.fetch_metadata(["aircAruvnKk", "deletedVid1"])
        assert "deletedVid1" not in out
        assert "aircAruvnKk" in out

    def test_fetch_metadata_survives_a_nonzero_exit(self, fake_ytdlp, fixture_text):
        # yt-dlp exits 1 if any id failed, even when others succeeded.
        fake_ytdlp.stdout = fixture_text("video.info.json")
        fake_ytdlp.returncode = 1
        fake_ytdlp.stderr = "ERROR: [youtube] deletedVid1: Video unavailable"
        out = ytdlp.fetch_metadata(["aircAruvnKk", "deletedVid1"])
        assert "aircAruvnKk" in out

    def test_fetch_metadata_handles_an_empty_id_list(self, fake_ytdlp):
        assert ytdlp.fetch_metadata([]) == {}
        assert fake_ytdlp.calls == []

    def test_download_invokes_with_the_built_args(self, fake_ytdlp, config):
        ytdlp.download("aircAruvnKk", config)
        assert fake_ytdlp.last_args == ytdlp.build_download_args("aircAruvnKk", config)

    def test_dry_run_does_not_invoke(self, fake_ytdlp, config):
        ytdlp.download("aircAruvnKk", config, dry_run=True)
        assert fake_ytdlp.calls == []

    def test_dry_run_still_validates_the_arguments(self, fake_ytdlp, config):
        from ytaf.config import DownloadConfig

        cfg = config.with_(download=DownloadConfig(format_selector="bv*+ba"))
        # A dry run is how you check a config before 04:00; it has to be the
        # same validation the real run does.
        result = ytdlp.download("aircAruvnKk", cfg, dry_run=True)
        assert result is not None

    def test_update_calls_dash_u(self, fake_ytdlp):
        ytdlp.update()
        assert "-U" in fake_ytdlp.last_args or "--update" in fake_ytdlp.last_args
