"""Command line entry points.

Exit codes are the contract: a scheduler reads them, and so does anyone
running a stage by hand to see why last night was quiet. A stage that failed
must not exit 0.

The sentiment commands get real attention because they are the whole
like/dislike interface -- Jellyfin's heart covers likes, and nothing in any
client covers dislikes.
"""

from __future__ import annotations

import json

import pytest

from ytaf import cli, spec
from ytaf.ledger import Ledger


@pytest.fixture
def config_file(tmp_path, config):
    path = tmp_path / "config.toml"
    path.write_text(
        f"""
        [paths]
        root = "{config.paths.root}"

        [discovery]
        strategy = "rss"
        channel_ids = ["UCYO_jab_esuFRV4b17AJtAw"]
        """,
        encoding="utf-8",
    )
    return path


@pytest.fixture
def saving_config_file(tmp_path, config, permanent_library):
    path = tmp_path / "saving.toml"
    path.write_text(
        f"""
        [paths]
        root = "{config.paths.root}"
        permanent_library = "{permanent_library}"

        [discovery]
        strategy = "rss"
        channel_ids = ["UCYO_jab_esuFRV4b17AJtAw"]
        """,
        encoding="utf-8",
    )
    return path


@pytest.fixture
def initialised(config_file):
    cli.main(["--config", str(config_file), "init"])
    return config_file


def open_ledger(config):
    return Ledger.open(config.paths.ledger)


class TestParser:
    def test_builds(self):
        assert cli.build_parser() is not None

    @pytest.mark.parametrize("command", cli.COMMANDS)
    def test_every_command_is_accepted(self, command):
        extra = ["aircAruvnKk"] if command in (
            "like", "dislike", "clear", "save", "verify-format"
        ) else (["/tmp/takeout"] if command == "import-takeout" else [])
        args = cli.build_parser().parse_args([command, *extra])
        assert args.command == command

    def test_help_exits_zero(self):
        with pytest.raises(SystemExit) as excinfo:
            cli.build_parser().parse_args(["--help"])
        assert excinfo.value.code == 0

    def test_an_unknown_command_exits_two(self):
        with pytest.raises(SystemExit) as excinfo:
            cli.build_parser().parse_args(["frobnicate"])
        assert excinfo.value.code == 2

    @pytest.mark.parametrize("flag", ["--dry-run", "--verbose", "--json"])
    def test_global_flags_parse(self, flag):
        assert cli.build_parser().parse_args([flag, "status"]) is not None

    def test_config_flag_parses(self, tmp_path):
        args = cli.build_parser().parse_args(["--config", str(tmp_path / "c.toml"), "status"])
        assert args.config == str(tmp_path / "c.toml")

    def test_root_flag_parses(self, tmp_path):
        assert cli.build_parser().parse_args(["--root", str(tmp_path), "status"]).root == (
            str(tmp_path)
        )

    @pytest.mark.parametrize("command", ["like", "dislike", "clear", "save"])
    def test_sentiment_commands_take_several_ids(self, command):
        args = cli.build_parser().parse_args([command, "aircAruvnKk", "WUvTyaaNkzM"])
        assert list(args.video_ids) == ["aircAruvnKk", "WUvTyaaNkzM"]

    @pytest.mark.parametrize("command", ["like", "dislike", "clear", "save"])
    def test_sentiment_commands_require_an_id(self, command):
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args([command])

    def test_stage_commands_are_distinct_from_manual_ones(self):
        assert set(cli.STAGE_COMMANDS).isdisjoint(cli.MANUAL_COMMANDS)
        assert set(cli.COMMANDS) == set(cli.STAGE_COMMANDS) | set(cli.MANUAL_COMMANDS)


class TestMain:
    def test_no_arguments_exits_nonzero(self):
        assert cli.main([]) != 0

    def test_init_creates_the_tree_and_ledger(self, tmp_path):
        root = tmp_path / "data"
        assert cli.main(["--root", str(root), "init"]) == cli.EXIT_OK
        assert (root / "library").is_dir()
        assert (root / "state" / "ledger.db").exists()

    def test_init_is_idempotent(self, tmp_path):
        root = tmp_path / "data"
        cli.main(["--root", str(root), "init"])
        assert cli.main(["--root", str(root), "init"]) == cli.EXIT_OK

    def test_a_bad_config_exits_two(self, tmp_path):
        bad = tmp_path / "bad.toml"
        bad.write_text("[selection]\nn_random = 0\n", encoding="utf-8")
        assert cli.main(["--config", str(bad), "status"]) == cli.EXIT_USAGE

    def test_a_missing_config_exits_two(self, tmp_path):
        assert cli.main(["--config", str(tmp_path / "ghost.toml"), "status"]) == (
            cli.EXIT_USAGE
        )

    def test_root_overrides_the_config_file(self, tmp_path, config_file):
        other = tmp_path / "elsewhere"
        cli.main(["--config", str(config_file), "--root", str(other), "init"])
        assert (other / "library").is_dir()


class TestSentimentCommands:
    @pytest.fixture
    def stocked(self, initialised, config, make_candidate, library_video):
        video = library_video(video_id="aircAruvnKk")
        with open_ledger(config) as led:
            led.upsert(
                make_candidate(
                    "aircAruvnKk", downloaded=1, filepath=str(video), source="rss"
                )
            )
            led.upsert(make_candidate("fromhistory", source="history", label=spec.LABEL_HISTORY))
        return initialised

    def label_of(self, config, video_id):
        with open_ledger(config) as led:
            return led.get(video_id).label

    def test_like_sets_the_label(self, stocked, config):
        assert cli.main(["--config", str(stocked), "like", "aircAruvnKk"]) == cli.EXIT_OK
        assert self.label_of(config, "aircAruvnKk") == spec.LABEL_LIKE

    def test_dislike_sets_the_label(self, stocked, config):
        assert cli.main(["--config", str(stocked), "dislike", "aircAruvnKk"]) == cli.EXIT_OK
        assert self.label_of(config, "aircAruvnKk") == spec.LABEL_DISLIKE

    def test_clear_returns_to_neutral(self, stocked, config):
        cli.main(["--config", str(stocked), "like", "aircAruvnKk"])
        cli.main(["--config", str(stocked), "clear", "aircAruvnKk"])
        assert self.label_of(config, "aircAruvnKk") is None

    def test_clear_restores_a_history_label(self, stocked, config):
        # Un-liking something from your watch history returns it to history,
        # not to nothing -- it really was watched once.
        cli.main(["--config", str(stocked), "like", "fromhistory"])
        cli.main(["--config", str(stocked), "clear", "fromhistory"])
        assert self.label_of(config, "fromhistory") == spec.LABEL_HISTORY

    def test_clear_is_the_way_out_of_a_dislike(self, stocked, config):
        cli.main(["--config", str(stocked), "dislike", "aircAruvnKk"])
        cli.main(["--config", str(stocked), "clear", "aircAruvnKk"])
        assert self.label_of(config, "aircAruvnKk") is None

    @pytest.mark.parametrize(
        "sequence,expected",
        [
            (["like", "dislike"], spec.LABEL_DISLIKE),
            (["dislike", "like"], spec.LABEL_LIKE),
            (["like", "clear", "dislike"], spec.LABEL_DISLIKE),
            (["dislike", "clear", "like"], spec.LABEL_LIKE),
            (["like", "like"], spec.LABEL_LIKE),
        ],
    )
    def test_sentiment_switches_freely(self, stocked, config, sequence, expected):
        for command in sequence:
            assert cli.main(["--config", str(stocked), command, "aircAruvnKk"]) == (
                cli.EXIT_OK
            )
        assert self.label_of(config, "aircAruvnKk") == expected

    def test_accepts_several_ids_at_once(self, stocked, config):
        cli.main(["--config", str(stocked), "like", "aircAruvnKk", "fromhistory"])
        assert self.label_of(config, "aircAruvnKk") == spec.LABEL_LIKE
        assert self.label_of(config, "fromhistory") == spec.LABEL_LIKE

    def test_an_unknown_id_exits_nonzero(self, stocked):
        assert cli.main(["--config", str(stocked), "like", "ghost000000"]) != cli.EXIT_OK

    def test_dry_run_changes_nothing(self, stocked, config):
        cli.main(["--config", str(stocked), "--dry-run", "like", "aircAruvnKk"])
        assert self.label_of(config, "aircAruvnKk") is None

    def test_a_label_change_does_not_touch_playback(self, stocked, config):
        with open_ledger(config) as led:
            led.record_watch("aircAruvnKk", 0.8, play_count=2)
        cli.main(["--config", str(stocked), "dislike", "aircAruvnKk"])
        with open_ledger(config) as led:
            row = led.get("aircAruvnKk")
        assert row.watch_pct == pytest.approx(0.8)
        assert row.play_count == 2


class TestSentimentAndJellyfin:
    @pytest.fixture
    def jf_config_file(self, tmp_path, config):
        path = tmp_path / "jf.toml"
        path.write_text(
            f"""
            [paths]
            root = "{config.paths.root}"

            [discovery]
            strategy = "rss"
            channel_ids = ["UCaaa"]

            [jellyfin]
            base_url = "https://jellyfin.test"
            api_key = "K"
            user_id = "user-1"
            """,
            encoding="utf-8",
        )
        cli.main(["--config", str(path), "init"])
        return path

    @pytest.fixture
    def stocked(self, jf_config_file, config, make_candidate, library_video):
        video = library_video(video_id="aircAruvnKk")
        with open_ledger(config) as led:
            led.upsert(
                make_candidate("aircAruvnKk", downloaded=1, filepath=str(video))
            )
        return jf_config_file

    def test_like_hearts_the_item_in_jellyfin(self, stocked, fake_http, fixture_json):
        # So a like at the CLI and a heart on the phone are the same state,
        # not two competing sources of truth.
        fake_http.route("/Items", json_body=fixture_json("jellyfin-items.json"))
        fake_http.route("/FavoriteItems/", status=200, json_body={})
        cli.main(["--config", str(stocked), "like", "aircAruvnKk"])
        assert fake_http.called("/FavoriteItems/", "POST")

    def test_a_jellyfin_failure_still_writes_the_ledger(self, stocked, config, fake_http):
        fake_http.route("/Items", status=500, body="boom")
        fake_http.route("/FavoriteItems/", status=500, body="boom")
        assert cli.main(["--config", str(stocked), "like", "aircAruvnKk"]) == cli.EXIT_OK
        with open_ledger(config) as led:
            assert led.get("aircAruvnKk").label == spec.LABEL_LIKE

    def test_works_with_no_jellyfin_configured(self, initialised, config, make_candidate):
        with open_ledger(config) as led:
            led.upsert(make_candidate("aircAruvnKk"))
        assert cli.main(["--config", str(initialised), "like", "aircAruvnKk"]) == (
            cli.EXIT_OK
        )


class TestSaveCommand:
    @pytest.fixture
    def stocked(self, saving_config_file, config, make_candidate, library_video):
        cli.main(["--config", str(saving_config_file), "init"])
        video = library_video(video_id="aircAruvnKk")
        with open_ledger(config) as led:
            led.upsert(
                make_candidate("aircAruvnKk", downloaded=1, filepath=str(video))
            )
        return saving_config_file

    def test_moves_the_video(self, stocked, config, permanent_library):
        assert cli.main(["--config", str(stocked), "save", "aircAruvnKk"]) == cli.EXIT_OK
        assert list(permanent_library.rglob("*.mp4"))

    def test_marks_the_row_saved(self, stocked, config):
        cli.main(["--config", str(stocked), "save", "aircAruvnKk"])
        with open_ledger(config) as led:
            assert led.get("aircAruvnKk").saved == 1

    def test_exits_two_without_a_permanent_library(
        self, initialised, config, make_candidate, library_video
    ):
        video = library_video(video_id="aircAruvnKk")
        with open_ledger(config) as led:
            led.upsert(make_candidate("aircAruvnKk", downloaded=1, filepath=str(video)))
        assert cli.main(["--config", str(initialised), "save", "aircAruvnKk"]) == (
            cli.EXIT_USAGE
        )

    def test_an_unknown_id_exits_nonzero(self, stocked):
        assert cli.main(["--config", str(stocked), "save", "ghost000000"]) != cli.EXIT_OK

    def test_dry_run_moves_nothing(self, stocked, permanent_library):
        cli.main(["--config", str(stocked), "--dry-run", "save", "aircAruvnKk"])
        assert list(permanent_library.rglob("*.mp4")) == []


class TestStageCommands:
    def test_discover_succeeds(self, initialised, fake_http, fake_ytdlp, fixture_text):
        fake_http.route("feeds/videos.xml", body=fixture_text("channel-feed.xml"))
        fake_ytdlp.stdout = json.dumps(
            {
                "id": "aircAruvnKk",
                "title": "T",
                "channel": "3Blue1Brown",
                "duration": 900,
                "view_count": 10_000,
                "upload_date": "20260914",
            }
        )
        assert cli.main(["--config", str(initialised), "discover"]) == cli.EXIT_OK

    def test_discover_exits_nonzero_on_a_total_source_failure(self, initialised, fake_http):
        fake_http.route("feeds/videos.xml", status=500, body="boom")
        assert cli.main(["--config", str(initialised), "discover"]) != cli.EXIT_OK

    def test_discover_dry_run_writes_nothing(
        self, initialised, config, fake_http, fake_ytdlp, fixture_text
    ):
        fake_http.route("feeds/videos.xml", body=fixture_text("channel-feed.xml"))
        fake_ytdlp.stdout = ""
        cli.main(["--config", str(initialised), "--dry-run", "discover"])
        with open_ledger(config) as led:
            assert led.count() == 0

    def test_score_runs_on_an_empty_ledger(self, initialised):
        assert cli.main(["--config", str(initialised), "score"]) == cli.EXIT_OK

    def test_download_runs_with_nothing_selected(self, initialised, fake_ytdlp):
        assert cli.main(["--config", str(initialised), "download"]) == cli.EXIT_OK

    def test_feedback_runs_without_jellyfin(self, initialised, fake_http):
        assert cli.main(["--config", str(initialised), "feedback"]) == cli.EXIT_OK

    def test_retain_runs_on_an_empty_library(self, initialised):
        assert cli.main(["--config", str(initialised), "retain"]) == cli.EXIT_OK

    def test_monitor_runs(self, initialised, fake_http):
        # A stalled discovery on an empty ledger is a genuine finding, so a
        # nonzero exit here is correct rather than a bug.
        assert cli.main(["--config", str(initialised), "monitor"]) in (
            cli.EXIT_OK,
            cli.EXIT_FAILURE,
        )

    def test_retrain_with_no_labels_exits_nonzero(self, initialised):
        assert cli.main(["--config", str(initialised), "retrain"]) == cli.EXIT_FAILURE

    def test_every_stage_command_has_an_entry_point(self):
        for command in cli.STAGE_COMMANDS:
            assert hasattr(cli, f"main_{command.replace('-', '_')}")

    def test_the_dedicated_entry_point_works(
        self, initialised, monkeypatch, fake_http, fake_ytdlp, fixture_text
    ):
        monkeypatch.setenv("YTAF_CONFIG", str(initialised))
        fake_http.route("feeds/videos.xml", body=fixture_text("channel-feed.xml"))
        fake_ytdlp.stdout = ""
        assert cli.main_discover([]) in (cli.EXIT_OK, cli.EXIT_FAILURE)


class TestStatus:
    def test_prints_something(self, initialised, capsys):
        assert cli.main(["--config", str(initialised), "status"]) == cli.EXIT_OK
        assert capsys.readouterr().out.strip()

    def test_json_is_parseable(self, initialised, capsys):
        cli.main(["--config", str(initialised), "--json", "status"])
        assert isinstance(json.loads(capsys.readouterr().out), dict)

    def test_reports_all_three_arms(self, initialised, config, make_candidate, capsys):
        with open_ledger(config) as led:
            for arm in ("model", "repeat", "random"):
                led.upsert(
                    make_candidate(
                        f"{arm}0000001"[:11],
                        selected=1,
                        selection=arm,
                        watched=1,
                        watch_pct=0.5,
                        downloaded=1,
                        filepath=f"/{arm}.mp4",
                    )
                )
        cli.main(["--config", str(initialised), "--json", "status"])
        out = json.loads(capsys.readouterr().out)
        assert "model" in json.dumps(out)
        assert "random" in json.dumps(out)


class TestImportTakeout:
    def test_imports_an_export(self, initialised, takeout_dir, config):
        assert cli.main(
            ["--config", str(initialised), "import-takeout", str(takeout_dir)]
        ) == cli.EXIT_OK
        with open_ledger(config) as led:
            assert led.count() > 0

    def test_a_missing_directory_exits_nonzero(self, initialised, tmp_path):
        assert cli.main(
            ["--config", str(initialised), "import-takeout", str(tmp_path / "ghost")]
        ) != cli.EXIT_OK

    def test_reports_the_discovered_channel_ids(self, initialised, takeout_dir, capsys):
        # These are what you paste into discovery.channel_ids, so the command
        # has to actually print them.
        cli.main(["--config", str(initialised), "import-takeout", str(takeout_dir)])
        assert "UCYO_jab_esuFRV4b17AJtAw" in capsys.readouterr().out


class TestVerifyFormat:
    def test_has_no_per_stage_entry_point(self):
        # It downloads a real video over the network. Every stage with a
        # `main_*` entry point is something a scheduler can fire; this one
        # deliberately has none.
        assert "verify-format" in cli.COMMANDS
        assert "verify-format" not in cli.STAGE_COMMANDS
        assert not hasattr(cli, "main_verify_format")

    def test_passes_when_the_result_is_h264(self, initialised, config, fake_ytdlp):
        channel = config.paths.library / "C"
        channel.mkdir(parents=True, exist_ok=True)
        stem = "20260914 - T [aircAruvnKk]"
        (channel / f"{stem}.mp4").write_bytes(b"\0")
        (channel / f"{stem}.info.json").write_text(
            json.dumps(
                {
                    "id": "aircAruvnKk",
                    "title": "T",
                    "channel": "C",
                    "upload_date": "20260914",
                    "vcodec": "avc1.640028",
                    "acodec": "mp4a.40.2",
                    "height": 1080,
                }
            ),
            encoding="utf-8",
        )
        assert cli.main(
            ["--config", str(initialised), "verify-format", "aircAruvnKk"]
        ) == cli.EXIT_OK

    def test_fails_when_the_result_is_not_h264(self, initialised, config, fake_ytdlp):
        channel = config.paths.library / "C"
        channel.mkdir(parents=True, exist_ok=True)
        stem = "20260914 - T [badCodecVid]"
        (channel / f"{stem}.mp4").write_bytes(b"\0")
        (channel / f"{stem}.info.json").write_text(
            json.dumps(
                {
                    "id": "badCodecVid",
                    "title": "T",
                    "channel": "C",
                    "upload_date": "20260914",
                    "vcodec": "vp09.00.50.08",
                    "acodec": "opus",
                    "height": 1440,
                }
            ),
            encoding="utf-8",
        )
        assert cli.main(
            ["--config", str(initialised), "verify-format", "badCodecVid"]
        ) == cli.EXIT_FAILURE
