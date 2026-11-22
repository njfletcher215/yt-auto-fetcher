"""The Jellyfin client: scans, playback reads, and transcode detection.

The watch-percent arithmetic gets the most attention. Jellyfin resets
PlaybackPositionTicks to 0 once an item is marked Played, so reading position
literally turns every finished video into a hard negative -- which would train
the model to find things you will not finish.
"""

from __future__ import annotations

import pytest

from ytaf import jellyfin
from ytaf.config import JellyfinConfig
from ytaf.errors import JellyfinError
from ytaf.jellyfin import TICKS_PER_SECOND, JellyfinClient, PlaybackItem


@pytest.fixture
def client(jellyfin_config):
    return JellyfinClient(jellyfin_config)


class TestUrlAndHeaders:
    def test_joins_the_base_url(self, client):
        assert client.url("/Sessions").startswith("https://jellyfin.test/")
        assert client.url("/Sessions").endswith("/Sessions")

    def test_does_not_double_the_slash(self):
        c = JellyfinClient(JellyfinConfig(base_url="https://jellyfin.test/", api_key="K"))
        assert "//Sessions" not in c.url("/Sessions")

    def test_accepts_a_path_without_a_leading_slash(self, client):
        assert client.url("Sessions").endswith("/Sessions")

    def test_appends_query_parameters(self, client):
        url = client.url("/Items", Recursive="true", Limit=10)
        assert "Recursive=true" in url
        assert "Limit=10" in url

    def test_url_encodes_query_values(self, client):
        assert "%20" in client.url("/Items", SearchTerm="a b") or "+" in client.url(
            "/Items", SearchTerm="a b"
        )

    def test_never_puts_the_api_key_in_the_url(self, client):
        # Keys in query strings end up in access logs and proxy caches.
        assert "TESTKEY" not in client.url("/Sessions")

    def test_sends_the_api_key_as_a_header(self, client):
        headers = client.headers()
        assert headers.get("X-Emby-Token") == "TESTKEY"

    def test_sends_the_mediabrowser_authorization_form(self, client):
        auth = client.headers().get("Authorization", "")
        assert auth.startswith("MediaBrowser ")
        assert "TESTKEY" in auth

    def test_raises_when_unconfigured(self):
        c = JellyfinClient(JellyfinConfig())
        with pytest.raises(JellyfinError):
            c.url("/Sessions")


class TestWatchPct:
    def test_computes_position_over_runtime(self):
        item = PlaybackItem(
            item_id="x",
            position_ticks=10 * TICKS_PER_SECOND,
            runtime_ticks=100 * TICKS_PER_SECOND,
            play_count=1,
        )
        assert item.watch_pct == pytest.approx(0.1)

    def test_played_is_one_regardless_of_position(self):
        # Jellyfin zeroes the position once an item is marked played, so
        # reading position literally would make every finished video a
        # mid-band negative -- the strongest one available.
        item = PlaybackItem(
            item_id="x",
            position_ticks=0,
            runtime_ticks=100 * TICKS_PER_SECOND,
            played=True,
            play_count=1,
        )
        assert item.watch_pct == pytest.approx(1.0)

    def test_a_zero_position_with_plays_is_a_quick_exit(self):
        # Opened and closed below Jellyfin's threshold for storing a resume
        # point. A real signal, just a weak one.
        item = PlaybackItem(
            item_id="x",
            position_ticks=0,
            runtime_ticks=100 * TICKS_PER_SECOND,
            play_count=1,
        )
        assert item.watch_pct == pytest.approx(0.0)

    def test_never_opened_is_none_not_zero(self):
        # No playback signal in either direction; zero would be a negative we
        # have not earned.
        item = PlaybackItem(
            item_id="x",
            position_ticks=0,
            runtime_ticks=100 * TICKS_PER_SECOND,
            play_count=0,
        )
        assert item.watch_pct is None

    def test_clamps_above_one(self):
        item = PlaybackItem(
            item_id="x",
            position_ticks=120 * TICKS_PER_SECOND,
            runtime_ticks=100 * TICKS_PER_SECOND,
            play_count=1,
        )
        assert item.watch_pct == pytest.approx(1.0)

    def test_clamps_below_zero(self):
        item = PlaybackItem(
            item_id="x",
            position_ticks=-5,
            runtime_ticks=100 * TICKS_PER_SECOND,
            play_count=1,
        )
        assert item.watch_pct == pytest.approx(0.0)

    def test_zero_runtime_is_none_not_a_division_error(self):
        item = PlaybackItem(
            item_id="x", position_ticks=50, runtime_ticks=0, play_count=1
        )
        assert item.watch_pct is None

    def test_opened_reflects_play_count_and_position(self):
        assert PlaybackItem(item_id="x", play_count=1).opened is True
        assert PlaybackItem(item_id="x", position_ticks=5).opened is True
        assert PlaybackItem(item_id="x", played=True).opened is True
        assert PlaybackItem(item_id="x").opened is False

    def test_transcoding_flag(self):
        assert PlaybackItem(item_id="x", play_method="Transcode").transcoding is True
        assert PlaybackItem(item_id="x", play_method="DirectPlay").transcoding is False
        assert PlaybackItem(item_id="x", play_method="DirectStream").transcoding is False
        assert PlaybackItem(item_id="x", play_method=None).transcoding is False


class TestTriggerScan:
    def test_posts_to_library_refresh(self, client, fake_http):
        fake_http.route("/Library/Refresh", status=204)
        result = client.trigger_scan()
        assert result.ok
        assert fake_http.called("/Library/Refresh", "POST")

    def test_sends_the_auth_header(self, client, fake_http):
        fake_http.route("/Library/Refresh", status=204)
        client.trigger_scan()
        _, _, kwargs = fake_http.calls[0]
        assert kwargs["headers"]["X-Emby-Token"] == "TESTKEY"

    def test_a_server_error_is_reported_not_raised(self, client, fake_http):
        # A failed scan is cosmetic; the files are on disk either way.
        fake_http.route("/Library/Refresh", status=500, body="boom")
        result = client.trigger_scan()
        assert not result.ok
        assert result.status == 500

    def test_a_transport_failure_is_reported_not_raised(self, client, monkeypatch):
        from ytaf import http as http_mod
        from ytaf.errors import YtafError

        def boom(*args, **kwargs):
            raise YtafError("connection refused")

        monkeypatch.setattr(http_mod, "post", boom)
        result = client.trigger_scan()
        assert not result.ok


class TestSessions:
    def test_parses_active_sessions(self, client, fake_http, fixture_json):
        fake_http.route("/Sessions", json_body=fixture_json("jellyfin-sessions.json"))
        assert {i.item_id for i in client.sessions()} == {"jf-item-1", "jf-item-3"}

    def test_skips_sessions_with_nothing_playing(self, client, fake_http, fixture_json):
        fake_http.route("/Sessions", json_body=fixture_json("jellyfin-sessions.json"))
        assert len(client.sessions()) == 2

    def test_reads_the_play_method(self, client, fake_http, fixture_json):
        fake_http.route("/Sessions", json_body=fixture_json("jellyfin-sessions.json"))
        methods = {i.item_id: i.play_method for i in client.sessions()}
        assert methods["jf-item-1"] == "DirectPlay"
        assert methods["jf-item-3"] == "Transcode"

    def test_reads_position_and_runtime(self, client, fake_http, fixture_json):
        fake_http.route("/Sessions", json_body=fixture_json("jellyfin-sessions.json"))
        item = next(i for i in client.sessions() if i.item_id == "jf-item-1")
        assert item.watch_pct == pytest.approx(0.9)

    def test_reads_the_path(self, client, fake_http, fixture_json):
        fake_http.route("/Sessions", json_body=fixture_json("jellyfin-sessions.json"))
        item = next(i for i in client.sessions() if i.item_id == "jf-item-1")
        assert item.path.endswith(".mp4")

    def test_a_server_error_raises(self, client, fake_http):
        fake_http.route("/Sessions", status=401, body="unauthorized")
        with pytest.raises(JellyfinError):
            client.sessions()

    def test_unparseable_json_raises(self, client, fake_http):
        fake_http.route("/Sessions", body="<html>not json</html>")
        with pytest.raises(JellyfinError):
            client.sessions()


class TestTranscodingSessions:
    def test_finds_only_transcoding_sessions(self, client, fake_http, fixture_json):
        fake_http.route("/Sessions", json_body=fixture_json("jellyfin-sessions.json"))
        assert [i.item_id for i in client.transcoding_sessions()] == ["jf-item-3"]

    def test_empty_when_everything_direct_plays(self, client, fake_http):
        fake_http.route(
            "/Sessions",
            json_body=[
                {
                    "Id": "s",
                    "PlayState": {"PlayMethod": "DirectPlay", "PositionTicks": 0},
                    "NowPlayingItem": {"Id": "i", "RunTimeTicks": 100},
                }
            ],
        )
        assert client.transcoding_sessions() == []


class TestPlayedItems:
    def test_parses_the_items_payload(self, client, fake_http, fixture_json):
        fake_http.route("/Items", json_body=fixture_json("jellyfin-items.json"))
        assert len(client.played_items()) == 6

    def test_queries_the_configured_user(self, client, fake_http, fixture_json):
        fake_http.route("/Items", json_body=fixture_json("jellyfin-items.json"))
        client.played_items()
        assert fake_http.called("/Users/user-1/Items")

    def test_requests_recursively(self, client, fake_http, fixture_json):
        # Episodes live two levels down from the library root; without this
        # the query returns only the channel folders.
        fake_http.route("/Items", json_body=fixture_json("jellyfin-items.json"))
        client.played_items()
        assert fake_http.called("Recursive=true") or fake_http.called("Recursive=True")

    def test_reads_playback_position_from_userdata(self, client, fake_http, fixture_json):
        fake_http.route("/Items", json_body=fixture_json("jellyfin-items.json"))
        item = next(i for i in client.played_items() if i.item_id == "jf-item-1")
        assert item.watch_pct == pytest.approx(0.95)

    def test_an_item_with_no_last_played_date_survives_filtering(
        self, client, fake_http, fixture_json
    ):
        # A hearted but never-played item has no LastPlayedDate, and dropping
        # it would lose the like.
        fake_http.route("/Items", json_body=fixture_json("jellyfin-items.json"))
        items = {i.item_id for i in client.played_items(since="2026-09-14T00:00:00+00:00")}
        assert "jf-item-6" in items

    def test_reads_the_played_flag(self, client, fake_http, fixture_json):
        fake_http.route("/Items", json_body=fixture_json("jellyfin-items.json"))
        item = next(i for i in client.played_items() if i.item_id == "jf-item-2")
        assert item.played is True
        assert item.watch_pct == pytest.approx(1.0)

    def test_reads_the_play_count(self, client, fake_http, fixture_json):
        fake_http.route("/Items", json_body=fixture_json("jellyfin-items.json"))
        item = next(i for i in client.played_items() if i.item_id == "jf-item-2")
        assert item.play_count == 3

    def test_reads_the_favorite_flag(self, client, fake_http, fixture_json):
        fake_http.route("/Items", json_body=fixture_json("jellyfin-items.json"))
        items = {i.item_id: i.is_favorite for i in client.played_items()}
        assert items["jf-item-2"] is True
        assert items["jf-item-1"] is False

    def test_filters_client_side_on_last_played(self, client, fake_http, fixture_json):
        fake_http.route("/Items", json_body=fixture_json("jellyfin-items.json"))
        ids = {i.item_id for i in client.played_items(since="2026-09-14T00:00:00+00:00")}
        assert "jf-item-3" not in ids  # played 09-13
        assert "jf-item-1" in ids

    def test_an_empty_library_returns_nothing(self, client, fake_http):
        fake_http.route("/Items", json_body={"Items": [], "TotalRecordCount": 0})
        assert client.played_items() == []

    def test_requires_a_user_id(self, fake_http):
        c = JellyfinClient(JellyfinConfig(base_url="https://jellyfin.test", api_key="K"))
        with pytest.raises(JellyfinError):
            c.played_items()


class TestVideoIdFromItem:
    def test_reads_the_id_out_of_the_path(self):
        # A Home Videos library does not read the NFO uniqueid a TV library
        # would, so the path is the only channel for this.
        item = PlaybackItem(
            item_id="x", path="/library/C/20260914 - A Title [aircAruvnKk].mp4"
        )
        assert jellyfin.video_id_from_item(item) == "aircAruvnKk"

    def test_ignores_brackets_in_the_title(self):
        item = PlaybackItem(
            item_id="x", path="/library/C/20260914 - A [Remastered] [aircAruvnKk].mp4"
        )
        assert jellyfin.video_id_from_item(item) == "aircAruvnKk"

    def test_returns_none_for_an_unrelated_item(self):
        # The library has plenty of things this pipeline did not put there.
        item = PlaybackItem(item_id="x", path="/mnt/other/movies/Some Movie.mkv")
        assert jellyfin.video_id_from_item(item) is None

    def test_returns_none_with_no_path(self):
        assert jellyfin.video_id_from_item(PlaybackItem(item_id="x")) is None


class TestMatchItemsToLedger:
    def test_matches_on_filepath(self):
        item = PlaybackItem(item_id="x", path="/library/a.mp4")
        matched = jellyfin.match_items_to_ledger([item], {"/library/a.mp4": "vid00000001"})
        assert matched == {"vid00000001": item}

    def test_falls_back_to_the_id_in_the_path(self):
        # Covers a video that was moved to the permanent library, where the
        # ledger filepath and Jellyfin's path may disagree for one run.
        item = PlaybackItem(item_id="x", path="/elsewhere/20260914 - A [aircAruvnKk].mp4")
        assert "aircAruvnKk" in jellyfin.match_items_to_ledger([item], {})

    def test_path_index_match_wins(self):
        item = PlaybackItem(
            item_id="x", path="/library/20260914 - A [wrongIdHere].mp4"
        )
        matched = jellyfin.match_items_to_ledger(
            [item], {"/library/20260914 - A [wrongIdHere].mp4": "vid00000001"}
        )
        assert set(matched) == {"vid00000001"}

    def test_unmatched_items_are_omitted(self):
        item = PlaybackItem(item_id="x", path="/movies/a.mkv")
        assert jellyfin.match_items_to_ledger([item], {}) == {}

    def test_handles_an_empty_input(self):
        assert jellyfin.match_items_to_ledger([], {}) == {}


class TestBuildClient:
    def test_builds_when_configured(self, jellyfin_config):
        assert jellyfin.build_client(jellyfin_config) is not None

    def test_returns_none_when_unconfigured(self):
        assert jellyfin.build_client(JellyfinConfig()) is None

    def test_returns_none_without_an_api_key(self):
        assert jellyfin.build_client(JellyfinConfig(base_url="https://x")) is None


class TestFavorites:
    def test_lists_hearted_items(self, client, fake_http, fixture_json):
        items = fixture_json("jellyfin-items.json")["Items"]
        hearted = [i for i in items if i["UserData"]["IsFavorite"]]
        fake_http.route("/Items", json_body={"Items": hearted, "TotalRecordCount": len(hearted)})
        assert {i.item_id for i in client.favorites()} == {"jf-item-2", "jf-item-6"}

    def test_queries_favorites_separately_from_playback(self, client, fake_http, fixture_json):
        # You can heart something without playing it, and that still means you
        # like it -- so this cannot be a filter over played items.
        fake_http.route("/Items", json_body=fixture_json("jellyfin-items.json"))
        client.favorites()
        assert fake_http.called("IsFavorite=true") or fake_http.called("Filters=IsFavorite")

    def test_sets_a_favorite(self, client, fake_http):
        fake_http.route("/FavoriteItems/", status=200, json_body={})
        assert client.set_favorite("jf-item-1", True) is True
        assert fake_http.called("/Users/user-1/FavoriteItems/jf-item-1")

    def test_clears_a_favorite(self, client, fake_http):
        fake_http.route("/FavoriteItems/", status=200, json_body={})
        assert client.set_favorite("jf-item-1", False) is True

    def test_a_failure_returns_false_rather_than_raising(self, client, fake_http):
        # `ytaf like` must still write the ledger when Jellyfin is down; the
        # next feedback run reconciles the heart.
        fake_http.route("/FavoriteItems/", status=500, body="boom")
        assert client.set_favorite("jf-item-1", True) is False

    def test_a_transport_failure_returns_false(self, client, monkeypatch):
        from ytaf import http as http_mod
        from ytaf.errors import YtafError

        monkeypatch.setattr(
            http_mod, "post", lambda *a, **k: (_ for _ in ()).throw(YtafError("down"))
        )
        assert client.set_favorite("jf-item-1", True) is False

    def test_requires_a_user_id(self, fake_http):
        c = JellyfinClient(JellyfinConfig(base_url="https://jellyfin.test", api_key="K"))
        with pytest.raises(JellyfinError):
            c.set_favorite("jf-item-1", True)


class TestFindByPath:
    def test_locates_an_item(self, client, fake_http, fixture_json):
        fake_http.route("/Items", json_body=fixture_json("jellyfin-items.json"))
        item = client.find_by_path(
            "/library/3Blue1Brown/20260914 - But what is a neural network [aircAruvnKk].mp4"
        )
        assert item is not None
        assert item.item_id == "jf-item-1"

    def test_returns_none_for_an_unknown_path(self, client, fake_http, fixture_json):
        fake_http.route("/Items", json_body=fixture_json("jellyfin-items.json"))
        assert client.find_by_path("/library/C/nothing [zzzzzzzzzzz].mp4") is None
