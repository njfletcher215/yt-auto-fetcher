"""Library paths -- the Python mirror of the yt-dlp output template.

Two things get tested hard: that the derived path matches what yt-dlp will
actually write (the download stage has to find the file it just made), and
that a hostile channel name cannot aim a write -- or a retention delete --
outside the library root.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ytaf import naming, spec
from ytaf.config import Paths
from ytaf.errors import ConfigError, MetadataError, UnsafePathError


@pytest.fixture
def metadata():
    return {
        "id": "aircAruvnKk",
        "channel": "3Blue1Brown",
        "title": "But what is a neural network?",
        "upload_date": "20260914",
    }


class TestSanitizeComponent:
    def test_leaves_an_ordinary_name_alone(self):
        assert naming.sanitize_component("3Blue1Brown") == "3Blue1Brown"

    @pytest.mark.parametrize("char", list('/\\:*?"<>|'))
    def test_replaces_every_unsafe_character(self, char):
        out = naming.sanitize_component(f"Before{char}After")
        assert char not in out
        assert "Before" in out and "After" in out

    def test_a_slash_cannot_create_a_directory_level(self):
        assert "/" not in naming.sanitize_component("Tom/Jerry")

    def test_strips_control_characters(self):
        assert "\n" not in naming.sanitize_component("Line\nBreak")

    def test_collapses_whitespace(self):
        assert naming.sanitize_component("Too    many   spaces") == "Too many spaces"

    def test_strips_padding(self):
        assert naming.sanitize_component("  padded  ") == "padded"

    def test_strips_a_leading_dot(self):
        # A leading dot hides the file from Jellyfin's scanner entirely.
        assert not naming.sanitize_component(".hidden").startswith(".")

    def test_strips_trailing_dots_and_spaces(self):
        # SMB and exFAT shares reject these.
        assert naming.sanitize_component("Trailing...") == "Trailing"
        assert naming.sanitize_component("Trailing ") == "Trailing"

    def test_truncates_long_names(self):
        assert len(naming.sanitize_component("x" * 500, max_length=100)) <= 100

    def test_never_returns_empty(self):
        for value in ("", "   ", "...", "///"):
            assert naming.sanitize_component(value)

    def test_preserves_unicode(self):
        assert naming.sanitize_component("日本語チャンネル") == "日本語チャンネル"

    def test_preserves_emoji(self):
        assert "🧠" in naming.sanitize_component("Brain 🧠 Stuff")

    def test_neutralises_a_parent_directory_reference(self):
        assert "/" not in naming.sanitize_component("../../etc")

    def test_preserves_square_brackets_in_a_title(self):
        # The id suffix is matched by a trailing-bracket pattern, so brackets
        # inside a title must survive without breaking id recovery.
        assert "[" in naming.sanitize_component("A Title [Remastered]")


class TestVideoNaming:
    def test_filename_is_date_title_id(self, metadata):
        assert naming.naming_for(metadata).filename == (
            "20260914 - But what is a neural network_ [aircAruvnKk].mp4"
        )

    def test_relative_path_is_channel_then_filename(self, metadata):
        obj = naming.naming_for(metadata)
        assert obj.relative_path == Path("3Blue1Brown") / obj.filename

    def test_there_is_no_season_directory(self, metadata):
        # Home Videos needs no season folders, and a flatter tree is one less
        # thing for retention to prune.
        assert len(naming.naming_for(metadata).relative_path.parts) == 2

    def test_falls_back_to_uploader_for_the_channel(self):
        obj = naming.naming_for(
            {"id": "x" * 11, "uploader": "Computerphile", "title": "T", "upload_date": "20260914"}
        )
        assert obj.channel == "Computerphile"

    def test_extension_is_configurable(self, metadata):
        assert naming.naming_for(metadata, ext="mkv").filename.endswith(".mkv")

    def test_sanitises_the_channel_and_title(self):
        obj = naming.naming_for(
            {"id": "x" * 11, "channel": "Tom/Jerry", "title": "A: B?", "upload_date": "20260914"}
        )
        assert "/" not in obj.channel
        assert ":" not in obj.filename

    def test_a_missing_upload_date_is_tolerated(self):
        # Unlike an episode number, the date prefix only affects sort order.
        # Nothing collides and nothing is mis-indexed.
        obj = naming.naming_for({"id": "x" * 11, "channel": "C", "title": "T"})
        assert obj.filename.endswith("[xxxxxxxxxxx].mp4")

    def test_a_missing_id_raises(self):
        # Without it the feedback stage can never map the file back.
        with pytest.raises(MetadataError):
            naming.naming_for({"channel": "C", "title": "T", "upload_date": "20260914"})

    def test_date_prefix_sorts_chronologically(self):
        early = naming.naming_for(
            {"id": "a" * 11, "channel": "C", "title": "Z", "upload_date": "20260101"}
        )
        late = naming.naming_for(
            {"id": "b" * 11, "channel": "C", "title": "A", "upload_date": "20261231"}
        )
        assert early.filename < late.filename


class TestVideoIdFromPath:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("20260914 - A Title [aircAruvnKk].mp4", "aircAruvnKk"),
            ("/library/C/20260914 - A [WUvTyaaNkzM].mp4", "WUvTyaaNkzM"),
            ("20260914 - Dashes [OmJ-4B-mS_Y].mp4", "OmJ-4B-mS_Y"),
            ("20260914 - A Title [aircAruvnKk].info.json", "aircAruvnKk"),
        ],
    )
    def test_reads_the_id_back(self, name, expected):
        assert naming.video_id_from_path(name) == expected

    def test_prefers_the_trailing_id_over_brackets_in_the_title(self):
        assert naming.video_id_from_path(
            "20260914 - Episode [Remastered] [aircAruvnKk].mp4"
        ) == "aircAruvnKk"

    @pytest.mark.parametrize(
        "name",
        [
            "Some Movie (2011).mkv",
            "20260914 - No id here.mp4",
            "20260914 - Too short [abc].mp4",
        ],
    )
    def test_returns_none_for_an_unrelated_file(self, name):
        assert naming.video_id_from_path(name) is None

    def test_round_trips_with_naming_for(self, metadata):
        obj = naming.naming_for(metadata)
        assert naming.video_id_from_path(obj.filename) == metadata["id"]


class TestLibraryPath:
    def test_is_rooted_in_the_library(self, config, metadata):
        assert naming.library_path(config, metadata).is_relative_to(config.paths.library)

    def test_matches_the_template_structure(self, config, metadata):
        path = naming.library_path(config, metadata)
        assert path.parent == config.paths.library / "3Blue1Brown"
        assert path.name.startswith("20260914 - ")
        assert path.name.endswith("[aircAruvnKk].mp4")

    @pytest.mark.parametrize(
        "channel", ["../../etc", "/etc/passwd", "..", "....//....//etc"]
    )
    def test_a_hostile_channel_name_cannot_escape(self, config, metadata, channel):
        path = naming.library_path(config, {**metadata, "channel": channel})
        assert path.is_relative_to(config.paths.library)

    def test_a_hostile_title_cannot_escape(self, config, metadata):
        path = naming.library_path(
            config, {**metadata, "title": "../../../../tmp/pwned"}
        )
        assert path.is_relative_to(config.paths.library)


class TestOutputTemplate:
    def test_joins_the_library_root_to_the_spec_template(self, config):
        assert naming.output_template(config) == str(
            config.paths.library / spec.OUTPUT_TEMPLATE
        )

    def test_keeps_the_ytdlp_placeholders_intact(self, config):
        template = naming.output_template(config)
        for placeholder in ("%(channel)s", "%(upload_date)s", "%(title)s", "%(id)s", "%(ext)s"):
            assert placeholder in template


class TestEnsureWithin:
    def test_accepts_a_path_inside_the_root(self, tmp_path):
        inside = tmp_path / "a" / "b.mp4"
        assert naming.ensure_within(tmp_path, inside) == inside.resolve()

    def test_accepts_the_root_itself(self, tmp_path):
        naming.ensure_within(tmp_path, tmp_path)

    @pytest.mark.parametrize("escape", ["../outside.mp4", "../../etc/passwd", "/etc/passwd"])
    def test_rejects_a_path_outside_the_root(self, tmp_path, escape):
        with pytest.raises(UnsafePathError):
            naming.ensure_within(tmp_path / "library", tmp_path / "library" / escape)

    def test_rejects_a_symlink_that_escapes(self, tmp_path):
        # Retention unlinks what this returns; following a symlink out of the
        # library would delete something that is not ours.
        root = tmp_path / "library"
        root.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.mp4").write_bytes(b"x")
        link = root / "link.mp4"
        link.symlink_to(outside / "secret.mp4")
        with pytest.raises(UnsafePathError):
            naming.ensure_within(root, link)

    def test_rejects_a_sibling_with_a_shared_prefix(self, tmp_path):
        # "/library" vs "/library-backup": a string prefix check would pass
        # this and let retention eat the backup.
        root = tmp_path / "library"
        root.mkdir()
        sibling = tmp_path / "library-backup"
        sibling.mkdir()
        with pytest.raises(UnsafePathError):
            naming.ensure_within(root, sibling / "x.mp4")


class TestSidecars:
    def test_finds_the_info_json_and_thumbnail(self, config, library_video):
        found = {p.name for p in naming.sidecars(library_video())}
        assert any(n.endswith(".info.json") for n in found)
        assert any(n.endswith(".jpg") for n in found)

    def test_returns_only_existing_files(self, config, library_video):
        assert naming.sidecars(
            library_video(with_info_json=False, with_thumbnail=False)
        ) == []

    def test_does_not_return_the_video_itself(self, config, library_video):
        video = library_video()
        assert video not in naming.sidecars(video)

    def test_does_not_match_a_different_video_with_a_shared_prefix(self, config):
        channel = config.paths.library / "C"
        channel.mkdir(parents=True)
        (channel / "20260914 - A [aaaaaaaaaa1].mp4").write_bytes(b"x")
        (channel / "20260914 - A Longer Title [bbbbbbbbbb2].mp4").write_bytes(b"x")
        (channel / "20260914 - A Longer Title [bbbbbbbbbb2].info.json").write_text("{}")
        assert naming.sidecars(channel / "20260914 - A [aaaaaaaaaa1].mp4") == []


class TestFindDownloaded:
    def test_finds_the_expected_mp4(self, config, library_video, metadata):
        video = library_video(
            video_id="aircAruvnKk", channel="3Blue1Brown", title="But what is a neural network_"
        )
        assert naming.find_downloaded(config, metadata) == video

    def test_falls_back_to_another_container(self, config, metadata):
        channel = config.paths.library / "3Blue1Brown"
        channel.mkdir(parents=True)
        actual = channel / "20260914 - But what is a neural network_ [aircAruvnKk].mkv"
        actual.write_bytes(b"x")
        assert naming.find_downloaded(config, metadata) == actual

    def test_falls_back_to_an_id_glob_when_the_title_differs(self, config, metadata):
        # yt-dlp's own sanitization of a pathological title may not match ours
        # character for character, but the id always will.
        channel = config.paths.library / "3Blue1Brown"
        channel.mkdir(parents=True)
        actual = channel / "20260914 - Differently Sanitised Title [aircAruvnKk].mp4"
        actual.write_bytes(b"x")
        assert naming.find_downloaded(config, metadata) == actual

    def test_returns_none_when_nothing_was_written(self, config, metadata):
        assert naming.find_downloaded(config, metadata) is None


class TestMoveToPermanent:
    @pytest.fixture
    def saving_config(self, config, permanent_library):
        return config.with_(
            paths=Paths.under(config.paths.root, permanent_library=permanent_library)
        )

    def test_moves_the_video(self, saving_config, library_video):
        video = library_video()
        new_path = naming.move_to_permanent(saving_config, video)
        assert new_path.exists()
        assert not video.exists()

    def test_lands_in_the_permanent_library(self, saving_config, library_video):
        new_path = naming.move_to_permanent(saving_config, library_video())
        assert new_path.is_relative_to(saving_config.paths.permanent_library)

    def test_preserves_the_channel_subdirectory(self, saving_config, library_video):
        new_path = naming.move_to_permanent(
            saving_config, library_video(channel="3Blue1Brown")
        )
        assert new_path.parent.name == "3Blue1Brown"

    def test_moves_the_sidecars(self, saving_config, library_video):
        new_path = naming.move_to_permanent(saving_config, library_video())
        assert (new_path.parent / f"{new_path.stem}.info.json").exists()
        assert (new_path.parent / f"{new_path.stem}.jpg").exists()

    def test_keeps_the_filename_so_the_id_survives(self, saving_config, library_video):
        video = library_video(video_id="aircAruvnKk")
        new_path = naming.move_to_permanent(saving_config, video)
        assert naming.video_id_from_path(new_path) == "aircAruvnKk"

    def test_refuses_without_a_permanent_library(self, config, library_video):
        with pytest.raises(ConfigError):
            naming.move_to_permanent(config, library_video())

    def test_refuses_a_source_outside_the_managed_library(
        self, saving_config, tmp_path
    ):
        stray = tmp_path / "elsewhere.mp4"
        stray.write_bytes(b"x")
        with pytest.raises(UnsafePathError):
            naming.move_to_permanent(saving_config, stray)
        assert stray.exists()
