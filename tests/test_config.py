"""Config: path derivation, TOML loading, and hard-constraint validation.

The validation tests are the point of this module. This runs unattended on a
schedule, so a config that is wrong in a way nothing notices is worse than one
that refuses to load.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ytaf import config as config_mod
from ytaf.config import (
    CONFIG_ENV_VAR,
    Config,
    DiscoveryConfig,
    DownloadConfig,
    FeedbackConfig,
    JellyfinConfig,
    ModelConfig,
    NotifyConfig,
    Paths,
    PrefilterConfig,
    RetentionConfig,
    SelectionConfig,
)
from ytaf.errors import ConfigError


class TestPaths:
    def test_derives_the_directory_tree(self, tmp_path: Path):
        paths = Paths.under(tmp_path / "youtube")
        assert paths.library == tmp_path / "youtube" / "library"
        assert paths.state == tmp_path / "youtube" / "state"
        assert paths.logs == tmp_path / "youtube" / "logs"

    def test_the_ledger_lives_in_state(self, tmp_path: Path):
        assert Paths.under(tmp_path).ledger == tmp_path / "state" / "ledger.db"

    def test_permanent_library_is_unset_by_default(self, tmp_path: Path):
        assert Paths.under(tmp_path).permanent_library is None

    def test_permanent_library_is_kept_outside_root(self, tmp_path: Path):
        # Nothing under root survives retention, so a permanent library
        # inside it would be deleted along with everything else.
        paths = Paths.under(tmp_path / "managed", permanent_library=tmp_path / "keep")
        assert not paths.permanent_library.is_relative_to(paths.root)

    def test_permanent_library_expands_a_tilde(self):
        paths = Paths.under("/tmp/x", permanent_library="~/keep")
        assert "~" not in str(paths.permanent_library)

    def test_model_dir_lives_in_state(self, tmp_path: Path):
        paths = Paths.under(tmp_path)
        assert paths.model_dir == tmp_path / "state" / "model"
        assert paths.centroid_path == paths.model_dir / "centroid.pkl"
        assert paths.classifier_path == paths.model_dir / "classifier.pkl"

    def test_expands_a_tilde(self):
        paths = Paths.under("~/yt-data")
        assert "~" not in str(paths.root)
        assert paths.root.is_absolute()

    def test_ensure_creates_everything(self, tmp_path: Path):
        paths = Paths.under(tmp_path / "fresh")
        paths.ensure()
        for d in (paths.root, paths.library, paths.state, paths.logs, paths.model_dir):
            assert d.is_dir()

    def test_ensure_is_idempotent(self, tmp_path: Path):
        paths = Paths.under(tmp_path / "fresh")
        paths.ensure()
        paths.ensure()
        assert paths.library.is_dir()

    def test_ensure_does_not_create_the_ledger_file(self, tmp_path: Path):
        # Creating an empty ledger.db here would mask unmounted storage.
        paths = Paths.under(tmp_path / "fresh")
        paths.ensure()
        assert not paths.ledger.exists()

    def test_ensure_does_not_create_the_permanent_library(self, tmp_path: Path):
        # It is an existing Jellyfin library. Silently making an empty
        # directory at a mistyped path would send saved videos nowhere.
        paths = Paths.under(tmp_path / "fresh", permanent_library=tmp_path / "typo")
        paths.ensure()
        assert not paths.permanent_library.exists()

    def test_is_frozen(self, tmp_path: Path):
        import dataclasses

        paths = Paths.under(tmp_path)
        with pytest.raises(dataclasses.FrozenInstanceError):
            paths.root = tmp_path  # type: ignore[misc]


class TestDefaults:
    def test_selection_defaults_to_five_two_and_three(self):
        cfg = SelectionConfig()
        assert (cfg.n_model_new, cfg.n_model_repeat, cfg.n_random) == (5, 2, 3)
        assert cfg.total == 10
        assert cfg.n_model == 7

    def test_repeat_cooldown_defaults_to_ninety_days(self):
        assert SelectionConfig().repeat_cooldown_days == 90

    def test_feedback_bands_default_to_the_spec(self):
        cfg = FeedbackConfig()
        assert cfg.watch_threshold == pytest.approx(0.60)
        assert cfg.partial_floor == pytest.approx(0.15)
        assert cfg.sync_favorites is True

    def test_prefilter_defaults_are_the_documented_bounds(self):
        pf = PrefilterConfig()
        assert (pf.min_duration, pf.max_duration, pf.min_view_count) == (300, 7200, 500)
        assert pf.allow_live is False

    def test_download_defaults_to_the_spec_selector(self):
        from ytaf import spec

        assert DownloadConfig().format_selector == spec.FORMAT_SELECTOR
        assert DownloadConfig().merge_output_format == "mp4"

    def test_retention_defaults_to_fourteen_days(self):
        r = RetentionConfig()
        assert r.max_age_days == 14
        assert r.log_retention_days == 90

    def test_retention_has_no_watch_or_label_exemptions(self):
        # Age is the only criterion. `ytaf save` is the only reprieve.
        assert not hasattr(RetentionConfig(), "watch_pct_threshold")
        assert not hasattr(RetentionConfig(), "keep_labeled")

    def test_jellyfin_is_unconfigured_by_default(self):
        # Discovery, scoring and download must all work with no Jellyfin.
        assert JellyfinConfig().configured is False

    def test_notifications_are_off_by_default(self):
        assert NotifyConfig().configured is False

    def test_config_for_root_fills_in_every_section(self, tmp_path: Path):
        cfg = Config.for_root(tmp_path)
        assert isinstance(cfg.discovery, DiscoveryConfig)
        assert isinstance(cfg.selection, SelectionConfig)
        assert isinstance(cfg.retention, RetentionConfig)
        assert cfg.paths.root == tmp_path


class TestDiscoveryStrategy:
    @pytest.mark.parametrize(
        "strategy,rss,search",
        [
            ("rss", True, False),
            ("search", False, True),
            ("both", True, True),
        ],
    )
    def test_strategy_flags(self, strategy, rss, search):
        d = DiscoveryConfig(strategy=strategy)
        assert d.uses_rss is rss
        assert d.uses_search is search


class TestFindConfigFile:
    def test_prefers_an_explicit_path(self, tmp_path: Path, monkeypatch):
        explicit = tmp_path / "explicit.toml"
        explicit.write_text("", encoding="utf-8")
        other = tmp_path / "env.toml"
        other.write_text("", encoding="utf-8")
        monkeypatch.setenv(CONFIG_ENV_VAR, str(other))
        assert config_mod.find_config_file(explicit) == explicit

    def test_missing_explicit_path_is_an_error_not_a_fallthrough(self, tmp_path: Path):
        with pytest.raises(ConfigError):
            config_mod.find_config_file(tmp_path / "nope.toml")

    def test_falls_back_to_the_env_var(self, tmp_path: Path, monkeypatch):
        target = tmp_path / "env.toml"
        target.write_text("", encoding="utf-8")
        monkeypatch.setenv(CONFIG_ENV_VAR, str(target))
        monkeypatch.chdir(tmp_path)
        assert config_mod.find_config_file() == target

    def test_env_var_pointing_at_nothing_is_an_error(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv(CONFIG_ENV_VAR, str(tmp_path / "ghost.toml"))
        with pytest.raises(ConfigError):
            config_mod.find_config_file()

    def test_falls_back_to_cwd(self, tmp_path: Path, monkeypatch):
        monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)
        local = tmp_path / "config.toml"
        local.write_text("", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        assert config_mod.find_config_file() == local

    def test_returns_none_when_nothing_exists(self, tmp_path: Path, monkeypatch):
        monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(config_mod, "SYSTEM_CONFIG_PATH", tmp_path / "etc.toml")
        assert config_mod.find_config_file() is None


class TestLoadConfig:
    def test_parses_a_full_file(self, tmp_path: Path):
        path = tmp_path / "config.toml"
        path.write_text(
            f"""
            [paths]
            root = "{tmp_path / 'data'}"

            [discovery]
            strategy = "rss"
            channel_ids = ["UCaaa", "UCbbb"]

            [selection]
            n_model_new = 3
            n_random = 2

            [retention]
            max_age_days = 14
            """,
            encoding="utf-8",
        )
        cfg = config_mod.load_config(path)
        assert cfg.paths.root == tmp_path / "data"
        assert cfg.discovery.strategy == "rss"
        assert cfg.discovery.channel_ids == ("UCaaa", "UCbbb")
        assert cfg.selection.n_model_new == 3
        assert cfg.retention.max_age_days == 14

    def test_records_where_it_came_from(self, tmp_path: Path):
        path = tmp_path / "config.toml"
        path.write_text(f'[paths]\nroot = "{tmp_path}"\n', encoding="utf-8")
        assert config_mod.load_config(path).source_path == path

    def test_unspecified_sections_keep_their_defaults(self, tmp_path: Path):
        path = tmp_path / "config.toml"
        path.write_text(f'[paths]\nroot = "{tmp_path}"\n', encoding="utf-8")
        cfg = config_mod.load_config(path)
        assert cfg.selection.n_model_new == 5
        assert cfg.prefilter.min_duration == 300

    def test_explicit_subpath_overrides_derivation(self, tmp_path: Path):
        path = tmp_path / "config.toml"
        path.write_text(
            f"""
            [paths]
            root = "{tmp_path / 'data'}"
            library = "{tmp_path / 'elsewhere'}"
            """,
            encoding="utf-8",
        )
        cfg = config_mod.load_config(path)
        assert cfg.paths.library == tmp_path / "elsewhere"
        assert cfg.paths.state == tmp_path / "data" / "state"

    def test_tilde_in_the_file_is_expanded(self, tmp_path: Path):
        path = tmp_path / "config.toml"
        path.write_text('[paths]\nroot = "~/yt-data"\n', encoding="utf-8")
        assert "~" not in str(config_mod.load_config(path).paths.root)

    def test_unknown_key_is_rejected(self, tmp_path: Path):
        path = tmp_path / "config.toml"
        path.write_text(
            f'[paths]\nroot = "{tmp_path}"\n\n[selection]\nn_modle = 7\n',
            encoding="utf-8",
        )
        with pytest.raises(ConfigError, match="n_modle"):
            config_mod.load_config(path)

    def test_unknown_section_is_rejected(self, tmp_path: Path):
        path = tmp_path / "config.toml"
        path.write_text(
            f'[paths]\nroot = "{tmp_path}"\n\n[dicsovery]\nstrategy = "rss"\n',
            encoding="utf-8",
        )
        with pytest.raises(ConfigError, match="dicsovery"):
            config_mod.load_config(path)

    def test_malformed_toml_is_a_config_error(self, tmp_path: Path):
        path = tmp_path / "config.toml"
        path.write_text("[paths\nroot =", encoding="utf-8")
        with pytest.raises(ConfigError):
            config_mod.load_config(path)

    def test_wrong_type_is_a_config_error(self, tmp_path: Path):
        path = tmp_path / "config.toml"
        path.write_text(
            f'[paths]\nroot = "{tmp_path}"\n\n[selection]\nn_model_new = "five"\n',
            encoding="utf-8",
        )
        with pytest.raises(ConfigError):
            config_mod.load_config(path)

    def test_no_config_anywhere_yields_usable_defaults(self, tmp_path: Path, monkeypatch):
        monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(config_mod, "SYSTEM_CONFIG_PATH", tmp_path / "etc.toml")
        cfg = config_mod.load_config()
        assert cfg.selection.n_model_new == 5
        assert cfg.source_path is None

    def test_the_shipped_example_config_loads(self):
        # If config.example.toml drifts out of sync with the parser, the
        # first thing anyone does with this project fails.
        example = Path(__file__).resolve().parents[1] / "config.example.toml"
        assert example.exists()
        cfg = config_mod.load_config(example)
        assert cfg.paths.root.is_absolute()


class TestValidation:
    def test_accepts_the_defaults(self, tmp_path: Path):
        config_mod.validate(
            Config.for_root(
                tmp_path,
                discovery=DiscoveryConfig(
                    strategy="both", channel_ids=("UCaaa",), keywords=("x",)
                ),
            )
        )

    def test_rejects_zero_exploration(self, tmp_path: Path):
        # The random arm is the control group the model-vs-chance check is
        # measured against. With none of them that check is not noisy, it is
        # uncomputable -- so this is an error, not a warning.
        cfg = Config.for_root(
            tmp_path, selection=SelectionConfig(n_model_new=7, n_random=0)
        )
        with pytest.raises(ConfigError, match="random|exploration|control"):
            config_mod.validate(cfg)

    def test_rejects_negative_exploration(self, tmp_path: Path):
        cfg = Config.for_root(tmp_path, selection=SelectionConfig(n_random=-1))
        with pytest.raises(ConfigError):
            config_mod.validate(cfg)

    def test_rejects_zero_new_model_picks(self, tmp_path: Path):
        cfg = Config.for_root(tmp_path, selection=SelectionConfig(n_model_new=0))
        with pytest.raises(ConfigError):
            config_mod.validate(cfg)

    def test_allows_zero_repeat_picks(self, tmp_path: Path):
        # Unlike exploration, the repeat arm is genuinely optional: nothing
        # downstream depends on it existing.
        config_mod.validate(
            Config.for_root(
                tmp_path,
                selection=SelectionConfig(n_model_repeat=0),
                discovery=DiscoveryConfig(strategy="rss", channel_ids=("UCaaa",)),
            )
        )

    def test_rejects_a_negative_cooldown(self, tmp_path: Path):
        cfg = Config.for_root(
            tmp_path, selection=SelectionConfig(repeat_cooldown_days=-1)
        )
        with pytest.raises(ConfigError):
            config_mod.validate(cfg)

    @pytest.mark.parametrize(
        "threshold,floor",
        [(0.6, 0.8), (0.0, 0.0), (1.5, 0.1), (0.6, -0.1), (0.6, 0.6)],
    )
    def test_rejects_incoherent_feedback_bands(self, tmp_path: Path, threshold, floor):
        cfg = Config.for_root(
            tmp_path,
            feedback=FeedbackConfig(watch_threshold=threshold, partial_floor=floor),
        )
        with pytest.raises(ConfigError):
            config_mod.validate(cfg)

    def test_rejects_a_nonpositive_retention_window(self, tmp_path: Path):
        cfg = Config.for_root(tmp_path, retention=RetentionConfig(max_age_days=0))
        with pytest.raises(ConfigError):
            config_mod.validate(cfg)

    @pytest.mark.parametrize(
        "selector",
        [
            "bv*+ba/b",
            "bestvideo+bestaudio",
            "bv*[vcodec^=vp9]+ba",
            "bv*[vcodec^=avc1][height<=1440]+ba[acodec^=mp4a]/b[ext=mp4]",
            "bv*[vcodec^=avc1]+ba[acodec^=mp4a]/b[ext=mp4]",
            "bv*[vcodec^=avc1][height<=1080]+ba[acodec^=opus]",
        ],
    )
    def test_rejects_a_selector_that_risks_transcoding(self, tmp_path: Path, selector):
        cfg = Config.for_root(tmp_path, download=DownloadConfig(format_selector=selector))
        with pytest.raises(ConfigError):
            config_mod.validate(cfg)

    @pytest.mark.parametrize(
        "selector",
        [
            "bv*[vcodec^=avc1][height<=1080]+ba[acodec^=mp4a]/b[ext=mp4]",
            "bv*[vcodec^=h264][height<=1080]+ba[acodec^=mp4a]/b[ext=mp4]",
            "bv*[vcodec^=avc1][height<=720]+ba[acodec^=mp4a]/b[ext=mp4]",
        ],
    )
    def test_accepts_a_safe_selector(self, tmp_path: Path, selector):
        config_mod.validate(
            Config.for_root(
                tmp_path,
                download=DownloadConfig(format_selector=selector),
                discovery=DiscoveryConfig(strategy="rss", channel_ids=("UCaaa",)),
            )
        )

    def test_rejects_a_non_mp4_merge_target(self, tmp_path: Path):
        cfg = Config.for_root(tmp_path, download=DownloadConfig(merge_output_format="mkv"))
        with pytest.raises(ConfigError):
            config_mod.validate(cfg)

    def test_rejects_inverted_duration_bounds(self, tmp_path: Path):
        cfg = Config.for_root(
            tmp_path, prefilter=PrefilterConfig(min_duration=7200, max_duration=300)
        )
        with pytest.raises(ConfigError):
            config_mod.validate(cfg)

    def test_rejects_negative_duration(self, tmp_path: Path):
        cfg = Config.for_root(tmp_path, prefilter=PrefilterConfig(min_duration=-1))
        with pytest.raises(ConfigError):
            config_mod.validate(cfg)

    @pytest.mark.parametrize("strategy", ["", "rss-only", "RSS", "feeds"])
    def test_rejects_an_unknown_strategy(self, tmp_path: Path, strategy):
        cfg = Config.for_root(tmp_path, discovery=DiscoveryConfig(strategy=strategy))
        with pytest.raises(ConfigError):
            config_mod.validate(cfg)

    def test_rejects_an_unknown_model_kind(self, tmp_path: Path):
        cfg = Config.for_root(tmp_path, model=ModelConfig(kind="transformer"))
        with pytest.raises(ConfigError):
            config_mod.validate(cfg)

    def test_rejects_an_unknown_embedder(self, tmp_path: Path):
        cfg = Config.for_root(tmp_path, model=ModelConfig(embedder="word2vec"))
        with pytest.raises(ConfigError):
            config_mod.validate(cfg)

    def test_rss_strategy_needs_at_least_one_channel(self, tmp_path: Path):
        cfg = Config.for_root(
            tmp_path, discovery=DiscoveryConfig(strategy="rss", channel_ids=())
        )
        with pytest.raises(ConfigError, match="channel"):
            config_mod.validate(cfg)

    def test_search_strategy_needs_at_least_one_keyword(self, tmp_path: Path):
        cfg = Config.for_root(
            tmp_path, discovery=DiscoveryConfig(strategy="search", keywords=())
        )
        with pytest.raises(ConfigError, match="keyword"):
            config_mod.validate(cfg)

    def test_load_config_validates(self, tmp_path: Path):
        path = tmp_path / "config.toml"
        path.write_text(
            f'[paths]\nroot = "{tmp_path}"\n\n[selection]\nn_random = 0\n',
            encoding="utf-8",
        )
        with pytest.raises(ConfigError):
            config_mod.load_config(path)


class TestOverrides:
    def test_with_replaces_one_section(self, tmp_path: Path):
        cfg = Config.for_root(tmp_path)
        tweaked = cfg.with_(selection=SelectionConfig(n_model_new=1, n_random=1))
        assert tweaked.selection.n_model_new == 1
        assert cfg.selection.n_model_new == 5
        assert tweaked.paths == cfg.paths
