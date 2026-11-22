"""Static checks on the three seams the rest of the suite depends on.

These are structural, not behavioural. They exist because every other test
module fakes the network and yt-dlp by monkeypatching `ytaf.http.get` and
`ytaf.ytdlp.run` -- and a module that does `from .http import get` captures
the real function at import time, slipping past the fake and reaching the
actual internet from a test run.

Checks run against code only: docstrings and comments are stripped first, so
prose explaining a constraint is never mistaken for a violation of it.
"""

from __future__ import annotations

import ast
import importlib
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "ytaf"
MODULES = sorted(p for p in SRC.rglob("*.py"))
MODULE_IDS = [p.name for p in MODULES]
# The seam modules are allowed to define and use these names directly.
SEAM_OWNERS = {"http.py", "ytdlp.py"}


def _docstring_lines(tree: ast.AST) -> set[int]:
    """Line numbers occupied by module, class and function docstrings."""
    occupied: set[int] = set()
    doc_owners = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    for node in ast.walk(tree):
        if not isinstance(node, doc_owners) or not node.body:
            continue
        if ast.get_docstring(node, clean=False) is None:
            continue
        first = node.body[0]
        occupied.update(range(first.lineno, (first.end_lineno or first.lineno) + 1))
    return occupied


def code_lines(path: Path) -> list[tuple[int, str]]:
    """(lineno, text) for every line of actual code, docstrings and comments
    stripped. Blank results are dropped."""
    source = path.read_text(encoding="utf-8")
    skip = _docstring_lines(ast.parse(source))
    out = []
    for lineno, line in enumerate(source.splitlines(), start=1):
        if lineno in skip:
            continue
        code = line.split("#", 1)[0]
        if code.strip():
            out.append((lineno, code))
    return out


def code_contains(path: Path, needle: str) -> list[str]:
    return [f"{path.name}:{n}: {t.strip()}" for n, t in code_lines(path) if needle in t]


class TestHelperItself:
    """The checks below are only as good as the stripping, so prove it."""

    def test_strips_module_docstrings(self, tmp_path):
        path = tmp_path / "m.py"
        path.write_text('"""A docstring mentioning FORBIDDEN."""\nx = 1\n', encoding="utf-8")
        assert code_contains(path, "FORBIDDEN") == []

    def test_strips_multiline_docstrings(self, tmp_path):
        path = tmp_path / "m.py"
        path.write_text(
            '"""Line one.\n\nLine two mentions FORBIDDEN.\n"""\nx = 1\n', encoding="utf-8"
        )
        assert code_contains(path, "FORBIDDEN") == []

    def test_strips_function_docstrings(self, tmp_path):
        path = tmp_path / "m.py"
        path.write_text(
            'def f():\n    """Mentions FORBIDDEN."""\n    return 1\n', encoding="utf-8"
        )
        assert code_contains(path, "FORBIDDEN") == []

    def test_strips_comments(self, tmp_path):
        path = tmp_path / "m.py"
        path.write_text("x = 1  # FORBIDDEN\n", encoding="utf-8")
        assert code_contains(path, "FORBIDDEN") == []

    def test_still_catches_a_string_literal(self, tmp_path):
        path = tmp_path / "m.py"
        path.write_text('x = "FORBIDDEN"\n', encoding="utf-8")
        assert code_contains(path, "FORBIDDEN")

    def test_still_catches_code(self, tmp_path):
        path = tmp_path / "m.py"
        path.write_text("FORBIDDEN = 1\n", encoding="utf-8")
        assert code_contains(path, "FORBIDDEN")


@pytest.mark.parametrize("path", MODULES, ids=MODULE_IDS)
class TestSourceHygiene:
    def test_parses(self, path):
        ast.parse(path.read_text(encoding="utf-8"))

    def test_does_not_import_http_functions_by_name(self, path):
        if path.name in SEAM_OWNERS:
            pytest.skip("owns the seam")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").endswith("http"):
                names = {alias.name for alias in node.names}
                assert not names & {"get", "post", "raise_for_status"}, (
                    f"{path.name} imports {names} by name; monkeypatching "
                    "ytaf.http cannot reach it. Use `http.get(...)` instead."
                )

    def test_does_not_import_ytdlp_run_by_name(self, path):
        if path.name in SEAM_OWNERS:
            pytest.skip("owns the seam")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").endswith("ytdlp"):
                names = {alias.name for alias in node.names}
                assert "run" not in names, (
                    f"{path.name} imports ytdlp.run by name; use `ytdlp.run(...)`."
                )

    def test_does_not_reach_the_network_outside_the_http_module(self, path):
        if path.name in SEAM_OWNERS:
            pytest.skip("owns the seam")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        banned = {"requests", "urllib", "urllib.request", "http.client", "socket", "httpx"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found = {a.name for a in node.names} & banned
                assert not found, f"{path.name} imports {found} directly"
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert module not in banned, f"{path.name} imports from {module}"

    def test_does_not_spawn_processes_outside_the_ytdlp_module(self, path):
        if path.name in SEAM_OWNERS:
            pytest.skip("owns the seam")
        for needle in ("subprocess", "os.system", "os.popen", "os.exec"):
            assert code_contains(path, needle) == [], code_contains(path, needle)

    @pytest.mark.parametrize("literal", ["/mnt/raid1", "/usr/local/bin/yt-", "/etc/systemd"])
    def test_hardcodes_no_deployment_paths(self, path, literal):
        # Everything is configurable so the pipeline runs out of a scratch
        # directory on natgentoo and out of a RAID mount on the Pi. A literal
        # path in code is how that stops being true.
        assert code_contains(path, literal) == [], code_contains(path, literal)

    @pytest.mark.parametrize("host", ["natpi", "natgentoo", "natwork"])
    def test_hardcodes_no_hostnames(self, path, host):
        assert code_contains(path, host) == [], code_contains(path, host)

    def test_passes_no_cookie_or_credential_flag_to_yt_dlp(self, path):
        # The guide's anonymity constraint, enforced across the whole source.
        # spec.py is the one place allowed to name them -- it owns the
        # blocklist that everything else is checked against.
        if path.name == "spec.py":
            pytest.skip("declares the blocklist")
        for flag in ("--cookies", "--netrc", "--username", "--password"):
            assert code_contains(path, flag) == [], code_contains(path, flag)

    def test_does_not_read_a_browser_cookie_store(self, path):
        for needle in ("cookies.sqlite", "Cookies.binarycookies", "cookiejar"):
            assert code_contains(path, needle) == []


class TestSpecIsTheSingleSourceOfTruth:
    @pytest.mark.parametrize("needle", ["vcodec^=avc1", "vcodec^=h264"])
    def test_only_spec_declares_a_format_selector(self, needle):
        offenders = [
            hit
            for path in MODULES
            if path.name != "spec.py"
            for hit in code_contains(path, needle)
        ]
        assert offenders == [], offenders

    def test_only_spec_declares_the_output_template(self):
        offenders = [
            hit
            for path in MODULES
            if path.name != "spec.py"
            for hit in code_contains(path, "%(upload_date>")
        ]
        assert offenders == [], offenders

    def test_only_spec_declares_the_feed_url(self):
        offenders = [
            hit
            for path in MODULES
            if path.name != "spec.py"
            for hit in code_contains(path, "feeds/videos.xml")
        ]
        assert offenders == [], offenders

    def test_only_spec_declares_the_search_prefix(self):
        offenders = [
            hit
            for path in MODULES
            if path.name != "spec.py"
            for hit in code_contains(path, "ytsearchdate")
        ]
        assert offenders == [], offenders

    def test_nothing_uses_relevance_ordered_search(self):
        # ytsearch (no "date") returns the same relevance-ranked results every
        # day, so discovery would stop finding anything new.
        offenders = [
            hit
            for path in MODULES
            for hit in code_contains(path, '"ytsearch:')
            + code_contains(path, "'ytsearch:")
        ]
        assert offenders == [], offenders


class TestPackageImports:
    def test_imports_without_optional_dependencies(self):
        # The Pi gets numpy and nothing else; torch and sklearn are extras.
        script = (
            "import sys; sys.path.insert(0, 'src');"
            "import ytaf, ytaf.cli, ytaf.discover, ytaf.score, ytaf.download,"
            "ytaf.feedback, ytaf.retention, ytaf.monitor, ytaf.training;"
            "assert 'torch' not in sys.modules, 'torch imported eagerly';"
            "assert 'sklearn' not in sys.modules, 'sklearn imported eagerly';"
            "print('ok')"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            cwd=str(SRC.parents[1]),
        )
        assert result.returncode == 0, result.stderr
        assert "ok" in result.stdout

    @pytest.mark.parametrize("path", MODULES, ids=MODULE_IDS)
    def test_every_module_is_importable(self, path):
        if path.name == "__init__.py":
            pytest.skip("the package itself")
        importlib.import_module(f"ytaf.{path.stem}")

    def test_no_module_does_work_at_import_time(self):
        # A stage that touches the filesystem or the network on import makes
        # `ytaf --help` fail on a box where the RAID is not mounted.
        for path in MODULES:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in tree.body:
                if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                    func = node.value.func
                    name = getattr(func, "id", getattr(func, "attr", ""))
                    assert name == "", f"{path.name} calls {name}() at import time"
