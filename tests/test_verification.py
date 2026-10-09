"""The self-verification scripts.

Two scripts in ``scripts/`` exist to check the deliverable rather than to
produce part of it:

* ``verify_readme.py`` re-derives every number quoted in ``README.md`` from the
  artifacts and the config dataclasses. If the document drifts away from the
  code, this fails.
* ``verify_live_api.py`` drives a running uvicorn over HTTP, covering the parts
  of the API that in-process ASGI tests cannot reach.

They are part of the deliverable, so they are exercised here too. The full runs
are slow and need real state, so what is asserted is that they import, expose
``main``, behave sensibly with nothing running, and that the README verifier
agrees with the current README.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import re
import sys
from pathlib import Path

import pytest

from ml.config import MODELS_DIR, PROJECT_ROOT

README = PROJECT_ROOT / "README.md"


def load_script(name: str):
    """Import a script by path - they are CLIs, not an importable package."""
    path = PROJECT_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_verify_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def readme_verifier():
    return load_script("verify_readme")


@pytest.fixture(scope="module")
def live_verifier():
    return load_script("verify_live_api")


class TestScriptsLoad:
    def test_the_readme_exists_and_is_substantial(self):
        assert README.is_file()
        assert README.stat().st_size > 10_000

    @pytest.mark.parametrize("name", ["verify_readme", "verify_live_api"])
    def test_each_script_imports_and_exposes_main(self, name):
        module = load_script(name)
        assert callable(module.main)

    def test_each_script_has_a_docstring(self):
        for name in ("verify_readme", "verify_live_api"):
            module = load_script(name)
            assert module.__doc__ and len(module.__doc__) > 50


class TestLiveVerifierWithoutAServer:
    """Pointed at a dead port it must say so, not fail obscurely."""

    def test_it_reports_no_server_and_exits_two(self, live_verifier, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["verify_live_api.py", "--base-url", "http://127.0.0.1:1"])
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            code = live_verifier.main()
        assert code == 2
        assert "No server" in captured.getvalue()
        assert "uvicorn" in captured.getvalue()

    def test_a_base_url_with_a_trailing_slash_still_works(self, live_verifier, monkeypatch):
        monkeypatch.setattr(
            sys, "argv", ["verify_live_api.py", "--base-url", "http://127.0.0.1:1/"]
        )
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            live_verifier.main()
        assert "No server at http://127.0.0.1:1" in captured.getvalue()


class TestReadmeVerifier:
    """The real check: does the README still describe this repository?"""

    def test_it_passes_against_the_current_document(self, readme_verifier, active_run_dir):
        """Skips when no model is trained, since it re-runs evaluation."""
        if not (MODELS_DIR / "expD_lr2e3_long" / "metrics.json").is_file():
            pytest.skip("No evaluated run to compare the README against.")
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            # argv is passed explicitly: under pytest, sys.argv belongs to the
            # runner, and argparse would try to interpret its own flags.
            code = readme_verifier.main(["--quiet"])
        assert code == 0, captured.getvalue()
        assert "README claims verified" in captured.getvalue()

    def test_quiet_hides_failures_but_not_the_summary(self, readme_verifier, monkeypatch):
        """A quiet run still reports the count, so CI logs stay short but truthful."""
        monkeypatch.setattr(readme_verifier, "check", lambda *a, **k: None)
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            readme_verifier.main(["--quiet"])
        assert "0/0 README claims" in captured.getvalue()

    def test_a_verbose_run_prints_each_failure(self, readme_verifier, monkeypatch):
        monkeypatch.setattr(
            readme_verifier,
            "check",
            lambda name, passed, detail="": readme_verifier.RESULTS.append((False, name, detail)),
        )
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            readme_verifier.main([])
        assert "[FAIL]" in captured.getvalue()

    def test_an_unevaluated_repository_exits_two(self, readme_verifier, tmp_path, monkeypatch):
        monkeypatch.setattr(readme_verifier, "MODELS_DIR", tmp_path)
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            code = readme_verifier.main([])
        assert code == 2
        assert "No evaluated run" in captured.getvalue()

    def test_its_helpers_are_honest(self, readme_verifier):
        """The verifier's own primitives must not be rubber stamps."""
        assert readme_verifier.readme_has("Emotion Recognition from Speech")
        assert not readme_verifier.readme_has("a string that is definitely not in the file")

    def test_it_reads_the_checked_in_readme(self, readme_verifier):
        assert readme_verifier.README.startswith("# ")

    def test_every_referenced_figure_is_generated_by_the_pipeline(self):
        images = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", README.read_text(encoding="utf-8"))
        assert images, "the README should illustrate its results"
        for image in images:
            assert image.startswith("artifacts/figures/"), image
            assert (PROJECT_ROOT / image).is_file(), image

    def test_the_readme_names_no_placeholders(self):
        text = README.read_text(encoding="utf-8")
        assert not re.search(r"TBD|TODO|FIXME|XXX", text, re.IGNORECASE)

    def test_the_readme_reports_the_held_out_number_not_the_validation_one(self):
        """The headline must be the test result. This is the claim that matters."""
        text = README.read_text(encoding="utf-8")
        assert "45.37 %" in text
        assert "48.33 %" in text
        # It must not present the validation score as the final performance.
        assert "not the 59.47 % validation figure" in text

    def test_the_readme_states_the_licence_and_its_non_commercial_term(self):
        text = README.read_text(encoding="utf-8").lower()
        assert "cc by-nc-sa 4.0" in text
        assert "non-commercial" in text
        assert "zenodo" in text

    def test_the_readme_records_the_dataset_checksum(self):
        text = README.read_text(encoding="utf-8")
        assert "bc696df654c87fed845eb13823edef8a" in text
        assert "208,468,073" in text


class TestVerificationScriptsArentDeadWeight:
    """Guard against the verifiers silently checking nothing."""

    def test_readme_verifier_reports_a_meaningful_number_of_checks(self, readme_verifier):
        with contextlib.redirect_stdout(io.StringIO()):
            readme_verifier.main(["--quiet"])
        assert len(readme_verifier.RESULTS) >= 100

    def test_a_second_run_does_not_inherit_the_first(self, readme_verifier):
        """``main`` is called more than once in this session; results must reset."""
        with contextlib.redirect_stdout(io.StringIO()):
            readme_verifier.main(["--quiet"])
            first = len(readme_verifier.RESULTS)
            readme_verifier.main(["--quiet"])
        assert len(readme_verifier.RESULTS) == first

    def test_live_verifier_knows_the_expected_error_codes(self, live_verifier):
        """The live checks are only useful if they assert on real codes."""
        source = Path(live_verifier.__file__).read_text(encoding="utf-8")
        for code in (
            "EMPTY_FILE",
            "INVALID_AUDIO",
            "SILENT_AUDIO",
            "AUDIO_TOO_SHORT",
            "UNSUPPORTED_FORMAT",
            "FILE_TOO_LARGE",
            "VALIDATION_ERROR",
        ):
            assert code in source, code
