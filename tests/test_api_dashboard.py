"""Tests for api.main: the HTTP surface of the dashboard.

Two claims are being defended here beyond "the pages render":

1. **The dashboard holds no model and scores nothing.** There is no /search
   endpoint, importing the app does not drag torch in, and the only way to
   make it compute anything is to start an allowlisted subprocess.
2. **A benchmark number never comes from the web process.** Everything shown
   was read from a result file on disk.
"""

from __future__ import annotations

import json
import sys

import pytest
from fastapi.testclient import TestClient

from api.jobs import JobRunner
from api.main import REPO_ROOT, Settings, create_app
from tests.test_api_runs import make_detail, make_run

TEST_SPECS = {
    "ok": lambda o: ["-c", "print('hello from a subprocess')"],
    "slow": lambda o: ["-c", "import time; print('up', flush=True); time.sleep(30)"],
}


@pytest.fixture
def results_dir(tmp_path):
    out = tmp_path / "results"
    out.mkdir()
    details = [
        make_detail("a plain question", ["4 CFR § 2.3"], {"hybrid": 1, "vector": 2, "bm25": 3}),
        make_detail("one that missed", ["4 CFR § 9.9"], {}),
        make_detail("a negative", [], {}),
    ]
    for stamp, recall in (("20260101-000000", 0.90), ("20260102-000000", 0.85)):
        run = make_run(stamp, recall10=recall, details=details)
        run.pop("_file")
        (out / f"{stamp}.json").write_text(json.dumps(run), encoding="utf-8")
    return out


@pytest.fixture
def client(results_dir, tmp_path):
    # Real repo root so the corpus/params health panel reads the actual
    # corpus.toml and params.json; results come from the fixture. eval_dir is
    # redirected: opening /authoring writes draft_ids back into the queue
    # files, and a test must not edit tracked ones.
    app = create_app(
        # gold_path stays real — the health panel reads it, and nothing here
        # promotes. The guard test below fails loudly if that ever changes.
        Settings(root=REPO_ROOT, results_dir=results_dir, eval_dir=tmp_path / "eval"),
        runner=JobRunner(specs=TEST_SPECS, python=sys.executable),
    )
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def empty_client(tmp_path):
    app = create_app(
        Settings(root=REPO_ROOT, results_dir=tmp_path),
        runner=JobRunner(specs=TEST_SPECS, python=sys.executable),
    )
    with TestClient(app) as test_client:
        yield test_client


# --- what the service deliberately is not -----------------------------------


class TestBoundaries:
    def test_there_is_no_search_endpoint(self, client):
        """A live retrieval preview next to question authoring teaches authors to
        write questions the current engine already answers."""
        paths = {route.path for route in client.app.routes}
        assert not any(p.startswith(("/search", "/answer", "/query")) for p in paths)

    def test_the_web_process_never_imports_torch(self):
        """Everything heavy runs in a child process; the server stays light."""
        probe = (
            "import sys; import api.main; "
            "assert 'torch' not in sys.modules, 'torch imported'; "
            "assert 'sentence_transformers' not in sys.modules"
        )
        import subprocess

        done = subprocess.run(
            [sys.executable, "-c", probe], cwd=REPO_ROOT, capture_output=True, text=True
        )
        assert done.returncode == 0, done.stderr

    def test_mutating_routes_stay_inside_their_three_areas(self, client):
        """Phase 2 added gold writes under /authoring, phase 3 scoring under /sandbox.
        Nothing else mutates, and no route writes retrieval/params.json — adoption is
        the tuner's job alone, reachable only as a /jobs subprocess."""
        writable = [
            route.path
            for route in client.app.routes
            if getattr(route, "methods", set()) & {"POST", "PUT", "PATCH", "DELETE"}
        ]
        assert all(
            p.startswith(("/jobs", "/authoring", "/sandbox", "/api/sandbox"))
            for p in writable
        ), writable
        assert not any("param" in p or p.endswith("/tune") for p in writable), writable

    def test_no_route_can_reach_the_params_writer(self, client, monkeypatch):
        """The guarantee behind the rule, enforced rather than described."""
        from retrieval import params as params_mod

        monkeypatch.setattr(
            params_mod, "save",
            lambda *a, **k: pytest.fail("an HTTP request wrote retrieval/params.json"),
        )
        for path in ("/", "/sandbox", "/authoring", "/jobs"):
            assert client.get(path).status_code == 200


# --- pages ------------------------------------------------------------------


class TestDashboard:
    def test_renders_the_latest_run_and_its_deltas(self, client):
        page = client.get("/")
        assert page.status_code == 200
        assert "20260102-000000" in page.text
        assert "recall@10" in page.text

    def test_flags_the_regression(self, client):
        assert "regression" in client.get("/").text

    def test_survives_an_empty_results_directory(self, empty_client):
        page = empty_client.get("/")
        assert page.status_code == 200
        assert "No runs" in page.text

    def test_shows_parameter_provenance_not_an_in_sample_score(self, client):
        page = client.get("/")
        assert "Cross-validated objective" in page.text
        assert "rrf_k" in page.text

    def test_health_json_reports_staleness_reasons(self, client):
        health = client.get("/api/health").json()
        assert health["error"] == ""
        assert isinstance(health["staleness"], list)
        assert health["titles"], "corpus.toml should declare titles"


class TestRunPage:
    def test_shows_the_rank_matrix(self, client):
        page = client.get("/runs/20260102-000000.json")
        assert page.status_code == 200
        assert "a plain question" in page.text
        assert "Per-question rank matrix" in page.text

    def test_marks_a_negative_as_not_applicable_rather_than_a_miss(self, client):
        page = client.get("/runs/20260102-000000.json")
        assert "cell negative" in page.text
        assert "no correct answer exists" in page.text

    def test_filters_to_problem_questions(self, client):
        page = client.get("/runs/20260102-000000.json?problems=1")
        assert "one that missed" in page.text
        assert "a plain question" not in page.text

    def test_unknown_run_is_a_404(self, client):
        assert client.get("/runs/20990101-000000.json").status_code == 404

    def test_path_traversal_is_refused(self, client):
        for name in ("..%2F..%2Fcorpus.toml", "....//corpus.toml"):
            assert client.get(f"/runs/{name}").status_code in (400, 404)

    def test_diff_lists_the_questions_that_moved(self, client):
        page = client.get("/runs/20260102-000000.json/diff?mode=hybrid")
        assert page.status_code == 200
        assert "Questions that moved" in page.text


# --- jobs over HTTP ---------------------------------------------------------


class TestJobEndpoints:
    def test_starting_a_job_redirects_to_its_page_and_streams(self, client):
        started = client.post("/jobs/ok", follow_redirects=False)
        assert started.status_code == 303
        job_id = started.headers["location"].rsplit("/", 1)[-1]

        stream = client.get(f"/jobs/{job_id}/stream")
        assert stream.status_code == 200
        assert "text/event-stream" in stream.headers["content-type"]
        assert "data: hello from a subprocess" in stream.text
        assert "event: done" in stream.text

        detail = client.get(f"/api/jobs/{job_id}").json()
        assert detail["status"] == "succeeded" and detail["returncode"] == 0

    def test_an_unknown_job_kind_is_a_404(self, client):
        assert client.post("/jobs/rm-rf").status_code == 404

    def test_a_rejected_option_is_a_422_not_a_command(self):
        app = create_app(Settings(root=REPO_ROOT), runner=JobRunner())
        with TestClient(app) as isolated:
            assert isolated.post("/jobs/eval", data={"mode": "; rm -rf /"}).status_code == 422

    def test_a_second_job_is_refused_while_one_runs(self, client):
        first = client.post("/jobs/slow", follow_redirects=False)
        job_id = first.headers["location"].rsplit("/", 1)[-1]
        try:
            assert client.post("/jobs/ok").status_code == 409
        finally:
            client.post(f"/jobs/{job_id}/cancel", follow_redirects=False)

    def test_job_list_partial_is_bare_for_htmx(self, client):
        client.post("/jobs/ok", follow_redirects=False)
        partial = client.get("/jobs", headers={"HX-Request": "true"})
        full = client.get("/jobs")
        assert "<html" not in partial.text and "<html" in full.text
        assert "<table" in partial.text  # the job table, not a whole page
        assert 'class="log"' not in partial.text  # and not the output either

    def test_healthz(self, client):
        assert client.get("/healthz").json() == {"ok": True}


def test_the_suite_does_not_edit_tracked_files():
    """A guard, not a feature test.

    Opening /authoring assigns draft_ids and writes them back, which is
    correct in production and must never happen to a committed file during a
    test run. This caught the dashboard fixture pointing at the real eval/.
    """
    import subprocess

    dirty = subprocess.run(
        ["git", "status", "--porcelain", "eval/", "retrieval/params.json", "corpus.toml"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    ).stdout.strip()
    assert not dirty, f"the test run modified tracked files:\n{dirty}"
