"""Tests for the parameter sandbox: the prescore cache, scoring, and governance.

Three properties carry the design, and each gets its own class.

**It is fast because it cannot compute.** The cache holds candidates and
cross-encoder scores; the scoring path is handed a reranker that raises. If a
refactor ever lets a model call back into a web request, these tests fail
rather than the dashboard hanging for eight seconds.

**It agrees with the tuner.** The sandbox calls ``tune.score`` on questions
the tuner's own ``_collect`` produced, so a number here is a number there. The
round-trip test pins that: scoring from a serialised cache must equal scoring
from the live objects.

**It cannot adopt.** No route writes ``params.json``, and the honesty
machinery — in-sample labelling, the 1/n margin, the probe counter, the
spent-holdout counter — is behaviour, not decoration, so it is tested.
"""

from __future__ import annotations

import random

import pytest

from api import sandbox as sandbox_mod
from api.sandbox import HYBRID, RERANK, NotWarmed, Sandbox
from retrieval import prescore, search
from retrieval import tune as tune_mod
from retrieval.params import FusionParams
from retrieval.prescore import PrescoreCache

BASE = FusionParams()


# --- fixtures: a tiny synthetic corpus, no models ---------------------------


def _hit(chunk_id: str, section: str, score: float, retrievers: list[str]) -> search.SearchResult:
    return search.SearchResult(
        chunk_id=chunk_id, citation=section, section_citation=section, score=score,
        retrievers=retrievers, heading="h", heading_path=["t", "h"], text=f"body of {section}",
    )


N_SECTIONS = 12


def _question(n: int, *, kind: str = "prose", answer_rank: int = 1) -> tune_mod.Question:
    """A question the two retrievers disagree about.

    The orders are independent seeded shuffles rather than one being the
    other's reverse. That matters: with mirrored orders every section's RRF
    score is symmetric in its rank, the ranking comes out identical for every
    ``rrf_k``, and a sandbox that ignored its parameters entirely would pass.
    BM25's score decay varies per question so the contrast band bites too.
    """
    rng = random.Random(n)
    sections = [f"4 CFR § {n}.{i}" for i in range(1, N_SECTIONS + 1)]
    chunk_of = {s: f"c{n}-{i}" for i, s in enumerate(sections)}

    vector_order = sections[:]
    rng.shuffle(vector_order)
    bm25_order = sections[:]
    rng.shuffle(bm25_order)

    vector = [
        _hit(chunk_of[s], s, 1.0 - 0.05 * i, ["vector"]) for i, s in enumerate(vector_order)
    ]
    decay = 0.80 + 0.03 * (n % 6)  # spreads BM25 contrast across the band
    bm25 = [
        _hit(chunk_of[s], s, 9.0 * decay**i, ["bm25"]) for i, s in enumerate(bm25_order)
    ]
    cands = search.Candidates(
        query=f"question {n}", vector_hits=vector, bm25_chunks=bm25,
        known=frozenset(), citing=[],
    )
    q = tune_mod.Question(
        f"question {n}", {vector_order[answer_rank - 1]}, kind, cands
    )
    for r in vector + bm25:
        q.chunks.setdefault(r.chunk_id, r)
    return q


class FakeReranker:
    """Deterministic cross-encoder stand-in: scores by passage, no model."""

    model_name = "fake-reranker-v1"

    def __init__(self):
        self.calls = 0

    def score(self, query, passages):
        self.calls += 1
        return [float(abs(hash(p)) % 1000) / 1000 for p in passages]


def _questions() -> list[tune_mod.Question]:
    return [
        _question(i, kind="prose" if i % 3 else "citation", answer_rank=(i % 7) + 1)
        for i in range(1, 13)
    ]


@pytest.fixture(scope="module")
def warm_cache(tmp_path_factory):
    """A cache warmed over the real grid with the fake reranker.

    Module-scoped: warming cross-encoder-scores the union over ~2400 fusion
    points, which is seconds even with a fake model. Tests that mutate the
    file take a copy.
    """
    questions = _questions()
    candidates = tune_mod.grid(BASE, rerank=True)
    reranker = FakeReranker()
    tune_mod._prescore(questions, candidates, conn=None, reranker=reranker)
    cache = PrescoreCache(
        questions=questions,
        provenance=prescore.Provenance(
            corpus=("4@2024-07-18",), gold_sha256="abcd", gold_n=len(questions),
            analyzer_version=prescore.ANALYZER_VERSION, tuned_at="2026-11-07 00:00:00",
            rerank_model=reranker.model_name,
        ),
        base_params=BASE,
        fusion_keys=frozenset(tune_mod._fusion_key(p) for p in candidates),
        reranked=True,
        warmed_at="2026-11-07 00:00:00",
        n_pairs=sum(len(q.ce) for q in questions),
    )
    path = tmp_path_factory.mktemp("sandbox") / "prescore.json.gz"
    prescore.dump(cache, path)
    return path


@pytest.fixture
def questions(warm_cache):
    """The cache's questions, reloaded — cross-encoder scores included."""
    return prescore.load(warm_cache).questions


@pytest.fixture
def scratch_cache(warm_cache, tmp_path):
    """A private copy, for tests that overwrite the file."""
    copy = tmp_path / "prescore.json.gz"
    copy.write_bytes(warm_cache.read_bytes())
    return copy


@pytest.fixture
def sandbox(warm_cache):
    return Sandbox(warm_cache)


def _live(path, *, rerank=True):
    """A one-point cache built from live Question objects. Cheap: no grid sweep."""
    qs = _questions()
    if rerank:
        tune_mod._prescore(qs, [BASE], conn=None, reranker=FakeReranker())
    cache = PrescoreCache(
        questions=qs,
        provenance=prescore.Provenance(corpus=(), gold_sha256="x", gold_n=len(qs),
                                       analyzer_version=1, tuned_at=""),
        base_params=BASE, fusion_keys=frozenset({tune_mod._fusion_key(BASE)}),
        reranked=rerank, warmed_at="", n_pairs=sum(len(q.ce) for q in qs),
    )
    prescore.dump(cache, path)
    return qs


# --- the cache --------------------------------------------------------------


class TestPrescoreCache:
    def test_round_trips_everything_scoring_needs(self, tmp_path):
        live = _live(tmp_path / "c.json.gz")
        loaded = prescore.load(tmp_path / "c.json.gz")
        assert loaded.n_questions == len(live)
        assert loaded.reranked and loaded.n_pairs > 0
        before, after = live[0], loaded.questions[0]
        assert after.text == before.text
        assert after.relevant == before.relevant
        assert after.ce == before.ce
        assert [r.section_citation for r in after.cands.vector_hits] == [
            r.section_citation for r in before.cands.vector_hits
        ]

    def test_scores_identically_before_and_after_serialisation(self, tmp_path):
        """The whole premise: a number from the cache is the tuner's number."""
        in_memory = _live(tmp_path / "c.json.gz")
        loaded = prescore.load(tmp_path / "c.json.gz")
        kw = {"rerank": True, "conn": None, "reranker": sandbox_mod._REFUSE}
        live = tune_mod.score(in_memory, BASE, **kw)
        from_disk = tune_mod.score(loaded.questions, BASE, **kw)
        assert from_disk.objective == pytest.approx(live.objective)
        assert from_disk.by_kind == live.by_kind

    def test_chunk_text_is_not_written_to_disk(self, warm_cache):
        """A reader of this cache can rank. It must not be able to rerank, and
        shipping the passages would be most of the file for no purpose."""
        import gzip

        raw = gzip.open(warm_cache, "rt", encoding="utf-8").read()
        assert "body of 4 CFR" not in raw
        assert prescore.load(warm_cache).questions[0].cands.vector_hits[0].text == ""

    def test_a_missing_cache_says_so(self, tmp_path):
        with pytest.raises(prescore.CacheError, match="warm it first"):
            prescore.load(tmp_path / "nope.json.gz")

    def test_a_foreign_version_is_refused_not_guessed_at(self, warm_cache, monkeypatch):  # noqa: E501
        monkeypatch.setattr(prescore, "CACHE_VERSION", prescore.CACHE_VERSION + 1)
        with pytest.raises(prescore.CacheError, match="re-warm"):
            prescore.load(warm_cache)

    def test_covered_fusion_keys_are_recorded(self, warm_cache):
        cache = prescore.load(warm_cache)
        assert cache.covers(BASE)
        assert not cache.covers(BASE.with_(rrf_k=7))  # off-grid

    def test_build_uses_the_tuners_own_collect(self, monkeypatch, questions):
        """Not a reimplementation: if `_collect` changes, the cache changes with it."""
        seen = {}

        def fake_collect(gold, *, conn, embedder):
            seen["called"] = True
            return questions

        monkeypatch.setattr(tune_mod, "_collect", fake_collect)
        monkeypatch.setattr(tune_mod, "_prescore", lambda *a, **k: None)
        cache = prescore.build(
            [{"question": "q", "relevant_citations": ["4 CFR § 1.1"]}],
            [], conn=None, embedder=None, reranker=FakeReranker(), base_params=BASE,
        )
        assert seen["called"] and cache.n_questions == len(questions)


# --- latency: the sandbox must not be able to compute -----------------------


class TestCannotCompute:
    def test_scoring_never_calls_the_reranker(self, sandbox, monkeypatch):
        """The guard that matters. A model call here is an 8-second web request."""
        calls = []
        monkeypatch.setattr(
            sandbox_mod._REFUSE, "score",
            lambda q, p: calls.append(1) or [0.0] * len(p),
        )
        sandbox.evaluate(BASE.with_(rrf_k=5), mode=RERANK)
        assert calls == []

    def test_an_off_grid_point_is_refused_rather_than_computed(self, sandbox):
        with pytest.raises(NotWarmed, match="outside the warmed region"):
            sandbox.evaluate(BASE.with_(rrf_k=7), mode=RERANK)

    def test_hybrid_mode_needs_no_cross_encoder_at_all(self, sandbox):
        """Fusion is arithmetic over candidates, so off-grid is fine without rerank."""
        result = sandbox.evaluate(BASE.with_(rrf_k=7), mode=HYBRID)
        assert result.metrics.n > 0

    def test_a_fusion_only_cache_refuses_rerank_mode(self, questions, tmp_path):
        cache = PrescoreCache(
            questions=questions,
            provenance=prescore.Provenance(corpus=(), gold_sha256="x", gold_n=0,
                                           analyzer_version=1, tuned_at=""),
            base_params=BASE,
            fusion_keys=frozenset({tune_mod._fusion_key(BASE)}),
            reranked=False, warmed_at="", n_pairs=0,
        )
        path = prescore.dump(cache, tmp_path / "fusion-only.json.gz")
        with pytest.raises(NotWarmed, match="no cross-encoder scores"):
            Sandbox(path).evaluate(BASE, mode=RERANK)
        assert Sandbox(path).evaluate(BASE, mode=HYBRID).metrics.n > 0

    def test_a_whole_sweep_is_fast(self, sandbox):
        """Twenty parameter sets, end to end, with no model in the process."""
        import time

        start = time.monotonic()
        for k in tune_mod.GRID["rrf_k"]:
            for weight in tune_mod.GRID["bm25_weight"]:
                sandbox.evaluate(
                    BASE.with_(rrf_k=k, bm25_weight=weight), mode=RERANK, with_movements=False
                )
        elapsed = time.monotonic() - start
        assert elapsed < 10.0, f"30 parameter sets took {elapsed:.1f}s"

    def test_the_cache_is_reloaded_when_the_file_changes(self, scratch_cache):
        """The warm job runs in another process and replaces the file underneath."""
        box = Sandbox(scratch_cache)
        first = box.cache()
        assert first.n_questions > 5
        shrunk = PrescoreCache(
            questions=first.questions[:5], provenance=first.provenance, base_params=BASE,
            fusion_keys=first.fusion_keys, reranked=True, warmed_at="later", n_pairs=1,
        )
        prescore.dump(shrunk, scratch_cache)
        assert box.cache().n_questions == 5


# --- the slider axes --------------------------------------------------------


class TestAxes:
    def test_axes_are_derived_from_the_tuners_grid(self):
        """A slider that reaches somewhere the tuner never looks is a trap."""
        for axis in sandbox_mod.AXES:
            key = "bm25_contrast_band" if axis.name == "bm25_contrast_band" else axis.name
            assert tuple(axis.values) == tuple(tune_mod.GRID[key])

    def test_every_reachable_point_is_warmed(self, sandbox):
        """The strong version: no slider combination can escape the cache."""
        cache = sandbox.cache()
        for axis in sandbox_mod.AXES:
            for key in axis.keys:
                point = sandbox_mod.params_from({axis.name: key}, BASE)
                assert cache.covers(point), f"{axis.name}={key} is not warmed"

    def test_a_band_is_parsed_into_lo_and_hi(self):
        point = sandbox_mod.params_from({"bm25_contrast_band": "1.6,3.0"}, BASE)
        assert (point.bm25_contrast_lo, point.bm25_contrast_hi) == (1.6, 3.0)

    def test_an_off_axis_value_is_refused_not_snapped(self):
        """Snapping would show numbers for settings the user did not choose."""
        with pytest.raises(NotWarmed, match="not a position"):
            sandbox_mod.params_from({"rrf_k": "7"}, BASE)

    def test_an_unknown_parameter_name_is_refused(self):
        with pytest.raises(NotWarmed, match="not a sandbox parameter"):
            sandbox_mod.params_from({"learning_rate": "0.1"}, BASE)

    def test_positions_round_trip_through_their_form_value(self):
        for axis in sandbox_mod.AXES:
            for value, key in zip(axis.values, axis.keys, strict=True):
                assert sandbox_mod._coerce(axis, key) == value


# --- the numbers ------------------------------------------------------------


class TestScoring:
    def test_reports_recall_and_mrr_against_production(self, sandbox):
        result = sandbox.evaluate(BASE.with_(rrf_k=2), mode=RERANK, baseline=BASE)
        assert 0.0 <= result.metrics.recall10 <= 1.0
        assert 0.0 <= result.metrics.mrr10 <= 1.0
        assert result.baseline.objective > 0
        assert result.delta == pytest.approx(
            result.metrics.objective - result.baseline.objective
        )

    def test_overall_means_match_the_per_kind_breakdown(self, sandbox):
        """The flattening must be exact, not approximate: kinds partition the set."""
        result = sandbox.evaluate(BASE, mode=RERANK)
        rows = result.metrics.by_kind.values()
        expected = sum(r["mrr@10"] * r["n"] for r in rows) / sum(r["n"] for r in rows)
        assert result.metrics.mrr10 == pytest.approx(expected)
        assert result.metrics.n == sum(r["n"] for r in rows)

    def test_parameters_actually_change_the_outcome(self, sandbox):
        """Guards the fixture as much as the code: a sandbox where nothing moves
        would pass every other test here."""
        scores = {
            k: sandbox.evaluate(BASE.with_(rrf_k=k), mode=HYBRID).metrics.objective
            for k in tune_mod.GRID["rrf_k"]
        }
        assert len(set(scores.values())) > 1, scores

    def test_lists_the_questions_that_moved(self, sandbox):
        result = sandbox.evaluate(BASE.with_(rrf_k=60, bm25_weight=0.25), mode=HYBRID)
        assert result.moved
        assert {m.direction for m in result.moved} <= {"better", "worse", "fixed", "broken"}

    def test_bad_news_is_listed_first(self, sandbox):
        result = sandbox.evaluate(BASE.with_(rrf_k=60, bm25_weight=0.25), mode=HYBRID)
        order = ["broken", "worse", "fixed", "better"]
        ranks = [order.index(m.direction) for m in result.moved]
        assert ranks == sorted(ranks)

    def test_the_explore_split_excludes_the_holdout(self, sandbox):
        explore = {q.text for q in sandbox.questions("explore")}
        holdout = {q.text for q in sandbox.questions("holdout")}
        assert explore and holdout
        assert not (explore & holdout)
        assert len(explore | holdout) == sandbox.cache().n_questions

    def test_the_split_is_deterministic(self, warm_cache):
        first = [q.text for q in Sandbox(warm_cache).questions("holdout")]
        second = [q.text for q in Sandbox(warm_cache).questions("holdout")]
        assert first == second


# --- governance -------------------------------------------------------------


class TestGovernance:
    def test_the_noise_margin_is_one_question(self, sandbox):
        result = sandbox.evaluate(BASE, mode=RERANK)
        assert result.margin == pytest.approx(1 / result.metrics.n)

    def test_an_identical_setting_reads_as_no_evidence(self, sandbox):
        result = sandbox.evaluate(BASE, mode=RERANK, baseline=BASE)
        assert result.delta == pytest.approx(0.0)
        assert result.verdict == sandbox_mod.WITHIN_NOISE
        assert "no evidence" in result.reading

    def test_a_gain_inside_the_margin_is_not_called_better(self, sandbox):
        """One flipped question must never read as a finding."""
        result = sandbox.evaluate(BASE, mode=RERANK)
        nudged = sandbox_mod.Evaluation(
            params=BASE, baseline_params=BASE, mode=RERANK, split="explore",
            metrics=sandbox_mod.Metrics(
                result.baseline.objective + result.margin * 0.9, 0, 0, 0, result.metrics.n
            ),
            baseline=result.baseline, margin=result.margin, probes=1,
        )
        assert nudged.verdict == sandbox_mod.WITHIN_NOISE

    def test_a_real_gain_is_still_labelled_in_sample(self, sandbox):
        result = sandbox.evaluate(BASE, mode=RERANK)
        big = sandbox_mod.Evaluation(
            params=BASE, baseline_params=BASE, mode=RERANK, split="explore",
            metrics=sandbox_mod.Metrics(
                result.baseline.objective + 0.5, 0, 0, 0, result.metrics.n
            ),
            baseline=result.baseline, margin=result.margin, probes=1,
        )
        assert big.verdict == sandbox_mod.BETTER
        assert "not evidence it generalises" in big.reading
        assert "Run the tuner" in big.reading

    def test_distinct_parameter_sets_are_counted(self, sandbox):
        sandbox.evaluate(BASE.with_(rrf_k=2), mode=HYBRID)
        sandbox.evaluate(BASE.with_(rrf_k=3), mode=HYBRID)
        repeat = sandbox.evaluate(BASE.with_(rrf_k=2), mode=HYBRID)
        assert repeat.probes == 2, "re-visiting a point is not a new draw"

    def test_the_probe_count_survives_a_re_warm(self, sandbox):
        """A fresh cache does not un-bias the searching you already did."""
        sandbox.evaluate(BASE.with_(rrf_k=2), mode=HYBRID)
        sandbox.reload()
        assert sandbox.evaluate(BASE.with_(rrf_k=3), mode=HYBRID).probes == 2

    def test_looking_at_the_holdout_is_counted(self, sandbox):
        assert sandbox.holdout_peeks == 0
        sandbox.peek_holdout(BASE.with_(rrf_k=2), mode=RERANK)
        sandbox.peek_holdout(BASE.with_(rrf_k=3), mode=RERANK)
        assert sandbox.holdout_peeks == 2

    def test_the_holdout_is_scored_on_the_held_back_questions(self, sandbox):
        peek = sandbox.peek_holdout(BASE, mode=RERANK)
        assert peek.split == "holdout"
        assert peek.metrics.n == len(sandbox.questions("holdout"))

    def test_the_sandbox_never_writes_params(self, sandbox, monkeypatch):
        """The hard guarantee. Only retrieval.tune adopts."""
        from retrieval import params as params_mod

        monkeypatch.setattr(
            params_mod, "save",
            lambda *a, **k: pytest.fail("the sandbox wrote retrieval/params.json"),
        )
        sandbox.evaluate(BASE.with_(rrf_k=2), mode=RERANK)
        sandbox.peek_holdout(BASE.with_(rrf_k=2), mode=RERANK)

    def test_staleness_reuses_the_tuners_rule(self, sandbox, monkeypatch):
        from pipeline.manifest import TitleSpec

        gold = [{"question": "q", "relevant_citations": ["4 CFR § 1.1"], "kind": "prose"}]
        reasons = sandbox.staleness(
            [TitleSpec(26, "2024-07-18")], gold, "some-other-reranker"
        )
        assert any("corpus changed" in r for r in reasons)
        assert any("gold set changed" in r for r in reasons)
        assert any("reranker changed" in r for r in reasons)


# --- the HTTP surface -------------------------------------------------------


class TestSandboxRoutes:
    @pytest.fixture
    def client(self, warm_cache, tmp_path):
        from fastapi.testclient import TestClient

        from api.jobs import JobRunner
        from api.main import REPO_ROOT, Settings, create_app

        app = create_app(
            Settings(root=REPO_ROOT, prescore_path=warm_cache, results_dir=tmp_path,
                     eval_dir=tmp_path, gold_path=tmp_path / "gold.jsonl"),
            runner=JobRunner(specs={}),
            sandbox=Sandbox(warm_cache),
        )
        with TestClient(app) as test_client:
            yield test_client

    @pytest.fixture
    def cold_client(self, tmp_path):
        from fastapi.testclient import TestClient

        from api.jobs import JobRunner
        from api.main import REPO_ROOT, Settings, create_app

        app = create_app(
            Settings(root=REPO_ROOT, prescore_path=tmp_path / "absent.json.gz",
                     results_dir=tmp_path, eval_dir=tmp_path),
            runner=JobRunner(specs={}),
        )
        with TestClient(app) as test_client:
            yield test_client

    def test_the_page_renders_sliders_for_the_grid_axes(self, client):
        page = client.get("/sandbox")
        assert page.status_code == 200
        for axis in sandbox_mod.AXES:
            assert f'name="{axis.name}"' in page.text
        assert 'type="range"' in page.text

    def test_a_cold_sandbox_offers_to_warm_rather_than_failing(self, cold_client):
        page = cold_client.get("/sandbox")
        assert page.status_code == 200
        assert "No prescore cache" in page.text
        assert "/jobs/warm" in page.text

    def test_evaluating_returns_metrics_labelled_in_sample(self, client):
        response = client.post(
            "/sandbox/evaluate", data={"mode": "rerank", "rrf_k": "2"}
        )
        assert response.status_code == 200
        assert "In-sample" in response.text and "not adopted" in response.text
        assert "recall@10" in response.text and "MRR@10" in response.text

    def test_an_off_grid_value_is_refused_over_http(self, client):
        response = client.post("/sandbox/evaluate", data={"mode": "rerank", "rrf_k": "7"})
        assert "not a position" in response.text

    def test_the_json_view_marks_every_result_unadopted(self, client):
        payload = client.post(
            "/api/sandbox/evaluate", data={"mode": "hybrid", "rrf_k": "3"}
        ).json()
        assert payload["in_sample"] is True and payload["adopted"] is False
        assert payload["split"] == "explore"
        assert payload["margin"] == pytest.approx(1 / payload["metrics"]["n"])
        assert payload["verdict"] in (
            sandbox_mod.WITHIN_NOISE, sandbox_mod.BETTER, sandbox_mod.WORSE
        )

    def test_the_probe_counter_climbs_across_requests(self, client):
        first = client.post("/api/sandbox/evaluate", data={"mode": "hybrid", "rrf_k": "2"})
        second = client.post("/api/sandbox/evaluate", data={"mode": "hybrid", "rrf_k": "3"})
        assert second.json()["probes"] > first.json()["probes"]

    def test_the_page_shows_how_many_sets_have_been_tried(self, client):
        client.post("/sandbox/evaluate", data={"mode": "hybrid", "rrf_k": "2"})
        page = client.get("/sandbox")
        assert "parameter set" in page.text and "in this session" in page.text
        assert "Before you believe this" in page.text

    def test_checking_the_holdout_is_a_separate_deliberate_action(self, client):
        page = client.get("/sandbox")
        assert "Check the held-out fold" in page.text
        assert "looked at <strong>0</strong>" in page.text

        revealed = client.post("/sandbox/holdout", data={"mode": "rerank", "rrf_k": "2"})
        assert "Held-out fold" in revealed.text
        assert "1 time" in revealed.text

    def test_no_sandbox_route_can_write_params(self, client):
        writable = [
            route.path
            for route in client.app.routes
            if getattr(route, "methods", set()) & {"POST", "PUT", "PATCH", "DELETE"}
            and route.path.startswith("/sandbox")
        ]
        assert set(writable) == {"/sandbox/evaluate", "/sandbox/holdout"}

    def test_the_only_adoption_control_starts_the_real_tuner(self, client):
        page = client.get("/sandbox")
        assert 'action="/jobs/tune"' in page.text
        assert "cross-validated" in page.text

    def test_warm_is_an_allowlisted_job(self):
        from api.jobs import JobRunner

        assert JobRunner().build("warm") == ["-m", "retrieval.prescore"]
        assert JobRunner().build("warm", {"no_rerank": "1"})[-1] == "--no-rerank"


class TestInertChanges:
    """A changed ranking with unchanged metrics is the commonest confusion."""

    def test_fusion_output_changes_even_when_the_metrics_do_not(self, sandbox):
        """Proves the sliders are wired to fuse, independently of the metrics."""
        q = sandbox.cache().questions[0]
        light = search.fuse(q.cands, 10, params=BASE.with_(bm25_weight=0.25))
        heavy = search.fuse(q.cands, 10, params=BASE.with_(bm25_weight=1.25))
        assert [s for s, *_ in light] != [s for s, *_ in heavy]

    def test_identical_metrics_with_changed_params_are_flagged_as_inert(self, sandbox):
        result = sandbox.evaluate(BASE, mode=HYBRID, baseline=BASE.with_(rrf_k=2))
        if not result.moved:
            assert result.inert
        else:  # the fixture is lively enough that something moved — also fine
            assert not result.inert

    def test_an_unchanged_setting_is_not_called_inert(self, sandbox):
        assert not sandbox.evaluate(BASE, mode=HYBRID, baseline=BASE).inert
