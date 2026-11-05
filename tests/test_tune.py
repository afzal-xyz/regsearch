"""Tests for the parameter store (retrieval.params), the agreement gate, and the
cross-validated tuner (retrieval.tune) — including the one property that
matters most: it must refuse an in-sample win that does not generalise."""

from __future__ import annotations

import json

import pytest

from pipeline.manifest import TitleSpec
from retrieval import params as pm
from retrieval import search, tune
from retrieval.params import Provenance
from retrieval.search import Candidates, SearchResult

SPEC = TitleSpec(4, "2024-07-18")


# --- params store -----------------------------------------------------------


def test_defaults_match_the_hand_tuned_constants():
    d = pm.DEFAULTS
    assert (d.rrf_k, d.bm25_weight, d.bm25_contrast_lo, d.bm25_contrast_hi) == (
        search.RRF_K, search.BM25_WEIGHT, search.BM25_CONTRAST_LO, search.BM25_CONTRAST_HI
    )
    assert d.agreement_scale == 0.0, "the gate must be off until the tuner turns it on"


def test_missing_file_yields_defaults_and_no_provenance(tmp_path):
    params, prov = pm.load(tmp_path / "none.json")
    assert params == pm.DEFAULTS and prov is None


def test_round_trip(tmp_path):
    p = pm.DEFAULTS.with_(rrf_k=5, agreement_scale=0.3)
    prov = Provenance(("4@2024-07-18",), "abc", 10, 5, "now", 0.8, "note")
    path = pm.save(p, prov, tmp_path / "params.json")
    got_p, got_prov = pm.load(path)
    assert got_p == p and got_prov == prov


def test_corrupt_file_falls_back_to_defaults(tmp_path):
    path = tmp_path / "params.json"
    path.write_text("{not json", encoding="utf-8")
    assert pm.load(path) == (pm.DEFAULTS, None)


def test_unknown_keys_are_ignored_for_forward_compat(tmp_path):
    path = tmp_path / "params.json"
    path.write_text(json.dumps({"params": {"rrf_k": 7, "future_knob": 1}}), encoding="utf-8")
    assert pm.load(path)[0].rrf_k == 7


def test_current_is_cached_until_reset(tmp_path):
    path = tmp_path / "params.json"
    pm.save(pm.DEFAULTS.with_(rrf_k=5), Provenance((), "", 0, 5, ""), path)
    assert pm.current(path).rrf_k == 5
    pm.save(pm.DEFAULTS.with_(rrf_k=9), Provenance((), "", 0, 5, ""), path)
    assert pm.current(path).rrf_k == 5, "cached"
    pm.reset()
    assert pm.current(path).rrf_k == 9


# --- agreement gate ---------------------------------------------------------


def _hit(section):
    return SearchResult(section, section, section, 1.0, ["x"], "", [], "")


def test_agreement_scales_with_overlap():
    vec = [_hit(f"s{i}") for i in range(10)]
    assert search.agreement(vec, [_hit(f"s{i}") for i in range(3)], 10, 0.3) == pytest.approx(1.0)
    assert search.agreement(vec, [_hit("s0")], 10, 0.3) == pytest.approx(1 / 3)
    assert search.agreement(vec, [_hit("other")], 10, 0.3) == 0.0


def test_agreement_disabled_at_scale_zero():
    assert search.agreement([_hit("a")], [_hit("b")], 10, 0.0) == 1.0


def _cands(query, vec_sections, bm25_sections, known=frozenset(), citing=()):
    return Candidates(
        query,
        [_hit(s) for s in vec_sections],
        [
            SearchResult(s, s, s, 10.0 - i, ["bm25"], "", [], "")
            for i, s in enumerate(bm25_sections)
        ],
        frozenset(known),
        list(citing),
    )


def test_gate_silences_a_disagreeing_bm25_on_prose():
    """Vector has the answer first. BM25's list is noise except that its 10th
    hit is vector's 3rd — enough co-occurrence under RRF to lift that section
    over the answer when BM25 is trusted. With the gate on, one shared section
    out of ten (overlap 0.1 against scale 0.5) cuts BM25 to a fifth and the
    answer stays first."""
    c = _cands("q", ["answer", "v2", "v3"], [f"n{i}" for i in range(9)] + ["v3"])
    # Force contrast-based confidence to 1 so only the agreement gate acts.
    loud = pm.DEFAULTS.with_(bm25_contrast_lo=0.0, bm25_contrast_hi=0.0001, bm25_contrast_anchor=1)
    off = [s for s, *_ in search.fuse(c, 3, params=loud.with_(agreement_scale=0.0))]
    on = [s for s, *_ in search.fuse(c, 3, params=loud.with_(agreement_scale=0.5))]
    assert off[0] == "v3", "sanity: without the gate, co-occurring v3 outranks the answer"
    assert on[0] == "answer"


def test_gate_never_applies_to_citation_lookups():
    """A cited section is a lexical anchor; vector cannot see citations, so the
    retrievers are *expected* to disagree there."""
    c = _cands("q", ["v1", "v2"], ["4 CFR § 7.3", "n2"], known={"§7.3"})
    on = search.fuse(c, 2, params=pm.DEFAULTS.with_(agreement_scale=0.5))
    assert on[0][0] == "4 CFR § 7.3" and on[0][4] is True


# --- tuner mechanics --------------------------------------------------------


def test_grid_contains_the_incumbent_and_every_axis():
    g = tune.grid(pm.DEFAULTS, rerank=True)
    assert pm.DEFAULTS in g
    assert {p.rerank_candidates for p in g} >= set(tune.GRID["rerank_candidates"])
    assert {p.agreement_scale for p in g} >= set(tune.GRID["agreement_scale"])
    assert len(set(g)) == len(g)


def test_grid_without_rerank_does_not_vary_candidates():
    assert {p.rerank_candidates for p in tune.grid(pm.DEFAULTS, rerank=False)} == {10}


def test_folds_are_stratified_and_deterministic():
    qs = [tune.Question("q", set(), "prose" if i % 4 else "citation", None) for i in range(40)]
    a, b = tune.folds(qs, 5), tune.folds(qs, 5)
    assert a == b
    assert sorted(i for f in a for i in f) == list(range(40))
    for f in a:
        assert any(qs[i].kind == "citation" for i in f), "every fold needs citation lookups"


def test_objective_blends_recall_and_mrr():
    assert tune.objective(["a", "b"], {"a"}) == pytest.approx(1.0)
    assert tune.objective(["b", "a"], {"a"}) == pytest.approx(0.5 * 1 + 0.5 * 0.5)
    assert tune.objective(["b"], {"a"}) == 0.0


def _q(answer, vec, bm, kind="prose"):
    return tune.Question("q", {answer}, kind, _cands("q", vec, bm))


def test_tuner_keeps_current_when_nothing_is_better():
    """Vector alone is right everywhere and BM25 agrees: no change can help."""
    qs = [_q(f"a{i}", [f"a{i}", "x"], [f"a{i}", "x"]) for i in range(10)]
    out = tune.tune(qs, pm.DEFAULTS, rerank=False, n_folds=2)
    assert not out.adopted
    assert out.chosen == pm.DEFAULTS
    assert "already the best" in out.reason


def test_tuner_adopts_a_change_that_generalises():
    """Every question has the same shape — vector right, BM25 confidently
    wrong and disjoint — so 'ignore disagreeing BM25' wins on every fold."""
    qs = [_q(f"a{i}", [f"a{i}", "v2"], ["n1", "n2", "n3", "v2"]) for i in range(20)]
    loud = pm.DEFAULTS.with_(bm25_contrast_lo=0.0, bm25_contrast_hi=0.0001, bm25_contrast_anchor=1)
    out = tune.tune(qs, loud, rerank=False, n_folds=4)
    assert out.adopted, out.reason
    assert out.tuned_cv > out.current_cv + out.margin
    assert out.chosen != loud


def test_tuner_rejects_an_in_sample_win_that_does_not_generalise(monkeypatch):
    """The central guarantee. Candidate B is brilliant on the two questions of
    one fold and slightly worse than current everywhere else: averaged over
    all questions it *wins* (0.56 vs 0.50), so a naive tuner would ship it.
    Held out properly, every fold that trains on those two questions picks B
    and then loses on its test fold — CV says 0.46 vs 0.50. Must not adopt."""
    qs = [_q(f"a{i}", [f"a{i}"], [f"a{i}"]) for i in range(10)]
    lucky = set(tune.folds(qs, 5)[0])
    special = pm.DEFAULTS.with_(rrf_k=60)

    def rigged(subset, p, **kw):
        if p == pm.DEFAULTS:
            return [0.5] * len(subset)
        if p == special:
            return [1.0 if j in lucky else 0.45 for j in range(len(subset))]
        return [0.4] * len(subset)

    monkeypatch.setattr(tune, "per_question", rigged)
    out = tune.tune(qs, pm.DEFAULTS, rerank=False, n_folds=5)
    assert out.best == special
    assert out.tuned_cv < out.current_cv, "held out, the 'win' is a loss"
    assert not out.adopted
    assert "does not generalise" in out.reason


def test_tie_break_prefers_the_nearest_parameters():
    a, b = pm.DEFAULTS.with_(rrf_k=20), pm.DEFAULTS.with_(rrf_k=60)
    assert tune.distance(a, pm.DEFAULTS) < tune.distance(b, pm.DEFAULTS)


# --- staleness --------------------------------------------------------------


def _prov(**kw):
    base = dict(corpus=("4@2024-07-18",), gold_sha256="g", gold_n=1, analyzer_version=5,
                tuned_at="t")
    base.update(kw)
    return Provenance(**base)


GOLD = [{"question": "q", "relevant_citations": ["4 CFR § 7.3"], "kind": "prose"}]


def test_never_tuned_is_stale():
    assert tune.staleness(None, [SPEC], GOLD)


def test_changing_the_reranker_makes_params_stale():
    from eval.run import gold_fingerprint
    from index.tokenizer import ANALYZER_VERSION

    fresh = _prov(gold_sha256=gold_fingerprint(GOLD)["sha256"], analyzer_version=ANALYZER_VERSION,
                  rerank_model="m1")
    assert tune.staleness(fresh, [SPEC], GOLD, "m1") == []
    assert any("reranker changed" in r for r in tune.staleness(fresh, [SPEC], GOLD, "m2"))
    assert tune.staleness(fresh, [SPEC], GOLD, None) == [], "hybrid-only tuning ignores it"


def test_fresh_provenance_is_not_stale():
    from eval.run import gold_fingerprint
    from index.tokenizer import ANALYZER_VERSION

    prov = _prov(gold_sha256=gold_fingerprint(GOLD)["sha256"], analyzer_version=ANALYZER_VERSION)
    assert tune.staleness(prov, [SPEC], GOLD) == []


def test_each_kind_of_drift_is_named():
    from eval.run import gold_fingerprint
    from index.tokenizer import ANALYZER_VERSION

    fresh = _prov(gold_sha256=gold_fingerprint(GOLD)["sha256"], analyzer_version=ANALYZER_VERSION)
    new_title = [SPEC, TitleSpec(1, "2024-07-18")]
    assert any("corpus changed" in r for r in tune.staleness(fresh, new_title, GOLD))
    assert any("gold set changed" in r for r in tune.staleness(fresh, [SPEC], GOLD + GOLD))
    old = Provenance(fresh.corpus, fresh.gold_sha256, 1, ANALYZER_VERSION - 1, "t")
    assert any("analyzer changed" in r for r in tune.staleness(old, [SPEC], GOLD))


def test_integrity_warns_on_stale_params(tmp_path, monkeypatch):
    from pipeline import integrity

    monkeypatch.setattr(pm, "DEFAULT_PARAMS_PATH", tmp_path / "none.json")
    problems = integrity.check_params([SPEC])
    assert [p.check for p in problems] == ["params:stale"]
    assert problems[0].severity == integrity.WARNING


# --- --check (the CI gate) --------------------------------------------------


def _write_corpus(path, titles):
    path.write_text(
        "".join(f'[[titles]]\nnumber = {n}\ndate = "2024-07-18"\n\n' for n in titles),
        encoding="utf-8",
    )


def _write_gold(path, n):
    import json

    path.write_text(
        "\n".join(
            json.dumps({"question": f"q{i}", "relevant_citations": ["4 CFR § 7.3"]})
            for i in range(n)
        ),
        encoding="utf-8",
    )


def _fresh_params(tmp_path, titles, gold_n):
    """A params file whose provenance matches the given corpus and gold set."""
    from eval.run import gold_fingerprint, load_gold
    from index.tokenizer import ANALYZER_VERSION
    from pipeline import manifest

    corpus = tmp_path / "corpus.toml"
    gold = tmp_path / "gold.jsonl"
    params = tmp_path / "params.json"
    _write_corpus(corpus, titles)
    _write_gold(gold, gold_n)
    from rag.reranker import DEFAULT_RERANK_MODEL

    fp = gold_fingerprint(load_gold(gold))
    pm.save(
        pm.DEFAULTS,
        Provenance(
            tuple(manifest.corpus_state(manifest.load(corpus))),
            fp["sha256"], fp["n"], ANALYZER_VERSION, "t", rerank_model=DEFAULT_RERANK_MODEL,
        ),
        params,
    )
    return corpus, gold, params


def _check(corpus, gold, params):
    return tune.main(
        ["--check", "--manifest", str(corpus), "--gold", str(gold), "--params", str(params)]
    )


def test_check_passes_when_params_match_committed_files(tmp_path, capsys):
    assert _check(*_fresh_params(tmp_path, [4], 3)) == 0
    assert "fresh" in capsys.readouterr().out


def test_check_fails_when_a_title_is_added(tmp_path, capsys):
    corpus, gold, params = _fresh_params(tmp_path, [4], 3)
    _write_corpus(corpus, [4, 12])
    assert _check(corpus, gold, params) == 1
    assert "corpus changed" in capsys.readouterr().out


def test_check_fails_when_the_gold_set_changes(tmp_path, capsys):
    corpus, gold, params = _fresh_params(tmp_path, [4], 3)
    _write_gold(gold, 4)
    assert _check(corpus, gold, params) == 1
    assert "gold set changed" in capsys.readouterr().out


def test_check_fails_when_the_reranker_changed(tmp_path, capsys):
    """A stronger or weaker cross-encoder changes what rerank_candidates should be."""
    corpus, gold, params = _fresh_params(tmp_path, [4], 3)
    rc = tune.main(["--check", "--manifest", str(corpus), "--gold", str(gold),
                    "--params", str(params), "--rerank-model", "some/other-model"])
    assert rc == 1
    assert "reranker changed" in capsys.readouterr().out


def test_check_fails_when_never_tuned(tmp_path, capsys):
    corpus, gold, params = _fresh_params(tmp_path, [4], 3)
    params.unlink()
    assert _check(corpus, gold, params) == 1
    assert "never been tuned" in capsys.readouterr().out


def test_check_loads_no_models(tmp_path, monkeypatch):
    """CI has no GPU, no cache and no patience; --check must stay file-only."""
    import index.embedder as emb

    def boom(self):
        raise AssertionError("a model was loaded during --check")

    monkeypatch.setattr(emb.Embedder, "model", property(boom))
    assert _check(*_fresh_params(tmp_path, [4], 3)) == 0
