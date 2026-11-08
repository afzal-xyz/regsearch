"""Subprocess probe: exercise the whole dashboard and prove it stayed model-free.

Run by ``tests/test_api_boundary.py`` in a fresh interpreter, because the
assertion is about ``sys.modules`` and the pytest process has no guarantee of
being clean — any other test may legitimately have imported torch.

Exits 0 with a one-line summary on stdout, or non-zero naming the module that
crept in and, where possible, the route that pulled it.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

# Modules that mean a model, or the machinery to run one, has entered the web
# process. Any of these is a latency and memory regression measured in
# seconds and hundreds of megabytes.
FORBIDDEN = (
    "torch",
    "transformers",
    "sentence_transformers",
    "onnxruntime",
    "accelerate",
    "safetensors",
)


def loaded() -> list[str]:
    return sorted(m for m in FORBIDDEN if m in sys.modules)


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> int:
    start = time.monotonic()
    from api.main import Settings, create_app  # noqa: PLC0415

    import_seconds = time.monotonic() - start
    if loaded():
        fail(f"importing api.main pulled in {loaded()}")

    from api.jobs import JobRunner
    from api.sandbox import Sandbox
    from eval import goldset
    from index import store
    from retrieval import prescore, tune
    from retrieval.params import FusionParams
    from tests.test_index import FIXTURE_CHUNKS, FakeEmbedder

    if loaded():
        fail(f"importing the api/retrieval modules pulled in {loaded()}")

    tmp = Path(tempfile.mkdtemp())
    db = tmp / "regsearch.db"
    store.build_index(FIXTURE_CHUNKS, db_path=db, embedder=FakeEmbedder(), show_progress=False)

    eval_dir = tmp / "eval"
    eval_dir.mkdir()
    (eval_dir / "gold_title4_draft.jsonl").write_text(
        json.dumps({
            "question": "Who must file paperwork each year?",
            "relevant_citations": ["4 CFR § 7.3"], "kind": "prose", "status": "approved",
        }) + "\n",
        encoding="utf-8",
    )
    goldset.write(
        [{"question": "seed", "relevant_citations": ["4 CFR § 8.1"],
          "kind": "prose", "status": "approved"}],
        eval_dir / "gold.jsonl",
    )

    # A prescore cache built by hand: the real warm path needs a cross-encoder,
    # which is exactly what must not be in this process.
    base = FusionParams()
    questions = []
    for text, section in (("q one", "4 CFR § 7.3"), ("q two", "4 CFR § 8.1")):
        from retrieval import search

        hit = search.SearchResult(
            chunk_id=f"c-{section}", citation=section, section_citation=section,
            score=1.0, retrievers=["vector"], heading="h", heading_path=[], text="",
        )
        cands = search.Candidates(query=text, vector_hits=[hit], bm25_chunks=[hit],
                                  known=frozenset(), citing=[])
        q = tune.Question(text, {section}, "prose", cands)
        q.chunks[hit.chunk_id] = hit
        q.ce = {hit.chunk_id: 0.5}
        questions.append(q)
    cache_path = tmp / "prescore.json.gz"
    prescore.dump(
        prescore.PrescoreCache(
            questions=questions,
            provenance=prescore.Provenance(corpus=(), gold_sha256="x", gold_n=2,
                                           analyzer_version=1, tuned_at=""),
            base_params=base,
            fusion_keys=frozenset({tune._fusion_key(base)}),
            reranked=True, warmed_at="now", n_pairs=2,
        ),
        cache_path,
    )
    if loaded():
        fail(f"building the fixtures pulled in {loaded()}")

    from fastapi.testclient import TestClient

    app = create_app(
        Settings(root=Path.cwd(), results_dir=tmp / "results", eval_dir=eval_dir,
                 gold_path=eval_dir / "gold.jsonl", db_path=db, prescore_path=cache_path),
        runner=JobRunner(specs={}),
        sandbox=Sandbox(cache_path),
    )

    sliders = {
        "mode": "rerank", "rrf_k": "10", "bm25_weight": "1.0",
        "bm25_contrast_anchor": "20", "bm25_contrast_band": "1.4,2.2",
        "agreement_scale": "0.0", "rerank_candidates": "10",
    }
    with TestClient(app) as client:
        draft_id = client.get("/api/authoring/4").json()[0]["draft_id"]
        calls = [
            ("GET", "/", None),
            ("GET", "/healthz", None),
            ("GET", "/api/health", None),
            ("GET", "/jobs", None),
            ("GET", "/authoring", None),
            ("GET", "/authoring/4", None),
            ("GET", "/api/authoring/4", None),
            ("GET", f"/authoring/4/reveal/{draft_id}", None),
            ("POST", "/authoring/4/meter",
             {"question": "who files paperwork", "citation": "4 CFR § 7.3", "kind": "prose"}),
            ("POST", f"/authoring/4/entry/{draft_id}",
             {"question": "Who files paperwork yearly?", "notes": "", "kind": "prose",
              "status": "approved", "citations": "4 CFR § 7.3"}),
            ("POST", "/authoring/4/promote", {}),
            ("GET", "/sandbox", None),
            ("POST", "/sandbox/evaluate", sliders),
            ("POST", "/sandbox/holdout", sliders),
            ("POST", "/api/sandbox/evaluate", sliders),
        ]
        for method, path, data in calls:
            response = (
                client.get(path) if method == "GET"
                else client.post(path, data=data, follow_redirects=False)
            )
            if response.status_code >= 500:
                fail(f"{method} {path} returned {response.status_code}: {response.text[:300]}")
            if loaded():
                fail(f"{method} {path} pulled in {loaded()}")

    print(
        f"OK: {len(calls)} routes exercised, api.main imported in {import_seconds:.2f}s, "
        f"none of {list(FORBIDDEN)} loaded"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
