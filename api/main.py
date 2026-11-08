"""RegSearch dashboard: read the eval record, run the gates, never fake a number.

Phase 0/1 scope. This service is deliberately *read-only with respect to the
benchmark*. It can:

* show every run in ``eval/results/``, the deltas between two of them, and
  where each individual gold question ranked;
* start the same ``python -m ...`` commands CI runs, as supervised
  subprocesses, and stream their output.

It cannot score a question, edit the gold set or write ``retrieval/params.json``.
Those arrive in later phases, and the parameter sandbox will stay explicitly
in-sample and explicitly unable to adopt anything — only ``retrieval.tune``
writes parameters, because only it cross-validates.

There is no ``/search`` endpoint on purpose. A live retrieval preview beside a
question-authoring form teaches the author to write questions the current
engine already answers, which is how a benchmark quietly stops measuring
anything. It is also why nothing here imports torch: the web process holds no
model, and every heavy thing it does runs in a child process.

Run it with::

    uv run uvicorn api.main:app --reload
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from api import authoring as authoring_mod
from api import jobs as jobs_mod
from api import runs as runs_mod
from api import sandbox as sandbox_mod
from api.jobs import JobBusy, JobRunner, UnknownJob
from eval import goldset
from eval import run as eval_run
from index import store
from pipeline import manifest
from rag import reranker as reranker_mod
from retrieval import params as params_mod
from retrieval import prescore
from retrieval import tune as tune_mod

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent


@dataclass(frozen=True)
class Settings:
    """Where the dashboard reads from. All relative to the repo root."""

    root: Path = REPO_ROOT
    results_dir: Path = field(default=None)  # type: ignore[assignment]
    manifest_path: Path = field(default=None)  # type: ignore[assignment]
    params_path: Path = field(default=None)  # type: ignore[assignment]
    gold_path: Path = field(default=None)  # type: ignore[assignment]
    report_path: Path = field(default=None)  # type: ignore[assignment]
    eval_dir: Path = field(default=None)  # type: ignore[assignment]
    db_path: Path = field(default=None)  # type: ignore[assignment]
    prescore_path: Path = field(default=None)  # type: ignore[assignment]
    # None = eval.compare's adaptive 1.5/n threshold, which is what CI uses.
    threshold: float | None = None

    def __post_init__(self) -> None:
        defaults = {
            "results_dir": self.root / "eval" / "results",
            "manifest_path": self.root / "corpus.toml",
            "params_path": self.root / "retrieval" / "params.json",
            "gold_path": self.root / "eval" / "gold.jsonl",
            "report_path": self.root / "eval" / "report.html",
            "eval_dir": self.root / "eval",
            "db_path": self.root / store.DEFAULT_DB,
            "prescore_path": self.root / prescore.DEFAULT_CACHE,
        }
        for name, value in defaults.items():
            if getattr(self, name) is None:
                object.__setattr__(self, name, value)

    @classmethod
    def from_env(cls) -> Settings:
        """Settings from the environment, so the dashboard can be pointed at a
        scratch corpus without editing code (``REGSEARCH_DB=… uvicorn …``)."""

        def path_from(name: str) -> Path | None:
            value = os.environ.get(name)
            return Path(value) if value else None

        return cls(
            root=Path(os.environ.get("REGSEARCH_ROOT", REPO_ROOT)),
            results_dir=path_from("REGSEARCH_RESULTS_DIR"),
            eval_dir=path_from("REGSEARCH_EVAL_DIR"),
            gold_path=path_from("REGSEARCH_GOLD"),
            db_path=path_from("REGSEARCH_DB"),
            prescore_path=path_from("REGSEARCH_PRESCORE"),
        )


# --- health panel -----------------------------------------------------------


def corpus_health(settings: Settings) -> dict:
    """The declared corpus and whether the tuned parameters still describe it.

    Reuses :func:`retrieval.tune.staleness` rather than re-deriving the rule,
    so the banner says exactly what ``retrieval.tune --check`` would fail on.
    Entirely file-based: no index, no model, no query.
    """
    health: dict = {"titles": [], "params": {}, "provenance": None, "staleness": [], "error": ""}
    try:
        specs = manifest.load(settings.manifest_path)
        health["titles"] = [
            {"number": s.number, "date": s.date, "name": s.name, "label": s.label}
            for s in specs
        ]
        params, prov = params_mod.load(settings.params_path)
        health["params"] = params.to_dict()
        health["provenance"] = prov.to_dict() if prov else None
        gold = eval_run.load_gold(settings.gold_path)
        health["gold_n"] = len(gold)
        health["staleness"] = tune_mod.staleness(
            prov, specs, gold, reranker_mod.DEFAULT_RERANK_MODEL
        )
    except (OSError, ValueError, KeyError, manifest.ManifestError) as exc:
        health["error"] = f"{type(exc).__name__}: {exc}"
        logger.warning("could not read corpus health: %s", exc)
    return health


# --- app --------------------------------------------------------------------


def create_app(
    settings: Settings | None = None,
    runner: JobRunner | None = None,
    sandbox: sandbox_mod.Sandbox | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    runner = runner or JobRunner(cwd=settings.root)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        # A job outliving the server would keep writing result files nobody is
        # watching, and hold the GPU/CPU for the next one.
        await runner.shutdown()

    app = FastAPI(title="RegSearch dashboard", lifespan=lifespan)
    app.state.settings = settings
    app.state.runner = runner
    app.state.sandbox = sandbox or sandbox_mod.Sandbox(settings.prescore_path)

    templates = Jinja2Templates(directory=str(HERE / "templates"))
    templates.env.globals.update(
        JOB_LABELS=jobs_mod.LABELS,
        ALL=runs_mod.ALL,
        METRICS=runs_mod.METRICS,
        fmt=_fmt,
        fmt_delta=_fmt_delta,
    )
    app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")

    def page(request: Request, name: str, status_code: int = 200, **ctx) -> HTMLResponse:
        return templates.TemplateResponse(request, name, ctx, status_code=status_code)

    # -- dashboard --

    @app.get("/", response_class=HTMLResponse)
    def dashboard(request: Request, against: str = "previous"):
        all_runs = runs_mod.load_runs(settings.results_dir)
        comparison = None
        if all_runs:
            try:
                reference = runs_mod.compare.pick_reference(all_runs, against)
            except FileNotFoundError:
                reference = None
            comparison = runs_mod.compare_runs(
                all_runs[-1], reference, threshold=settings.threshold
            )
        history = [runs_mod.meta_of(r) for r in reversed(all_runs)]
        trends = _trends(all_runs)
        return page(
            request,
            "dashboard.html",
            comparison=comparison,
            current=runs_mod.meta_of(all_runs[-1]) if all_runs else None,
            reference=(
                runs_mod.meta_of(comparison.reference)
                if comparison and comparison.reference
                else None
            ),
            history=history,
            trends=trends,
            health=corpus_health(settings),
            jobs=runner.list(),
            active=runner.active(),
            report_exists=settings.report_path.exists(),
        )

    # -- one run --

    @app.get("/runs/{name}", response_class=HTMLResponse)
    def run_detail(
        request: Request,
        name: str,
        kind: str = "",
        problems: bool = False,
    ):
        run = _load(settings, name)
        modes = tuple(run.get("modes") or runs_mod.groups_of(run))
        matrix = runs_mod.status_matrix(
            run, modes=modes, kind=kind or None, only_problems=problems
        )
        return page(
            request,
            "run.html",
            run=run,
            meta=runs_mod.meta_of(run),
            modes=modes,
            groups=runs_mod.groups_of(run),
            summaries={g: runs_mod.summary_table(run, g) for g in runs_mod.groups_of(run)},
            matrix=matrix,
            kinds=sorted({e.get("kind", "prose") for e in run.get("details", [])}),
            kind=kind,
            problems=problems,
            n_total=len(run.get("details", [])),
        )

    @app.get("/runs/{name}/diff", response_class=HTMLResponse)
    def run_diff(request: Request, name: str, against: str = "previous", mode: str = ""):
        all_runs = runs_mod.load_runs(settings.results_dir)
        current = _load(settings, name)
        index = next((i for i, r in enumerate(all_runs) if r["_file"] == name), None)
        if against == "previous":
            reference = all_runs[index - 1] if index else None
        else:
            reference = _load(settings, against)
        modes = tuple(current.get("modes") or ())
        mode = mode if mode in modes else (modes[-1] if modes else "")
        comparison = runs_mod.compare_runs(current, reference, threshold=settings.threshold)
        moved = (
            runs_mod.question_deltas(current, reference, mode=mode)
            if reference and mode
            else []
        )
        return page(
            request,
            "diff.html",
            meta=runs_mod.meta_of(current),
            reference=runs_mod.meta_of(reference) if reference else None,
            comparison=comparison,
            moved=moved,
            modes=modes,
            mode=mode,
            others=[r["_file"] for r in all_runs if r["_file"] != name],
        )

    # -- jobs --

    @app.get("/jobs", response_class=HTMLResponse)
    def job_list(request: Request):
        """The job table. Bare partial for an HTMX poll, full page otherwise."""
        name = "partials/jobs.html" if request.headers.get("hx-request") else "jobs.html"
        return page(request, name, jobs=runner.list(), active=runner.active())

    @app.post("/jobs/{kind}")
    async def job_start(request: Request, kind: str):
        form = await request.form()
        opts = {k: v for k, v in form.items() if not k.startswith("_")}
        try:
            job = await runner.start(kind, opts)
        except JobBusy as exc:
            raise HTTPException(409, str(exc)) from exc
        except UnknownJob as exc:
            raise HTTPException(404, str(exc)) from exc
        except jobs_mod.InvalidOption as exc:
            raise HTTPException(422, str(exc)) from exc
        if "application/json" in request.headers.get("accept", ""):
            return JSONResponse(job.to_dict(), status_code=202)
        return RedirectResponse(f"/jobs/{job.id}", status_code=303)

    @app.get("/jobs/{job_id}", response_class=HTMLResponse)
    def job_detail(request: Request, job_id: str):
        job = runner.get(job_id)
        if job is None:
            raise HTTPException(404, f"no job {job_id!r}")
        _, lines = job.tail()
        return page(request, "job.html", job=job, lines=lines)

    @app.post("/jobs/{job_id}/cancel")
    async def job_cancel(job_id: str):
        try:
            job = await runner.cancel(job_id)
        except UnknownJob as exc:
            raise HTTPException(404, str(exc)) from exc
        return RedirectResponse(f"/jobs/{job.id}", status_code=303)

    @app.get("/jobs/{job_id}/stream")
    async def job_stream(request: Request, job_id: str, after: int = 0):
        job = runner.get(job_id)
        if job is None:
            raise HTTPException(404, f"no job {job_id!r}")
        # Resume where a dropped connection left off rather than replaying.
        last = request.headers.get("last-event-id")
        if last and last.isdigit():
            after = int(last)

        async def events() -> AsyncIterator[str]:
            async for item in runner.stream(job_id, after=after):
                if item is None:
                    yield ": keep-alive\n\n"
                    continue
                number, line = item
                yield f"id: {number + 1}\ndata: {line}\n\n"
            yield f"event: done\ndata: {job.status} rc={job.returncode}\n\n"

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # -- gold-set authoring (phase 2) --
    #
    # These routes are sync on purpose: they touch SQLite and tokenize text,
    # so Starlette runs them in a threadpool and the event loop stays free for
    # the job log stream. None of them query the retriever.

    def _specs() -> list[manifest.TitleSpec]:
        try:
            return manifest.load(settings.manifest_path)
        except manifest.ManifestError as exc:
            raise HTTPException(500, f"cannot read the corpus: {exc}") from exc

    def _spec(title: int) -> manifest.TitleSpec:
        spec = next((s for s in _specs() if s.number == title), None)
        if spec is None:
            raise HTTPException(404, f"Title {title} is not in the declared corpus")
        return spec

    def _draft(title: int) -> list[dict]:
        return authoring_mod.load_draft(title, settings.eval_dir)

    @app.get("/authoring", response_class=HTMLResponse)
    def authoring_index(request: Request):
        specs = _specs()
        gold = goldset.load(settings.gold_path)
        cov = goldset.coverage(gold, specs)
        rows = []
        for spec in specs:
            draft = _draft(spec.number)
            counts = cov.per_title.get(spec.number, {})
            rows.append(
                {
                    "spec": spec,
                    "prose": counts.get("prose", 0),
                    "citation": counts.get("citation", 0),
                    "draft_total": len(draft),
                    "draft_approved": sum(1 for e in draft if e.get("status") == "approved"),
                    "draft_written": sum(
                        1 for e in draft if str(e.get("question", "")).strip()
                    ),
                    "has_draft": authoring_mod.draft_path(
                        spec.number, settings.eval_dir
                    ).exists(),
                }
            )
        return page(
            request,
            "authoring.html",
            rows=rows,
            negatives=cov.negatives,
            min_prose=goldset.MIN_PROSE_PER_TITLE,
            min_citation=goldset.MIN_CITATION_PER_TITLE,
            min_negatives=goldset.MIN_NEGATIVES_PER_TITLE * len(specs),
            index_ready=settings.db_path.exists(),
        )

    @app.get("/authoring/{title}", response_class=HTMLResponse)
    def authoring_title(request: Request, title: int):
        spec = _spec(title)
        draft = authoring_mod.read_draft(title, settings.eval_dir)
        items = authoring_mod.queue(
            title, eval_dir=settings.eval_dir, db_path=settings.db_path
        )
        return page(
            request,
            "authoring_title.html",
            spec=spec,
            title=title,
            items=items,
            file_problems=draft.problems,
            skipped_lines=draft.skipped,
            unverified=[i for i in items if i.unverified],
            has_draft=draft.path.exists(),
            index_ready=settings.db_path.exists(),
            staleness=corpus_health(settings)["staleness"],
        )

    @app.post("/authoring/{title}/scaffold")
    def authoring_scaffold(title: int):
        spec = _spec(title)
        try:
            authoring_mod.scaffold(
                spec, eval_dir=settings.eval_dir, db_path=settings.db_path
            )
        except FileExistsError as exc:
            raise HTTPException(409, str(exc)) from exc
        except ValueError as exc:  # title not indexed
            raise HTTPException(422, str(exc)) from exc
        return RedirectResponse(f"/authoring/{title}", status_code=303)

    @app.post("/authoring/{title}/new")
    def authoring_new(title: int):
        _spec(title)
        draft = _draft(title)
        draft.append(authoring_mod.new_entry())
        authoring_mod.save_draft(title, draft, settings.eval_dir)
        return RedirectResponse(f"/authoring/{title}", status_code=303)

    @app.post("/authoring/{title}/entry/{draft_id}", response_class=HTMLResponse)
    async def authoring_save(request: Request, title: int, draft_id: str):
        _spec(title)
        form = await request.form()
        draft = _draft(title)
        entry = authoring_mod.find_draft(draft, draft_id)
        if entry is None:
            raise HTTPException(404, f"no draft entry {draft_id!r}")

        entry["question"] = str(form.get("question", "")).strip()
        entry["notes"] = str(form.get("notes", ""))
        kind = str(form.get("kind", entry.get("kind", "prose")))
        if kind not in goldset.KINDS:
            raise HTTPException(422, f"kind must be one of {goldset.KINDS}")
        entry["kind"] = kind
        status = str(form.get("status", "draft"))
        if status not in goldset.STATUSES:
            raise HTTPException(422, f"status must be one of {goldset.STATUSES}")
        if "citations" in form:
            entry["relevant_citations"] = [
                c.strip() for c in str(form["citations"]).split(",") if c.strip()
            ]
        # Approving an entry whose citation is not in the index is the mistake
        # the unchecked meter exists to prevent. Promotion already refuses it;
        # refusing it here too means the author finds out while they are
        # looking at the row, not at merge time with twenty other entries.
        if status == "approved" and _item(settings, entry).unverified:
            raise HTTPException(
                422,
                f"cannot approve: {(entry['relevant_citations'] or ['?'])[0]} is not in the "
                f"index, so the overlap meter could not check this question and promotion "
                f"would refuse it",
            )
        entry["status"] = status
        authoring_mod.save_draft(title, draft, settings.eval_dir)

        item = _item(settings, entry)
        return page(request, "partials/draft_row.html", item=item, title=title, saved=True)

    @app.post("/authoring/{title}/delete/{draft_id}")
    def authoring_delete(title: int, draft_id: str):
        _spec(title)
        draft = [e for e in _draft(title) if e.get("draft_id") != draft_id]
        authoring_mod.save_draft(title, draft, settings.eval_dir)
        return RedirectResponse(f"/authoring/{title}", status_code=303)

    @app.post("/authoring/{title}/meter", response_class=HTMLResponse)
    async def authoring_meter(request: Request, title: int):
        """Live overlap meter. Returns the gauge only — never the section text."""
        form = await request.form()
        meter = _meter(
            settings,
            question=str(form.get("question", "")),
            citation=str(form.get("citation", "")),
            kind=str(form.get("kind", "prose")),
        )
        return page(request, "partials/overlap.html", overlap=meter, inline=True)

    @app.get("/authoring/{title}/reveal/{draft_id}", response_class=HTMLResponse)
    def authoring_reveal(request: Request, title: int, draft_id: str):
        """Fetch a section's body on demand.

        The body is not in the page until this is called, so "collapsed by
        default" is a property of the payload rather than of a CSS rule. The
        reveal is recorded on the entry: a reviewer can then see which
        questions were written with the text open.
        """
        _spec(title)
        draft = _draft(title)
        entry = authoring_mod.find_draft(draft, draft_id)
        if entry is None:
            raise HTTPException(404, f"no draft entry {draft_id!r}")
        if not entry.get("revealed"):
            entry["revealed"] = True
            authoring_mod.save_draft(title, draft, settings.eval_dir)
        cites = entry.get("relevant_citations") or []
        sections = authoring_mod.load_sections(settings.db_path, cites[:1])
        section = sections.get(cites[0]) if cites else None
        return page(
            request,
            "partials/section_text.html",
            section=section or authoring_mod.MISSING_SECTION,
        )

    @app.post("/authoring/{title}/promote", response_class=HTMLResponse)
    def authoring_promote(request: Request, title: int):
        _spec(title)
        result = authoring_mod.promote(
            title,
            specs=_specs(),
            eval_dir=settings.eval_dir,
            gold_path=settings.gold_path,
            db_path=settings.db_path,
        )
        return page(
            request,
            "promotion.html",
            result=result,
            title=title,
            status_code=200 if result.ok else 422,
        )

    @app.get("/api/authoring/{title}")
    def api_authoring(title: int):
        _spec(title)
        items = authoring_mod.queue(
            title, eval_dir=settings.eval_dir, db_path=settings.db_path
        )
        return [
            {
                "draft_id": i.draft_id,
                "question": i.entry.get("question", ""),
                "status": i.entry.get("status"),
                "kind": i.entry.get("kind"),
                "relevant_citations": i.entry.get("relevant_citations"),
                "heading": i.section.heading,
                "breadcrumb": list(i.section.breadcrumb),
                "overlap": i.overlap.to_dict() if i.overlap else None,
            }
            for i in items
        ]

    # -- the parameter sandbox (phase 3) --
    #
    # Reads retrieval/params.json for the baseline and never writes it. The
    # only adoption-shaped control on the page is a button that starts
    # `retrieval.tune` as a subprocess, because only the tuner cross-validates.

    def _sandbox_context(request: Request, evaluation=None, error: str = "", **extra) -> dict:
        sandbox = app.state.sandbox
        base = params_mod.load(settings.params_path)[0]
        position = evaluation.params if evaluation else base
        health = corpus_health(settings)
        staleness: list[str] = []
        warmed_at = ""
        covered = 0
        if sandbox.loaded:
            cache = sandbox.cache()
            warmed_at = cache.warmed_at
            covered = len(cache.fusion_keys)
            try:
                gold = eval_run.load_gold(settings.gold_path)
                staleness = sandbox.staleness(
                    manifest.load(settings.manifest_path), gold,
                    reranker_mod.DEFAULT_RERANK_MODEL,
                )
            except (OSError, ValueError, manifest.ManifestError) as exc:
                staleness = [f"could not check: {exc}"]
        return {
            "axes": sandbox_mod.AXES,
            "position": position,
            "base": base,
            "axis_value_of": sandbox_mod.axis_value_of,
            "axis_key": sandbox_mod._axis_key,
            "evaluation": evaluation,
            "error": error,
            "loaded": sandbox.loaded,
            "warmed_at": warmed_at,
            "covered": covered,
            "cache_staleness": staleness,
            "params_staleness": health["staleness"],
            "probes": len(sandbox.probes),
            "holdout_peeks": sandbox.holdout_peeks,
            "active_job": runner.active(),
            **extra,
        }

    def _sandbox_eval(form, *, holdout: bool = False):
        """Parse slider positions and score them. Returns (evaluation, error)."""
        sandbox = app.state.sandbox
        mode = str(form.get("mode", sandbox_mod.RERANK))
        if mode not in (sandbox_mod.HYBRID, sandbox_mod.RERANK):
            raise HTTPException(422, "mode must be hybrid or rerank")
        base = params_mod.load(settings.params_path)[0]
        values = {a.name: form[a.name] for a in sandbox_mod.AXES if a.name in form}
        try:
            point = sandbox_mod.params_from(values, base)
            if holdout:
                return sandbox.peek_holdout(point, mode=mode), ""
            return sandbox.evaluate(point, mode=mode, baseline=base), ""
        except (sandbox_mod.NotWarmed, prescore.CacheError) as exc:
            return None, str(exc)

    @app.get("/sandbox", response_class=HTMLResponse)
    def sandbox_page(request: Request):
        sandbox = app.state.sandbox
        evaluation, error = None, ""
        try:
            base = params_mod.load(settings.params_path)[0]
            evaluation = sandbox.evaluate(base, mode=sandbox_mod.RERANK, baseline=base)
        except (sandbox_mod.NotWarmed, prescore.CacheError) as exc:
            error = str(exc)
        return page(request, "sandbox.html", **_sandbox_context(request, evaluation, error))

    @app.post("/sandbox/evaluate", response_class=HTMLResponse)
    async def sandbox_evaluate(request: Request):
        evaluation, error = _sandbox_eval(await request.form())
        return page(
            request, "partials/sandbox_result.html",
            **_sandbox_context(request, evaluation, error),
        )

    @app.post("/sandbox/holdout", response_class=HTMLResponse)
    async def sandbox_holdout(request: Request):
        evaluation, error = _sandbox_eval(await request.form(), holdout=True)
        return page(
            request, "partials/sandbox_holdout.html",
            **_sandbox_context(request, evaluation, error),
        )

    @app.post("/api/sandbox/evaluate")
    async def api_sandbox_evaluate(request: Request):
        evaluation, error = _sandbox_eval(await request.form())
        if evaluation is None:
            raise HTTPException(409, error)
        return {
            "params": evaluation.params.to_dict(),
            "mode": evaluation.mode,
            "split": evaluation.split,
            "in_sample": True,
            "adopted": False,
            "metrics": {
                "objective": evaluation.metrics.objective,
                "recall@5": evaluation.metrics.recall5,
                "recall@10": evaluation.metrics.recall10,
                "mrr@10": evaluation.metrics.mrr10,
                "n": evaluation.metrics.n,
                "by_kind": evaluation.metrics.by_kind,
            },
            "baseline_objective": evaluation.baseline.objective,
            "delta": evaluation.delta,
            "margin": evaluation.margin,
            "verdict": evaluation.verdict,
            "reading": evaluation.reading,
            "probes": evaluation.probes,
        }

    # -- small JSON surface (used by the tests and the later phases) --

    @app.get("/api/runs")
    def api_runs():
        return [m.__dict__ for m in runs_mod.list_runs(settings.results_dir)]

    @app.get("/api/runs/{name}")
    def api_run(name: str):
        return _load(settings, name)

    @app.get("/api/jobs")
    def api_jobs():
        return [j.to_dict() for j in runner.list()]

    @app.get("/api/jobs/{job_id}")
    def api_job(job_id: str):
        job = runner.get(job_id)
        if job is None:
            raise HTTPException(404, f"no job {job_id!r}")
        _, lines = job.tail()
        return {**job.to_dict(), "dropped": job.dropped, "output": lines}

    @app.get("/api/health")
    def api_health():
        return corpus_health(settings)

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    return app


def _meter(settings: Settings, *, question: str, citation: str, kind: str):
    """Score a question against one section's body, server-side.

    The body never leaves the server for this: the browser sends the question
    and a citation, and gets back a gauge. That is what keeps "Reveal Text"
    meaningful — the meter can read the section even when the author has not.
    """
    entry = {
        "question": question,
        "relevant_citations": [citation] if citation else [],
        "kind": kind if kind in goldset.KINDS else "prose",
    }
    section = _section_for(settings, entry)
    return authoring_mod.meter_for(entry, section, authoring_mod.vocabulary(settings.db_path))


def _section_for(settings: Settings, entry: dict) -> authoring_mod.Section:
    cites = entry.get("relevant_citations") or []
    if not cites:
        return authoring_mod.MISSING_SECTION
    return authoring_mod.load_sections(settings.db_path, cites[:1]).get(
        cites[0], authoring_mod.MISSING_SECTION
    )


def _item(settings: Settings, entry: dict) -> authoring_mod.DraftItem:
    """One draft entry joined to its section and a fresh meter."""
    section = _section_for(settings, entry)
    return authoring_mod.DraftItem(
        entry=entry,
        section=section,
        overlap=authoring_mod.meter_for(
            entry, section, authoring_mod.vocabulary(settings.db_path)
        ),
    )


def _load(settings: Settings, name: str) -> dict:
    try:
        return runs_mod.load_run(name, settings.results_dir)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


def _trends(all_runs: list[dict]) -> list[dict]:
    """Sparkline per mode for the headline metric, oldest run first."""
    if len(all_runs) < 2:
        return []
    out = []
    for mode in ("vector", "bm25", "hybrid", "rerank"):
        for metric in ("recall@10", "mrr@10"):
            series = runs_mod.metric_series(all_runs, runs_mod.ALL, mode, metric)
            if any(v is not None for v in series):
                out.append(
                    {
                        "mode": mode,
                        "metric": metric,
                        "svg": runs_mod.sparkline(series),
                        "last": next((v for v in reversed(series) if v is not None), None),
                    }
                )
    return out


def _fmt(value: float | None, places: int = 3) -> str:
    return "—" if value is None else f"{value:.{places}f}"


def _fmt_delta(value: float | None) -> str:
    return "—" if value is None else f"{value:+.3f}"


app = create_app()
