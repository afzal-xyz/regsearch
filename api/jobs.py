"""Supervised subprocess execution for the pipeline CLIs.

The dashboard never evaluates, tunes or onboards in-process. It shells out to
the same ``python -m ...`` entry points CI runs, and streams their output.

Why a subprocess rather than calling ``eval.run.evaluate()`` directly:

* **One code path.** The number the dashboard shows is produced by the command
  the gate runs, not by a second in-process route that can quietly drift from
  it. A dashboard that disagrees with CI is worse than no dashboard.
* **Stale process state.** :mod:`retrieval.params` caches ``params.json`` for
  the life of the process (its ``_CACHE``), so a long-lived API server would
  go on using the parameters it read at boot after a tune rewrote them. A
  fresh interpreter per job cannot.
* **Memory and blast radius.** A cross-encoder checkpoint is hundreds of MB
  and is never freed; a crash in torch should end a job, not the dashboard.
* **Cancellation.** Killing a process is a signal. Cancelling a thread in the
  middle of a torch forward pass is a polite request it may decline.

Only the commands in :data:`SPECS` can be started, and each spec validates its
options into an explicit argv list. Request data is never interpolated into a
command line, and no spec accepts a free-form argument.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
import signal
import sys
import uuid
from collections import deque
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from eval.run import MODES as EVAL_MODES
from pipeline.onboard import STAGES as ONBOARD_STAGES

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent

# Lines kept per job. Enough for a full onboard; the head is dropped rather
# than the tail, because the verdict is always at the end.
MAX_LINES = 4000
# A single "line" longer than this is flushed as-is. Guards against a command
# that writes megabytes without a newline.
MAX_LINE_BYTES = 8192
# How long a stream subscriber waits before emitting a keep-alive.
HEARTBEAT_SECONDS = 15.0
# Grace period between SIGTERM and SIGKILL when cancelling.
KILL_AFTER_SECONDS = 5.0

RUNNING = "running"
SUCCEEDED = "succeeded"
FAILED = "failed"
CANCELLED = "cancelled"


class JobError(RuntimeError):
    """Base class for anything that stops a job from starting."""


class UnknownJob(JobError):
    """No such job kind, or no such job id."""


class JobBusy(JobError):
    """Another job is already running."""


class InvalidOption(JobError):
    """An option was missing, unrecognised, or out of range."""


# --- option validation ------------------------------------------------------
#
# Every helper below is total: it either returns a safe value or raises.
# Nothing reaches argv that has not passed through one of them.

_RESULT_FILE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.json$")


def _reject_unknown(opts: Mapping[str, object], allowed: set[str]) -> None:
    extra = sorted(set(opts) - allowed)
    if extra:
        raise InvalidOption(f"unrecognised option(s): {', '.join(extra)}")


def _choice(opts: Mapping[str, object], name: str, allowed, default: str) -> str:
    value = str(opts.get(name, default))
    if value not in allowed:
        raise InvalidOption(f"{name}={value!r} is not one of {tuple(allowed)}")
    return value


def _int(opts: Mapping[str, object], name: str, lo: int, hi: int) -> str:
    try:
        value = int(str(opts[name]))
    except (TypeError, ValueError) as exc:
        raise InvalidOption(f"{name} must be an integer") from exc
    if not lo <= value <= hi:
        raise InvalidOption(f"{name}={value} is outside {lo}..{hi}")
    return str(value)


def _flag(opts: Mapping[str, object], name: str) -> bool:
    value = opts.get(name, False)
    if isinstance(value, str):
        return value.lower() in ("1", "true", "on", "yes")
    return bool(value)


def _results_file(opts: Mapping[str, object], name: str) -> str:
    """A bare filename inside ``eval/results``; never a path."""
    value = str(opts[name])
    if not _RESULT_FILE.match(value):
        raise InvalidOption(
            f"{name}={value!r} must be a plain result filename such as 20261104-115758.json"
        )
    return value


# --- the allowlist ----------------------------------------------------------


def _spec_eval(opts: Mapping[str, object]) -> list[str]:
    _reject_unknown(opts, {"mode", "k", "candidates"})
    argv = ["-m", "eval.run", "--mode", _choice(opts, "mode", ("all", *EVAL_MODES), "all")]
    if "k" in opts:
        argv += ["--k", _int(opts, "k", 1, 100)]
    if "candidates" in opts:
        argv += ["--candidates", _int(opts, "candidates", 1, 200)]
    return argv


def _spec_compare(opts: Mapping[str, object]) -> list[str]:
    _reject_unknown(opts, {"against", "no_fail"})
    argv = ["-m", "eval.compare"]
    if "against" in opts:
        argv += ["--against", _results_file(opts, "against")]
    if _flag(opts, "no_fail"):
        argv += ["--no-fail"]
    return argv


def _spec_tune(opts: Mapping[str, object]) -> list[str]:
    """Re-tune fusion/rerank constants.

    ``--dry-run`` defaults to *on*: the honest-by-default choice is to report
    the cross-validated verdict and let the operator ask for the write.
    """
    _reject_unknown(opts, {"dry_run", "force", "no_rerank", "folds"})
    argv = ["-m", "retrieval.tune"]
    if _flag(opts, "dry_run"):
        argv += ["--dry-run"]
    if _flag(opts, "force"):
        argv += ["--force"]
    if _flag(opts, "no_rerank"):
        argv += ["--no-rerank"]
    if "folds" in opts:
        argv += ["--folds", _int(opts, "folds", 2, 20)]
    return argv


def _spec_tune_check(opts: Mapping[str, object]) -> list[str]:
    _reject_unknown(opts, set())
    return ["-m", "retrieval.tune", "--check"]


def _spec_goldset_check(opts: Mapping[str, object]) -> list[str]:
    _reject_unknown(opts, {"strict"})
    argv = ["-m", "eval.goldset", "--check"]
    if _flag(opts, "strict"):
        argv += ["--strict"]
    return argv


def _spec_integrity(opts: Mapping[str, object]) -> list[str]:
    _reject_unknown(opts, {"strict"})
    argv = ["-m", "pipeline.integrity"]
    if _flag(opts, "strict"):
        argv += ["--strict"]
    return argv


def _spec_onboard(opts: Mapping[str, object]) -> list[str]:
    """Re-run the pipeline over the whole declared corpus.

    Deliberately cannot pass ``--title``: adding a title rewrites
    ``corpus.toml`` and triggers a multi-hour fetch/chunk/index. That stays a
    considered action at a terminal, not a button on a web page.
    """
    _reject_unknown(opts, {"only", "start", "strict", "skip_tune"})
    argv = ["-m", "pipeline.onboard"]
    if "only" in opts:
        argv += ["--only", _choice(opts, "only", ONBOARD_STAGES, "integrity")]
    if "start" in opts:
        argv += ["--from", _choice(opts, "start", ONBOARD_STAGES, "integrity")]
    if _flag(opts, "strict"):
        argv += ["--strict"]
    if _flag(opts, "skip_tune"):
        argv += ["--skip-tune"]
    return argv


def _spec_warm(opts: Mapping[str, object]) -> list[str]:
    """Rebuild the sandbox's prescore cache.

    This is the only job that exists to make the UI fast rather than to gate
    anything. It writes a derived artifact under ``data/`` and touches no
    benchmark file.
    """
    _reject_unknown(opts, {"no_rerank"})
    argv = ["-m", "retrieval.prescore"]
    if _flag(opts, "no_rerank"):
        argv += ["--no-rerank"]
    return argv


SPECS: dict[str, Callable[[Mapping[str, object]], list[str]]] = {
    "eval": _spec_eval,
    "warm": _spec_warm,
    "compare": _spec_compare,
    "tune": _spec_tune,
    "tune-check": _spec_tune_check,
    "goldset-check": _spec_goldset_check,
    "integrity": _spec_integrity,
    "onboard": _spec_onboard,
}

# Shown on the dashboard buttons. Kept beside the specs so a new job kind
# cannot be exposed without a human-readable account of what it does.
LABELS: dict[str, str] = {
    "eval": "Run eval (all modes)",
    "warm": "Warm the sandbox cache",
    "compare": "Compare latest vs previous",
    "tune": "Re-tune parameters (dry run)",
    "tune-check": "Check parameter staleness",
    "goldset-check": "Gate the gold set",
    "integrity": "Gate the corpus",
    "onboard": "Re-run the pipeline",
}


# --- jobs -------------------------------------------------------------------


@dataclass
class Job:
    id: str
    kind: str
    argv: tuple[str, ...]
    started_at: str
    status: str = RUNNING
    returncode: int | None = None
    finished_at: str | None = None
    error: str = ""
    total_lines: int = 0
    lines: deque[str] = field(default_factory=lambda: deque(maxlen=MAX_LINES))
    # Replaced (not cleared) on every notification, so a subscriber that
    # snapshots it before reading state can never miss a wake-up.
    _updated: asyncio.Event = field(default_factory=asyncio.Event)
    _proc: asyncio.subprocess.Process | None = None
    _task: asyncio.Task | None = None

    @property
    def running(self) -> bool:
        return self.status == RUNNING

    @property
    def command(self) -> str:
        return " ".join(("python", *self.argv))

    @property
    def dropped(self) -> int:
        """Lines evicted from the head of the ring buffer."""
        return self.total_lines - len(self.lines)

    def append(self, line: str) -> None:
        self.lines.append(line)
        self.total_lines += 1
        self._notify()

    def tail(self, after: int = 0) -> tuple[int, list[str]]:
        """Lines numbered ``[after, total)``, plus the new cursor.

        Lines already evicted are skipped; ``dropped`` says how many.
        """
        start = max(after, self.dropped)
        if start >= self.total_lines:
            return self.total_lines, []
        return self.total_lines, list(self.lines)[start - self.dropped :]

    def _notify(self) -> None:
        event, self._updated = self._updated, asyncio.Event()
        event.set()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "command": self.command,
            "status": self.status,
            "returncode": self.returncode,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "lines": self.total_lines,
        }


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


async def _iter_lines(stream: asyncio.StreamReader) -> AsyncIterator[str]:
    r"""Yield output lines, splitting on ``\n`` *and* ``\r``.

    tqdm — which sentence-transformers and the tuner both use — redraws a
    progress bar with carriage returns and no newline. Splitting on ``\r`` too
    turns that into a series of short lines the UI can show the last of,
    instead of one unbounded line that only arrives when the job ends.
    """
    buf = bytearray()
    while True:
        chunk = await stream.read(4096)
        if not chunk:
            break
        buf += chunk
        while True:
            idx = min(
                (i for i in (buf.find(b"\n"), buf.find(b"\r")) if i >= 0),
                default=-1,
            )
            if idx < 0:
                break
            line, buf = buf[:idx], buf[idx + 1 :]
            yield bytes(line).decode("utf-8", "replace")
        if len(buf) > MAX_LINE_BYTES:
            yield bytes(buf[:MAX_LINE_BYTES]).decode("utf-8", "replace")
            del buf[:MAX_LINE_BYTES]
    if buf:
        yield bytes(buf).decode("utf-8", "replace")


class JobRunner:
    """Starts, supervises and streams one allowlisted subprocess at a time.

    Serialised on purpose. Eval, tune and onboard each saturate the CPU and
    load their own copy of the models; two at once does not finish sooner, and
    on a shared machine it makes both sets of timings meaningless.
    """

    def __init__(
        self,
        *,
        cwd: Path | str = REPO_ROOT,
        python: str | None = None,
        specs: Mapping[str, Callable[[Mapping[str, object]], list[str]]] | None = None,
        history: int = 50,
    ) -> None:
        self.cwd = Path(cwd)
        self.python = python or sys.executable
        self.specs = dict(SPECS if specs is None else specs)
        self._jobs: dict[str, Job] = {}
        self._order: deque[str] = deque(maxlen=history)

    # -- lookup --

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def require(self, job_id: str) -> Job:
        job = self.get(job_id)
        if job is None:
            raise UnknownJob(f"no job {job_id!r}")
        return job

    def list(self) -> list[Job]:
        """Newest first."""
        return [self._jobs[i] for i in reversed(self._order) if i in self._jobs]

    def active(self) -> Job | None:
        return next((j for j in self._jobs.values() if j.running), None)

    def build(self, kind: str, opts: Mapping[str, object] | None = None) -> list[str]:
        """Validate ``opts`` into an argv. Raises rather than guessing."""
        spec = self.specs.get(kind)
        if spec is None:
            raise UnknownJob(f"{kind!r} is not an allowed job (have: {sorted(self.specs)})")
        return spec(opts or {})

    # -- lifecycle --

    async def start(self, kind: str, opts: Mapping[str, object] | None = None) -> Job:
        busy = self.active()
        if busy is not None:
            raise JobBusy(f"{busy.kind} (job {busy.id}) is still running")
        argv = self.build(kind, opts)

        env = {
            **os.environ,
            # Without this the child block-buffers its stdout into a pipe and
            # the UI shows nothing until the job ends — which for eval is
            # half an hour of blank screen.
            "PYTHONUNBUFFERED": "1",
            "PYTHONIOENCODING": "utf-8",
        }
        proc = await asyncio.create_subprocess_exec(
            self.python,
            *argv,
            cwd=str(self.cwd),
            env=env,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            # Own process group, so cancelling reaches grandchildren too.
            start_new_session=os.name == "posix",
        )
        job = Job(
            id=uuid.uuid4().hex[:12],
            kind=kind,
            argv=tuple(argv),
            started_at=_now(),
            _proc=proc,
        )
        if len(self._order) == self._order.maxlen:
            self._jobs.pop(self._order[0], None)
        self._jobs[job.id] = job
        self._order.append(job.id)
        job._task = asyncio.create_task(self._pump(job, proc))
        logger.info("job %s started: %s", job.id, job.command)
        return job

    async def _pump(self, job: Job, proc: asyncio.subprocess.Process) -> None:
        try:
            if proc.stdout is not None:
                async for line in _iter_lines(proc.stdout):
                    job.append(line)
            job.returncode = await proc.wait()
        except asyncio.CancelledError:
            job.error = "runner shut down"
            raise
        except OSError as exc:  # pragma: no cover - the pipe died under us
            job.error = str(exc)
            logger.exception("job %s failed while reading output", job.id)
        finally:
            if job.status == RUNNING:
                if job.returncode == 0:
                    job.status = SUCCEEDED
                elif job.returncode is not None and job.returncode < 0:
                    # Killed by a signal: cancel() is the only thing that sends
                    # one here, but a crash looks the same and is not a success.
                    job.status = CANCELLED
                else:
                    job.status = FAILED
            job.finished_at = _now()
            job._notify()
            logger.info("job %s %s (rc=%s)", job.id, job.status, job.returncode)

    async def cancel(self, job_id: str) -> Job:
        job = self.require(job_id)
        proc = job._proc
        if not job.running or proc is None:
            return job
        job.status = CANCELLED
        job.append("-- cancelled by the operator --")
        self._signal(proc, signal.SIGTERM)
        try:
            await asyncio.wait_for(proc.wait(), KILL_AFTER_SECONDS)
        except TimeoutError:
            self._signal(proc, signal.SIGKILL)
        job._notify()
        return job

    @staticmethod
    def _signal(proc: asyncio.subprocess.Process, sig: signal.Signals) -> None:
        try:
            if os.name == "posix":
                os.killpg(os.getpgid(proc.pid), sig)
            else:  # pragma: no cover - posix only in practice
                proc.terminate()
        except (ProcessLookupError, PermissionError):
            pass  # already gone

    async def shutdown(self) -> None:
        """Terminate anything still running; called from the app's lifespan."""
        for job in list(self._jobs.values()):
            if job.running:
                with contextlib.suppress(JobError):
                    await self.cancel(job.id)
            task = job._task
            if task is not None:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task

    # -- streaming --

    async def stream(
        self, job_id: str, *, after: int = 0, heartbeat: float = HEARTBEAT_SECONDS
    ) -> AsyncIterator[tuple[int, str] | None]:
        """Yield ``(line_number, line)`` until the job ends and is drained.

        ``None`` is a heartbeat tick, so the caller can keep an idle SSE
        connection alive through proxies without inventing output. The line
        number is the stream cursor: pass it back as ``after`` to resume.
        """
        job = self.require(job_id)
        while True:
            waiter = job._updated  # snapshot BEFORE reading state: no lost wake-ups
            cursor, lines = job.tail(after)
            first = cursor - len(lines)
            for offset, line in enumerate(lines):
                yield first + offset, line
            after = cursor
            if not job.running and after >= job.total_lines:
                return
            try:
                await asyncio.wait_for(waiter.wait(), heartbeat)
            except TimeoutError:
                yield None
