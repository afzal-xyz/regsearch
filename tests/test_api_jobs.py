"""Tests for api.jobs: the command allowlist, process supervision and streaming.

The allowlist tests matter most. Every job the dashboard can start is a
subprocess on the developer's machine, so "which argv can a web request
produce" is the whole security surface, and it is pure and cheap to test.

Process tests drive the runner with injected specs that run trivial ``-c``
scripts, so the suite never shells out to eval or torch.
"""

from __future__ import annotations

import asyncio
import sys
from collections import deque

import pytest

from api import jobs as jobs_mod
from api.jobs import (
    CANCELLED,
    FAILED,
    SUCCEEDED,
    InvalidOption,
    Job,
    JobBusy,
    JobRunner,
    UnknownJob,
)


def run(coro):
    """Drive one coroutine to completion (no pytest-asyncio dependency)."""
    return asyncio.run(coro)


TEST_SPECS = {
    "ok": lambda o: ["-c", "print('alpha'); print('beta')"],
    "fail": lambda o: ["-c", "import sys; print('nope'); sys.exit(3)"],
    "slow": lambda o: ["-c", "import time; print('start', flush=True); time.sleep(30)"],
    "cr": lambda o: ["-c", r"import sys; sys.stdout.write('10%\r50%\r100%\ndone\n')"],
    "flood": lambda o: ["-c", "[print(i) for i in range(50)]"],
}


def make_runner(**kw) -> JobRunner:
    return JobRunner(specs=TEST_SPECS, python=sys.executable, **kw)


# --- the allowlist ----------------------------------------------------------


class TestAllowlist:
    def test_unknown_kind_is_refused(self):
        with pytest.raises(UnknownJob):
            JobRunner().build("rm-rf")

    def test_eval_defaults_to_all_modes(self):
        assert JobRunner().build("eval") == ["-m", "eval.run", "--mode", "all"]

    def test_eval_accepts_a_known_mode(self):
        argv = JobRunner().build("eval", {"mode": "rerank", "k": "5"})
        assert argv == ["-m", "eval.run", "--mode", "rerank", "--k", "5"]

    def test_eval_refuses_an_unknown_mode(self):
        with pytest.raises(InvalidOption):
            JobRunner().build("eval", {"mode": "; rm -rf /"})

    def test_unrecognised_option_is_refused_rather_than_ignored(self):
        """Silently dropping an option would make the UI lie about what it ran."""
        with pytest.raises(InvalidOption):
            JobRunner().build("eval", {"gold": "/etc/passwd"})

    def test_out_of_range_integer_is_refused(self):
        with pytest.raises(InvalidOption):
            JobRunner().build("eval", {"k": "100000"})
        with pytest.raises(InvalidOption):
            JobRunner().build("eval", {"k": "not a number"})

    @pytest.mark.parametrize(
        "against",
        ["../../etc/passwd", "/etc/passwd", "a.json;rm -rf .", "eval/results/x.json", "x.txt"],
    )
    def test_compare_baseline_must_be_a_bare_result_filename(self, against):
        with pytest.raises(InvalidOption):
            JobRunner().build("compare", {"against": against})

    def test_compare_accepts_a_real_stamp(self):
        argv = JobRunner().build("compare", {"against": "20261104-115758.json"})
        assert argv == ["-m", "eval.compare", "--against", "20261104-115758.json"]

    def test_tune_is_dry_run_when_asked(self):
        assert "--dry-run" in JobRunner().build("tune", {"dry_run": "1"})
        assert "--dry-run" not in JobRunner().build("tune", {})

    def test_onboard_cannot_add_a_title(self):
        """Adding a title rewrites corpus.toml and triggers a full re-fetch."""
        with pytest.raises(InvalidOption):
            JobRunner().build("onboard", {"title": "26", "date": "2024-07-18"})

    def test_onboard_stage_must_be_a_real_stage(self):
        assert JobRunner().build("onboard", {"only": "eval"})[-1] == "eval"
        with pytest.raises(InvalidOption):
            JobRunner().build("onboard", {"only": "fetch-everything"})

    def test_every_exposed_job_has_a_label(self):
        assert set(jobs_mod.SPECS) == set(jobs_mod.LABELS)

    def test_no_spec_passes_a_free_form_string_through(self):
        """Each spec must reject options it does not know, with empty input working."""
        for kind, spec in jobs_mod.SPECS.items():
            assert spec({}), f"{kind} produced an empty argv"
            with pytest.raises(InvalidOption):
                spec({"nonsense_option": "x"})


# --- the ring buffer --------------------------------------------------------


class TestBuffer:
    def test_tail_returns_only_new_lines(self):
        job = Job(id="a", kind="ok", argv=(), started_at="now")
        for i in range(3):
            job.append(f"line {i}")
        cursor, lines = job.tail()
        assert lines == ["line 0", "line 1", "line 2"] and cursor == 3
        assert job.tail(cursor) == (3, [])
        job.append("line 3")
        assert job.tail(cursor) == (4, ["line 3"])

    def test_head_is_dropped_not_the_tail(self, monkeypatch):
        """The verdict is always the last line, so overflow must evict the head."""
        monkeypatch.setattr(jobs_mod, "MAX_LINES", 5)
        job = Job(id="a", kind="ok", argv=(), started_at="now")
        assert job.lines.maxlen == 5
        for i in range(12):
            job.append(str(i))
        assert job.total_lines == 12
        assert job.dropped == 7
        assert job.tail()[1] == ["7", "8", "9", "10", "11"]

    def test_cursor_behind_the_window_resumes_at_the_oldest_kept_line(self, monkeypatch):
        monkeypatch.setattr(jobs_mod, "MAX_LINES", 3)
        job = Job(id="a", kind="ok", argv=(), started_at="now")
        for i in range(10):
            job.append(str(i))
        cursor, lines = job.tail(after=0)
        assert cursor == 10 and lines == ["7", "8", "9"]

    def test_deque_is_per_job(self):
        a, b = (Job(id=i, kind="ok", argv=(), started_at="now") for i in ("a", "b"))
        a.append("x")
        assert isinstance(b.lines, deque) and not b.lines


# --- supervision ------------------------------------------------------------


class TestExecution:
    def test_captures_output_and_succeeds(self):
        async def go():
            runner = make_runner()
            job = await runner.start("ok")
            await job._task
            return job

        job = run(go())
        assert job.status == SUCCEEDED
        assert job.returncode == 0
        assert job.tail()[1] == ["alpha", "beta"]

    def test_nonzero_exit_is_a_failure_with_its_code(self):
        async def go():
            runner = make_runner()
            job = await runner.start("fail")
            await job._task
            return job

        job = run(go())
        assert job.status == FAILED
        assert job.returncode == 3
        assert "nope" in job.tail()[1]

    def test_carriage_returns_become_separate_lines(self):
        r"""tqdm redraws with \r; without splitting on it the UI sees one huge line."""

        async def go():
            runner = make_runner()
            job = await runner.start("cr")
            await job._task
            return job

        job = run(go())
        assert job.tail()[1] == ["10%", "50%", "100%", "done"]

    def test_only_one_job_runs_at_a_time(self):
        async def go():
            runner = make_runner()
            slow = await runner.start("slow")
            with pytest.raises(JobBusy):
                await runner.start("ok")
            await runner.cancel(slow.id)
            # Once it is done, the next job starts fine.
            second = await runner.start("ok")
            await second._task
            return second

        assert run(go()).status == SUCCEEDED

    def test_cancel_terminates_the_process(self):
        async def go():
            runner = make_runner()
            job = await runner.start("slow")
            while job.total_lines == 0:  # wait for it to actually be up
                await asyncio.sleep(0.02)
            await runner.cancel(job.id)
            await job._task
            return job

        job = run(go())
        assert job.status == CANCELLED
        assert job.returncode != 0
        assert "cancelled by the operator" in job.tail()[1][-1]

    def test_cancelling_a_finished_job_is_a_no_op(self):
        async def go():
            runner = make_runner()
            job = await runner.start("ok")
            await job._task
            return await runner.cancel(job.id)

        assert run(go()).status == SUCCEEDED

    def test_shutdown_stops_a_running_job(self):
        async def go():
            runner = make_runner()
            job = await runner.start("slow")
            await runner.shutdown()
            return job

        assert run(go()).status == CANCELLED

    def test_history_is_bounded_and_newest_first(self):
        async def go():
            runner = make_runner(history=3)
            for _ in range(5):
                job = await runner.start("ok")
                await job._task
            return runner.list()

        listed = run(go())
        assert len(listed) == 3
        assert [j.started_at for j in listed] == sorted(
            (j.started_at for j in listed), reverse=True
        )


class TestStreaming:
    def test_streams_until_the_job_ends(self):
        async def go():
            runner = make_runner()
            job = await runner.start("flood")
            seen = [item async for item in runner.stream(job.id, heartbeat=5.0)]
            return job, [line for item in seen if item for _, line in [item]]

        job, lines = run(go())
        assert job.status == SUCCEEDED
        assert lines[0] == "0" and lines[-1] == "49"

    def test_stream_resumes_from_a_cursor(self):
        async def go():
            runner = make_runner()
            job = await runner.start("flood")
            await job._task
            return [line async for item in runner.stream(job.id, after=45) if item
                    for _, line in [item]]

        assert run(go()) == ["45", "46", "47", "48", "49"]

    def test_stream_emits_heartbeats_while_idle(self):
        """An idle SSE connection must send something or proxies close it."""

        async def go():
            runner = make_runner()
            job = await runner.start("slow")
            ticks = 0
            async for item in runner.stream(job.id, heartbeat=0.05):
                if item is None:
                    ticks += 1
                    if ticks == 2:
                        await runner.cancel(job.id)
            return ticks

        assert run(go()) >= 2

    def test_streaming_an_unknown_job_raises(self):
        async def go():
            async for _ in make_runner().stream("nope"):
                pass

        with pytest.raises(UnknownJob):
            run(go())
