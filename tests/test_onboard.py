"""Tests for the onboarding orchestrator and the fetcher's retry policy."""

from __future__ import annotations

import pytest
import requests

from ingest import fetcher
from pipeline import onboard
from pipeline.manifest import TitleSpec

SPEC = TitleSpec(4, "2024-07-18")


# --- fetch retry ------------------------------------------------------------


class _Resp:
    """Minimal stand-in for a streamed requests.Response."""

    def __init__(self, status=200, body=b"<?xml version='1.0'?><DIV1/>", headers=None):
        self.status_code = status
        self._body = body
        self.headers = {"Content-Type": "application/xml", **(headers or {})}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}", response=self)

    def iter_content(self, chunk_size=None):
        yield self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _Session:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def get(self, url, **kw):
        self.calls += 1
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def close(self):
        pass


def _fetch(tmp_path, responses, **kw):
    session = _Session(responses)
    slept = []
    path = fetcher.fetch_title_xml(
        4,
        "2024-07-18",
        data_dir=tmp_path,
        session=session,
        min_interval=0,
        sleep=slept.append,
        **kw,
    )
    return path, session, slept


def test_504_is_retried_and_then_succeeds(tmp_path):
    """The motivating case: eCFR 504s on a large title, then serves it."""
    path, session, slept = _fetch(tmp_path, [_Resp(504), _Resp(504), _Resp()])
    assert path.exists()
    assert session.calls == 3


def test_backoff_doubles(tmp_path):
    _, _, slept = _fetch(tmp_path, [_Resp(504), _Resp(504), _Resp()], backoff=10)
    assert slept == [10, 20]


def test_retry_after_header_overrides_backoff(tmp_path):
    _, _, slept = _fetch(
        tmp_path, [_Resp(429, headers={"Retry-After": "7"}), _Resp()], backoff=60
    )
    assert slept == [7]


def test_timeout_is_retried(tmp_path):
    path, session, _ = _fetch(tmp_path, [requests.Timeout("slow"), _Resp()])
    assert path.exists() and session.calls == 2


def test_connection_error_is_retried(tmp_path):
    path, session, _ = _fetch(tmp_path, [requests.ConnectionError("reset"), _Resp()])
    assert path.exists() and session.calls == 2


def test_404_is_not_retried(tmp_path):
    """A title/date that does not exist will never exist; fail fast."""
    with pytest.raises(requests.HTTPError):
        _fetch(tmp_path, [_Resp(404), _Resp()])


def test_gives_up_after_max_attempts(tmp_path):
    with pytest.raises(requests.HTTPError):
        _fetch(tmp_path, [_Resp(503)] * 3, max_attempts=3)


def test_empty_body_is_treated_as_transient(tmp_path):
    path, session, _ = _fetch(tmp_path, [_Resp(body=b""), _Resp()])
    assert path.exists() and session.calls == 2


def test_html_error_page_is_not_cached_or_retried(tmp_path):
    with pytest.raises(fetcher.FetchError, match="not XML"):
        _fetch(tmp_path, [_Resp(body=b"<!doctype html><html>oops</html>"), _Resp()])
    assert not list(tmp_path.glob("*.xml"))


def test_failed_fetch_leaves_no_partial_file(tmp_path):
    with pytest.raises(requests.HTTPError):
        _fetch(tmp_path, [_Resp(500)] * 4)
    assert not list(tmp_path.iterdir())


def test_cached_file_short_circuits(tmp_path):
    dest = fetcher.cache_path(4, "2024-07-18", tmp_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"<?xml version='1.0'?><DIV1/>")
    _, session, _ = _fetch(tmp_path, [])
    assert session.calls == 0


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (requests.Timeout(), True),
        (requests.ConnectionError(), True),
        (requests.HTTPError(response=_Resp(503)), True),
        (requests.HTTPError(response=_Resp(429)), True),
        (requests.HTTPError(response=_Resp(404)), False),
        (requests.HTTPError(response=_Resp(400)), False),
        (ValueError(), False),
    ],
)
def test_is_transient(exc, expected):
    assert fetcher.is_transient(exc) is expected


# --- stage selection --------------------------------------------------------


def test_default_runs_every_stage():
    assert onboard.select_stages(None, None) == list(onboard.STAGES)


def test_only_runs_one_stage():
    assert onboard.select_stages("integrity", None) == ["integrity"]


def test_from_runs_the_tail():
    assert onboard.select_stages(None, "gold") == ["gold", "tune", "eval", "regression"]


def test_preflight_is_first_and_tune_precedes_eval():
    order = list(onboard.STAGES)
    assert order[0] == "preflight", "cheap failures must come before any download"
    assert order.index("gold") < order.index("tune") < order.index("eval"), (
        "tune needs the gold set and must run before the numbers are recorded"
    )


def test_stage_order_gates_before_measuring():
    """Integrity and the gold gate must both precede eval: measuring a corpus
    you have not verified produces a number nobody can trust."""
    order = list(onboard.STAGES)
    assert order.index("integrity") < order.index("eval")
    assert order.index("gold") < order.index("eval")
    assert order.index("eval") < order.index("regression")


def test_every_stage_has_an_implementation():
    assert set(onboard.STAGE_FUNCS) == set(onboard.STAGES)


# --- driver -----------------------------------------------------------------


def test_run_stops_at_the_first_failure(monkeypatch, capsys):
    calls = []

    def ok(name):
        def f(specs, **kw):
            calls.append(name)

        return f

    def boom(specs, **kw):
        calls.append("integrity")
        raise onboard.StageFailed("bad corpus")

    monkeypatch.setitem(onboard.STAGE_FUNCS, "preflight", ok("preflight"))
    monkeypatch.setitem(onboard.STAGE_FUNCS, "fetch", ok("fetch"))
    monkeypatch.setitem(onboard.STAGE_FUNCS, "chunk", ok("chunk"))
    monkeypatch.setitem(onboard.STAGE_FUNCS, "index", ok("index"))
    monkeypatch.setitem(onboard.STAGE_FUNCS, "integrity", boom)

    rc = onboard.run([SPEC], list(onboard.STAGES))
    assert rc == 1
    assert calls == ["preflight", "fetch", "chunk", "index", "integrity"]
    assert "eval" not in calls, "a failed gate must not be followed by a scored run"
    assert "not running" in capsys.readouterr().err


def test_run_passes_every_declared_title_to_every_stage(monkeypatch):
    seen = []
    for name in onboard.STAGES:
        monkeypatch.setitem(
            onboard.STAGE_FUNCS, name, lambda specs, _n=name, **kw: seen.append((_n, list(specs)))
        )
    specs = [TitleSpec(1, "2024-07-18"), SPEC, TitleSpec(31, "2024-07-18")]
    assert onboard.run(specs, list(onboard.STAGES)) == 0
    assert len(seen) == len(onboard.STAGES)
    for _name, got in seen:
        assert got == specs, "every stage must see the whole corpus"


def test_main_requires_a_date_with_title(capsys):
    assert onboard.main(["--title", "1"]) == 1
    assert "requires --date" in capsys.readouterr().err
