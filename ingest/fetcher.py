"""Download full-title XML from the eCFR Versioner API, with on-disk caching.

The eCFR "versioner" service serves the full XML of a CFR title as it stood on a
given issue date:

    GET https://www.ecfr.gov/api/versioner/v1/full/{date}/title-{title}.xml

Downloads are cached under ``data/raw/`` keyed by title + date and are not
re-fetched if already present.

CLI:
    python -m ingest.fetcher --title 40 --date 2026-07-01
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

ECFR_BASE = "https://www.ecfr.gov"
FULL_XML_PATH = "/api/versioner/v1/full/{date}/title-{title}.xml"
FULL_XML_URL = ECFR_BASE + FULL_XML_PATH

DEFAULT_USER_AGENT = "RegSearch/0.1 (+https://github.com/afzal-xyz/regsearch)"
DEFAULT_MIN_INTERVAL = 1.0  # seconds between network requests (be polite to eCFR)
DATA_RAW = Path("data/raw")

_CHUNK_SIZE = 64 * 1024
# (connect, read) seconds. The read timeout is generous because eCFR renders a
# full title on demand: Title 31 takes over a minute before the first byte.
_REQUEST_TIMEOUT = (10, 300)

# Retry policy. A full-title render is slow and eCFR sheds load under it — a
# 504 Gateway Time-out on the first attempt at a large title is routine, and
# the identical request succeeds minutes later. Without retry, onboarding a
# big title is a coin flip, so the pipeline treats these as transient.
DEFAULT_MAX_ATTEMPTS = 4
DEFAULT_BACKOFF = 15.0  # seconds before the 2nd attempt; doubles each time
_BACKOFF_CAP = 300.0
_RETRY_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})

# Module-level timestamp used for simple cross-call rate limiting.
_last_request_at = 0.0


class FetchError(RuntimeError):
    """Raised when the eCFR response is missing or is not valid title XML."""


def _validate_date(date: str) -> str:
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"date must be YYYY-MM-DD, got {date!r}") from exc
    return date


def cache_path(title: int, date: str, data_dir: Path = DATA_RAW) -> Path:
    """Return the on-disk cache path for a title/date, e.g.
    ``data/raw/title-40-2026-07-01.xml``."""
    return Path(data_dir) / f"title-{int(title)}-{date}.xml"


def build_url(title: int, date: str) -> str:
    return FULL_XML_URL.format(date=date, title=int(title))


def _respect_rate_limit(min_interval: float) -> None:
    """Sleep just long enough to keep at least ``min_interval`` seconds between
    successive network requests across calls."""
    global _last_request_at
    if min_interval <= 0:
        return
    wait = min_interval - (time.monotonic() - _last_request_at)
    if wait > 0:
        time.sleep(wait)
    _last_request_at = time.monotonic()


def _retry_after(resp: requests.Response) -> float | None:
    """``Retry-After`` in seconds, if the server sent a usable one.

    Only the delta-seconds form is honored; the HTTP-date form is rare here and
    not worth the clock-skew risk. A server's own number beats our backoff
    schedule, so it wins when present.
    """
    raw = resp.headers.get("Retry-After", "").strip()
    if not raw:
        return None
    try:
        seconds = float(raw)
    except ValueError:
        return None
    return min(max(seconds, 0.0), _BACKOFF_CAP) if seconds >= 0 else None


def is_transient(exc: BaseException) -> bool:
    """Is this failure worth retrying the identical request for?

    Timeouts and connection drops are transient by nature. For HTTP errors
    only :data:`_RETRY_STATUS` qualifies: a 404 means the title/date does not
    exist and will never exist, so retrying it just burns minutes before
    showing the user the same error.
    """
    if isinstance(exc, requests.Timeout | requests.ConnectionError):
        return True
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return exc.response.status_code in _RETRY_STATUS
    return False


def _looks_like_xml(head: bytes, content_type: str) -> bool:
    """Heuristic guard against 200-status error/HTML pages: the eCFR full-title
    endpoint returns an ``<?xml ...>`` document with an XML content type."""
    lead = head.lstrip()[:512].lower()
    if lead.startswith(b"<!doctype html") or lead.startswith(b"<html"):
        return False
    if lead.startswith(b"<?xml"):
        return True
    return "xml" in content_type.lower() and lead.startswith(b"<")


class TransientFetchError(FetchError):
    """A fetch failure that an identical retry could plausibly fix."""


def _download_once(
    url: str,
    dest: Path,
    *,
    session: requests.Session,
    user_agent: str,
    min_interval: float,
) -> None:
    """One streamed download attempt, committed atomically on success."""
    headers = {"User-Agent": user_agent, "Accept": "application/xml"}
    tmp = dest.with_name(dest.name + ".tmp")
    try:
        _respect_rate_limit(min_interval)
        with session.get(url, headers=headers, stream=True, timeout=_REQUEST_TIMEOUT) as resp:
            resp.raise_for_status()
            content_type = resp.headers.get("Content-Type", "")
            wrote_any = False
            with open(tmp, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=_CHUNK_SIZE):
                    if not chunk:
                        continue
                    if not wrote_any:
                        # Validate the first bytes before committing anything.
                        if not _looks_like_xml(chunk, content_type):
                            raise FetchError(
                                f"response for {url} is not XML "
                                f"(Content-Type={content_type!r}); refusing to cache"
                            )
                        wrote_any = True
                    fh.write(chunk)
            if not wrote_any:
                # A 200 with no body is eCFR giving up mid-render, not a real
                # answer about this title — worth retrying.
                raise TransientFetchError(f"empty response body for {url}")
        os.replace(tmp, dest)
    except BaseException:
        # Never leave a partial or invalid file behind in the cache.
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def fetch_title_xml(
    title: int,
    date: str,
    *,
    data_dir: Path = DATA_RAW,
    session: requests.Session | None = None,
    user_agent: str = DEFAULT_USER_AGENT,
    min_interval: float = DEFAULT_MIN_INTERVAL,
    force: bool = False,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    backoff: float = DEFAULT_BACKOFF,
    sleep=time.sleep,
) -> Path:
    """Fetch the full XML for ``title`` as of ``date`` and cache it under
    ``data_dir``. Returns the path to the cached file.

    If the file is already cached (and ``force`` is False) it is returned
    without any network request. The download is streamed to a temporary file,
    validated as XML, then atomically moved into place, so a failed or bogus
    response never poisons the cache.

    Transient failures (timeouts, dropped connections, 5xx, 429 — see
    :func:`is_transient`) are retried up to ``max_attempts`` times with
    exponential backoff, honouring ``Retry-After`` when the server sends one.
    Everything else fails immediately: a 404 means that title/date does not
    exist, and retrying it only delays the error. ``sleep`` is injectable so
    tests can exercise the backoff schedule without waiting.
    """
    date = _validate_date(date)
    dest = cache_path(title, date, data_dir)

    if dest.exists() and not force:
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    url = build_url(title, date)

    owns_session = session is None
    session = session or requests.Session()
    try:
        delay = backoff
        for attempt in range(1, max(1, max_attempts) + 1):
            try:
                _download_once(
                    url,
                    dest,
                    session=session,
                    user_agent=user_agent,
                    min_interval=min_interval,
                )
                return dest
            except (requests.RequestException, TransientFetchError) as exc:
                transient = isinstance(exc, TransientFetchError) or is_transient(exc)
                if not transient or attempt >= max_attempts:
                    raise
                wait = delay
                response = getattr(exc, "response", None)
                if response is not None:
                    wait = _retry_after(response) or delay
                print(
                    f"warning: {type(exc).__name__} fetching title {title} "
                    f"(attempt {attempt}/{max_attempts}); retrying in {wait:.0f}s",
                    file=sys.stderr,
                )
                sleep(wait)
                delay = min(delay * 2, _BACKOFF_CAP)
    finally:
        if owns_session:
            session.close()

    raise AssertionError("unreachable")  # pragma: no cover


def _use_os_trust_store() -> None:
    """Best-effort: trust the OS certificate store so the CLI works behind a
    TLS-inspecting proxy. No-op if ``truststore`` isn't installed (certifi's
    bundle is used, which is fine on machines without cert interception)."""
    try:
        import truststore

        truststore.inject_into_ssl()
    except ImportError:
        pass


def main(argv: list[str] | None = None) -> int:
    _use_os_trust_store()
    parser = argparse.ArgumentParser(
        prog="python -m ingest.fetcher",
        description="Download full-title CFR XML from the eCFR Versioner API.",
    )
    parser.add_argument("--title", type=int, required=True, help="CFR title number, e.g. 40")
    parser.add_argument("--date", required=True, help="Issue date, YYYY-MM-DD")
    parser.add_argument(
        "--data-dir", type=Path, default=DATA_RAW, help="Cache directory (default: data/raw)"
    )
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    parser.add_argument(
        "--min-interval",
        type=float,
        default=DEFAULT_MIN_INTERVAL,
        help="Minimum seconds between network requests",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if cached")
    parser.add_argument(
        "--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS,
        help="Attempts before giving up on a transient failure",
    )
    parser.add_argument(
        "--backoff", type=float, default=DEFAULT_BACKOFF,
        help="Seconds before the 2nd attempt (doubles each retry)",
    )
    args = parser.parse_args(argv)

    try:
        dest = fetch_title_xml(
            args.title,
            args.date,
            data_dir=args.data_dir,
            user_agent=args.user_agent,
            min_interval=args.min_interval,
            force=args.force,
            max_attempts=args.max_attempts,
            backoff=args.backoff,
        )
    except (ValueError, FetchError, requests.RequestException) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
