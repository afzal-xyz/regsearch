"""Tests for ingest.fetcher.

The unit tests mock the HTTP layer entirely (no network). The integration test
is marked ``slow`` and really fetches a tiny title (Title 4, "Accounts").
"""

from __future__ import annotations

from unittest import mock

import pytest

from ingest import fetcher

XML_BODY = b'<?xml version="1.0" encoding="UTF-8"?><CFR><DIV1>Title 4</DIV1></CFR>'


class FakeResponse:
    """Minimal stand-in for a streaming ``requests.Response``."""

    def __init__(self, chunks, *, status=200, content_type="application/xml"):
        self._chunks = chunks
        self.status_code = status
        self.headers = {"Content-Type": content_type}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests_http_error(self.status_code)

    def iter_content(self, chunk_size=None):
        yield from self._chunks


def requests_http_error(status):
    from requests import HTTPError

    return HTTPError(f"status {status}")


def make_session(response):
    session = mock.MagicMock()
    session.get.return_value = response
    return session


def test_cache_path_naming(tmp_path):
    assert fetcher.cache_path(40, "2026-07-01", data_dir=tmp_path) == (
        tmp_path / "title-40-2026-07-01.xml"
    )


def test_build_url():
    assert fetcher.build_url(4, "2024-07-18") == (
        "https://www.ecfr.gov/api/versioner/v1/full/2024-07-18/title-4.xml"
    )


def test_invalid_date_raises(tmp_path):
    with pytest.raises(ValueError):
        fetcher.fetch_title_xml(40, "07/01/2026", data_dir=tmp_path)


def test_fetch_writes_file_mocked(tmp_path):
    session = make_session(FakeResponse([XML_BODY]))

    dest = fetcher.fetch_title_xml(
        40, "2026-07-01", data_dir=tmp_path, session=session, min_interval=0
    )

    assert dest.exists()
    assert dest.read_bytes() == XML_BODY

    args, kwargs = session.get.call_args
    assert args[0] == (
        "https://www.ecfr.gov/api/versioner/v1/full/2026-07-01/title-40.xml"
    )
    assert kwargs["headers"]["User-Agent"] == fetcher.DEFAULT_USER_AGENT
    assert "xml" in kwargs["headers"]["Accept"]
    assert kwargs["stream"] is True


def test_cache_skips_redownload(tmp_path):
    session = make_session(FakeResponse([XML_BODY]))
    fetcher.fetch_title_xml(
        40, "2026-07-01", data_dir=tmp_path, session=session, min_interval=0
    )

    session.get.reset_mock()
    dest = fetcher.fetch_title_xml(
        40, "2026-07-01", data_dir=tmp_path, session=session, min_interval=0
    )

    session.get.assert_not_called()
    assert dest.exists()


def test_force_redownloads(tmp_path):
    session = make_session(FakeResponse([XML_BODY]))
    fetcher.fetch_title_xml(
        40, "2026-07-01", data_dir=tmp_path, session=session, min_interval=0
    )

    session.get.return_value = FakeResponse([XML_BODY])
    session.get.reset_mock()
    fetcher.fetch_title_xml(
        40, "2026-07-01", data_dir=tmp_path, session=session, min_interval=0, force=True
    )

    session.get.assert_called_once()


def test_html_error_page_not_cached(tmp_path):
    """A 200-status HTML error page must not be written to the cache."""
    session = make_session(
        FakeResponse(
            [b"<!DOCTYPE html><html><body>Not found</body></html>"],
            content_type="text/html",
        )
    )

    with pytest.raises(fetcher.FetchError):
        fetcher.fetch_title_xml(
            40, "2026-07-01", data_dir=tmp_path, session=session, min_interval=0
        )

    assert not fetcher.cache_path(40, "2026-07-01", data_dir=tmp_path).exists()
    assert list(tmp_path.glob("*.tmp")) == []  # temp file cleaned up


@pytest.mark.slow
def test_fetch_title_4_integration(tmp_path):
    """Really fetch Title 4 (tiny) from the eCFR API."""
    dest = fetcher.fetch_title_xml(4, "2024-07-18", data_dir=tmp_path)

    assert dest.exists()
    data = dest.read_bytes()
    assert data.lstrip().startswith(b"<?xml")
    assert b"<CFR" in data or b"<DIV" in data
