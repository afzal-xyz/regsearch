"""Pytest fixtures / setup shared across the suite.

Use the operating system's certificate trust store for TLS. This lets network
tests (e.g. the ``slow`` eCFR integration test) work on machines behind a
TLS-inspecting proxy whose root cert is in the OS store but not in certifi's
bundle. No-op for the mocked unit tests.
"""

from __future__ import annotations

try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:  # pragma: no cover - truststore is a dev dependency
    pass
