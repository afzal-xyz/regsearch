"""The architectural boundary: the web process holds no models, ever.

This is the invariant the dashboard's whole design rests on. Everything
expensive — embedding, retrieval, cross-encoding, tuning — happens in a
subprocess (`api/jobs.py`) or was frozen to disk ahead of time
(`retrieval/prescore.py`). If torch ever gets imported by a request handler,
the symptom is not a test failure: it is a dashboard that takes eight seconds
to answer and holds half a gigabyte for the life of the server.

Three overlapping checks, because each misses what the others catch:

* a **static scan** of ``api/`` finds a module-level import even in a file
  nothing currently reaches;
* an **import probe** catches anything pulled in at import time;
* a **route probe** exercises every endpoint in a fresh interpreter, which is
  the only way to catch a lazy import inside a handler.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from tests import _boundary_probe

REPO_ROOT = Path(__file__).resolve().parent.parent
API_DIR = REPO_ROOT / "api"

# One list, shared with the subprocess probe, so the static scan and the
# runtime check can never disagree about what counts as heavy.
HEAVY = set(_boundary_probe.FORBIDDEN)

# Importing api.main must stay interactive-fast. The real figure is ~0.3s;
# torch alone is several seconds, so this bound only fires on a real
# regression rather than on a slow machine.
IMPORT_BUDGET_SECONDS = 5.0


def _imported_names(path: Path) -> set[str]:
    """Top-level package names imported anywhere in a module, statically."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


class TestStaticImports:
    @pytest.mark.parametrize(
        "module", sorted(p.name for p in API_DIR.glob("*.py")), ids=lambda n: n
    )
    def test_no_api_module_names_a_model_library(self, module):
        """Not even lazily inside a function — api/ has no business with torch."""
        offenders = _imported_names(API_DIR / module) & HEAVY
        assert not offenders, f"api/{module} imports {sorted(offenders)}"

    def test_the_heavy_work_lives_behind_the_job_allowlist(self):
        """The dashboard reaches models only by starting a subprocess."""
        from api import jobs

        for kind, spec in jobs.SPECS.items():
            argv = spec({})
            assert argv[0] == "-m", f"{kind} is not a `python -m` subprocess: {argv}"


class TestRuntimeBoundary:
    def test_importing_and_exercising_the_app_loads_no_model_library(self):
        """The strong version: every route is called, then sys.modules is checked."""
        done = subprocess.run(
            # -m, so sys.path[0] is the repo root rather than tests/.
            [sys.executable, "-m", "tests._boundary_probe"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=300,
        )
        assert done.returncode == 0, done.stderr or done.stdout
        assert "OK:" in done.stdout

    def test_importing_the_app_is_fast(self):
        """A proxy for "no model library", with a different failure mode: an
        import that is slow for some *other* reason is also worth knowing about."""
        probe = (
            "import time; t=time.perf_counter(); import api.main; "
            "print(time.perf_counter()-t)"
        )
        done = subprocess.run(
            [sys.executable, "-c", probe], cwd=REPO_ROOT, capture_output=True, text=True,
            timeout=120,
        )
        assert done.returncode == 0, done.stderr
        elapsed = float(done.stdout.strip().splitlines()[-1])
        assert elapsed < IMPORT_BUDGET_SECONDS, f"api.main took {elapsed:.2f}s to import"

    def test_the_sandbox_scoring_path_cannot_reach_a_cross_encoder(self):
        """Belt and braces for the stub reranker, from the outside."""
        from api.sandbox import _REFUSE, NotWarmed

        with pytest.raises(NotWarmed):
            _REFUSE.score("a query", ["a passage"])
