"""import_sentence_transformers turns OS code-integrity blocks into a clear error."""

from __future__ import annotations

import builtins
import sys

import pytest

from index import embedder as embedder_mod


def _block_import(monkeypatch, message: str):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise ImportError(message)
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "sentence_transformers", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_app_control_block_is_explained(monkeypatch):
    _block_import(
        monkeypatch,
        "DLL load failed while importing _highs_options: "
        "An Application Control policy has blocked this file.",
    )
    with pytest.raises(ImportError) as exc:
        embedder_mod.import_sentence_transformers()
    text = str(exc.value)
    assert "Smart App Control" in text
    assert "WSL" in text
    assert "_highs_options" in text  # the original cause is kept in the message
    assert isinstance(exc.value.__cause__, ImportError)


def test_ordinary_import_error_passes_through(monkeypatch):
    _block_import(monkeypatch, "No module named 'sentence_transformers'")
    with pytest.raises(ImportError, match="No module named") as exc:
        embedder_mod.import_sentence_transformers()
    assert "Smart App Control" not in str(exc.value)


def test_embedder_model_uses_the_helper(monkeypatch):
    _block_import(monkeypatch, "DLL load failed: An Application Control policy has blocked this file.")
    with pytest.raises(ImportError, match="Smart App Control"):
        embedder_mod.Embedder().model
