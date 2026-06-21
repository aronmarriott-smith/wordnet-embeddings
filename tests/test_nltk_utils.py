"""Tests for the shared NLTK download-on-demand helper."""

from __future__ import annotations

import pytest

from wordnet_embeddings.sources._nltk_utils import ensure_nltk_corpus


def test_noop_when_resource_present(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr("nltk.data.find", lambda path: calls.append(path))
    monkeypatch.setattr("nltk.download", lambda *_a, **_k: pytest.fail("should not download"))

    ensure_nltk_corpus("words")

    assert calls == ["corpora/words"]


def test_downloads_when_resource_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(_path: str) -> None:
        raise LookupError("missing")

    downloaded: list[str] = []
    monkeypatch.setattr("nltk.data.find", missing)
    monkeypatch.setattr("nltk.download", lambda package_id: downloaded.append(package_id) or True)

    ensure_nltk_corpus("words")

    assert downloaded == ["words"]


def test_raises_clear_error_when_download_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(_path: str) -> None:
        raise LookupError("missing")

    monkeypatch.setattr("nltk.data.find", missing)
    monkeypatch.setattr("nltk.download", lambda *_a, **_k: False)

    with pytest.raises(RuntimeError, match="Failed to download.*'words'"):
        ensure_nltk_corpus("words")


def test_custom_resource_path_is_probed(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr("nltk.data.find", lambda path: calls.append(path))

    ensure_nltk_corpus("english_wordnet", resource_path="corpora/english_wordnet")

    assert calls == ["corpora/english_wordnet"]


def test_noop_when_resource_present_but_still_zipped(monkeypatch: pytest.MonkeyPatch) -> None:
    # nltk.download() unzips by default, but a corpus already present from
    # elsewhere (e.g. baked into a base image) may still be a bare .zip —
    # find() only resolves that via the "<path>.zip/<package_id>/" form.
    def find_only_zip_form(path: str) -> None:
        if path != "corpora/wordnet.zip/wordnet/":
            raise LookupError("missing")

    monkeypatch.setattr("nltk.data.find", find_only_zip_form)
    monkeypatch.setattr("nltk.download", lambda *_a, **_k: pytest.fail("should not download"))

    ensure_nltk_corpus("wordnet")
