"""Shared download-on-demand helper for NLTK-backed sources."""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def ensure_nltk_corpus(package_id: str, resource_path: str | None = None) -> None:
    """Ensure an NLTK corpus package is downloaded, raising a clear error on failure.

    Args:
        package_id: the NLTK downloader package id (e.g. "wordnet", "words").
        resource_path: the nltk.data resource path to probe for presence
            (defaults to f"corpora/{package_id}").
    """
    if _is_present(resource_path or f"corpora/{package_id}", package_id):
        return

    import nltk

    log.info("NLTK corpus %r not found; downloading (one-time setup)...", package_id)
    if not nltk.download(package_id):
        raise RuntimeError(
            f"Failed to download the NLTK {package_id!r} corpus — check network access "
            f"and try `python -c \"import nltk; nltk.download('{package_id}')\"` manually."
        ) from None


def _is_present(path: str, package_id: str) -> bool:
    """Check both the unzipped and still-zipped forms nltk.data.find() can resolve.

    nltk.download() unzips by default, but a corpus already present from
    elsewhere (e.g. pre-downloaded in an image, or fetched by an older NLTK)
    may still be sitting as a .zip — find() only auto-descends into it given
    the exact "<path>.zip/<package_id>/" form, not the bare path.
    """
    import nltk

    for candidate in (path, f"{path}.zip/{package_id}/"):
        try:
            nltk.data.find(candidate)
            return True
        except LookupError:
            continue
    return False
