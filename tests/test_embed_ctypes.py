"""Python integration tests for the C engine via ctypes.

See CUSTOM_EMBEDDINGS_RESEARCH.md (green-ai repo), Part 4, "Testing the
inference engine". These tests load `engine/libembed.so` via `ctypes` and
check `embed_text` against a tiny fixture table — they verify the engine's
mechanics, not the trained model's quality (that's a separate evaluation
process, Parts 5/6).

Currently skipped: the engine (engine/src/embed.c) is a stub.
"""

import ctypes
from pathlib import Path

import pytest

ENGINE_LIB = Path(__file__).parent.parent / "engine" / "libembed.so"

pytestmark = pytest.mark.skip(
    reason="engine/src/embed.c is a stub - nothing to test yet (see TODOs)"
)


@pytest.fixture
def lib():
    return ctypes.CDLL(str(ENGINE_LIB))


def test_known_word_embedding(lib):
    # TODO: build a tiny fixture table, load it via embed_load, and assert
    # embed_text() returns the expected 128-dim, L2-normalised vector for a
    # known single-word input.
    raise NotImplementedError


def test_multi_word_mean_pooling(lib):
    # TODO: assert embed_text() correctly mean-pools a multi-word chunk.
    raise NotImplementedError


def test_oov_uses_undefined_vector(lib):
    # TODO: assert an OOV token contributes the `undefined` row.
    raise NotImplementedError


def test_empty_input_returns_undefined(lib):
    # TODO: assert empty/punctuation-only input doesn't crash and returns
    # the `undefined` vector.
    raise NotImplementedError
