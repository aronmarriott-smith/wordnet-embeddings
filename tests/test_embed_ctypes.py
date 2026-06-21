"""Python integration tests for the C engine via ctypes.

Tests load engine/libembed.so and verify embed_text() mechanics against a tiny
synthetic fixture — not the trained model's quality.
"""

import os
import ctypes
import math
import struct
import subprocess
from pathlib import Path

import pytest

from wordnet_embeddings.config import ENGINE_BUILD_COMMAND

REPO_ROOT = Path(__file__).parent.parent
ENGINE_DIR = REPO_ROOT / "engine"
ENGINE_LIB = ENGINE_DIR / "libembed.so"
EMBED_DIM = 128  # must match engine/include/embed.h

# Synthetic fixture matching the C tests:
#   row 0 "undefined": [  0, 100,   0, 0 ]  — non-zero for OOV testing
#   row 1 "cat":       [127,   0,   0, 0 ]
#   row 2 "dog":       [  0,   0, 127, 0 ]
FX_DIM   = 4
FX_SCALE = 1.0 / 127.0
FX_VOCAB = ["undefined", "cat", "dog"]
FX_ROWS  = [[0, 100, 0, 0], [127, 0, 0, 0], [0, 0, 127, 0]]


@pytest.fixture(scope="session")
def lib():
    build_command = ENGINE_BUILD_COMMAND.get(os.name)
    if build_command is not None:
        result = subprocess.run(
            build_command, cwd=ENGINE_DIR, capture_output=True, text=True
        )
        assert result.returncode == 0, f"{' '.join(build_command)} failed:\n{result.stderr}"
    elif not ENGINE_LIB.exists():
        pytest.fail(
            f"{ENGINE_LIB} does not exist and os.name={os.name!r} has no "
            "auto-build command configured (see ENGINE_BUILD_COMMAND in "
            "wordnet_embeddings/config.py). Run bin/build_windows.ps1 "
            "(or .sh) first."
        )

    so = ctypes.CDLL(str(ENGINE_LIB))

    so.embed_load.restype  = ctypes.c_void_p
    so.embed_load.argtypes = [ctypes.c_char_p]

    so.embed_free.restype  = None
    so.embed_free.argtypes = [ctypes.c_void_p]

    so.embed_text.restype  = ctypes.c_int
    so.embed_text.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_float),
    ]

    return so


@pytest.fixture
def fixture_dir(tmp_path):
    (tmp_path / "vocab.txt").write_text("\n".join(FX_VOCAB) + "\n")
    with (tmp_path / "embeddings.bin").open("wb") as f:
        f.write(b"WNEB")
        f.write(struct.pack("<IIf", len(FX_VOCAB), FX_DIM, FX_SCALE))
        for row in FX_ROWS:
            f.write(struct.pack(f"{FX_DIM}b", *row))
    return tmp_path


@pytest.fixture
def model(lib, fixture_dir):
    handle = lib.embed_load(str(fixture_dir).encode())
    assert handle is not None, "embed_load returned NULL"
    yield handle
    lib.embed_free(handle)


def _embed(lib, model, text):
    out = (ctypes.c_float * EMBED_DIM)()
    assert lib.embed_text(model, text.encode(), out) == 0
    return list(out[:FX_DIM])


def _norm(v):
    return math.sqrt(sum(x * x for x in v))


def test_known_word_cat(lib, model):
    v = _embed(lib, model, "cat")
    assert abs(v[0] - 1.0) < 1e-5
    assert abs(v[1]) < 1e-5
    assert abs(_norm(v) - 1.0) < 1e-5


def test_known_word_dog(lib, model):
    v = _embed(lib, model, "dog")
    assert abs(v[2] - 1.0) < 1e-5
    assert abs(_norm(v) - 1.0) < 1e-5


def test_oov_uses_undefined(lib, model):
    # OOV -> row 0 [0, 100, 0, 0] -> normalised [0, 1.0, 0, 0]
    v = _embed(lib, model, "xyz")
    assert abs(v[1] - 1.0) < 1e-5
    assert abs(_norm(v) - 1.0) < 1e-5


def test_empty_input_uses_undefined(lib, model):
    v = _embed(lib, model, "!!!")
    assert abs(v[1] - 1.0) < 1e-5


def test_multi_word_mean_pooling(lib, model):
    # "cat dog" -> mean of rows 1 + 2 -> [1/√2, 0, 1/√2, 0]
    v = _embed(lib, model, "cat dog")
    expected = 1.0 / math.sqrt(2.0)
    assert abs(v[0] - expected) < 1e-4
    assert abs(v[2] - expected) < 1e-4
    assert abs(_norm(v) - 1.0) < 1e-5


def test_output_is_l2_normalised(lib, model):
    v = _embed(lib, model, "cat")
    assert abs(_norm(v) - 1.0) < 1e-5
