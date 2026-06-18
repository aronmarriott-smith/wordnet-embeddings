#!/usr/bin/env bash
# Benchmark the exported model against MTEB's English STS task family.
# Run bin/export_sentence_transformer.sh first.
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON=${PYTHON:-$(command -v python3.12 || command -v python3.11 || command -v python3.10 || command -v python3)}

if [ ! -d "venv" ]; then
    echo "Creating venv with $($PYTHON --version)..."
    $PYTHON -m venv venv
fi

VENV_BIN=venv/bin
[ -f "$VENV_BIN/python" ] || VENV_BIN=venv/Scripts

"$VENV_BIN/pip" install -q -r requirements-dev.txt
"$VENV_BIN/python" -m wordnet_embeddings.benchmark "$@"
