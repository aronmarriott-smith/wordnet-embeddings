#!/usr/bin/env bash
# Train WordNet embeddings (TransE, see config.py for hyperparameters).
#
# Quick smoke-test (1 epoch):        ./bin/train.sh --epochs 1
# Full training run:                 ./bin/train.sh
# With evaluation (GPU recommended): ./bin/train.sh --evaluate
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON=${PYTHON:-$(command -v python3.12 || command -v python3.11 || command -v python3.10 || command -v python3)}

if [ ! -d "venv" ]; then
    echo "Creating venv with $($PYTHON --version)..."
    $PYTHON -m venv venv
fi

VENV_BIN=venv/bin
[ -f "$VENV_BIN/python" ] || VENV_BIN=venv/Scripts

"$VENV_BIN/pip" install -q -r requirements.txt
"$VENV_BIN/python" -m wordnet_embeddings.train "$@"
