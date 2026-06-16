#!/usr/bin/env bash
# Train WordNet embeddings (TransE, 128-dim, 200 epochs by default).
#
# Quick smoke-test (1 epoch):   ./train.sh --epochs 1
# Full training run (~5-20min): ./train.sh
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

if [ ! -d "venv" ]; then
    echo "Creating venv and installing dependencies..."
    python3 -m venv venv
    venv/bin/pip install -q -r requirements.txt
fi

venv/bin/python -m wordnet_embeddings.train "$@"
