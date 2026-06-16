#!/usr/bin/env bash
# Export trained embeddings to the device-ready binary table.
# Run this after train.sh has completed.
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

if [ ! -d "venv" ]; then
    echo "Creating venv and installing dependencies..."
    python3 -m venv venv
    venv/bin/pip install -q -r requirements.txt
fi

venv/bin/python -m wordnet_embeddings.export "$@"
