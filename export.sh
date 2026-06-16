#!/usr/bin/env bash
# Export trained embeddings to the device-ready binary table.
# Run this after train.sh has completed.
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

# Prefer Python 3.12 (homebrew) over the system Python (Xcode 3.9 is too old
# for some pykeen/class_resolver dependency combinations).
PYTHON=${PYTHON:-$(command -v python3.12 || command -v python3.11 || command -v python3.10 || command -v python3)}

if [ ! -d "venv" ]; then
    echo "Creating venv with $($PYTHON --version)..."
    $PYTHON -m venv venv
fi

# Always sync requirements in case they changed since the venv was created.
venv/bin/pip install -q -r requirements.txt

venv/bin/python -m wordnet_embeddings.export "$@"
