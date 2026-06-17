#!/usr/bin/env bash
# Run Python ctypes tests for the C inference engine.
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
"$VENV_BIN/pytest" tests/ -v "$@"
