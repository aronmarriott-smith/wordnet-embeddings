#!/usr/bin/env bash
# Cross-compiles engine/libembed.so (a Windows DLL) using a Linux Docker
# container, instead of native Windows MinGW. See build_windows.ps1's
# docstring for why; this is the same thing for bash (e.g. CI, or running
# from Linux/macOS to produce a Windows binary).
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."

IMAGE=wordnet-embeddings-windows-cross

docker build -t "$IMAGE" -f engine/Dockerfile.windows engine

# MSYS_NO_PATHCONV: Git Bash auto-converts /engine (a container-side path)
# into a Windows host path otherwise, breaking -w/-v's destination side.
MSYS_NO_PATHCONV=1 docker run --rm -v "$(pwd)/engine:/engine" -w /engine "$IMAGE" \
    make CC=x86_64-w64-mingw32-gcc clean lib

echo "Built engine/libembed.so (Windows DLL) via Docker cross-compile."
