#Requires -Version 5.1
<#
.SYNOPSIS
    Cross-compiles engine/libembed.so (a Windows DLL) using a Linux Docker
    container, instead of native Windows MinGW.

.DESCRIPTION
    Avoids the native-Windows MinGW toolchain/environment issues this project
    has hit repeatedly (PATH loss, temp-directory resolution failures). Builds
    a small Debian + mingw-w64 image (engine/Dockerfile.windows) and runs the
    cross-compile with engine/ bind-mounted, so output lands directly at
    engine/libembed.so on the host — no image rebuild needed for source
    changes.

    Only builds the library (`make lib`), not `make test` — the resulting
    .exe is a Windows binary and can't run inside the Linux container.
    Verify it for real via `bin/test.sh` on Windows, or pytest tests/.

.EXAMPLE
    .\bin\build_windows.ps1
#>

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$Image = "wordnet-embeddings-windows-cross"

docker build -t $Image -f engine/Dockerfile.windows engine
if ($LASTEXITCODE -ne 0) { throw "docker build failed" }

docker run --rm -v "${PWD}/engine:/engine" -w /engine $Image `
    make CC=x86_64-w64-mingw32-gcc clean lib
if ($LASTEXITCODE -ne 0) { throw "cross-compile failed" }

Write-Output "Built engine/libembed.so (Windows DLL) via Docker cross-compile."
