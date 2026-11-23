#!/usr/bin/env bash
# Only writes inside this project. Never calls sudo, conda install, or global pip.
set -euo pipefail
cd -- "$(dirname -- "$0")"
python_bin="${WALLBOY_PYTHON:-python3}"
"$python_bin" -m venv .venv
.venv/bin/python -m pip install --cache-dir .cache/pip torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install --cache-dir .cache/pip -r requirements.txt
printf '\nReady. Run: ./run.sh\n'
