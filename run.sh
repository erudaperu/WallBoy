#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
if [[ ! -x .venv/bin/python ]]; then
    printf 'Run ./setup.sh once to create the isolated environment.\n' >&2
    exit 1
fi
exec .venv/bin/python -m wallboy "$@"
