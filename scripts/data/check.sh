#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")/../.."
if [ -n "${IT3160_PYTHON:-}" ]; then
    data_python="$IT3160_PYTHON"
elif [ -x .venv/bin/python ]; then
    data_python=.venv/bin/python
else
    data_python=python3
fi
"$data_python" -m pytest "$@"
"$data_python" -m ruff check src tests scripts
