#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
uv sync --quiet
source .venv/bin/activate
python train.py "$@"
