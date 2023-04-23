#!/usr/bin/env bash
set -Eeuo pipefail
source /webui/venv/bin/activate
exec python ./launch.py "$@"
