#!/usr/bin/env bash
set -euo pipefail
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [ "$#" -lt 2 ]; then
  echo "Usage: $0 INPUT_FOLDER OUTPUT_FOLDER [SETTINGS_JSON]"
  echo "Uses the one-page A4 dense 48-marker board."
  exit 2
fi
args=(run --input "$1" --out "$2" --board "$project_root/calibration/a4-dense/board.json")
shift 2
if [ "$#" -gt 0 ] && [[ "$1" != --* ]]; then args+=(--settings "$1"); shift; fi
args+=("$@")
exec "$project_root/.venv/bin/packing-assets" "${args[@]}"
