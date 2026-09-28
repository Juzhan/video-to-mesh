#!/usr/bin/env bash
set -euo pipefail
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [ "$#" -lt 2 ]; then
  echo "Usage: $0 INPUT_FOLDER OUTPUT_FOLDER [SETTINGS_JSON] [--with-blender]"
  echo "A4 fast profile: up to 48 views, 640-pixel stereo; geometry still needs review."
  exit 2
fi
args=(run --input "$1" --out "$2" --board "$project_root/calibration/a4-dense/board.json" --quality fast --pybullet)
shift 2
if [ "$#" -gt 0 ] && [[ "$1" != --* ]]; then args+=(--settings "$1"); shift; fi
args+=("$@")
exec "$project_root/.venv/bin/packing-assets" "${args[@]}"
