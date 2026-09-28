#!/usr/bin/env bash
set -euo pipefail
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_root"
with_collision=false
with_pybullet=false
for option in "$@"; do
  case "$option" in
    --with-collision) with_collision=true ;;
    --with-pybullet) with_pybullet=true ;;
    --help)
      echo 'Usage: ./scripts/setup_all.sh [--with-collision] [--with-pybullet]'
      echo 'Default: model reconstruction/OBJ only. Blender is an optional external application.'
      exit 0 ;;
    *) echo "Unknown option: $option" >&2; exit 2 ;;
  esac
done
./scripts/setup_quality.sh
if "$with_pybullet"; then
  .venv/bin/pip install -e '.[pybullet]'
elif "$with_collision"; then
  .venv/bin/pip install -e '.[collision]'
fi
.venv/bin/pip check
echo 'Ready: ./run-videos.sh INPUT_FOLDER [OUTPUT_FOLDER] [SETTINGS_JSON]'
echo 'Optional: --with-blender for scenes; --with-collision for URDF; --pybullet for simulation check.'
