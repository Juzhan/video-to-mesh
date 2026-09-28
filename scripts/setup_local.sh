#!/usr/bin/env bash
set -euo pipefail
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_root"
python3 -m venv .venv
# The CPU and CUDA distributions expose the SAME Python module and must not coexist.
if .venv/bin/python -m pip show pycolmap >/dev/null 2>&1; then
  .venv/bin/python -m pip uninstall -y pycolmap
  .venv/bin/python -m pip install --timeout 180 --force-reinstall --no-deps pycolmap-cuda12==4.2.0
fi
.venv/bin/python -m pip install --timeout 180 -e '.[reconstruction]'
.venv/bin/python -c 'import pycolmap; assert pycolmap.has_cuda; print("COLMAP", pycolmap.__version__, "CUDA ready")'
echo 'Ready: .venv/bin/packing-assets reconstruct --help'
