#!/usr/bin/env bash
set -euo pipefail
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
bind_host=${PACKING_UPLOAD_HOST:-127.0.0.1}
if [ "$bind_host" = "127.0.0.1" ] && ip -4 addr show 2>/dev/null | rg -q 'inet 10\.42\.0\.1/'; then
  bind_host=10.42.0.1
fi
exec "$project_root/.venv/bin/python" -m packing_assets.upload_server --host "$bind_host" "$@"
