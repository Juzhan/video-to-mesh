"""Offline Meshy input preparation; no upload or credits."""

import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from packing_assets.meshy_views import prepare_meshy_views

p = argparse.ArgumentParser()
p.add_argument("--result", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
r = prepare_meshy_views(a.result, a.out)
print(
    f"Prepared {len(r['views'])} views; no API submission. Preview: {a.out.resolve()}/preview.jpg"
)
