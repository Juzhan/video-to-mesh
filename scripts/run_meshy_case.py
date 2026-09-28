"""Compatibility entry point; prefer packing-assets meshy --help."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from packing_assets.meshy_workflow import reference_signature  # legacy import
from packing_assets.cli import main

if __name__ == "__main__":
    sys.argv.insert(1, "meshy")
    main()
