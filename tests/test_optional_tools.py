import json
import subprocess
import sys
from pathlib import Path

import pytest

from packing_assets import cli


@pytest.mark.parametrize(
    "flags,render,obj_only,physics",
    [
        ([], False, True, False),
        (["--with-blender"], True, True, False),
        (["--with-collision"], False, False, False),
        (["--pybullet", "--with-blender"], True, False, True),
        (["--no-blender", "--obj-only"], False, True, False),
    ],
)
def test_cli_output_options_are_explicit(
    monkeypatch, tmp_path, flags, render, obj_only, physics
):
    def run(*args, **kwargs):
        assert args[5] is render and args[6] is obj_only
        assert kwargs["physics_test"] is physics
        return {"status": "complete_needs_review", "exported": 1, "failed": 0}

    monkeypatch.setattr("packing_assets.automation.run_automatic", run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "packing-assets",
            "run",
            "--input",
            str(tmp_path),
            "--board",
            "board.json",
            "--out",
            str(tmp_path / "out"),
            *flags,
        ],
    )
    cli.main()


def test_obj_simplification_and_smoothing_without_display_or_physics(tmp_path):
    pytest.importorskip("open3d")
    code = r"""
import importlib.abc, sys
class Unavailable(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'pybullet', 'bpy', 'coacd', 'shapely'} or fullname == 'packing_assets.pybullet_asset':
            raise ModuleNotFoundError('optional dependency deliberately unavailable: '+fullname)
sys.meta_path.insert(0, Unavailable())
from pathlib import Path
import subprocess, trimesh, json
from packing_assets.assets import export_asset
from packing_assets.visual_batch import assemble_visual
def forbidden(*args, **kwargs):
    raise AssertionError('OBJ export must not invoke Blender or other subprocesses')
subprocess.run = forbidden
out = Path(sys.argv[1])
source = out/'source'
export_asset(trimesh.creation.icosphere(subdivisions=5, radius=.08), source, {})
report = assemble_visual([{'id':'ball','asset':str(source)}], out/'delivery')
assert report['status']=='complete_needs_review', report
assert report['blender']=='skipped' and report['physics_test']=='not_requested'
status=json.loads((out/'delivery/models/ball/status.json').read_text())
for variant in status['exported_mesh_checks'].values():
    assert variant['closed_volume'] and variant['faces']<20480
assert not list((out/'delivery').rglob('*.urdf'))
assert not list((out/'delivery').glob('*.blend'))
print('OBJ, smooth, simplification and volume audit succeeded without optional tools')
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_setup_default_has_no_optional_install(tmp_path):
    # Run the installer entry point with stubs: inspect commands without installing anything.
    root = tmp_path / "project"
    (root / "scripts").mkdir(parents=True)
    (root / ".venv/bin").mkdir(parents=True)
    from shutil import copy2

    copy2(
        Path(__file__).resolve().parents[1] / "scripts/setup_all.sh",
        root / "scripts/setup_all.sh",
    )
    quality = root / "scripts/setup_quality.sh"
    quality.write_text('#!/bin/sh\nprintf "quality-setup\\n"\n')
    quality.chmod(0o755)
    pip = root / ".venv/bin/pip"
    pip.write_text('#!/bin/sh\nprintf "%s\\n" "$*"\n')
    pip.chmod(0o755)
    default = subprocess.run(
        ["bash", str(root / "scripts/setup_all.sh")],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "install -e" not in default.stdout
    optional = subprocess.run(
        ["bash", str(root / "scripts/setup_all.sh"), "--with-pybullet"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "install -e .[pybullet]" in optional.stdout
