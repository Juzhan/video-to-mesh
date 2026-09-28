# Development and release preparation

Use the repository root as the working directory. `setup_all.sh` creates an editable local installation and downloads pinned SAM code/weights into ignored directories. Only the explicitly curated mug GIF and result assets are included; full captures and pretrained weights are excluded.

## Tests

```bash
.venv/bin/pytest -q  # full locally installed environment, including optional libraries
```

The default user setup does not install test, collision or simulation extras. Developers can install `.[test,collision,evaluation,pybullet]` in the reconstruction environment. Blender is needed only for synthetic rendering and independent Blender audits, not ordinary OBJ export. `tests/test_optional_tools.py` verifies geometry export while display/simulation imports and subprocess launches are blocked.

The GitHub Actions CPU job installs only `.[test]` and runs the small portable subset:

```bash
python -m pytest -q tests/test_pipeline.py tests/test_metric.py \
  tests/test_folder_batch.py tests/test_fast_capture.py \
  tests/test_dense_board.py tests/test_progress.py tests/test_meshy_branch.py \
  tests/test_meshy_case.py tests/test_meshy_views.py
```

This CI job does **not** validate CUDA stereo, download SAM, call paid APIs or certify geometry. Full tests additionally need the reconstruction, collision, evaluation and PyBullet extras plus installed SAM. Keep tests deterministic and use procedural fixtures.

## Synthetic captures

Blender scripts are included; no truth mesh is passed into the RGB reconstruction path:

```bash
blender -b --python scripts/render_a4_table_cases.py -- \
  --out outputs/synthetic-cases --cases keyboard
./run-videos.sh outputs/synthetic-cases/input outputs/synthetic-models
```

The example generator produces photo folders, not phone footage. `evaluation_only/` holds truth for separate evaluation, outside the input folders. It can take several minutes. The board texture is shipped in `calibration/a4-dense/board-preview.png`.

`check_blender_mesh.py` independently reimports OBJ files, writes a BMesh topology audit and renders the underside of the last file:

```bash
blender -b --python scripts/check_blender_mesh.py -- \
  --mesh output_me/models/keyboard/keyboard.obj \
  --out output_me/verification/keyboard
```

## Source-only release

```bash
python scripts/package_source.py --check
python scripts/package_source.py
```

The explicit source allowlist includes code, tests, documentation, config and calibration resources. Except for the explicitly allowlisted mug GIF/OBJ and documented result image, captures, generated objects, logs, credentials, downloaded dependencies/weights and local experiment history are excluded. The packager rejects symlinks, oversized files, obvious credentials and local absolute paths in the selected source files. It does not upload anything.

Before publishing, inspect `git status` and `git diff --cached`, run the relevant tests, and inspect the ZIP contents. Add your own GitHub remote; no remote URL or hosting credentials are embedded. CI execution on GitHub itself still needs the repository to be published by its owner.

`local-history/` on the development machine preserves old experiments and a move manifest. It is ignored, not part of the source release; historical scripts may reference the old layout.
