# Changelog

## Unreleased

- Prepare public documentation with a real video-to-metric-mug GIF/result example, a checked sample OBJ, aligned Chinese/English backend defaults and an explicit demo-media packaging allowlist. Ignore the local `om/` output tree.

- Preserve dense Meshy frontend routing on retries and freeze submitted mask caches. A hybrid retry no longer overwrites paid dense input masks with sparse masks and fails with a changed metric-reference error. Local measurement keeps its independent upgrade path.

- Hybrid outputs also include an admitted `_local.obj` when dimension fitting reports an error (including a greater-than-10% discrepancy), even if the original Meshy silhouette passes. Harmless rejected refinements do not trigger this fallback; original models and review labels remain unchanged.

- Retained hybrid Meshy outputs now include a separate low-poly `_local.obj` backup when their local measurement produced an admitted closed reconstruction. Named exports, metadata and ZIP packaging preserve both candidates and retain the original review status.

- Added a separate phone-job recovery worker with priority plans, SQLite ownership, restart isolation, original-delivery backups and persistent geometry overrides. Hybrid upgrades now refit cached Meshy geometry against dense metric references and validate with recomputed local masks, preserving paid inputs and reporting unresolved shape failures.

- Added an isolated, optional Hunyuan3D-2mv Turbo/standard experiment with pinned model downloads and checksums, calibrated RGBA inputs, offline GPU inference, metric OBJ export and preserved failure reports. RTX 5080 trials cover bottle, keyboard and plane; current shape/dimension/topology failures are documented. This is not yet a production phone/batch backend.

- Meshy now defaults to hybrid local dimension measurement with reusable camera/depth checkpoints, isolated measurement work and no second paid generation. Phone upload offers hybrid, fast photo calibration and local modes. Existing fast provider inputs remain immutable. Local dimension constraints allow at most about 0.5% photo-guided adjustment.

- Added separate low-poly Meshy dimension variants with bounded axis scaling, held-out silhouette/boundary checks, verified same-video local reconstruction dimensions when available, cache invalidation and original-model preservation. No new provider task is required.

- Recover insufficient sparse Meshy foreground with dense prompts automatically, preserving camera/A4 caches and skipping local surface reconstruction. Phone jobs no longer require a manual full-frontend rerun for this failure.

- Added a phone-friendly LAN upload portal (`run-upload.sh`) with native-camera/file selection, 4 MiB resumable chunks, progress/speed, paired access, durable single-worker queue, per-object output downloads and retry using existing reconstruction/API caches. No cloud relay or new web framework dependencies.

- Meshy now defaults to a sparse metric frontend, skipping dense stereo, fusion and local final surfaces. Sparse geometry is used only to prompt SAM and initialize uniform metric fitting; delivered shape remains Meshy. Keep the previous route with `--meshy-frontend full`.
- Meshy smoothing and the second Blender scene are now opt-in (`--with-smooth`). Existing paid generations can migrate to the fast branch only after verifying original video/board fingerprints, preserving exact submitted crops and task state.

- Retain Meshy generations that fail shape checks: export closed metric candidates as `retained_unvalidated` with original quality results and Blender labels; retain unscaled raw GLB/OBJ even if later fitting fails. Unvalidated candidates remain visual-only. Added real-video timing breakdown and explicit fast-frontend limitations.

- Check required Meshy credentials before expensive reconstruction. Added `configure-meshy --key-file` to remember a credential file path in user configuration without copying its contents; completed provider caches remain usable offline.

- Publish each object's final OBJ/smoothed model immediately after its reconstruction and optional Meshy branch, before starting the next object. Incremental status/index updates preserve completed models on later failure/interruption; shared scenes and ZIPs are built once at the end.

- Added `--objects` selection by filename or unambiguous object name; default processes all. Selected runs reuse common caches and keep their delivery/reports separate from full-batch results.

- Default installation and `run`/`meshy` output now require neither Blender nor PyBullet. Opt in to display with `--with-blender`, collision/URDF export with `--with-collision`, or collision export plus simulation with `--pybullet`. Geometry checks remain enabled.
- Decoupled low-poly geometry export from the PyBullet exporter. Legacy `--obj-only`/`--no-blender` remain valid; `run-a4-fast.sh` remains an explicit PyBullet entry point.

- Integrated optional `run --backend local|meshy`, per-object backend settings, and a `meshy` CLI for existing metric reconstructions. The default remains local; Meshy is an explicit paid branch with no silent fallback.
- Meshy progress stages, per-case reports, provider cache resume, interrupted-stage recovery and final cached-geometry audits. Existing script remains a compatibility entry point.
- Published the synthetic pump benchmark summary and documented incomplete fast-frontend/interior validation. API time is not presented as end-to-end video latency.

## 0.2.0

- Live per-phase progress with current input, cached stages, actual frame/view counters and elapsed time. Native/Python algorithm output is captured in `run.log`; `progress.json` updates during long native calls.
- Source-only project layout, MIT license, third-party notices, CPU CI and reproducible source packaging. Historical experiments and private inputs/results remain local and ignored.
- Coherent geometry prompts avoid selecting surrounding floor/reference paper.
- Out-of-frame silhouette projections are treated as unknown rather than background, preserving partially cropped objects.
- Final OBJ and smooth variants are reimported for topology/volume checks, including cached exports; raw diagnostic surfaces are clearly separated.

All geometric outputs remain review candidates. These fixes do not certify real dimensions, hidden surfaces or thin-feature accuracy.
