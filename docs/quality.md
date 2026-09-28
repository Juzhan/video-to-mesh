# Geometry, validation and limits

Pipeline: video frame selection → COLMAP camera estimation and stereo → board-based metric alignment → support-plane removal → SAM 2 object segmentation → surface and silhouette candidates → observed free-space carving → quality checks → low-poly and smooth exports. The default route has no learned image-to-3D generative API.

## What is checked

- Camera registration and held-out board alignment controls.
- Visible silhouette agreement: median recall ≥ 0.95, 10th percentile recall ≥ 0.90, median IoU ≥ 0.90, plus a closed, consistently oriented material volume.
- Simplification samples against the high-resolution reconstruction, with bounded geometry and extent changes.
- Reimported final OBJ topology, orientation and volume, including cached and smoothed variants.
- Optional Blender import dimensions, compound collision checks and PyBullet scene tests, only when explicitly enabled. Default OBJ geometry checks do not require Blender or PyBullet.

These are internal checks. A board fitting residual is **not** a whole-object accuracy guarantee; agreement with inferred masks is **not** independent ground truth. Closed topology does not certify absence of all self-intersections or correct physical geometry. Default masses and homogeneous-density inertia are approximations.

## General fixes included

Automatic prompts use coherent elevated measurements rather than scattered floor noise. This assumes a recognizable object body and can miss distant or very thin parts.

A projection outside the camera image is unknown, not background. Each voxel needs at least 12 actual observations; the 97% silhouette agreement threshold uses those observations as its denominator. A one-pixel rasterization allowance is used during volume construction; validation uses the original undilated masks.

Cup/bowl mode requires an observed opening. Material walls may be watertight while leaving the opening accessible; the program does not simply cap the rim. Hidden underside detail and unobserved cavities cannot be recovered reliably.

## Validation status

Local real captures of a keyboard, small box and toy aircraft have completed the fast pipeline after the above fixes. Their personal source videos and meshes are not distributed here. Fine key gaps, aircraft fins, landing gear, reflective parts and thin edges remain approximate. Previous synthetic experiments also exposed substantial cavity-floor and small-feature errors; do not treat the examples as broad accuracy certification.

Fast mode targets practical throughput but does not promise a fixed per-object time. Report cold runs separately from cache resumes, separate collision decomposition from OBJ-only timing, and always state hardware and image settings. A failed item remains failed rather than receiving a silently relaxed quality threshold or global convex replacement.
