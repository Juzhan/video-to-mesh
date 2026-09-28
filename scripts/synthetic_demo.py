"""End-to-end metric integration fixture: synthetic photos -> ArUco -> COLMAP poses -> scaled mesh.

This validates transforms and unit conventions, NOT physical capture/reconstruction accuracy.
No photogrammetric estimation or generated object shape is involved in this fixture.
"""

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

from packing_assets.board import create_board
from packing_assets.metric import transform_points
from packing_assets.pipeline import run_batch


def main(out):
    out.mkdir(parents=True, exist_ok=True)
    create_board(out / "board")
    board_image = cv2.imread(str(out / "board/board-preview.png"))
    images = out / "images"
    model = out / "sparse"
    images.mkdir(exist_ok=True)
    model.mkdir(exist_ok=True)
    width, height = 1600, 1200
    fx = fy = 1400
    cx, cy = (width - 1) / 2, (height - 1) / 2
    k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], float)
    # Ground truth maps arbitrary reconstruction units to meters.
    s = 0.071
    rotation = Rotation.from_euler("xyz", [21, 33, 48], degrees=True).as_matrix()
    offset = np.array([0.5, -0.2, 0.13])
    truth = np.eye(4)
    truth[:3, :3] = s * rotation
    truth[:3, 3] = offset
    lines = []
    # Warp the flat calibration board from metric coordinates into eight perspective views.
    corners = np.array(
        [
            [-0.210, 0.1485, 0],
            [0.210, 0.1485, 0],
            [0.210, -0.1485, 0],
            [-0.210, -0.1485, 0],
        ]
    )
    source_pixels = np.float32([[0, 0], [1680, 0], [1680, 1188], [0, 1188]])
    for index, angle in enumerate(np.linspace(0, 2 * np.pi, 8, endpoint=False), 1):
        center = np.array([0.23 * np.cos(angle), 0.23 * np.sin(angle), 0.65])
        forward = -center / np.linalg.norm(center)
        right = np.cross(forward, [0, 1, 0])
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        r = np.stack([right, down, forward])
        t = -r @ center
        cam = corners @ r.T + t
        pixels = cam @ k.T
        pixels = pixels[:, :2] / pixels[:, 2:]
        h = cv2.getPerspectiveTransform(source_pixels, pixels.astype(np.float32))
        image = cv2.warpPerspective(
            board_image, h, (width, height), borderValue=(235, 235, 235)
        )
        name = f"view_{index:02d}.png"
        cv2.imwrite(str(images / name), image)
        r_source = r @ rotation
        t_source = (r @ offset + t) / s
        qx, qy, qz, qw = Rotation.from_matrix(r_source).as_quat()
        fields = [index, qw, qx, qy, qz, *t_source, 1, name]
        lines.append(" ".join(map(str, fields)) + "\n\n")
    (model / "cameras.txt").write_text(
        f"1 PINHOLE {width} {height} {fx} {fy} {cx + 0.5} {cy + 0.5}\n"
    )
    (model / "images.txt").write_text("".join(lines))
    (model / "points3D.txt").write_text("")
    mesh = trimesh.creation.box([0.12, 0.065, 0.19])
    mesh.apply_translation([0, 0, 0.095])
    mesh.apply_transform(np.linalg.inv(truth))
    mesh.export(out / "source.obj")
    manifest = out / "jobs.json"
    manifest.write_text(
        json.dumps(
            {
                "objects": [
                    {
                        "id": "synthetic_bottle_box",
                        "backend": "colmap_mesh",
                        "mesh": "source.obj",
                        "colmap_model": "sparse",
                        "images_dir": "images",
                        "board": "board/board.json",
                        "expected_extents_m": [0.12, 0.065, 0.19],
                        "collision": False,
                    }
                ]
            },
            indent=2,
        )
    )
    results = run_batch(manifest, out / "results")
    if results[0]["status"] == "failed":
        raise RuntimeError(results[0]["error"])
    report = json.loads((out / "results/synthetic_bottle_box/asset.json").read_text())
    errors = np.abs(np.array(report["extents_m_xyz_z_up"]) - [0.12, 0.065, 0.19]) * 1000
    assert errors.max() < 1.0, errors
    print(
        json.dumps(
            {
                "fixture": "synthetic_only",
                "true_dimensions_mm": [120, 65, 190],
                "recovered_dimensions_mm": (
                    np.array(report["extents_m_xyz_z_up"]) * 1000
                ).tolist(),
                "max_dimension_error_mm": float(errors.max()),
                "held_out_board_rms_mm": report["provenance"]["calibration"][
                    "held_out_rms_mm"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    import sys

    main(Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/synthetic"))
