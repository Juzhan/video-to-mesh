import json
from pathlib import Path

import cv2
import numpy as np


def dictionary():
    return cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


def create_board(out: Path, scale: float = 1.0):
    """A3 vector print, a preview, and authoritative metric corner coordinates."""
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("board scale must be positive")
    out.mkdir(parents=True, exist_ok=True)
    positions = [(x, y) for y in (15, 250) for x in (15, 85, 155, 225, 295, 373)]
    positions += [(x, y) for x in (15, 373) for y in (93, 172)]
    side = 32
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{420 * scale}mm" height="{297 * scale}mm" viewBox="0 0 420 297">',
        '<rect width="420" height="297" fill="white"/>',
    ]
    preview = np.full((1188, 1680), 255, np.uint8)
    markers = {}
    for idx, (x, y) in enumerate(positions):
        bits = cv2.aruco.generateImageMarker(dictionary(), idx, 6)
        for row, col in np.argwhere(bits == 0):
            svg.append(
                f'<rect x="{x + col * side / 6:.6f}" y="{y + row * side / 6:.6f}" width="{side / 6:.6f}" height="{side / 6:.6f}" fill="black"/>'
            )
        preview[y * 4 : (y + side) * 4, x * 4 : (x + side) * 4] = (
            cv2.aruco.generateImageMarker(dictionary(), idx, side * 4)
        )
        # OpenCV corner order: TL, TR, BR, BL. Board X right, Y up, Z out of paper.
        corners = np.array(
            [[x, y, 0], [x + side, y, 0], [x + side, y + side, 0], [x, y + side, 0]],
            float,
        )
        corners[:, 0] -= 210
        corners[:, 1] = 148.5 - corners[:, 1]
        markers[str(idx)] = (corners * scale / 1000).tolist()
    svg.extend(
        [
            '<rect x="58" y="48.5" width="304" height="200" fill="none" stroke="#bbb" stroke-dasharray="2 2"/>',
            '<text x="210" y="134" text-anchor="middle" font-size="5" fill="#aaa">OBJECT AREA - keep markers visible</text>',
            '<text x="210" y="143" text-anchor="middle" font-size="3.5" fill="#aaa">X right / Y up on paper / Z above board</text>',
            '<path d="M160 220h100 M160 217v6 M260 217v6" stroke="black" stroke-width="0.3"/>',
            f'<text x="210" y="216" text-anchor="middle" font-size="3.5">CHECK: {100 * scale:g} mm - PRINT AT 100%</text>',
            "</svg>",
        ]
    )
    (out / "board.svg").write_text("\n".join(svg))
    cv2.imwrite(str(out / "board-preview.png"), preview)
    spec = {
        "version": 1,
        "dictionary": "DICT_4X4_50",
        "units": "m",
        "page_mm": [420 * scale, 297 * scale],
        "marker_side_m": side * scale / 1000,
        "markers": markers,
        "suggested_roi_m": [
            [-0.150 * scale, -0.100 * scale, 0],
            [0.150 * scale, 0.100 * scale, 0.350 * scale],
        ],
        "print_note": "Print board.svg at 100%; verify horizontal and vertical distances once. PNG is a detection preview, not a calibrated print.",
    }
    (out / "board.json").write_text(json.dumps(spec, indent=2))
    return spec


def detect(image):
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary(), params)
    corners, ids, _ = detector.detectMarkers(image)
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2).astype(float) for i, c in zip(ids.ravel(), corners)}


def create_dense_a4(out: Path):
    """48 unique 20 mm ArUco codes; occluded IDs are simply not observations."""
    out.mkdir(parents=True, exist_ok=True)
    width, height, side = 297.0, 210.0, 20.0
    positions = [
        (9.0 + 37.0 * col, 9.0 + 34.4 * row) for row in range(6) for col in range(8)
    ]
    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="297mm" height="210mm" viewBox="0 0 297 210">',
        '<rect width="297" height="210" fill="white"/>',
    ]
    markers = {}
    for idx, (x, y) in enumerate(positions):
        bits = cv2.aruco.generateImageMarker(dictionary(), idx, 6)
        for row, col in np.argwhere(bits == 0):
            svg.append(
                f'<rect x="{x + col * side / 6:.8f}" y="{y + row * side / 6:.8f}" width="{side / 6:.8f}" height="{side / 6:.8f}" fill="black"/>'
            )
        corners = np.array(
            [[x, y, 0], [x + side, y, 0], [x + side, y + side, 0], [x, y + side, 0]],
            float,
        )
        corners[:, 0] -= width / 2
        corners[:, 1] = height / 2 - corners[:, 1]
        markers[str(idx)] = (corners / 1000).tolist()
    svg.append("</svg>")
    (out / "board.svg").write_text("\n".join(svg))
    spec = {
        "version": 1,
        "layout": "dense_a4_48",
        "dictionary": "DICT_4X4_50",
        "units": "m",
        "page_mm": [width, height],
        "marker_side_m": side / 1000,
        "markers": markers,
        "suggested_roi_m": [[-0.7, -0.7, 0], [0.7, 0.7, 0.5]],
        "proxy_crop_to_observed": True,
        "print_note": "One A4 landscape page, actual size 100%; never fit/shrink. Use matching board.json. Keep visible codes spread across both axes.",
        "placement_note": "Object may cover some codes or rest beside the paper on the same flat support, within the scan region. Board and object must remain rigidly fixed.",
    }
    (out / "board.json").write_text(json.dumps(spec, indent=2))
    return spec
