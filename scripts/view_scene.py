"""Portable PyBullet scene viewer. Place next to scene.json and assets/."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pybullet as p
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--smooth", action="store_true")
    parser.add_argument(
        "--scene", type=Path, default=Path(__file__).resolve().parent / "scene.json"
    )
    args = parser.parse_args()
    root = args.scene.resolve().parent
    scene = json.loads(args.scene.read_text())
    if not scene["items"]:
        raise ValueError("scene has no exported items")
    cid = p.connect(p.DIRECT if args.headless else p.GUI)
    try:
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1 / 240)
        p.setPhysicsEngineParameter(numSolverIterations=100)
        floor = p.createMultiBody(0, p.createCollisionShape(p.GEOM_PLANE))
        bodies = []
        colors = [
            [0.13, 0.42, 0.67, 1],
            [0.78, 0.35, 0.13, 1],
            [0.22, 0.58, 0.39, 1],
            [0.53, 0.32, 0.69, 1],
        ]
        for index, item in enumerate(scene["items"]):
            position = np.array(item["position_m"]) + [0, 0, 0.025]
            urdf = (
                (item.get("smooth_urdf") or item["urdf"])
                if args.smooth
                else item["urdf"]
            )
            body = p.loadURDF(
                str(root / urdf),
                basePosition=position,
                globalScaling=1,
                flags=p.URDF_USE_INERTIA_FROM_FILE,
            )
            p.changeVisualShape(body, -1, rgbaColor=colors[index % len(colors)])
            p.changeDynamics(
                body, -1, lateralFriction=0.5, restitution=0, collisionMargin=0.001
            )
            shapes = len(p.getCollisionShapeData(body, -1))
            if shapes != item["collision_parts"]:
                raise AssertionError(
                    f"{item['id']}: expected {item['collision_parts']} shapes, got {shapes}"
                )
            bodies.append((body, item))
        # Probe actual collision geometry before falling, along observed cavity columns.
        cavity_checks = []
        sphere = p.createMultiBody(
            0, p.createCollisionShape(p.GEOM_SPHERE, radius=0.002)
        )
        for body, item in bodies:
            for column in item.get("cavity", {}).get("probe_columns", []):
                for z in np.linspace(
                    column["floor_z_m"] + 0.008, column["rim_z_m"] + 0.006, 6
                ):
                    point = (
                        np.r_[column["xy_m"], z]
                        + np.array(item["position_m"])
                        + [0, 0, 0.025]
                    )
                    p.resetBasePositionAndOrientation(sphere, point, [0, 0, 0, 1])
                    blocked = bool(p.getClosestPoints(sphere, body, distance=0))
                    cavity_checks.append({"id": item["id"], "clear": not blocked})
        p.removeBody(sphere)
        if any(not c["clear"] for c in cavity_checks):
            raise AssertionError("collision geometry blocks a required interior probe")
        bounds = np.array(
            [np.array(item["bounds_m"]) + item["position_m"] for item in scene["items"]]
        )
        lo, hi = bounds[:, 0].min(0), bounds[:, 1].max(0)
        target = (lo + hi) / 2
        span = max(hi - lo)
        p.resetDebugVisualizerCamera(max(span * 1.5, 0.6), 25, -50, target)
        # 10 cm ground grid and labels are also visible in GUI.
        for x in np.arange(-0.1, hi[0] + 0.2, 0.1):
            p.addUserDebugLine(
                [float(x), -0.1, 0.001],
                [float(x), float(hi[1] + 0.1), 0.001],
                [0.5, 0.5, 0.5],
            )
        for y in np.arange(-0.1, hi[1] + 0.2, 0.1):
            p.addUserDebugLine(
                [-0.1, float(y), 0.001],
                [float(hi[0] + 0.1), float(y), 0.001],
                [0.5, 0.5, 0.5],
            )
        for body, item in bodies:
            anchor = np.array(item["position_m"]) + [
                0,
                0,
                item["dimensions_m"][2] + 0.035,
            ]
            p.addUserDebugText(item["id"], anchor, [0.15, 0.15, 0.15], textSize=1)
        if not args.headless:
            while p.isConnected():
                p.stepSimulation()
                time.sleep(1 / 240)
            return
        for _ in range(2400):
            p.stepSimulation()
        results = []
        for body, item in bodies:
            position, orientation = p.getBasePositionAndOrientation(body)
            linear, angular = p.getBaseVelocity(body)
            contacts = p.getContactPoints(body, floor)
            good = bool(
                contacts
                and np.isfinite(position).all()
                and np.linalg.norm(linear) < 0.015
                and np.linalg.norm(angular) < 0.1
                and min(c[8] for c in contacts) > -0.005
            )
            results.append(
                {
                    "id": item["id"],
                    "passed": good,
                    "floor_contacts": len(contacts),
                    "linear_speed_m_s": float(np.linalg.norm(linear)),
                    "angular_speed_rad_s": float(np.linalg.norm(angular)),
                    "position_m": position,
                    "orientation_xyzw": orientation,
                    "max_contact_penetration_m": max(
                        0, -min((c[8] for c in contacts), default=0)
                    ),
                    "collision_parts": len(p.getCollisionShapeData(body, -1)),
                }
            )
        camera = target + np.array([0.6, -1, 1.2]) * max(span, 0.4)
        view = p.computeViewMatrix(camera, target, [0, 0, 1])
        projection = p.computeProjectionMatrixFOV(45, 1.5, 0.01, 10)
        _, _, rgba, _, _ = p.getCameraImage(
            1200, 800, view, projection, renderer=p.ER_TINY_RENDERER
        )
        Image.fromarray(np.asarray(rgba, dtype="uint8").reshape(800, 1200, 4)).save(
            root / "pybullet-scene.png"
        )
        passed = all(r["passed"] for r in results)
        report = {
            "status": "passed" if passed else "failed",
            "simulated_seconds": 10,
            "items": results,
            "cavity_probes": cavity_checks,
            "friction": 0.5,
            "mass_sources": {i["id"]: i["mass_source"] for i in scene["items"]},
            "scope": "scale=1 import, per-body compound shape count, floor contact and settling; not measured material dynamics",
        }
        (root / "physics-check.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        if not passed:
            raise SystemExit(1)
    finally:
        if p.isConnected(cid):
            p.disconnect(cid)


if __name__ == "__main__":
    main()
