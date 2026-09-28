"""Headless contact/scale smoke test; example mass/friction are not measurements."""

import argparse, json, sys
from pathlib import Path
import numpy as np
import pybullet as p
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from packing_assets.assets import load_mesh

parser = argparse.ArgumentParser()
parser.add_argument("--package", type=Path, required=True)
args = parser.parse_args()
root = args.package.resolve()
cid = p.connect(p.DIRECT)
try:
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1 / 240)
    p.setPhysicsEngineParameter(numSolverIterations=100)
    ground = p.createMultiBody(0, p.createCollisionShape(p.GEOM_PLANE))
    body = p.loadURDF(
        str(root / "asset.urdf"),
        basePosition=[0, 0, 0.06],
        flags=p.URDF_USE_INERTIA_FROM_FILE,
    )
    shape_count = len(p.getCollisionShapeData(body, -1))
    expected = json.loads((root / "package.json").read_text())["collision_parts"]
    assert shape_count == expected, (shape_count, expected)
    measured = np.ptp(np.array(p.getAABB(body)), axis=0)
    source = load_mesh(root / "visual.obj").extents
    # Compound broadphase AABBs are conservative, especially after inertia-axis rotation.
    assert np.all(np.abs(measured - source) < 0.020), (measured, source)
    _, initial_inertial_orientation = p.getBasePositionAndOrientation(body)
    p.changeDynamics(
        body, -1, lateralFriction=0.5, restitution=0, collisionMargin=0.001
    )
    for _ in range(1200):
        p.stepSimulation()
    aabb = np.array(p.getAABB(body))
    contacts = p.getContactPoints(body, ground)
    linear, angular = p.getBaseVelocity(body)
    assert contacts, "object did not contact the floor"
    assert aabb[0, 2] > -0.006 and aabb[0, 2] < 0.01, aabb
    assert np.linalg.norm(linear) < 0.01 and np.linalg.norm(angular) < 0.05, (
        linear,
        angular,
    )
    # Check both sides of this pump's neck: a single global hull would fill these gaps.
    # Probe coordinates are in the source mesh frame, not supplied shape truth.
    mesh = load_mesh(root / "visual.obj")
    z = mesh.bounds[1, 2] * 0.88
    probe_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.0015)
    probe = p.createMultiBody(0, probe_shape)
    # Use an undisplaced body for repeatable geometry-only queries.
    p.resetBasePositionAndOrientation(body, [0, 0, 0], initial_inertial_orientation)
    # resetBase places the inertial COM; translate test points to this frame explicitly.
    com = np.array(
        json.loads((root / "package.json").read_text())["estimated_center_of_mass_m"]
    )
    p.resetBasePositionAndOrientation(body, com, initial_inertial_orientation)
    trials = []
    for x in [-0.025, 0.025]:
        position = np.array([x, 0, z])
        p.resetBasePositionAndOrientation(probe, position, [0, 0, 0, 1])
        p.performCollisionDetection()
        collisions = p.getClosestPoints(probe, body, distance=0)
        trials.append({"source_point_m": [x, 0, z], "colliding": bool(collisions)})
    assert not any(t["colliding"] for t in trials), (
        "compound collision unexpectedly filled neck recess"
    )
    p.removeBody(probe)
    # Restore the body above the floor for the final rendered settled scene.
    p.resetBasePositionAndOrientation(
        body, com + [0, 0, 0.06], initial_inertial_orientation
    )
    p.resetBaseVelocity(body, [0, 0, 0], [0, 0, 0])
    for _ in range(1200):
        p.stepSimulation()
    view = p.computeViewMatrix([0.37, -0.55, 0.32], [0, 0, 0.105], [0, 0, 1])
    projection = p.computeProjectionMatrixFOV(40, 1.3, 0.01, 3)
    _, _, rgba, _, _ = p.getCameraImage(
        780,
        600,
        view,
        projection,
        renderer=p.ER_TINY_RENDERER,
        lightDirection=[1, -1, 2],
    )
    Image.fromarray(np.asarray(rgba, dtype=np.uint8).reshape(600, 780, 4)).save(
        root / "pybullet-preview.png"
    )
    report = {
        "status": "passed",
        "engine": "PyBullet",
        "collision_shapes_loaded": shape_count,
        "joints": p.getNumJoints(body),
        "mass_kg": p.getDynamicsInfo(body, -1)[0],
        "example_parameters_not_measured": {
            "mass_kg": p.getDynamicsInfo(body, -1)[0],
            "friction": 0.5,
        },
        "source_dimensions_m": source.tolist(),
        "conservative_broadphase_aabb_dimensions_m": measured.tolist(),
        "settled_floor_contacts": len(contacts),
        "settled_linear_speed_m_s": float(np.linalg.norm(linear)),
        "neck_clearance_probes": trials,
        "scope": "import, shared scale, floor contact and retained neck clearance; not validated material dynamics",
    }
    (root / "pybullet-test.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
finally:
    p.disconnect(cid)
