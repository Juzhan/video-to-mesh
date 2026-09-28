import numpy as np
from packing_assets.round_container import measured_inner_floor


def observations(angle_span=2 * np.pi, slope=0):
    rng = np.random.default_rng(5)
    angle = np.linspace(-angle_span / 2, angle_span / 2, 600, endpoint=False)
    radius = rng.uniform(0.032, 0.038, len(angle))
    xy = np.c_[radius * np.cos(angle), radius * np.sin(angle)]
    points = np.c_[xy, 0.011 + slope * xy[:, 0] + rng.normal(0, 0.00025, len(angle))]
    return points, np.tile([0.0, 0.0, 1.0], (len(points), 1))


def test_annulus_support_can_recover_unobserved_central_flat_floor():
    points, normals = observations()
    fit = measured_inner_floor(points, normals, np.zeros(2), 0.094, 0, 0.020)
    assert fit["accepted"] and fit["azimuth_sectors_of_12"] == 12
    assert abs(fit["z_m"] - 0.011) < 0.0003
    assert 0.032 < fit["radius_m"] < 0.039


def test_one_sided_patch_or_sloping_wall_cannot_supply_a_flat_floor():
    for points, normals in [
        observations(angle_span=np.pi / 2),
        observations(slope=0.2),
    ]:
        assert not measured_inner_floor(points, normals, np.zeros(2), 0.094, 0, 0.020)[
            "accepted"
        ]


def test_sparse_points_and_already_lower_floor_are_preserved():
    points, normals = observations()
    assert not measured_inner_floor(
        points[:50], normals[:50], np.zeros(2), 0.094, 0, 0.020
    )["accepted"]
    assert not measured_inner_floor(points, normals, np.zeros(2), 0.094, 0, 0.010)[
        "accepted"
    ]
