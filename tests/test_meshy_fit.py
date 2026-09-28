import numpy as np
from scipy.spatial.transform import Rotation
from packing_assets.meshy_fit import fit_upright_similarity


def test_registration_recovers_uniform_metric_scale_without_axis_stretching():
    rng = np.random.default_rng(43)
    src = rng.uniform([-1, -0.4, -0.1], [1, 0.4, 0.1], (1200, 3))
    src = src[~((src[:, 0] > 0.2) & (src[:, 1] > 0.05))]
    rotation = Rotation.from_euler("z", 60, degrees=True).as_matrix()
    target = src @ rotation.T * 0.08 + [0.15, -0.2, 0.05]
    matrix, report = fit_upright_similarity(src, target)
    moved = src @ matrix[:3, :3].T + matrix[:3, 3]
    assert np.sqrt(np.mean((moved - target) ** 2)) < 1e-5
    np.testing.assert_allclose(
        np.linalg.svd(matrix[:3, :3], compute_uv=False), 0.08, atol=1e-5
    )
    assert (
        np.linalg.det(matrix[:3, :3]) > 0 and abs(report["uniform_scale"] - 0.08) < 1e-5
    )
