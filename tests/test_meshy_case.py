from scripts.run_meshy_case import reference_signature


def test_reference_cache_ignores_timing_but_detects_camera_and_mask_changes(tmp_path):
    reference = tmp_path / "depth_proxy"
    reference.mkdir()
    (reference / "visual.obj").write_text("mesh")
    (reference / "asset.json").write_text("{}")
    for name in ["sparse", "images", "masks"]:
        (tmp_path / "work" / name).mkdir(parents=True)
    (tmp_path / "work/metric.json").write_text("{}")
    camera = tmp_path / "work/sparse/cameras.bin"
    camera.write_bytes(b"camera-v1")
    mask = tmp_path / "work/masks/frame.jpg.png"
    mask.write_bytes(b"mask-v1")
    first = reference_signature(tmp_path, reference)
    (tmp_path / "reconstruction.json").write_text('{"last_run_seconds":2}')
    assert reference_signature(tmp_path, reference) == first
    camera.write_bytes(b"camera-v2")
    second = reference_signature(tmp_path, reference)
    assert second != first
    mask.write_bytes(b"mask-v2")
    assert reference_signature(tmp_path, reference) != second
