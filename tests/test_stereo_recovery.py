from packing_assets.reconstruct import recover_interrupted_stereo


def test_interrupted_maps_cannot_be_mistaken_for_completed_cache(tmp_path):
    stereo = tmp_path / "dense/stereo"
    stereo.mkdir(parents=True)
    (stereo / "patch-match.cfg").write_text("camera neighbors")
    image = tmp_path / "dense/images/a.jpg"
    image.parent.mkdir()
    image.write_bytes(b"intact image")
    for name in ("depth_maps", "normal_maps", "consistency_graphs"):
        folder = stereo / name
        folder.mkdir()
        (folder / "a.jpg.geometric.bin").write_bytes(b"\x00" * 40)
    state = {
        "active_stage": "multi_view_stereo",
        "stages": {"metric_scale": {"done": True}},
    }
    moved = recover_interrupted_stereo(tmp_path, state)
    assert len(moved) == 3
    assert not list((stereo / "depth_maps").iterdir())
    assert (stereo / "patch-match.cfg").read_text() == "camera neighbors"
    assert image.read_bytes() == b"intact image"
    from pathlib import Path

    assert (Path(moved[0]) / "a.jpg.geometric.bin").read_bytes() == b"\x00" * 40
    assert state["stages"]["metric_scale"]["done"]
    assert recover_interrupted_stereo(tmp_path, state) == []


def test_finished_stereo_and_other_active_stages_are_preserved(tmp_path):
    folder = tmp_path / "dense/stereo/depth_maps"
    folder.mkdir(parents=True)
    data = folder / "a.jpg.geometric.bin"
    data.write_bytes(b"complete checkpoint")
    for state in [
        {
            "active_stage": "multi_view_stereo",
            "stages": {"multi_view_stereo": {"done": True}},
        },
        {"active_stage": "fusion", "stages": {}},
    ]:
        assert recover_interrupted_stereo(tmp_path, state) == []
        assert data.read_bytes() == b"complete checkpoint"
