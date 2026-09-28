"""Public releases must include curated examples without leaking local captures."""

import pytest

from scripts.package_source import source_files


def write(root, name, content="example"):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def test_curated_demo_does_not_allow_other_captures(tmp_path):
    expected = {
        write(tmp_path, "README.md"),
        write(tmp_path, "docs/media/mug-input.gif"),
        write(tmp_path, "examples/mug/mug.obj"),
    }
    for name in [
        "examples/private.obj",
        "docs/media/private.gif",
        "docs/media/raw.mp4",
        "phone-workspace/access-code.txt",
        "om/scene.json",
        "input/capture.mp4",
    ]:
        write(tmp_path, name)
    assert set(source_files(tmp_path)) == expected


def test_public_asset_symlink_is_rejected(tmp_path):
    secret = write(tmp_path, "private.txt")
    public = tmp_path / "examples/mug/mug.obj"
    public.parent.mkdir(parents=True)
    public.symlink_to(secret)
    with pytest.raises(ValueError, match="symlink"):
        source_files(tmp_path)


def test_credentials_in_public_text_are_rejected(tmp_path):
    write(tmp_path, "examples/settings.json", "msy_" + "A" * 32)
    with pytest.raises(ValueError, match="credential"):
        source_files(tmp_path)
