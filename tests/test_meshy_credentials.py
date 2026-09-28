import json
import pytest
from packing_assets.meshy import read_api_key, configure_key_file, preflight_credentials
from packing_assets.automation import run_automatic


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.delenv("MESHY_API_KEY", raising=False)
    monkeypatch.delenv("MESHY_API_KEY_FILE", raising=False)


def test_config_stores_path_only_and_explicit_env_takes_precedence(
    tmp_path, monkeypatch
):
    key = tmp_path / "credential.txt"
    key.write_text("test-local-placeholder\n")
    config = configure_key_file(key)
    assert json.loads(config.read_text()) == {"api_key_file": str(key)}
    assert "test-local-placeholder" not in config.read_text()
    assert read_api_key() == "test-local-placeholder"
    monkeypatch.setenv("MESHY_API_KEY", "test-env-placeholder")
    assert read_api_key() == "test-env-placeholder"
    assert read_api_key(key) == "test-local-placeholder"


def test_missing_credentials_fail_before_reconstruction(tmp_path, monkeypatch):
    source = tmp_path / "input"
    source.mkdir()
    (source / "box.mp4").write_bytes(b"video")
    board = tmp_path / "board.json"
    board.write_text("{}")
    monkeypatch.setattr(
        "packing_assets.automation.run_batch",
        lambda *a, **k: pytest.fail("must fail before expensive reconstruction"),
    )
    with pytest.raises(ValueError, match="configure-meshy"):
        run_automatic(source, board, tmp_path / "out", backend="meshy")


def test_downloaded_cache_needs_no_credentials(tmp_path):
    provider = tmp_path / "provider"
    provider.mkdir()
    (provider / "meshy-task.json").write_text(json.dumps({"downloaded": True}))
    (provider / "source.glb").write_bytes(b"glTFfixture")
    preflight_credentials([provider])
    (provider / "source.glb").unlink()
    with pytest.raises(ValueError, match="missing"):
        preflight_credentials([provider])


def test_unreadable_saved_path_has_actionable_error(tmp_path):
    key = tmp_path / "credential.txt"
    key.write_text("test-local-placeholder")
    configure_key_file(key)
    key.unlink()
    with pytest.raises(ValueError, match="Cannot read Meshy key file"):
        read_api_key()
