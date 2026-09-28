import hashlib
import json
import threading
from pathlib import Path

import pytest
import requests

from packing_assets.upload_server import CHUNK, Intake, Problem, Server


@pytest.fixture
def intake(tmp_path):
    board = tmp_path / "board.json"
    board.write_text("{}")
    store = Intake(tmp_path / "intake", board, worker=False)
    yield store
    store.close()


def create(store, **kw):
    return store.create(
        dict(name="键盘", filename="clip.mov", size=12, backend="meshy", **kw)
    )


def test_resumable_offsets_duplicate_chunk_and_completion(intake, monkeypatch):
    monkeypatch.setattr(
        "packing_assets.upload_server.probe_video",
        lambda p: {"width": 1920, "height": 1080},
    )
    job = create(intake)
    uid = job["id"]
    assert intake.append(uid, 0, b"first-")["received"] == 6
    assert intake.append(uid, 0, b"first-")["received"] == 6
    with pytest.raises(Problem, match="内容不一致"):
        intake.append(uid, 0, b"wrong!")
    with pytest.raises(Problem, match="位置不一致"):
        intake.append(uid, 8, b"last")
    with pytest.raises(Problem, match="尚未上传完整"):
        intake.complete(uid)
    intake.append(uid, 6, b"second")
    row = intake.complete(uid)
    assert row["state"] == "uploaded"
    assert row["sha256"] == hashlib.sha256(b"first-second").hexdigest()
    assert intake.complete(uid)["state"] == "uploaded"
    assert len(intake.listing()) == 1
    assert (intake.folder(uid) / "input/键盘.mov").read_bytes() == b"first-second"


def test_recovery_uses_disk_offset_and_queued_state(intake, monkeypatch):
    monkeypatch.setattr(
        "packing_assets.upload_server.probe_video", lambda p: {"width": 2, "height": 2}
    )
    uid = create(intake)["id"]
    (intake.folder(uid) / "upload.part").write_bytes(b"first-second")
    assert intake.public(intake.get(uid))["received"] == 12
    # Crash after file rename but before queue DB commit.
    (intake.folder(uid) / "upload.part").replace(intake.folder(uid) / "input/键盘.mov")
    intake.update(uid, state="verifying")
    recovered = Intake(intake.root, intake.board, worker=False)
    assert recovered.get(uid)["state"] == "uploading"
    assert recovered.complete(uid)["state"] == "uploaded"
    recovered.update(uid, state="processing")
    another = Intake(intake.root, intake.board, worker=False)
    assert another.get(uid)["state"] == "queued"


def test_names_isolation_and_limits(intake):
    first = create(intake)["id"]
    second = create(intake)["id"]
    assert first != second
    assert intake.folder(first) != intake.folder(second)
    with pytest.raises(Problem):
        intake.folder("../../secrets")
    with pytest.raises(Problem):
        intake.create({"name": "bad", "filename": "x.py", "size": 12})
    with pytest.raises(Problem):
        intake.append(first, 0, b"a" * (CHUNK + 1))
    with pytest.raises(Problem):
        intake.create(
            {"name": "big", "filename": "x.mp4", "size": intake.max_bytes + 1}
        )


def test_restart_preserves_external_recovery_ownership(intake):
    uid = create(intake)["id"]
    intake.update(uid, state="processing")
    with intake.connect() as db:
        db.execute(
            "INSERT INTO recovery_claims(id,plan) VALUES (?,?)", (uid, "test-plan")
        )
    restarted = Intake(intake.root, intake.board, worker=False)
    assert restarted.get(uid)["state"] == "processing"
    restarted.close()


def test_retry_command_keeps_corrected_geometry_settings(intake):
    uid = create(intake)["id"]
    settings = intake.folder(uid) / "recovery-settings.json"
    settings.write_text('{"objects": {"键盘.mov": {"geometry": "outer_shape"}}}')
    command = intake.command(intake.get(uid))
    assert command[command.index("--settings") + 1] == str(settings)


def test_worker_uses_cli_and_reuses_output_on_retry(intake, monkeypatch):
    uid = create(intake)["id"]
    calls = []

    class FakeProcess:
        def __init__(self, cmd, **kwargs):
            calls.append(cmd)
            output = Path(cmd[cmd.index("--out") + 1])
            output.mkdir(exist_ok=True)
            (output / "run-report.json").write_text(
                json.dumps({"exported": 1, "retained_unvalidated": 1})
            )

        def wait(self):
            return 0

    monkeypatch.setattr("packing_assets.upload_server.subprocess.Popen", FakeProcess)
    intake.run_one(intake.get(uid))
    assert intake.get(uid)["state"] == "review"
    assert calls[0][calls[0].index("--backend") + 1] == "meshy"
    assert "--with-smooth" not in calls[0]
    assert "--with-blender" not in calls[0]
    intake.update(uid, state="failed")
    intake.worker_enabled = True
    assert intake.retry(uid)["state"] == "queued"
    assert intake.command(intake.get(uid)) == calls[0]


def test_http_auth_stream_upload_and_download_allowlist(intake, monkeypatch):
    monkeypatch.setattr(
        "packing_assets.upload_server.probe_video", lambda p: {"width": 2, "height": 2}
    )
    server = Server(("127.0.0.1", 0), intake)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    session = requests.Session()
    session.trust_env = False
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        assert session.get(url + "/").status_code == 200
        assert session.get(url + "/api/jobs").status_code == 401
        session.headers["Authorization"] = "Bearer " + intake.code
        job = session.post(
            url + "/api/uploads",
            json={"name": "test", "filename": "clip.mp4", "size": 12},
        ).json()
        uid = job["id"]
        assert (
            session.put(
                url + "/api/jobs/" + uid,
                data=b"first-second",
                headers={"Upload-Offset": "0"},
            ).status_code
            == 200
        )
        assert (
            session.post(url + "/api/jobs/" + uid + "/complete").json()["state"]
            == "uploaded"
        )
        output = intake.folder(uid) / "output"
        (output / "models/test").mkdir(parents=True)
        (output / "models/test/test.obj").write_text("valid model fixture")
        assert (
            session.get(
                url + "/api/jobs/" + uid + "/file",
                params={"path": "models/test/test.obj"},
            ).text
            == "valid model fixture"
        )
        assert (
            session.get(
                url + "/api/jobs/" + uid + "/file",
                params={"path": "../../access-code.txt"},
            ).status_code
            == 404
        )
        assert (
            session.get(
                url + "/api/jobs/" + uid + "/file", params={"path": "../input/test.mp4"}
            ).status_code
            == 404
        )
        assert (
            session.get(url + "/api/jobs").json()["jobs"][0]["files"][0]["name"]
            == "test.obj"
        )
    finally:
        session.close()
        server.shutdown()
        server.server_close()
        thread.join(2)


def test_meshy_default_hybrid_and_explicit_fast_commands(intake):
    default = create(intake)
    command = intake.command(default)
    assert command[command.index("--meshy-frontend") + 1] == "hybrid"
    fast = intake.create(
        {"name": "fast", "filename": "fast.mp4", "size": 12, "backend": "meshy-fast"}
    )
    command = intake.command(fast)
    assert command[command.index("--backend") + 1] == "meshy"
    assert command[command.index("--meshy-frontend") + 1] == "fast"
