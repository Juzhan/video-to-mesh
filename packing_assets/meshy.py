"""Persistent Meshy task IDs; ambiguous POST failures never trigger automatic resubmission."""

import base64
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import time

import requests
from .progress import event

BASE = "https://api.meshy.ai/openapi/v1/multi-image-to-3d"


def credential_config_path():
    root = Path(os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config"))
    return root / "packing-assets/meshy.json"


def read_api_key(key_file=None):
    """Resolve credentials without logging or persisting their contents."""
    key_path = key_file or os.environ.get("MESHY_API_KEY_FILE")
    key = os.environ.get("MESHY_API_KEY")
    if not key_path and not key:
        config = credential_config_path()
        if config.is_file():
            key_path = json.loads(config.read_text()).get("api_key_file")
    if key_path:
        try:
            key = Path(key_path).expanduser().read_text().strip()
        except OSError:
            raise ValueError(
                "Cannot read Meshy key file; check --key-file or the saved local configuration"
            ) from None
    if not key or not key.strip():
        raise ValueError(
            "Meshy API key is missing. Run: packing-assets configure-meshy --key-file /path/to/key.txt; or pass --key-file / set MESHY_API_KEY_FILE"
        )
    if "\n" in key or "\r" in key:
        raise ValueError("API key file must contain only the key on one line")
    return key.strip()


def configure_key_file(path):
    path = Path(path).expanduser().resolve()
    read_api_key(path)  # validate readability; does not call the service
    config = credential_config_path()
    config.parent.mkdir(parents=True, exist_ok=True)
    write_json(config, {"api_key_file": str(path)})
    config.chmod(0o600)
    return config


def preflight_credentials(provider_dirs, key_file=None):
    """Fail before expensive reconstruction; fully downloaded runs remain offline."""
    for out in provider_dirs:
        out = Path(out)
        state_file = out / "meshy-task.json"
        state = json.loads(state_file.read_text()) if state_file.is_file() else {}
        if not (state.get("downloaded") and (out / "source.glb").is_file()):
            read_api_key(key_file)
            return


def write_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def request_payload(images, model="meshy-7.1", geometry_resolution="standard"):
    if not 1 <= len(images) <= 4:
        raise ValueError("Meshy needs 1-4 object-only photos, first one front view")
    if geometry_resolution not in {"standard", "2k"}:
        raise ValueError("multi-image geometry_resolution must be standard or 2k")
    urls = []
    for path in images:
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            raise ValueError("Meshy accepts JPG/PNG here")
        mime = mimetypes.guess_type(path.name)[0]
        urls.append(
            f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()
        )
    return {
        "image_urls": urls,
        "ai_model": model,
        "geometry_resolution": geometry_resolution,
        "should_texture": False,
        "should_remesh": False,
        "image_enhancement": False,
        "auto_size": False,
        "target_formats": ["glb"],
    }


def generate(
    images,
    out: Path,
    model="meshy-7.1",
    timeout_s=1800,
    session=None,
    geometry_resolution="standard",
    key_file=None,
):
    payload = request_payload(images, model, geometry_resolution)
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "meshy-task.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    if state and state.get("fingerprint") != fingerprint:
        raise ValueError("inputs or model changed; use a new output directory")
    result = out / "source.glb"
    if state.get("downloaded") and result.exists():
        event("stage", "meshy_generate", cached=True)
        return result
    key = read_api_key(key_file)
    headers = {"Authorization": f"Bearer {key}"}
    session = session or requests.Session()
    if not state.get("task_id"):
        if state.get("submission_started"):
            raise RuntimeError(
                "previous submission outcome is unknown; find its task ID in Meshy and add task_id to meshy-task.json before resuming (avoid double charge)"
            )
        state = {"fingerprint": fingerprint, "submission_started": True, "model": model}
        write_json(state_path, state)
        event("stage", "meshy_submit")
        # Do not retry POST: the service may have accepted the task before a connection failure.
        try:
            response = session.post(BASE, json=payload, headers=headers, timeout=90)
            response.raise_for_status()
            state["task_id"] = response.json()["result"]
        except Exception as error:
            state["submission_error_type"] = type(error).__name__
            write_json(state_path, state)
            raise RuntimeError(
                "submission failed or response uncertain; inspect meshy-task.json before retry"
            ) from error
        write_json(state_path, state)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            response = session.get(
                BASE + "/" + state["task_id"], headers=headers, timeout=60
            )
            if response.status_code == 429 or response.status_code >= 500:
                time.sleep(min(10, max(0, deadline - time.monotonic())))
                continue
            response.raise_for_status()
            task = response.json()
        except (requests.Timeout, requests.ConnectionError):
            time.sleep(min(10, max(0, deadline - time.monotonic())))
            continue
        if state.get("status") != task["status"]:
            print("Meshy task: " + task["status"], flush=True)
            if task["status"] in {"PENDING", "IN_PROGRESS"}:
                event(
                    "stage",
                    "meshy_pending" if task["status"] == "PENDING" else "meshy_running",
                )
        if isinstance(task.get("progress"), (float, int)):
            event("advance", max(0, min(100, task["progress"])), 100, "服务进度")
        state["status"] = task["status"]
        write_json(state_path, state)
        if task["status"] == "SUCCEEDED":
            event("stage", "meshy_download")
            url = task.get("model_urls", {}).get("glb")
            if not url or not url.startswith("https://"):
                raise RuntimeError("successful task did not include an HTTPS GLB URL")
            # Signed asset URL uses no Authorization header. Never write signed URLs to metadata.
            download = session.get(url, timeout=180)
            download.raise_for_status()
            if download.content[:4] != b"glTF":
                raise RuntimeError("download is not a GLB")
            temp = result.with_suffix(".part")
            temp.write_bytes(download.content)
            temp.replace(result)
            state["downloaded"] = True
            write_json(state_path, state)
            return result
        if task["status"] in {"FAILED", "CANCELED", "CANCELLED", "EXPIRED"}:
            raise RuntimeError(f"Meshy task ended: {task['status']}; task ID retained")
        time.sleep(min(10, max(0, deadline - time.monotonic())))
    raise TimeoutError(
        "Meshy wait timed out; rerun to resume saved task without resubmission"
    )
