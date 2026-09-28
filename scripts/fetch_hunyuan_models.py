"""Download only pinned FP16 shape weights; no textures, pickle checkpoints or API key."""

import argparse
import hashlib
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1] / "vendor/hunyuan-models"
MODELS = [
    (
        "Hunyuan3D-2mv",
        "3a761b539b29fe4ff64714813aa9560fd66f5de0",
        "hunyuan3d-dit-v2-mv-turbo",
        "172d7a989b99f66af760da0080d8c6f6350c72609536cf0c55b87a4afa574e45",
    ),
    (
        "Hunyuan3D-2",
        "9cd649ba6913f7a852e3286bad86bfa9a2d83dcf",
        "hunyuan3d-vae-v2-0-turbo",
        "66b7af697118db2de00c22f63a385c75f9049709739281971837af9e4ce31cb4",
    ),
]


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-standard", action="store_true")
    args = parser.parse_args()
    models = list(MODELS)
    if args.include_standard:
        models.append(
            (
                "Hunyuan3D-2mv",
                "3a761b539b29fe4ff64714813aa9560fd66f5de0",
                "hunyuan3d-dit-v2-mv",
                "d36f5881bcdc56726b73e517cd444c13c60732431622da7268145355c8d38e9c",
            )
        )
    for repo, revision, folder, expected in models:
        for filename in ("config.yaml", "model.fp16.safetensors"):
            target = ROOT / folder / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and (
                filename == "config.yaml" or sha256(target) == expected
            ):
                print("Verified cache:", folder, filename, flush=True)
                continue
            temporary = target.with_suffix(target.suffix + ".partial")
            url = f"https://huggingface.co/tencent/{repo}/resolve/{revision}/{folder}/{filename}"
            print("Downloading:", folder, filename, flush=True)
            with requests.get(url, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                with temporary.open("wb") as stream:
                    for chunk in response.iter_content(8 * 1024 * 1024):
                        stream.write(chunk)
            if filename.endswith("safetensors") and sha256(temporary) != expected:
                raise ValueError(f"checksum mismatch: {temporary}")
            temporary.replace(target)


if __name__ == "__main__":
    main()
