"""Create an allowlisted, source-only ZIP. Includes only explicitly curated demo media, never raw captures or weights."""

import argparse
import hashlib
import json
import re
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOP = {
    ".gitignore",
    ".gitattributes",
    ".editorconfig",
    "README.md",
    "LICENSE",
    "CONTRIBUTING.md",
    "CHANGELOG.md",
    "SECURITY.md",
    "THIRD_PARTY_NOTICES.md",
    "pyproject.toml",
    "requirements-tested.txt",
    "run-videos.sh",
    "run-upload.sh",
    "run-a4-fast.sh",
    "run-a4.sh",
}
DIRECTORIES = {
    "packing_assets",
    "scripts",
    "tests",
    "docs",
    "examples",
    "calibration",
    ".github",
}
PUBLIC_DEMO_FILES = {
    "docs/media/mug-input.gif",
    "examples/mug/mug.obj",
}

EXTENSIONS = {
    ".html",
    ".css",
    ".js",
    ".py",
    ".sh",
    ".md",
    ".toml",
    ".txt",
    ".json",
    ".yml",
    ".yaml",
    ".svg",
    ".png",
    ".pdf",
}
SECRETS = re.compile(
    r"(?:sk-[A-Za-z0-9]{24,}|msy_[A-Za-z0-9]{24,}|gh[pousr]_[A-Za-z0-9]{24,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)"
)
LOCAL_PATH = re.compile(r"/(?:home|Users)/[A-Za-z0-9_.-]+/")


def source_files(root=ROOT):
    files = [root / name for name in TOP if (root / name).is_file()]
    for folder in sorted(DIRECTORIES):
        for path in (root / folder).rglob("*"):
            if any(
                part.startswith(".") or part == "__pycache__"
                for part in path.relative_to(root / folder).parts
            ):
                continue
            if path.is_symlink():
                raise ValueError(f"symlink is not source: {path.relative_to(root)}")
            if path.is_file() and (
                path.suffix in EXTENSIONS
                or path.relative_to(root).as_posix() in PUBLIC_DEMO_FILES
            ):
                files.append(path)
    for path in files:
        if path.is_symlink():
            raise ValueError(f"symlink is not source: {path.relative_to(root)}")
        if path.stat().st_size > 5 * 1024 * 1024:
            raise ValueError(f"oversized source file: {path.relative_to(root)}")
        if path.suffix not in {".png", ".pdf", ".gif"}:
            text = path.read_text()
            if SECRETS.search(text):
                raise ValueError(f"possible credential: {path.relative_to(root)}")
            if LOCAL_PATH.search(text):
                raise ValueError(f"local absolute path: {path.relative_to(root)}")
    return sorted(set(files))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--out", type=Path, default=ROOT / "dist/packing-assets-source.zip"
    )
    args = parser.parse_args()
    files = source_files()
    if (ROOT / ".git").exists():
        tracked = (
            subprocess.run(
                ["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True
            )
            .stdout.decode()
            .split("\0")
        )
        allowed = {str(p.relative_to(ROOT)) for p in files}
        unexpected = set(filter(None, tracked)) - allowed
        if unexpected:
            raise ValueError(
                "tracked files outside source allowlist: "
                + ", ".join(sorted(unexpected))
            )
    size = sum(p.stat().st_size for p in files)
    print(
        f"Source audit passed: {len(files)} files, {size / 1024 / 1024:.2f} MiB before compression"
    )
    if args.check:
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in files
    }
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            info = zipfile.ZipInfo("packing-assets/" + str(path.relative_to(ROOT)))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (path.stat().st_mode & 0o777) << 16
            archive.writestr(info, path.read_bytes())
        archive.writestr(
            "packing-assets/SOURCE_MANIFEST.json", json.dumps(manifest, indent=2)
        )
    print(args.out.resolve())


if __name__ == "__main__":
    main()
