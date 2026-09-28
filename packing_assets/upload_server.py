"""Private LAN video intake with resumable uploads and a durable single-worker queue."""

import argparse
import fcntl
import hashlib
import json
import logging
import mimetypes
import os
import re
import secrets
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

PROJECT = Path(__file__).resolve().parents[1]
WEB = Path(__file__).with_name("web")
CHUNK = 4 * 1024 * 1024
EXTENSIONS = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm"}
logger = logging.getLogger(__name__)


class Problem(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def object_name(value):
    value = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", str(value)).strip(" .")[:80]
    if not value:
        raise Problem("请填写物体名称")
    return value


def probe_video(path):
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,codec_name:format=duration",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            timeout=20,
            check=True,
        )
        data = json.loads(result.stdout)
        stream = data["streams"][0]
        if stream["width"] <= 0 or stream["height"] <= 0:
            raise ValueError("empty video")
        return dict(
            stream, duration_seconds=float(data.get("format", {}).get("duration", 0))
        )
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, IndexError):
        raise Problem(
            "视频无法读取，请尝试手机相机的 1080p 普通视频（MP4 / MOV）"
        ) from None


class Intake:
    def __init__(self, root, board, worker=True, max_bytes=2 * 1024**3):
        self.root = Path(root).resolve()
        self.board = Path(board).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = self.root / "queue.sqlite3"
        self.max_bytes = max_bytes
        self.worker_enabled = worker
        self.stop = threading.Event()
        self.wake = threading.Event()
        self.locks = {}
        self.guard = threading.Lock()
        self.child = None
        self.thread = None
        code_file = self.root / "access-code.txt"
        if not code_file.exists():
            fd = os.open(code_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w") as f:
                f.write(str(secrets.randbelow(90_000_000) + 10_000_000))
        self.code = code_file.read_text().strip()
        with self.connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, name TEXT NOT NULL, filename TEXT NOT NULL,
                size INTEGER NOT NULL, received INTEGER DEFAULT 0, state TEXT NOT NULL,
                backend TEXT NOT NULL, blender INTEGER NOT NULL, created REAL NOT NULL,
                updated REAL NOT NULL, error TEXT, video TEXT, sha256 TEXT)""")
            db.execute(
                "CREATE TABLE IF NOT EXISTS recovery_claims (id TEXT PRIMARY KEY, plan TEXT NOT NULL)"
            )
            db.execute(
                "UPDATE jobs SET state='queued', error='服务重启，复用已完成步骤继续处理' WHERE state='processing' AND id NOT IN (SELECT id FROM recovery_claims)"
            )
            db.execute("UPDATE jobs SET state='uploading' WHERE state='verifying'")
        if worker:
            self.thread = threading.Thread(target=self.work, daemon=True)
            self.thread.start()

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.db, timeout=15)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def lock(self, uid):
        with self.guard:
            return self.locks.setdefault(uid, threading.Lock())

    def folder(self, uid):
        if not re.fullmatch(r"[a-f0-9]{24}", uid):
            raise Problem("任务不存在", 404)
        return self.root / "jobs" / uid

    def get(self, uid):
        self.folder(uid)
        with self.connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (uid,)).fetchone()
        if row is None:
            raise Problem("任务不存在", 404)
        return dict(row)

    def update(self, uid, **values):
        values["updated"] = time.time()
        with self.connect() as db:
            db.execute(
                "UPDATE jobs SET " + ",".join(k + "=?" for k in values) + " WHERE id=?",
                [*values.values(), uid],
            )

    def create(self, data):
        size = data.get("size")
        if type(size) is not int or not 0 < size <= self.max_bytes:
            raise Problem(f"单个视频需小于 {self.max_bytes // 1024**2} MB")
        suffix = Path(str(data.get("filename", ""))).suffix.lower()
        if suffix not in EXTENSIONS:
            raise Problem("请选择 MP4、MOV、M4V、MKV、AVI 或 WebM 视频")
        backend = data.get("backend", "meshy")
        if backend not in {"meshy", "meshy-fast", "local"}:
            raise Problem("不支持的生成方式")
        if shutil.disk_usage(self.root).free < size * 2 + 1024**3:
            raise Problem("电脑磁盘可用空间不足", 507)
        uid = secrets.token_hex(12)
        name = object_name(data.get("name", ""))
        folder = self.folder(uid)
        (folder / "input").mkdir(parents=True)
        (folder / "upload.part").touch()
        now = time.time()
        with self.connect() as db:
            db.execute(
                "INSERT INTO jobs (id,name,filename,size,state,backend,blender,created,updated) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    uid,
                    name,
                    name + suffix,
                    size,
                    "uploading",
                    backend,
                    int(bool(data.get("blender", False))),
                    now,
                    now,
                ),
            )
        return self.public(self.get(uid))

    def append(self, uid, offset, data):
        with self.lock(uid):
            row = self.get(uid)
            if row["state"] != "uploading":
                raise Problem("这个视频已提交处理，不能追加", 409)
            if not 0 < len(data) <= CHUNK or offset + len(data) > row["size"]:
                raise Problem("上传块长度不正确")
            part = self.folder(uid) / "upload.part"
            actual = part.stat().st_size
            if offset < actual and offset + len(data) <= actual:
                with part.open("rb") as f:
                    f.seek(offset)
                    if f.read(len(data)) != data:
                        raise Problem("续传文件内容不一致，请重新新建上传", 409)
                return {"received": actual}
            if offset != actual:
                raise Problem("上传位置不一致，请恢复续传", 409)
            with part.open("ab") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            self.update(uid, received=actual + len(data))
            return {"received": actual + len(data)}

    def complete(self, uid):
        with self.lock(uid):
            row = self.get(uid)
            if row["state"] != "uploading":
                return self.public(row)
            folder = self.folder(uid)
            part = folder / "upload.part"
            destination = folder / "input" / row["filename"]
            # Recover a crash after rename but before committing queue state.
            video = part if part.exists() else destination
            if not video.exists() or video.stat().st_size != row["size"]:
                raise Problem("视频尚未上传完整", 409)
            self.update(uid, state="verifying")
            try:
                info = probe_video(video)
                digest = self.digest(video)
                if video == part:
                    part.replace(destination)
                self.update(
                    uid,
                    state="queued" if self.worker_enabled else "uploaded",
                    received=row["size"],
                    video=json.dumps(info),
                    sha256=digest,
                    error=None,
                )
            except Exception:
                self.update(uid, state="uploading")
                raise
            self.wake.set()
            return self.public(self.get(uid))

    @staticmethod
    def digest(path):
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(CHUNK), b""):
                h.update(chunk)
        return h.hexdigest()

    def public(self, row):
        row = dict(row)
        folder = self.folder(row["id"])
        if row["state"] == "uploading":
            part = folder / "upload.part"
            # Actual bytes on disk are authoritative after an interrupted request.
            row["received"] = (
                part.stat().st_size
                if part.exists()
                else row["size"]
                if (folder / "input" / row["filename"]).exists()
                else 0
            )
        for name in ["progress", "run-report"]:
            path = folder / "output" / (name + ".json")
            if path.exists():
                try:
                    report = json.loads(path.read_text())
                    if name == "progress":
                        from .progress import LABELS

                        row["progress"] = {
                            k: report.get(k)
                            for k in [
                                "elapsed_seconds",
                                "stage_elapsed_seconds",
                                "cached",
                                "detail",
                            ]
                        }
                        row["progress"]["stage"] = LABELS.get(
                            report.get("stage"), report.get("stage")
                        )
                    else:
                        row["report"] = {
                            k: report.get(k)
                            for k in [
                                "status",
                                "exported",
                                "failed",
                                "retained_unvalidated",
                            ]
                        }
                except (OSError, ValueError):
                    pass
        files = []
        output = folder / "output"
        for path in sorted((output / "models").glob("*/*.obj")):
            files.append({"name": path.name, "path": str(path.relative_to(output))})
        for name in ["scene-package.zip", "scene.png", "scene.blend"]:
            if (output / name).is_file():
                files.append({"name": name, "path": name})
        row["files"] = files
        row["video"] = json.loads(row["video"]) if row.get("video") else None
        return row

    def listing(self):
        with self.connect() as db:
            rows = db.execute(
                "SELECT * FROM jobs ORDER BY created DESC LIMIT 200"
            ).fetchall()
        return [self.public(row) for row in rows]

    def retry(self, uid):
        with self.lock(uid):
            row = self.get(uid)
            if row["state"] not in {"failed", "uploaded"}:
                raise Problem("只有失败或仅上传的任务可重新排队", 409)
            if not self.worker_enabled:
                raise Problem("当前服务仅接收上传，不运行生成")
            self.update(uid, state="queued", error=None)
            self.wake.set()
            return self.public(self.get(uid))

    def command(self, row):
        folder = self.folder(row["id"])
        command = [
            sys.executable,
            "-m",
            "packing_assets.cli",
            "run",
            "--input",
            str(folder / "input"),
            "--out",
            str(folder / "output"),
            "--board",
            str(self.board),
            "--quality",
            "fast",
            "--backend",
            "meshy" if row["backend"] == "meshy-fast" else row["backend"],
            "--progress",
            "off",
        ]
        if row["backend"] in {"meshy", "meshy-fast"}:
            command.extend(
                [
                    "--meshy-frontend",
                    "fast" if row["backend"] == "meshy-fast" else "hybrid",
                ]
            )
        if row["blender"]:
            command.append("--with-blender")
        settings = folder / "recovery-settings.json"
        if settings.is_file():
            command.extend(["--settings", str(settings)])
        return command

    def run_one(self, row):
        uid = row["id"]
        self.update(uid, state="processing", error=None)
        folder = self.folder(uid)
        with (folder / "worker.log").open("ab") as log:
            self.child = subprocess.Popen(
                self.command(row), cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT
            )
            code = self.child.wait()
            self.child = None
        if self.stop.is_set():
            self.update(uid, state="queued", error="服务停止，启动后继续")
            return
        report_path = folder / "output/run-report.json"
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        if code == 0 and report.get("exported", 0):
            self.update(
                uid, state="review" if report.get("retained_unvalidated") else "done"
            )
        else:
            errors = [e.get("error", "") for e in report.get("errors", [])]
            self.update(
                uid,
                state="failed",
                error="；".join(errors)[:700]
                or "处理未完成，详细记录见电脑上的 worker.log",
            )

    def work(self):
        while not self.stop.is_set():
            with self.connect() as db:
                row = db.execute(
                    "SELECT * FROM jobs WHERE state='queued' AND id NOT IN (SELECT id FROM recovery_claims) ORDER BY updated LIMIT 1"
                ).fetchone()
            if row:
                try:
                    self.run_one(dict(row))
                except Exception as error:
                    logger.exception("Intake worker failed for job %s", row["id"])
                    self.update(row["id"], state="failed", error=str(error)[:700])
            else:
                self.wake.wait(2)
                self.wake.clear()

    def close(self):
        self.stop.set()
        self.wake.set()
        if self.child and self.child.poll() is None:
            self.child.terminate()
        if self.thread:
            self.thread.join(timeout=10)


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, intake):
        self.intake = intake
        self.auth_failures = {}
        self.auth_lock = threading.Lock()
        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        # Never log access codes, request bodies, or credentials.
        pass

    def send(
        self, status, body, mime="application/json; charset=utf-8", download=False
    ):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        if self.close_connection:
            self.send_header("Connection", "close")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' blob:; media-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'",
        )
        if download:
            self.send_header("Content-Disposition", "attachment")
        self.end_headers()
        self.wfile.write(body)

    def read(self, limit):
        if self.headers.get("Transfer-Encoding"):
            raise Problem("不支持该传输格式")
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise Problem("无效的内容长度") from None
        if not 0 < length <= limit:
            raise Problem("请求太大或内容为空", 413)
        self.connection.settimeout(90)
        data = self.rfile.read(length)
        if len(data) != length:
            raise Problem("上传中断，请续传", 409)
        return data

    def route(self):
        store = self.server.intake
        path = urlsplit(self.path).path
        if self.command == "GET" and path in {"/", "/app.js", "/style.css"}:
            filename = "index.html" if path == "/" else path[1:]
            return self.send(
                200,
                (WEB / filename).read_bytes(),
                mimetypes.guess_type(filename)[0] + "; charset=utf-8",
            )
        if not secrets.compare_digest(
            self.headers.get("Authorization", ""), "Bearer " + store.code
        ):
            with self.server.auth_lock:
                now = time.monotonic()
                failures = [
                    t
                    for t in self.server.auth_failures.get(self.client_address[0], [])
                    if now - t < 60
                ]
                self.server.auth_failures[self.client_address[0]] = [
                    *failures[-20:],
                    now,
                ]
            if len(failures) >= 15:
                time.sleep(1)
                raise Problem("配对码错误次数较多，请稍后再试", 429)
            raise Problem("请输入电脑显示的配对码", 401)
        if self.command == "GET" and path == "/api/jobs":
            return self.send(
                200,
                {
                    "jobs": store.listing(),
                    "chunk_size": CHUNK,
                    "worker_enabled": store.worker_enabled,
                    "max_bytes": store.max_bytes,
                },
            )
        if self.command == "POST" and path == "/api/uploads":
            return self.send(201, store.create(json.loads(self.read(16 * 1024))))
        match = re.fullmatch(
            r"/api/jobs/([a-f0-9]{24})(?:/(complete|retry|file))?", path
        )
        if not match:
            raise Problem("地址不存在", 404)
        uid, action = match.groups()
        if self.command == "GET" and not action:
            return self.send(200, store.public(store.get(uid)))
        if self.command == "PUT" and not action:
            try:
                offset = int(self.headers.get("Upload-Offset", "-1"))
            except ValueError:
                raise Problem("无效的续传位置") from None
            return self.send(200, store.append(uid, offset, self.read(CHUNK)))
        if self.command == "POST" and action in {"complete", "retry"}:
            return self.send(
                200, store.complete(uid) if action == "complete" else store.retry(uid)
            )
        if self.command == "GET" and action == "file":
            from urllib.parse import parse_qs

            relative = parse_qs(urlsplit(self.path).query).get("path", [""])[0]
            row = store.public(store.get(uid))
            if relative not in {f["path"] for f in row["files"]}:
                raise Problem("结果文件不存在", 404)
            root = store.folder(uid) / "output"
            target = (root / relative).resolve()
            if not target.is_relative_to(root.resolve()):
                raise Problem("无效文件地址", 400)
            # Only final model artifacts are exposed; input clips, logs, config and API credentials are never served.
            self.send_response(200)
            self.send_header(
                "Content-Type",
                mimetypes.guess_type(target.name)[0] or "application/octet-stream",
            )
            self.send_header("Content-Length", str(target.stat().st_size))
            self.send_header("Content-Disposition", "attachment")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            with target.open("rb") as source:
                shutil.copyfileobj(source, self.wfile, CHUNK)
            return
        raise Problem("不支持的操作", 405)

    def handle_request(self):
        try:
            self.route()
        except Problem as error:
            self.close_connection = True
            self.send(error.status, {"error": str(error)})
        except (ValueError, TypeError, KeyError):
            self.close_connection = True
            self.send(400, {"error": "请求格式不正确"})
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            self.close_connection = True
        except Exception:
            logger.exception("Intake request failed")
            self.close_connection = True
            self.send(500, {"error": "电脑端处理异常，请查看服务日志"})

    do_GET = do_POST = do_PUT = handle_request


def main():
    import signal

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="bind the hotspot/LAN address explicitly for phone access",
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--root", type=Path, default=PROJECT / "phone-workspace")
    parser.add_argument(
        "--board", type=Path, default=PROJECT / "calibration/a4-dense/board.json"
    )
    parser.add_argument(
        "--upload-only",
        action="store_true",
        help="receive videos without launching generation",
    )
    args = parser.parse_args()
    if not args.board.is_file():
        parser.error("board file missing")
    if not shutil.which("ffprobe"):
        parser.error("ffprobe missing; install FFmpeg to validate uploaded videos")
    args.root.mkdir(parents=True, exist_ok=True)
    with (args.root / ".server.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("this intake root already has a running server")
        store = Intake(args.root, args.board, worker=not args.upload_only)
        server = Server((args.host, args.port), store)

        def terminate(signum, frame):
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, terminate)
        print(
            f"手机上传页：http://{args.host}:{args.port}\n配对码：{store.code}\n文件目录：{store.root}",
            flush=True,
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
            store.close()


if __name__ == "__main__":
    main()
