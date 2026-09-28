"""CLI progress independent of noisy Python/C++ logs; no invented time estimates."""

from contextvars import ContextVar
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback
import unicodedata

_current = ContextVar("packing_assets_progress", default=None)
LABELS = {
    "starting": "准备",
    "objects": "逐件生成与导出",
    "meshy": "Meshy 分支",
    "meshy_reference": "检查米制参考",
    "meshy_views": "Meshy 选帧与抠图",
    "meshy_generate": "Meshy 生成或续跑",
    "meshy_submit": "提交 Meshy 付费任务",
    "meshy_pending": "Meshy 排队",
    "meshy_running": "Meshy 云端生成",
    "meshy_download": "下载 Meshy 模型",
    "meshy_metric": "恢复米制尺寸",
    "meshy_photo": "照片姿态与尺度校正",
    "meshy_check": "Meshy 几何与照片检查",
    "meshy_dimensions": "校正分方向尺寸并检查预留视角",
    "hybrid_measurement": "复用重建缓存测量本地尺寸",
    "reconstruction": "重建",
    "export": "导出",
    "scene": "场景与打包",
    "prepare": "抽帧与参照板检测",
    "features": "图像特征提取",
    "matching": "匹配照片",
    "camera_estimation": "估计相机",
    "metric_scale": "A4 尺寸定标",
    "sparse_object_prompts": "定位物体稀疏特征",
    "sparse_photo_masks": "Meshy 快速分割",
    "sparse_registration_reference": "准备尺度参考",
    "undistort": "校正图像",
    "multi_view_stereo": "多视图深度重建",
    "fusion": "融合深度",
    "remove_board": "分离桌面",
    "photo_masks_v3": "分割物体",
    "masked_surface_v3": "重建表面",
    "surface_check": "检查原始表面",
    "packing_proxy_v2": "构建封闭外形",
    "proxy_check": "核对照片轮廓",
    "depth_proxy_v2": "恢复外部凹凸与内腔",
    "round_container_v3": "优化杯碗形状",
    "candidate_check": "检查并选择候选",
    "mesh_export": "降面、平滑和水密检查",
    "collision_export": "碰撞分解、平滑和水密检查",
    "blender": "生成普通 Blender 场景",
    "blender_smooth": "生成平滑 Blender 场景",
    "physics_test": "PyBullet 检查",
    "package": "整理文件与打包",
}


def fit_line(text, columns):
    """Keep CJK terminal lines from wrapping and breaking cursor redraw."""
    clean = "".join(c for c in text if c.isprintable())
    width = lambda c: (
        0
        if unicodedata.combining(c)
        else (2 if unicodedata.east_asian_width(c) in "WF" else 1)
    )
    if sum(map(width, clean)) <= columns:
        return clean
    result = ""
    used = 0
    for char in clean:
        if used + width(char) > max(0, columns - 1):
            break
        result += char
        used += width(char)
    return result + "…"


def event(method, *args, **kwargs):
    reporter = _current.get()
    if reporter is not None:
        getattr(reporter, method)(*args, **kwargs)


class ProgressSession:
    def __init__(self, out, mode="auto", verbose=False, stream=None, interval=1.0):
        self.out = Path(out)
        self.mode = mode
        self.verbose = verbose
        self.stream = stream
        self.interval = interval
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.started = time.monotonic()
        self.stage_started = self.started
        self.last_emit = 0.0
        self.state = {
            "status": "running",
            "phase": "starting",
            "completed": 0,
            "total": 0,
            "failed": 0,
            "item": None,
            "item_index": 0,
            "stage": None,
            "cached": False,
            "detail": None,
            "pid": os.getpid(),
            "progress_basis": "completed items in current phase, not time remaining",
        }
        self.saved = []
        self.console_owned = False
        self.log = None
        self.thread = None
        self.drawn = False

    def __enter__(self):
        self.out.mkdir(parents=True, exist_ok=True)
        self.token = _current.set(self)
        if self.stream is None:
            self.stream = os.fdopen(os.dup(2), "w", buffering=1, encoding="utf-8")
            self.console_owned = True
        self.tty = self.mode == "auto" and self.stream.isatty() and not self.verbose
        if not self.verbose:
            sys.stdout.flush()
            sys.stderr.flush()
            self.log = (self.out / "run.log").open("a", buffering=1)
            self.log.write(
                f"\n=== Run {time.strftime('%Y-%m-%d %H:%M:%S')} PID {os.getpid()} ===\n"
            )
            for fd in (1, 2):
                self.saved.append((fd, os.dup(fd)))
                os.dup2(self.log.fileno(), fd)
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.thread.start()
        self._refresh(True)
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc is not None:
            if self.log:
                traceback.print_exception(exc_type, exc, tb, file=self.log)
            self.state["error"] = str(exc)
            self.finish(
                "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            )
        elif self.state["status"] == "running":
            self.finish("complete")
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=2.0)
        sys.stdout.flush()
        sys.stderr.flush()
        for fd, original in self.saved:
            os.dup2(original, fd)
            os.close(original)
        if self.log:
            self.log.close()
        if self.tty:
            self.stream.write("\n")
            self.stream.flush()
        if self.console_owned:
            self.stream.close()
        _current.reset(self.token)

    def _heartbeat(self):
        while not self.stop.wait(self.interval):
            self._refresh()

    def _refresh(self, force=False):
        with self.lock:
            now = time.monotonic()
            self.state["elapsed_seconds"] = round(now - self.started, 1)
            self.state["stage_elapsed_seconds"] = round(now - self.stage_started, 1)
            temp = self.out / "progress.tmp.json"
            temp.write_text(json.dumps(self.state, ensure_ascii=False, indent=2))
            temp.replace(self.out / "progress.json")
            if self.mode == "off":
                return
            if not force and now - self.last_emit < (0.15 if self.tty else 10.0):
                return
            self.last_emit = now
            s = self.state
            total = s["total"]
            done = s["completed"]
            ratio = done / total if total else 0
            bar = (
                "[" + "#" * int(ratio * 20) + "-" * (20 - int(ratio * 20)) + "]"
                if total
                else "[运行中]"
            )
            phase = LABELS.get(s["phase"], s["phase"])
            count = f" {done}/{total}" if total else ""
            item = f" | {s['item_index']}/{total} {s['item']}" if s["item"] else ""
            stage = LABELS.get(s["stage"], s["stage"] or "准备")
            detail = s["detail"]
            part = (
                f" {detail['done']}/{detail['total']} {detail['unit']}"
                if detail
                else ""
            )
            cache = " [缓存]" if s["cached"] else ""
            outcome = f" | {s['status']}" if s["status"] != "running" else ""
            line = f"{phase} {bar}{count}{item} | {stage}{cache}{part} | 本步 {s['stage_elapsed_seconds']:.0f}s / 总计 {s['elapsed_seconds']:.0f}s{outcome}"
            if self.tty:
                try:
                    columns = os.get_terminal_size(self.stream.fileno()).columns
                except (OSError, AttributeError):
                    columns = 80
                lines = [
                    f"{phase} {bar}{count}{item}",
                    f"{stage}{cache}{part}",
                    f"本步 {s['stage_elapsed_seconds']:.0f}s / 总计 {s['elapsed_seconds']:.0f}s{outcome}",
                ]
                self.stream.write(
                    ("\033[2A" if self.drawn else "")
                    + "\n".join(
                        "\r\033[2K" + fit_line(text, max(1, columns - 1))
                        for text in lines
                    )
                )
                self.drawn = True
            else:
                self.stream.write(line + "\n")
            self.stream.flush()

    def phase(self, name, total):
        with self.lock:
            self.state.update(
                phase=name,
                total=total,
                completed=0,
                item=None,
                item_index=0,
                detail=None,
                stage=None,
                cached=False,
            )
            self.stage_started = time.monotonic()
            self._refresh(True)

    def item(self, name, index):
        with self.lock:
            self.state.update(
                item=str(name), item_index=index, detail=None, stage=None, cached=False
            )
            self.stage_started = time.monotonic()
            self._refresh(True)

    def stage(self, name, cached=False):
        with self.lock:
            self.state.update(stage=name, cached=cached, detail=None)
            self.stage_started = time.monotonic()
            self._refresh(True)

    def advance(self, done, total, unit="帧"):
        with self.lock:
            self.state["detail"] = {
                "done": int(done),
                "total": int(total),
                "unit": unit,
            }
            self._refresh()

    def item_done(self, status="done"):
        with self.lock:
            self.state["completed"] += 1
            if status in ("failed", "rejected"):
                self.state["failed"] += 1
            self.state["last_item_status"] = status
            self._refresh(True)

    def finish(self, status):
        with self.lock:
            self.state["status"] = status
            self._refresh(True)
