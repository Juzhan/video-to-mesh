"""Run a fixed, prioritized phone recovery plan beside the upload worker.

Claims only completed jobs with a SQLite compare-and-set. It never starts a
second Intake instance (which would reset the live upload worker's state).
"""

import argparse
import fcntl
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temporary.replace(path)


def connect(root):
    db = sqlite3.connect(root / "queue.sqlite3", timeout=30)
    db.row_factory = sqlite3.Row
    return db


def process(root, plan_file):
    plan = json.loads(plan_file.read_text())
    plan_id = plan_file.parent.name
    while True:
        waiting = False
        for entry in plan["jobs"]:
            if entry.get("recovery_state") in {"finished", "skipped"}:
                continue
            uid = entry["id"]
            folder = root / "jobs" / uid
            output = folder / "output"
            with connect(root) as db:
                db.execute("BEGIN IMMEDIATE")
                row = db.execute("SELECT * FROM jobs WHERE id=?", (uid,)).fetchone()
                claim = db.execute(
                    "SELECT plan FROM recovery_claims WHERE id=?", (uid,)
                ).fetchone()
                owned = claim and claim["plan"] == plan_id
                if (
                    row["state"] in {"uploading", "verifying", "queued", "processing"}
                    and not owned
                ):
                    waiting = True
                    continue
                # A queued job may already have finished in hybrid mode through
                # the regular worker while higher-priority repairs were running.
                report_file = output / "run-report.json"
                previous = (
                    json.loads(report_file.read_text()) if report_file.exists() else {}
                )
                if (
                    entry["original_state"] in {"queued", "processing"}
                    and previous.get("meshy_frontend") == "hybrid"
                    and row["state"] == "done"
                ):
                    entry["recovery_state"] = "skipped"
                    entry["reason"] = "regular_worker_completed_hybrid"
                    save(plan_file, plan)
                    continue
                if claim and not owned:
                    waiting = True
                    continue
                if (
                    row["state"] not in {"done", "review", "failed", "uploaded"}
                    and not owned
                ):
                    waiting = True
                    continue
                db.execute(
                    "INSERT OR REPLACE INTO recovery_claims(id,plan) VALUES (?,?)",
                    (uid, plan_id),
                )
                db.execute(
                    "UPDATE jobs SET state='processing',backend=?,error=NULL,updated=? WHERE id=?",
                    (entry["target_backend"], time.time(), uid),
                )
            entry.update(recovery_state="running", started=time.time())
            plan["current"] = {"id": uid, "name": entry["name"]}
            save(plan_file, plan)
            backup = folder / "recovery-history" / plan_id
            backup.mkdir(parents=True, exist_ok=True)
            for name in (
                "models",
                "assets",
                "scene-package.zip",
                "scene.blend",
                "scene.png",
                "run-report.json",
                "input-manifest.json",
            ):
                source, target = output / name, backup / name
                if source.exists() and not target.exists():
                    shutil.copytree(
                        source, target
                    ) if source.is_dir() else shutil.copy2(source, target)
            command = [
                sys.executable,
                "-m",
                "packing_assets.cli",
                "run",
                "--input",
                str(folder / "input"),
                "--out",
                str(output),
                "--board",
                plan["board"],
                "--quality",
                "fast",
                "--backend",
                entry["target_backend"],
                "--progress",
                "off",
            ]
            if entry["target_backend"] == "meshy":
                command.extend(["--meshy-frontend", "hybrid"])
            if entry.get("geometry"):
                settings = folder / "recovery-settings.json"
                save(
                    settings,
                    {"objects": {row["filename"]: {"geometry": entry["geometry"]}}},
                )
                command.extend(["--settings", str(settings)])
            if row["blender"]:
                command.append("--with-blender")
            print("Processing", entry["name"], entry["target_backend"], flush=True)
            # Separate processes/GPU contexts; restrict CPU oversubscription.
            env = {
                **os.environ,
                "OMP_NUM_THREADS": "4",
                "OPENBLAS_NUM_THREADS": "4",
                "MKL_NUM_THREADS": "4",
            }
            with (folder / "recovery-worker.log").open("ab") as log:
                result = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            report = json.loads(report_file.read_text()) if report_file.exists() else {}
            if result.returncode == 0 and report.get("exported", 0):
                state = "review" if report.get("retained_unvalidated") else "done"
                error = None
            else:
                state = "failed"
                error = (
                    "；".join(e.get("error", "") for e in report.get("errors", []))[
                        :700
                    ]
                    or "补处理未完成，见 recovery-worker.log"
                )
            with connect(root) as db:
                db.execute(
                    "UPDATE jobs SET state=?,error=?,updated=? WHERE id=?",
                    (state, error, time.time(), uid),
                )
                db.execute(
                    "DELETE FROM recovery_claims WHERE id=? AND plan=?", (uid, plan_id)
                )
            entry.update(
                recovery_state="finished",
                final_state=state,
                error=error,
                finished=time.time(),
            )
            plan["current"] = None
            save(plan_file, plan)
            print("Finished", entry["name"], state, error or "", flush=True)
        if not waiting:
            plan.update(status="finished", finished=time.time())
            save(plan_file, plan)
            return
        time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    with (args.root / ".recovery.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        process(args.root.resolve(), args.plan.resolve())


if __name__ == "__main__":
    main()
