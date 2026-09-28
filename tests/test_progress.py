import io, json, os, subprocess, sys
from pathlib import Path
import pytest
from packing_assets.progress import ProgressSession, event


def test_progress_records_real_counts_and_cached_stages(tmp_path):
    stream = io.StringIO()
    with ProgressSession(tmp_path, "plain", verbose=True, stream=stream) as p:
        event("phase", "reconstruction", 2)
        event("item", "键盘.mp4", 1)
        event("stage", "prepare", cached=True)
        state = json.loads((tmp_path / "progress.json").read_text())
        assert state["cached"] and state["completed"] == 0
        event("stage", "photo_masks_v3")
        event("advance", 12, 48)
        assert (
            json.loads((tmp_path / "progress.json").read_text())["detail"]["done"] == 12
        )
        event("item_done")
        event("item", "broken.mp4", 2)
        event("item_done", "failed")
        event("finish", "partial")
    state = json.loads((tmp_path / "progress.json").read_text())
    assert (
        state["completed"] == 2
        and state["failed"] == 1
        and state["status"] == "partial"
    )
    assert "[缓存]" in stream.getvalue() and "\x1b" not in stream.getvalue()


def test_progress_error_is_not_reported_as_completion(tmp_path):
    with pytest.raises(ValueError):
        with ProgressSession(tmp_path, "off", verbose=True, stream=io.StringIO()):
            event("phase", "reconstruction", 2)
            raise ValueError("bad input")
    state = json.loads((tmp_path / "progress.json").read_text())
    assert (
        state["status"] == "failed"
        and state["completed"] == 0
        and state["error"] == "bad input"
    )
    event("stage", "no active reporter")


def test_native_logs_are_redirected_and_file_descriptors_restored(tmp_path):
    code = """
from pathlib import Path
from packing_assets.progress import ProgressSession,event
import os,sys
with ProgressSession(Path(sys.argv[1]),'plain'):
    event('phase','reconstruction',1)
    print('python detail',flush=True)
    os.write(2,b'native detail\\n')
    event('item_done');event('finish','complete')
print('stdout restored',flush=True)
os.write(2,b'stderr restored\\n')
"""
    run = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "python detail" not in run.stdout + run.stderr
    assert "native detail" not in run.stdout + run.stderr
    assert "stdout restored" in run.stdout and "stderr restored" in run.stderr
    assert "python detail" in (tmp_path / "run.log").read_text()
    assert "native detail" in (tmp_path / "run.log").read_text()


def test_tty_uses_a_bar_and_off_is_quiet(tmp_path):
    class Terminal(io.StringIO):
        def isatty(self):
            return True

    terminal = Terminal()
    with ProgressSession(tmp_path / "tty", "auto", verbose=False, stream=terminal):
        event("phase", "export", 2)
        event("item_done")
        event("finish", "complete")
    assert "[##########----------]" in terminal.getvalue()
    quiet = io.StringIO()
    with ProgressSession(tmp_path / "quiet", "off", verbose=True, stream=quiet):
        event("finish", "complete")
    assert not quiet.getvalue()


def test_terminal_width_handles_chinese_and_control_characters():
    from packing_assets.progress import fit_line

    assert fit_line("键盘模型", 5) == "键盘…"
    assert fit_line("a\nb\x1bc", 10) == "abc"
