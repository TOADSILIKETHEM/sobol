"""Job files under <output_root>/_jobs/<job_id>/ and detached launch. Shared by tui_app and job.py.

job.json (JobSpec) is written once by the TUI; job_status.json (JobStatus) is rewritten atomically
by job.py; job.log holds everything the job printed. The TUI only reads these, so closing it or
losing SSH never touches a running job.
"""
from __future__ import annotations

import csv
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

SOBOL_DIR = Path(__file__).resolve().parent
JOB_PY = SOBOL_DIR / "job.py"
JOBS_SUBDIR = "_jobs"


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


@dataclass
class JobSpec:
    label: str
    sim_argv: Optional[List[str]] = None  # runner argv; None = render-only job
    resume_batch_dir: Optional[str] = None  # set: run resume_batch.py on this batch instead of a new one
    render: Optional[Dict[str, object]] = None  # RenderFormValues as a dict (render_backend.form_to_dict)
    run_selection: object = "all"  # "all" | {"first": N} | {"ids": [...]} | {"paths": [...]}
    render_dumps_argv: Optional[List[str]] = None  # non-sobol run dir: render_dumps.py argv
    keep_dumps: bool = False  # after rendering
    omp_num_threads: int = 1
    skip_runs: List[str] = field(default_factory=list)  # run dir names not to render (Resume: rendered already)
    form_argv: Optional[List[str]] = None  # the form as typed (no forced --no-cleanup / 'd' --dry-run); Clone reads it
    host: str = field(default_factory=socket.gethostname)
    created: str = field(default_factory=_now_iso)


@dataclass
class Stage:
    name: str
    state: str = "running"  # running | done | failed | skipped
    started: str = ""
    ended: str = ""
    message: str = ""
    progress: Optional[Dict[str, object]] = None  # live phase of a running render stage (stage_progress)


@dataclass
class JobStatus:
    state: str = "running"  # running | done | partial | failed | cancelled
    pid: int = 0
    pgid: int = 0
    start_time: Optional[str] = None  # /proc start time of pid: a reused pid is not this job
    host: str = ""
    batch_dir: Optional[str] = None
    stages: List[Stage] = field(default_factory=list)


def jobs_root(output_root: Path) -> Path:
    return Path(output_root) / JOBS_SUBDIR


def new_job_dir(output_root: Path, label: str, now: Optional[datetime] = None) -> Path:
    stem = (now or datetime.now()).strftime("%Y%m%d_%H%M%S") + "_" + (
        re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("_") or "job")
    root = jobs_root(output_root)
    root.mkdir(parents=True, exist_ok=True)
    for n in range(1, 1000):
        d = root / (stem if n == 1 else f"{stem}_{n}")
        try:
            d.mkdir()
            return d
        except FileExistsError:
            continue
    raise RuntimeError(f"no free job dir for {stem} in {root}")


def _write_json_atomic(path: Path, data: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def write_spec(job_dir: Path, spec: JobSpec) -> None:
    _write_json_atomic(Path(job_dir) / "job.json", asdict(spec))


def read_spec(job_dir: Path) -> JobSpec:
    return JobSpec(**json.loads((Path(job_dir) / "job.json").read_text(encoding="utf-8")))


def write_status(job_dir: Path, status: JobStatus) -> None:
    _write_json_atomic(Path(job_dir) / "job_status.json", asdict(status))


def read_status(job_dir: Path) -> Optional[JobStatus]:
    try:
        d = json.loads((Path(job_dir) / "job_status.json").read_text(encoding="utf-8"))
        d["stages"] = [Stage(**s) for s in d.get("stages", [])]
        return JobStatus(**d)
    except (OSError, ValueError, TypeError):
        return None


def list_jobs(output_root: Path) -> List[Path]:
    root = jobs_root(output_root)
    if not root.is_dir():
        return []
    return sorted((d for d in root.iterdir() if (d / "job.json").is_file()), key=lambda d: d.name, reverse=True)


def proc_start_time(pid: int) -> Optional[str]:
    """Field 22 of /proc/<pid>/stat (start time in clock ticks since boot); None where /proc is missing."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    return stat[stat.rindex(")") + 2:].split()[19]  # fields after "(comm)" start at field 3


def pid_alive(pid: int, start_time: Optional[str] = None) -> bool:
    """pid is running and, when start_time is given, is still the process that recorded it."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # a zombie child of this process still answers kill(0); reap it if it is ours
    try:
        done, _ = os.waitpid(pid, os.WNOHANG)
        if done != 0:
            return False
    except ChildProcessError:
        pass
    if start_time is not None:
        now = proc_start_time(pid)
        if now is not None and now != start_time:
            return False  # pid reused by another process since the job died
    return True


def liveness(status: JobStatus) -> str:
    if status.state != "running":
        return status.state
    if status.host and status.host != socket.gethostname():
        return f"running on {status.host}"
    return "running" if pid_alive(status.pid, status.start_time) else "died"


def running_on_host(output_root: Path) -> List[Path]:
    out = []
    for d in list_jobs(output_root):
        st = read_status(d)
        if st is not None and liveness(st) == "running":
            out.append(d)
    return out


def launch(job_dir: Path, spec: JobSpec, script: Path = JOB_PY) -> int:
    """Start script on job_dir in its own session (no controlling terminal, so no SIGHUP on SSH drop)."""
    env = dict(os.environ, OMP_NUM_THREADS=str(spec.omp_num_threads))
    with open(Path(job_dir) / "job.log", "ab") as log:
        proc = subprocess.Popen([sys.executable, str(script), str(job_dir)], stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    return proc.pid


def cancel(status: JobStatus) -> None:
    """SIGTERM the job's process group. ProcessLookupError if the job is gone (its pid may be reused)."""
    if not pid_alive(status.pid, status.start_time):
        raise ProcessLookupError(status.pid)
    os.killpg(status.pgid or status.pid, signal.SIGTERM)


def tail(path: Path, n: int = 20) -> List[str]:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace").splitlines()[-n:]
    except OSError:
        return []


PROGRESS_FILE_ENV = "SOBOL_PROGRESS_FILE"  # render_dumps.py writes its progress_info() here


def progress_info(stage: str, resume: bool, output_dir=None, n_expected=None, path=None, index=None, total=None,
                  **_) -> Dict[str, object]:
    """Stage.progress for one on_stage(stage, **info) call of the render pipeline (tui_sim_render)."""
    d: Dict[str, object] = {"phase": stage}
    if path is not None:
        d.update(path=path, index=index, total=total)
    if output_dir is not None and stage in ("render", "compare"):
        d.update(n_expected=n_expected, dir=str(output_dir),
                 glob="frame_*.png" if stage == "render" else "compare_*.png")
        if stage == "render":
            d["log"] = str(Path(output_dir) / "render.log")
            if not resume:
                d["since"] = time.time()
    return d


def write_progress_file(path: Path, info: Dict[str, object]) -> None:
    _write_json_atomic(Path(path), info)


_PHASE_WORDS = {"convert": "converting", "preprocess": "preprocessing", "render": "rendering",
                "compare": "comparing"}


def last_line(path: Path, max_bytes: int = 4096) -> str:
    """Last non-empty line of a file that may be large (Blender's render.log); "" if unreadable."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - max_bytes))
            text = f.read().decode("utf-8", errors="replace")
    except OSError:
        return ""
    lines = [ln.strip() for ln in re.split(r"[\r\n]+", text) if ln.strip()]
    if not lines:
        return ""
    line = re.sub(r"^\d+:\d+\.\d+\s+\w+\s+\|\s*", "", lines[-1])  # Blender's "00:16.516  render  | " prefix
    return re.sub(r"'[^']*[\\/]([^'\\/]+)'", r"'\1'", line)  # a quoted path -> its file name


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:  # deleted between glob and stat
        return 0.0


def stage_progress(stage: Stage) -> str:
    """One line for a running render stage, e.g. "rendering instance_grains (path 1/2): 37/120 frames | Fra:38 ...".

    Counts the files in progress["dir"] on every call, so a poll every few seconds makes it tick."""
    p = stage.progress
    if p and p.get("file"):  # written by a child process (render_dumps.py)
        try:
            p = {**p, **json.loads(Path(str(p["file"])).read_text(encoding="utf-8"))}
        except (OSError, ValueError):
            return ""
    if not p or not p.get("phase"):
        return ""
    text = _PHASE_WORDS.get(str(p.get("phase")), str(p.get("phase")))
    if p.get("path"):
        text += f" {p['path']}" + (f" (path {p['index']}/{p['total']})" if p.get("total") else "")
    if p.get("dir"):
        d = Path(str(p["dir"]))
        since = p.get("since")  # set unless resuming: older files are left from an earlier render
        files = list(d.glob(str(p.get("glob", "*")))) if d.is_dir() else []
        n = sum(1 for f in files if since is None or _mtime(f) >= float(since))
        unit = "npz" if p.get("phase") == "convert" else "frames"
        expected = p.get("n_expected")
        text += f": {n}/{expected} {unit}" if expected is not None else f": {n} {unit}"
    if p.get("log"):
        line = last_line(Path(str(p["log"])))
        if line:
            text += " | " + (line if len(line) <= 100 else line[:99] + "…")
    return text


def _csv_rows(path: Path) -> List[Dict[str, str]]:
    try:
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except OSError:
        return []


def batch_progress(batch_dir: Path) -> str:
    total = len(_csv_rows(Path(batch_dir) / "sobol_mass_samples.csv"))
    rows = _csv_rows(Path(batch_dir) / "sobol_mass_outputs.csv")
    if rows and all(r.get("status") == "prepared_only" for r in rows):
        return f"{len(rows)}/{total} runs prepared (dry run)"
    ok = sum(1 for r in rows if r.get("status") == "ok")
    return f"{len(rows)}/{total} runs finished: {ok} ok, {len(rows) - ok} failed"
