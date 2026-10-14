#!/usr/bin/env python3
"""Run one job headlessly: sweep/single run (runner or resume_batch.py), then renders.

    python3 sobol/job.py <job_dir>     # launched detached by jobstore.launch(); stdout -> job.log

Reads <job_dir>/job.json, writes <job_dir>/job_status.json after every stage change. SIGTERM
(Cancel in the TUI, sent to the whole process group) marks the job cancelled.
"""
from __future__ import annotations

import csv
import os
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jobstore  # noqa: E402
import render_backend  # noqa: E402

SOBOL_DIR = jobstore.SOBOL_DIR
RESUME = SOBOL_DIR / "resume_batch.py"
RENDER_DUMPS = SOBOL_DIR / "render_dumps.py"
REPORT_ENV = "SOBOL_REPORT_BATCH_DIR"  # = run_mass_sobol_phantom.REPORT_BATCH_DIR_ENV (not imported: heavy)
BLENDER_NOTE = "cancelled; a Windows blender.exe started by this job may still be running (check Task Manager)"


class Cancelled(Exception):
    pass


def _on_sigterm(_signum, _frame):
    raise Cancelled()


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _runner_script() -> Path:
    return Path(os.environ.get("SOBOL_JOB_RUNNER", SOBOL_DIR / "run_mass_sobol_phantom.py"))


class _Job:
    def __init__(self, job_dir: Path) -> None:
        self.dir = job_dir
        self.spec = jobstore.read_spec(job_dir)
        self.status = jobstore.JobStatus(pid=os.getpid(), pgid=os.getpgid(0),
                                         start_time=jobstore.proc_start_time(os.getpid()),
                                         host=jobstore.socket.gethostname())

    def save(self) -> None:
        try:
            jobstore.write_status(self.dir, self.status)
        except OSError as exc:  # status is a view; never stop the job over it
            print(f"[job] status not written: {exc}", flush=True)

    def begin(self, name: str) -> jobstore.Stage:
        stage = jobstore.Stage(name, started=_now())
        self.status.stages.append(stage)
        self.save()
        print(f"[job] {name}", flush=True)
        return stage

    def progress(self, stage: jobstore.Stage, info: dict) -> None:
        stage.progress = info
        self.save()

    def end(self, stage: jobstore.Stage, state: str, message: str = "") -> None:
        stage.state, stage.ended, stage.message, stage.progress = state, _now(), message, None
        self.save()
        print(f"[job] {stage.name}: {state} {message}", flush=True)


def _ok_runs(batch_dir: Path) -> Tuple[List[Tuple[int, Path]], int]:
    """(run_id, run_dir) of ok rows, and the number of rows that were not ok."""
    try:
        with open(batch_dir / "sobol_mass_outputs.csv", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except OSError:
        return [], 0
    ok = [(int(r["run_id"]), Path(r["run_dir"])) for r in rows if r.get("status") == "ok"]
    return sorted(ok), len(rows) - len(ok)


def select_runs(ok: List[Tuple[int, Path]], selection) -> Tuple[List[Path], List[str]]:
    """Run dirs to render, and run ids asked for that are not ok (failed or missing)."""
    if selection == "all":
        return [p for _, p in ok], []
    if isinstance(selection, dict) and "first" in selection:
        return [p for _, p in ok[: int(selection["first"])]], []
    if isinstance(selection, dict) and "ids" in selection:
        by_id = dict(ok)
        return ([by_id[i] for i in selection["ids"] if i in by_id],
                [str(i) for i in selection["ids"] if i not in by_id])
    raise ValueError(f"unknown run selection {selection!r}")


def _sim_prefix(argv: List[str]) -> str:
    for i, a in enumerate(argv):
        if a.startswith("--prefix="):
            return a.split("=", 1)[1]
        if a == "--prefix" and i + 1 < len(argv):
            return argv[i + 1]
    return "sobol"


def _clean_unrendered(job: _Job, ok: List[Tuple[int, Path]], rendered: List[Path]) -> None:
    """Render after forced --no-cleanup on the whole sweep; delete the dumps of ok runs it did not render,
    as the runner would have (unless keep_dumps). Failed runs keep theirs, as with the runner."""
    if job.spec.keep_dumps:
        return
    chosen = {Path(p) for p in rendered}
    others = [p for _, p in ok if Path(p) not in chosen and Path(p).is_dir()]
    if not others:
        return
    from run_mass_sobol_phantom import _cleanup_run_dir  # heavy import, only when needed
    stage = job.begin("cleanup")
    prefix = _sim_prefix(job.spec.sim_argv or [])
    for run_dir in others:
        _cleanup_run_dir(run_dir, prefix)
    job.end(stage, "done", f"deleted dumps of {len(others)} run(s) not rendered")


def _close_running(job: _Job, message: str) -> None:
    for stage in job.status.stages:
        if stage.state == "running":
            stage.state, stage.ended, stage.message, stage.progress = "failed", _now(), message, None


def _sim(job: _Job) -> Tuple[Optional[Path], List[Tuple[int, Path]], bool]:
    """Returns (batch_dir, ok runs, any problem). Raises Cancelled."""
    spec = job.spec
    stage = job.begin("sim")
    report = job.dir / "batch_dir.txt"
    if spec.resume_batch_dir:
        cmd = [sys.executable, str(RESUME), "--batch-dir", spec.resume_batch_dir, *spec.sim_argv]
    else:
        cmd = [sys.executable, str(_runner_script()), *spec.sim_argv]
    rc = subprocess.call(cmd, env=dict(os.environ, **{REPORT_ENV: str(report)}))
    batch = Path(spec.resume_batch_dir) if spec.resume_batch_dir else (
        Path(report.read_text(encoding="utf-8").strip()) if report.is_file() else None)
    job.status.batch_dir = str(batch) if batch else None
    if "--dry-run" in spec.sim_argv:
        job.end(stage, "done" if rc == 0 else "failed", f"dry run, exit {rc}; prepared {batch}")
        return batch, [], rc != 0
    ok, not_ok = _ok_runs(batch) if batch else ([], 0)
    message = f"exit {rc}; {len(ok)} ok, {not_ok} not ok" + (f"; batch {batch}" if batch else "")
    job.end(stage, "done" if ok else "failed", message)
    return batch, ok, not_ok > 0 or rc != 0


def _render(job: _Job, run_dirs: List[Path], skipped: List[str]) -> bool:
    """Render each run dir; True when every one rendered."""
    spec = job.spec
    all_ok = True
    if skipped:
        stage = job.begin("select")
        job.end(stage, "skipped", "not rendered (failed or not in batch): run " + ", ".join(skipped))
        all_ok = False
    backend = render_backend.load()
    if isinstance(backend, str):
        stage = job.begin("render")
        job.end(stage, "failed", backend)
        return False
    form = render_backend.form_from_dict(backend, spec.render)
    for run_dir in run_dirs:
        stage = job.begin(f"render {run_dir.name}")
        try:
            if not render_backend.is_sobol_run(run_dir):
                job.end(stage, "failed", f"{run_dir} is not a sobol run with dumps")
                all_ok = False
                continue
            res = render_backend.render_run(backend, run_dir, form, keep_dumps=spec.keep_dumps,
                                            on_progress=lambda info, stage=stage: job.progress(stage, info))
        except Cancelled:
            job.end(stage, "failed", BLENDER_NOTE)
            raise
        except Exception as exc:  # one bad run must not stop the others
            job.end(stage, "failed", f"{type(exc).__name__}: {exc}")
            all_ok = False
            continue
        timing = "; ".join(f"{p.name} {p.n_frames} frames {p.elapsed_s:.0f} s" for p in res.paths)
        job.end(stage, "done" if res.ok else "failed", res.message + (f" | {timing}" if timing else ""))
        all_ok = all_ok and res.ok
    return all_ok


def _render_dumps(job: _Job) -> bool:
    stage = job.begin("render_dumps")
    progress_file = job.dir / "progress.json"  # render_dumps.py writes its phase here; the TUI reads it
    progress_file.unlink(missing_ok=True)
    job.progress(stage, {"file": str(progress_file)})
    try:
        rc = subprocess.call([sys.executable, str(RENDER_DUMPS), *job.spec.render_dumps_argv],
                             env=dict(os.environ, **{jobstore.PROGRESS_FILE_ENV: str(progress_file)}))
    except Cancelled:
        job.end(stage, "failed", BLENDER_NOTE)
        raise
    job.end(stage, "done" if rc == 0 else "failed", f"exit {rc}")
    return rc == 0


def main(argv: List[str]) -> int:
    job = _Job(Path(argv[0]))
    signal.signal(signal.SIGTERM, _on_sigterm)
    job.save()
    spec = job.spec
    try:
        problem = False
        if spec.render_dumps_argv:
            problem = not _render_dumps(job)
        elif spec.sim_argv is not None:
            batch, ok, problem = _sim(job)
            if not ok and "--dry-run" not in spec.sim_argv:
                job.status.state = "failed"
                job.save()
                return 1
            if spec.render and ok:
                run_dirs, skipped = select_runs(ok, spec.run_selection)
                done_before = set(spec.skip_runs)
                todo = [p for p in run_dirs if p.name not in done_before]
                if len(todo) < len(run_dirs):
                    print(f"[job] {len(run_dirs) - len(todo)} run(s) rendered by the original job; not again",
                          flush=True)
                problem = not _render(job, todo, skipped) or problem
                _clean_unrendered(job, ok, run_dirs)
        elif spec.render:
            problem = not _render(job, [Path(p) for p in spec.run_selection["paths"]], [])
        job.status.state = "partial" if problem else "done"
    except Cancelled:
        job.status.state = "cancelled"
        report = job.dir / "batch_dir.txt"
        if job.status.batch_dir is None and report.is_file():  # runner never returned; Resume needs the batch
            job.status.batch_dir = report.read_text(encoding="utf-8").strip() or None
        _close_running(job, "cancelled")
    except Exception as exc:
        _close_running(job, f"stopped by {type(exc).__name__}")
        stage = job.begin("error")
        job.end(stage, "failed", f"{type(exc).__name__}: {exc}")
        job.status.state = "failed"
    job.save()
    return 0 if job.status.state == "done" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
