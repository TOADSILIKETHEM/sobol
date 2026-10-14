"""Render stages for job.py and tui_app, borrowed from tui_sim_render.py (unchanged).

tui_sim_render sys.exits at import when Code/local_paths.json is missing (OzSTAR, a fresh PC),
so it is imported here on demand and a failure becomes a reason string instead of killing the
caller. render_run() is the post-sim half of tui_sim_render._run_pipeline_stages for a run dir
that already exists (a sweep run); phase 5 moves both into pipeline.py and drops this copy.
"""
from __future__ import annotations

import contextlib
import importlib
import io
import json
import math
import os
import platform
import re
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Callable, List, Optional, Union

import jobstore


def load() -> Union[ModuleType, str]:
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            t = importlib.import_module("tui_sim_render")
    except SystemExit as exc:
        detail = exc.code if isinstance(exc.code, str) else err.getvalue().strip()
        return f"render unavailable: {detail or 'tui_sim_render exited at import'}"
    except ImportError as exc:
        return f"render unavailable: {exc}"
    try:
        t.check_blender_exe()
    except FileNotFoundError as exc:
        return f"render unavailable: {exc}"
    return t


def is_sobol_run(run_dir: Path) -> bool:
    """A runner run dir the converter can read (it globs sobol_[0-9]*)."""
    run_dir = Path(run_dir)
    return (run_dir / "sobol.setup").is_file() and any(run_dir.glob("sobol_[0-9]*"))


def form_to_dict(form) -> dict:
    d = asdict(form)
    d["paths"] = list(form.paths)
    d["camera_modes"] = list(form.camera_modes)
    return d


def form_from_dict(t: ModuleType, d: dict):
    d = dict(d)
    if "camera_mode" in d:  # job.json written before camera_modes (one camera per job)
        d.setdefault("camera_modes", [d.pop("camera_mode")])
    d.pop("apophis_scale", None)
    return t.RenderFormValues(**{**d, "paths": tuple(d["paths"]),
                                 "camera_modes": tuple(d.get("camera_modes", ("auto",)))})


def _run_id(run_dir: Path) -> int:
    m = re.fullmatch(r"run_(\d+)", run_dir.name)
    return int(m.group(1)) if m else 1


def render_run(t: ModuleType, run_dir: Path, form, keep_dumps: bool = False,
               base_output_dir: Optional[Path] = None, on_line: Optional[Callable[[str], None]] = None,
               on_progress: Optional[Callable[[dict], None]] = None):
    """[convert | dump preprocess] -> each ticked path -> [compare] for one finished sobol run dir.

    Writes <run_dir>/pipeline_timing.json as tui_sim_render does (sim time null: the sim ran in a sweep).

    on_progress gets a dict per phase (jobstore.Stage.progress): where the phase writes its files and how many
    to expect, so a reader can count them while the phase runs.
    """
    report = on_progress or (lambda d: None)

    def on_stage(stage: str, **info):
        report(jobstore.progress_info(stage, form.resume, **info))

    if not form.paths:
        raise ValueError("at least one render path must be selected")
    if form.compare and len(form.paths) < 2:
        raise ValueError("compare needs at least two render paths")
    t.validate_render_form(form)
    started_at, t0 = datetime.now().astimezone(), time.monotonic()
    facts: dict = {}
    base = Path(base_output_dir) if base_output_dir is not None else t.DATA_DIR / "DEMCSVs"
    run_dir = Path(run_dir)
    record = t.RunRecord(run_id=_run_id(run_dir), mass_input_kg=math.nan, run_dir=str(run_dir), status="ok",
                         closest_approach_km=math.nan, closest_approach_au=math.nan, error="")
    # DEMDumpConvert names outputs <run dir's parent>/<run dir> (see tui_sim_render._grains_and_bodies_dirs)
    grains_dir, bodies_dir = t._grains_and_bodies_dirs(run_dir, base, run_dir.parent)
    n_dumps = t.count_matching(run_dir, "sobol_[0-9]*")
    facts["n_dumps"] = n_dumps
    if n_dumps == 0:
        return t.PipelineResult(stage="convert", ok=False, record=record,
                                message=f"no sobol_[0-9]* dumps in {run_dir} (cleaned up? rerun with --no-cleanup)")
    dump_route = not t.needs_npz(form.paths)
    if dump_route:
        n_frames, source = n_dumps, f"read {n_dumps} dump(s) directly (convert skipped)"
    else:
        report({"phase": "convert", "dir": str(grains_dir), "glob": "*.npz", "n_expected": n_dumps})
        t_convert = time.monotonic()
        try:
            t.run_convert_stage(record, base)
        except t.ConvertError as exc:
            return t.PipelineResult(stage="convert", ok=False, message=str(exc), record=record)
        facts["convert_s"] = time.monotonic() - t_convert
        n_npz = facts["n_npz"] = t.count_matching(grains_dir, "*.npz")
        if n_npz == 0:
            return t.PipelineResult(stage="convert", ok=False, record=record,
                                    message=f"no grain npz files found in {grains_dir}")
        if not keep_dumps:
            t._cleanup_run_dir(run_dir, "sobol")
        n_frames, source = n_npz, f"converted {n_npz} frame(s)"
    n_expected = facts["n_expected_frames"] = n_frames if form.max_frames is None else min(n_frames, form.max_frames)
    ctx = t.PathContext(grains_dir=grains_dir, bodies_dir=bodies_dir, batch_dir=base / run_dir.parent.name,
                        run_name=run_dir.name, render_form=form, n_expected_frames=n_expected)
    specs = [t.RENDER_PATHS[n] for n in t.RENDER_PATH_NAMES if n in form.paths]
    if dump_route:
        targets = {s.name: t._viz_dir(ctx, s) for s in specs}
        report({"phase": "preprocess"})
        t_pre = time.monotonic()
        ctx.preprocessed = t.run_dump_preprocess_stage(
            t.build_dump_preprocess_command(ctx, run_dir, targets), targets, on_line=on_line)
        facts["dump_preprocess_s"] = time.monotonic() - t_pre
    results = [r for i, s in enumerate(specs, start=1)
               for r in t.run_render_path_cameras(s, ctx, on_stage=on_stage, index=i, total=len(specs),
                                                  on_line=on_line)]
    all_ok = all(r.ok for r in results)
    message = source + "; " + "; ".join(t.format_path_result(r) for r in results)
    if dump_route and not keep_dumps:
        # same rule as the old TUI: keep the dumps while any path may need a retry from them
        pre_failed = [r.name for r in results if not r.ok and r.stage == "preprocess"]
        if not any(r.ok for r in results) or pre_failed:
            message += f"; dumps kept in {run_dir}"
        else:
            t._cleanup_run_dir(run_dir, "sobol")
    compare = None
    if form.compare:
        compare, compare_message = t.run_compare_for_paths(ctx, results, on_stage=on_stage)
        message += "; " + compare_message
        all_ok = all_ok and compare is not None
    result = t.PipelineResult(stage="done" if all_ok else "partial", ok=all_ok, message=message, record=record,
                              paths=results, compare=compare)
    if run_dir.is_dir():
        report_ = timing_report(t, form, result, facts, started_at=started_at,
                                total_s=time.monotonic() - t0, run_dir=run_dir)
        try:
            t._write_json(run_dir / "pipeline_timing.json", report_)
        except OSError as exc:
            result.message += f" (timing not written: {exc})"
        else:
            result.timing_path = run_dir / "pipeline_timing.json"
    return result


def timing_report(t: ModuleType, form, result, facts: dict, *, started_at: datetime, total_s: float,
                  run_dir: Path) -> dict:
    """tui_sim_render.build_timing_report's layout for a render of a run that already exists.

    The sim time and settings are kept from an earlier pipeline_timing.json (tui_sim_render single run)."""
    try:
        old = json.loads((run_dir / "pipeline_timing.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        old = {}
    if not isinstance(old, dict):
        old = {}
    return {
        "schema_version": t._TIMING_SCHEMA_VERSION,
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "outcome": {"stage": result.stage, "ok": result.ok},
        "wall_clock_s": {"total": total_s, "sim": (old.get("wall_clock_s") or {}).get("sim"), "convert": facts.get("convert_s"),
                         "dump_preprocess": facts.get("dump_preprocess_s")},
        "paths": [t._path_timing(r) for r in result.paths],
        "counts": {"np_apophis": t._np_apophis_from_setup(run_dir), "n_dumps": facts.get("n_dumps"),
                   "n_npz_frames": facts.get("n_npz"), "n_expected_frames": facts.get("n_expected_frames")},
        "sim": old.get("sim"),
        "render": asdict(form),
        "run_dir": str(run_dir),
        "environment": {"hostname": platform.node(), "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
                        "phantom_dir": None, "blender_exe": t.BLENDER_EXE},
        "git": {"sobol": t._git_state(t._SOBOL_REPO), "code": t._git_state(t.CODE_DIR)},
        "notes": "rendered by sobol/tui.py from an existing run: sim time and settings are in the batch "
                 "(sobol_mass_samples.csv); render_s includes the optional video encode.",
    }


_DUMP_ROUTE = ("instance_grains", "composite")


def render_dumps_argv(run_dir: Path, form, *, name: Optional[str] = None, prefix: str = "sobol",
                      dem_model: Optional[str] = None, np_apophis: Optional[int] = None, apophis_only: bool = False,
                      encounter: Optional[str] = None) -> List[str]:
    """render_dumps.py argv for a run dir that is not a sobol run (e.g. someone else's phantomflyby run)."""
    if any(p not in _DUMP_ROUTE for p in form.paths):
        raise ValueError("render_dumps.py renders instance_grains / composite only")
    if form.compare:
        raise ValueError("render_dumps.py has no compare; tick one path or render a sobol run")
    argv = [str(run_dir), "--prefix", prefix]
    if name:
        argv += ["--name", name]
    if dem_model:
        argv += ["--dem-model", dem_model]
    if np_apophis is not None:
        argv += ["--np-apophis", str(np_apophis)]
    if apophis_only:
        argv.append("--apophis-only")
    if encounter:
        argv += ["--encounter", encounter]
    argv += ["--paths", *form.paths, "--samples", str(form.samples), "--fps", str(form.fps),
             "--resolution", form.resolution, "--slowmo", str(form.slowmo)]
    if form.max_frames is not None:
        argv += ["--max-frames", str(form.max_frames)]
    if form.encode_video:
        argv.append("--video")
        if form.video_fps is not None:
            argv += ["--video-fps", str(form.video_fps)]
    if tuple(form.camera_modes) != ("auto",):
        argv += ["--camera-mode", *form.camera_modes]
    if form.camera_fov != 70.0:  # = tui_sim_render.DEFAULT_CAMERA_FOV (not imported here)
        argv += ["--camera-fov", repr(float(form.camera_fov))]
    if "composite" in form.paths and form.envelope_method != "hull":
        argv += ["--envelope-method", form.envelope_method]
    if form.resume:
        argv.append("--resume")
    return argv
