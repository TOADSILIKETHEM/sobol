#!/usr/bin/env python3
"""Render existing PHANTOM DEM dumps (no sim): dumps -> dump preprocess -> Blender.

Same stages and outputs as tui_sim_render.py's dump route (instance_grains /
composite). Dumps not named sobol_NNNNN, or a run dir without sobol.setup,
are staged first: dumps hardlinked (else symlinked, else copied) as
sobol_NNNNN next to a minimal sobol.setup, so the readers see a normal run.

    python3 sobol/render_dumps.py RUN_DIR [--name NAME] [--prefix sobol]
        [--dem-model particle|sink --np-apophis N [--apophis-only]]
        [--encounter hyperbola|ephemeris]
        [--paths instance_grains composite] [--samples 124] [--fps 24]
        [--resolution 1920x1080] [--max-frames N] [--video [--video-fps N]]
        [--camera-mode auto|grain_only] [--envelope-method hull|sdf] [--out-root DIR]

Outputs: <out-root>/<name>/<run>_viz_instance/, <run>_render_<path>/ (<run> =
the run dir's name if it is run_NNNN, else run_0001). Staged dumps go in
_staged_<run dir name>/ beside the dumps (hardlinks; symlink, then copy, as
fallbacks -- no symlinks for a Windows python.exe reader) and are removed once
the preprocess has read them.

--encounter hyperbola: the dumps' one sink is a phantomflyby Earth (as in
Mia's flybys): it is named Earth, placed and spun. Without it a lone sink comes
out named Sun and the run stops before Blender (--encounter ephemeris keeps it).
Paths and tools: Code/local_paths.json (python3 sobol/setup_paths.py).
"""
import argparse
import csv
import errno
import json
import os
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jobstore  # noqa: E402
import tui_sim_render as t  # noqa: E402

DUMP_ROUTE_PATHS = tuple(t.DUMP_PREPROCESS_FLAGS)


def link_or_copy(src: Path, dst: Path, allow_symlink: bool = True) -> str:
    """Hardlink, else symlink, else copy. A WSL symlink is unreadable from Windows
    python.exe, so allow_symlink=False goes straight to copy."""
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError as exc:
        if exc.errno not in (errno.EXDEV, errno.EPERM, errno.EACCES, errno.ENOTSUP):
            raise
    if allow_symlink:
        try:
            os.symlink(src.resolve(), dst)
            return "symlink"
        except OSError:
            pass
    shutil.copy2(src, dst)
    return "copy"


def _setup_text(dem_model: str, apophis_only: bool, np_apophis: int) -> str:
    # Only the keys viz/dump_frames.py parse_setup_flags() reads.
    return (
        f"apophis_only = {'T' if apophis_only else 'F'}\n"
        f"np_apophis = {np_apophis}\n"
        "use_dem = T\n"
        f"use_dem_as_sinks = {'T' if dem_model == 'sink' else 'F'}\n"
    )


_STAGE_MARKER = ".render_dumps_stage"


def _encounter_mode(run: Path):
    p = run / "encounter.json"
    return json.loads(p.read_text(encoding="utf-8")).get("mode", "ephemeris") if p.is_file() else None


def stage_run(src: Path, stage_root: Path, prefix: str, dem_model, apophis_only, np_apophis,
              allow_symlink: bool = True, run_name: str = "run_0001", encounter=None) -> Path:
    """Run dir the dump readers can use: src itself if it already has sobol_* dumps
    and a sobol.setup (and no --dem-model / differing --encounter override), else a
    staged copy in <stage_root>/_staged_<src name>/<run_name> -- a folder of its own
    (marked), so staging can never touch the source dumps or a real run beside them.
    encounter="hyperbola" writes encounter.json, so the readers name the lone
    perturber sink Earth (else the sink list starts at Sun)."""
    dumps = sorted(src.glob(f"{prefix}_[0-9]*"))
    if not dumps:
        sys.exit(f"no {prefix}_[0-9]* dumps in {src}")
    has_setup = (src / "sobol.setup").is_file()
    encounter_ok = encounter is None or (_encounter_mode(src) or "ephemeris") == encounter
    if prefix == "sobol" and has_setup and dem_model is None and encounter_ok:
        return src
    if dem_model is None and not has_setup:
        # dump_frames.dem_model() treats a missing setup as sink DEM; guessing
        # wrong silently misreads particle-DEM dumps, so make the user say.
        sys.exit(f"{src} has no sobol.setup (or dumps are not sobol_*): pass --dem-model particle|sink "
                 "and --np-apophis N (plus --apophis-only if there are no Earth/Moon/Sun sinks)")
    if dem_model is not None and np_apophis is None:
        sys.exit("--np-apophis is required with --dem-model (grain count; sets the min-grains filter)")
    stage = stage_root / f"_staged_{src.name}"
    if stage.exists() and not (stage / _STAGE_MARKER).is_file():
        sys.exit(f"{stage} exists and was not made by render_dumps.py; move it or pass another --stage-dir")
    run = stage / run_name
    run.mkdir(parents=True, exist_ok=True)
    (stage / _STAGE_MARKER).write_text(f"staged from {src}\n", encoding="utf-8")
    for old in run.glob("sobol_[0-9]*"):
        old.unlink()
    kinds = {link_or_copy(d, run / f"sobol_{i:05d}", allow_symlink) for i, d in enumerate(dumps)}
    (run / "sobol.setup").unlink(missing_ok=True)
    if dem_model is None:  # only --encounter differs: keep the run's own setup
        link_or_copy(src / "sobol.setup", run / "sobol.setup", allow_symlink)
    else:
        (run / "sobol.setup").write_text(_setup_text(dem_model, bool(apophis_only), np_apophis), encoding="utf-8")
    # hyperbola runs: the readers name the lone perturber sink "Earth" from this
    (run / "encounter.json").unlink(missing_ok=True)
    if encounter is not None:
        (run / "encounter.json").write_text(json.dumps({"mode": encounter}) + "\n", encoding="utf-8")
    elif (src / "encounter.json").is_file():
        link_or_copy(src / "encounter.json", run / "encounter.json", allow_symlink)
    print(f"staged {len(dumps)} dump(s) in {run} ({'/'.join(sorted(kinds))})", flush=True)
    return run


def lone_sun_sink(bodies_dir: Path) -> bool:
    """True when the first bodies CSV has exactly one body besides Apophis and it
    came out named Sun: a phantomflyby Earth read without --encounter hyperbola."""
    csvs = sorted(bodies_dir.glob("*.csv"))
    if not csvs:
        return False
    with open(csvs[0], newline="", encoding="utf-8") as fh:
        names = [row["name"] for row in csv.DictReader(fh) if row.get("name") != "Apophis"]
    return names == ["Sun"]


def _parse(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("run_dir", type=Path)
    p.add_argument("--name", help="output folder name (default: run dir's parent name)")
    p.add_argument("--prefix", default="sobol", help="dump file prefix (default sobol)")
    p.add_argument("--dem-model", choices=("particle", "sink"))
    p.add_argument("--apophis-only", action="store_true", help="dumps hold no Earth/Moon/Sun sinks")
    p.add_argument("--np-apophis", type=int, help="grain count (with --dem-model)")
    p.add_argument("--encounter", choices=("hyperbola", "ephemeris"),
                   help="hyperbola: the one sink is a phantomflyby Earth; ephemeris: sinks start at Sun")
    p.add_argument("--paths", nargs="+", default=["instance_grains"])
    p.add_argument("--samples", type=int, default=t.DEFAULT_SAMPLES)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--resolution", default="1920x1080")
    p.add_argument("--max-frames", type=int)
    p.add_argument("--video", action="store_true", help="also encode an mp4")
    p.add_argument("--video-fps", type=int, help="mp4 frame rate (default: --fps)")
    p.add_argument("--camera-mode", choices=("auto", "grain_only"), default="auto")
    p.add_argument("--envelope-method", choices=("hull", "sdf"), default="hull", help="composite envelope")
    p.add_argument("--slowmo", type=int, default=1,
                   help="instance paths: N frames per dump, in-betweens interpolated (own _slowmoN folder)")
    p.add_argument("--resume", action="store_true", help="keep frames already rendered, render the rest")
    p.add_argument("--out-root", type=Path, default=t.DATA_DIR / "DEMCSVs")
    p.add_argument("--stage-dir", type=Path,
                   help="where renamed dumps are staged, in _staged_<run dir name>/ (default: beside "
                        "the dumps, so hardlinks; removed after the preprocess)")
    a = p.parse_args(argv)
    if a.dem_model is None and (a.np_apophis is not None or a.apophis_only):
        p.error("--np-apophis / --apophis-only only apply with --dem-model "
                "(without it the run's own sobol.setup is used as is)")
    return a


def main(argv=None) -> int:
    a = _parse(argv)
    bad = [p for p in a.paths if p not in DUMP_ROUTE_PATHS]
    if bad:
        sys.exit(f"{bad}: only {list(DUMP_ROUTE_PATHS)} render straight from dumps; "
                 "per_sphere / instance_static need the npz convert -- use sobol/tui_sim_render.py")
    form = t.RenderFormValues(resolution=a.resolution, samples=a.samples, fps=a.fps,
                              max_frames=a.max_frames, encode_video=a.video, paths=tuple(a.paths),
                              slowmo=a.slowmo, resume=a.resume, camera_mode=a.camera_mode,
                              video_fps=a.video_fps, envelope_method=a.envelope_method)
    try:
        t.validate_render_form(form)
    except ValueError as exc:
        sys.exit(str(exc))
    t.check_blender_exe(t.BLENDER_EXE)
    src = a.run_dir.expanduser().resolve()
    # absolute: the preprocess runs with cwd=Code/ and Blender from its own dir
    out_root = a.out_root.expanduser().resolve()
    name = a.name or src.parent.name
    batch = out_root / name
    # sobol_mass_runs/<batch>/run_0007 -> DEMCSVs/<batch>/run_0007_*, as the TUI names it
    run_name = src.name if re.fullmatch(r"run_\d+", src.name) else "run_0001"
    # Windows python.exe (WSL host) cannot follow WSL symlinks
    windows_reader = t._paths.is_windows_tool(t.WIN_VENV_PYTHON)
    if a.stage_dir is not None:
        stage_root = a.stage_dir.expanduser().resolve()
    else:  # beside the dumps = same filesystem = hardlinks, no copy
        stage_root = src.parent if os.access(src.parent, os.W_OK) else batch
    run = stage_run(src, stage_root, a.prefix, a.dem_model, a.apophis_only, a.np_apophis,
                    allow_symlink=not windows_reader, run_name=run_name, encounter=a.encounter)
    n = len(list(run.glob("sobol_[0-9]*")))
    ctx = t.PathContext(grains_dir=batch / "unused_grains", bodies_dir=batch / f"{run_name}_bodies_output",
                        batch_dir=batch, run_name=run_name, render_form=form, n_expected_frames=n)
    targets = {}
    for path in a.paths:
        d = batch / f"{run_name}{t.RENDER_PATHS[path].viz_dir_suffix}"
        d.mkdir(parents=True, exist_ok=True)
        targets[path] = d
    say = lambda line: None if "Fra:" in line else print(line, flush=True)  # noqa: E731
    # a job (sobol/job.py) asks for progress in a file, so the TUI's Jobs tab can count frames
    progress_file = os.environ.get(jobstore.PROGRESS_FILE_ENV)

    def progress(stage, **info):
        try:
            jobstore.write_progress_file(Path(progress_file), jobstore.progress_info(stage, a.resume, **info))
        except OSError:
            pass  # progress is a view; never stop a render over it

    if progress_file:
        progress("preprocess")
    try:
        ctx.preprocessed = t.run_dump_preprocess_stage(t.build_dump_preprocess_command(ctx, run, targets),
                                                       targets, on_line=say)
    finally:
        if run != src and (run.parent / _STAGE_MARKER).is_file():
            shutil.rmtree(run.parent)  # the renders read only the viz dirs
    if a.encounter is None and lone_sun_sink(ctx.bodies_dir):
        print(f"{src}: the dumps have one perturber sink, read as Sun. If it is a phantomflyby Earth, "
              "pass --encounter hyperbola (Earth placed + spun); --encounter ephemeris renders it as Sun.",
              file=sys.stderr)
        print("RENDER_FAILED", flush=True)
        return 1
    ok = True
    extra = {"on_stage": progress} if progress_file else {}
    for i, path in enumerate(a.paths, start=1):
        if progress_file:
            extra.update(index=i, total=len(a.paths))
        r = t.run_render_path(t.RENDER_PATHS[path], ctx, on_line=say, **extra)
        print(t.format_path_result(r), flush=True)
        ok = ok and r.ok
    print("RENDER_DONE" if ok else "RENDER_FAILED", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
