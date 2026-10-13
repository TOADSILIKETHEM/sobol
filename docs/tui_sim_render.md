# Sim + render pipeline TUI (`tui_sim_render.py`)

Parent: `../CLAUDE.md`. Shared facts: `../docs/SHARED_FACTS.md`. Merge/rename caveats: `../docs/CONTEXT_CHANGELOG.md`.

`sobol/tui_sim_render.py` is a Textual TUI that takes **one** DEM run from configure → PHANTOM → dump conversion → headless Blender render (any combination of render paths), with no manual steps. It handles one run per launch, not a sweep; `tui_run.py` is the sweep TUI. Windows render-script how-to: `Code/CLAUDE.md`.

**Stages.** `run_pipeline()` calls each stage in order in one worker thread and stops at the first failure. Each stage reuses existing code:
1. **Sim.** `run_sim_stage()` calls `preflight()` and `run_one_case(run_id=1, ...)` from `run_mass_sobol_phantom.py`. That means DEM `nfulldump=1`, ephemeris copy, and metrics extraction all behave exactly as in a sweep. Any `record.status != "ok"` stops the pipeline; `prepared_only` is reported as a dry-run success.
2. **Convert.** `run_convert_stage()` calls `Analysis/run_demtocsv_batch.run_dem_dump_convert()` with `min_dem_grains` read automatically from the run's `sobol.setup`, writing Windows `DEMCSVs/<batch>/run_0001_{bodies,grains}_output/`. `DEMDumpConvert.py` never raises on bad or missing dumps, so the pipeline then counts `*.npz` and fails at `convert` if there are 0. The converter hardcodes the `sobol_[0-9]*` dump prefix, so keep `prefix = sobol`. Once the npz count is verified, raw dumps, `.ev` and `phantom.log` are **deleted by default** (renders read only the converted files); tick `keep_dumps` to keep them.
   **Skipped** when no ticked path needs npz (`needs_npz`: only `per_sphere` and `instance_static` read it). Then one `viz/viz_preprocess_dump.py` call (Windows `.venv`) reads each dump once for composite and instance_grains together and writes the bodies CSVs. If one path's builder fails, that path fails at preprocess and the other still renders. Dumps are deleted after the render paths (unless `keep_dumps`), but kept if every path failed or any path failed at preprocess (the dumps are then the only way to retry it). 0 dumps fails at `convert`.
3. **Render paths.** Tick any combination on the form; one sim + one convert feed all of them, run in fixed order `per_sphere`, `composite`, `instance_grains`, `instance_static`. Each path is `[preprocess] → DEMHeadlessRender.py --viz-path …`:
   - `per_sphere` — no preprocess; `DEMGrainsBlenderEarthCam.py` (≤~2000 grains).
   - `composite` (D3) — `viz/viz_preprocess.py --envelope-method hull|sdf` in the Windows `.venv` python → `DEMBulkVizBlender.py`.
   - `instance_grains` — `viz/viz_preprocess_grains_instance.py` in the Windows `.venv` python (real grains, per-grain radius, sorted by `grain_id`) → `DEMBulkInstanceBlender.py` (`static: false`).
   - `instance_static` — `viz/viz_preprocess_lite.py --shape-obj --n-points` under `blender.exe --background` → `DEMBulkInstanceBlender.py` (`static: true`; placeholder, not sim motion).
   A failed path is recorded and the remaining paths still run; result `done` only when every ticked path succeeds, else `partial`. Optional video encode (`encode_video`/`video_fps`) runs per path, inside `run_render_path()`, writing each path's `animation.mp4` next to its own `frame_*.png` sequence.
4. **Compare (optional).** Tick `compare` (needs ≥ 2 ticked paths). After the render paths, `run_compare_for_paths()` passes every path that rendered to `render_compare.run_compare_stage()` (WSL Pillow). It pairs frames by number, pastes them left to right in path order with a label box (width ≤ 3840, even dims), and writes `run_0001_compare/compare_####.png`. With `encode_video` it also writes `animation.mp4` via WSL `ffmpeg` (libx264, `video_fps` or `fps`). Differing frame counts → compares the shared frames and says so in the result. Fewer than 2 paths rendered → skipped. Any compare failure → result `partial`; per-path renders are never touched. Standalone on existing renders: `python3 sobol/render_compare.py DEMCSVs/<batch>/run_0001_render_composite DEMCSVs/<batch>/run_0001_render_instance_grains [--encode-video]`.

**Output layout.** Each launch creates `<output_root>/<prefix>_<YYYYmmdd_HHMMSS>_<batch_label>/run_0001/`; the label defaults to `sim_render` and is sanitised with `sanitize_batch_label()`. OneDrive `Code/DEMCSVs/<batch>/` (`tui_sim_render.DATA_DIR`): `run_0001_{grains,bodies}_output/`, `run_0001_viz/` (composite), `run_0001_viz_instance/` (instance_grains), `run_0001_viz_instance_static/` (instance_static) (dump route: no `run_0001_grains_output/`; `run_0001_viz/grains/` holds the x/y/z files the composite camera reads), and one `run_0001_render_<path>/` per ticked path with `frame_####.png` (+ `animation.mp4` if `encode_video`), and `run_0001_compare/` (`compare_####.png` + `animation.mp4`) when `compare` is ticked. Rename from v1 `run_0001_render/`: `../docs/CONTEXT_CHANGELOG.md`.

**Logs and timing.** Every preprocess/render subprocess runs through `_run_logged()`: stdout+stderr merged, streamed line by line to `preprocess.log` (in that path's `run_0001_viz*/`; the shared dump-route call logs to the first ticked path's viz dir) or `render.log` (in `run_0001_render_<path>/`), with only a bounded tail kept for error messages. The latest line is appended to the status bar. Windows `.venv` preprocess runs `python.exe -u` and `DEMHeadlessRender.py` line-buffers stdout so lines arrive live; `instance_static` under `blender.exe` logs at exit. After any non-dry run whose `run_dir` exists, `run_pipeline()` writes `<run_dir>/pipeline_timing.json` (`build_timing_report()`): wall-clock `total`/`sim`/`convert`/`dump_preprocess`; per path `preprocess_s` (null for dump-route paths), `render_s` (includes video encode), `n_frames`, `render_s_per_frame`; `np_apophis` from the run's `sobol.setup`; dump/npz counts; sim sample + render form; hostname, `OMP_NUM_THREADS`; `sobol` and `Code` git SHA + `dirty`. A write failure is noted in the status and never fails the run. Per-sphere `max_frames` is patched into `MAX_FRAMES` so only those frames are keyframed.

**Form.** Everything is on a single screen, with **Run Pipeline**, **Dry Run** and **Quit** buttons.
- **Paths:**
  - `prefix` (`sobol`) and `batch_label`.
  - `base_dir`, which defaults to `sobol/`.
  - `phantom_dir`, which defaults to `PHANTOM_DIR` if set, else `sobol/`.
  - `output_root`, which defaults to `Honours/sobol_mass_runs`. These defaults are computed from `__file__`, so they are the same whatever directory you launch from.
  - `ephemeris_cache_dir`. Set it to `/home/mboyle/Honours/sobol`. Its placeholder text is misleading: leaving it **blank means PHANTOM attempts a live Horizons download**, because no `*.txt` files get copied.
- **Sim:**
  - `np_apophis` (default 500).
  - `spin_period (hr)` and `spin_torque_align (deg)`.
  - `kt_cgs`: blank keeps the template value; `0` writes 0.
  - `dn_cohes_factor` (0.1). Passed only when `kt_cgs > 0`; the runner's `resolve_coh_gap_after_setup()` converts it after setup (`dn × 2 × R_grain`).
  - Friction / cohesion (Mia, merged 2026-10-06; particle DEM): `mu_s` (blank = `DEFAULT_MU_S` 0, frictionless; binary default 0.5), `mu_r`, `mu_t` (need `mu_s > 0`), `epsilon_t_dem` ((0,1]), `cohesion_pa` (Pa; not with `kt_cgs > 0`). Blank keeps the binary default for all but `mu_s`. Bad combinations are refused in the form; a binary without friction is refused before setup.
  - `tmax` (default `108` hr = template 4.5 days) and `dtmax` (default `0.5` hr = template 30 min). Clearing a box falls back to the template.
  - `sink_earth_id` (4) and `sink_apophis_id` (11).
  - `shape_file`: a non-blank value turns on `use_shape_crop`.
  - `keep_dumps` (default off): keep raw dumps + `.ev` in `sobol_mass_runs/<batch>/run_0001` after convert.
  - `use_dem` is hardcoded to `T`, because the convert and render stages need DEM grains.
- **Sim — body / encounter** (Sep 2026; CLI equivalent + cache detail: `../CLAUDE.md` § "Settled bodies and
  hyperbola encounters"):
  - `body_source` (`lattice` | `settled`) and `encounter` (`ephemeris` | `hyperbola`). `hyperbola` needs `settled`.
    `settled` + `ephemeris` runs the cached body on the real 2029 orbit via `packing_file` (CLI detail
    `../CLAUDE.md`); verified 2026-09-27 headless np300 → convert → `per_sphere`.
  - `flyby_rp_km` (default `38000`), `flyby_vinf_kms` (default `5.9`), `flyby_start_sep_km` (`4e5`): read only for
    `hyperbola` (prefilled values never reach an ephemeris sample/CSV). `shape_file` is ignored for `settled` (the body
    carries its own shape).
  - Hyperbola validation at click: tmax + dtmax set, spin fields blank, then `encounter.check_flyby_geometry()`
    (same rule as the CLI: values > 0, `R_EARTH_KM (6371) < rp < start_sep`, tmax ≥ 2 × time to pericentre).
  - `run_sim_stage()` builds/loads the settled body (`attach_settled_bodies`, cache
    `<output_root parent>/settled_bodies`, same cache as the CLI) before `run_one_case`; for hyperbola it also
    resolves `phantomflyby` and swaps `sink_earth_id` 4 → 1. `settle_tdyn`/`relax_tdyn`/`body_cache_dir` are `SimParams`
    fields (defaults 5.0 / 0.5 / shared cache), not form rows.
  - Verified 2026-09-27: headless form → np300 settled hyperbola (rp 38000, v_inf 5.9) → convert → `per_sphere`
    render, pericentre |r| 38050 km.
- **Sim — boulders (settled only)** (Oct 2026; CLI equivalent + semantics: `../CLAUDE.md` § `--boulders`):
  - `boulders` checkbox (default off). It can only be ticked on a settled body; if `body_source` changes afterwards it stays ticked and enabled (so it can be unticked) and Run refuses it.
  - Selects `boulder_shape` (`round` | `ellipsoid` | `grown`, random boulders) and `boulder_placed` (`none` | `binary` | `file`).
  - Inputs, all fixed values (no sweep in this one-run TUI): `f_boulder` (0-1, default 0.3), `rb_min` (1.5), `rb_max` (4), `q_slope` (3.5), `axis_min` (ellipsoid), `jagged` (grown), `lobe_ratio` (binary), `boulder_file` (file), `nmin_clump` (>= 2), `seed`, `bond_sigma_pa`, `tau/sigma` (`bond_tau_ratio`, default 1), `kb_cgs` (blank = `kn_cgs`), `bond_reach` (0.1). A blank box keeps the `BoulderSpec` default.
  - **`bond_sigma_pa` is required** (Pa; 0 = stretch-limit breaking only; no default). `boulders.validate_spec()` runs at click.
  - `run_sim_stage()` resolves `phantomclumps` and refuses a `phantom` without clump bonds (`boulders.attach_tools`) after the body is built; the run then clumps the cached body into `run_0001/` (`clumps.log`, `boulders.json`) on either encounter. Needs `make demtools`.
- **Render:**
  - `resolution` (1920x1080), `samples` (124), `fps` (24).
  - `max_frames` (blank means all).
  - `camera_mode` (`auto` | `grain_only`, which patches `CAMERA_MODE`).
  - `encode_video` checkbox (off by default) + `video_fps` (blank = same as `fps`) — optional final MP4-encode stage, applied per ticked render path.
  - `slowmo` (default 1) — instance paths only: N frames per dump over the same sim time (in-betweens interpolated on the grains' Bezier curves, not simulated). Renders (n−1)·N+1 frames into its own `run_0001_render_<path>_slowmoN/`; composite / per_sphere in the same run stay 1:1. Refused with no instance path ticked, and with `compare` (frame counts differ). Keep `video_fps` at 24 (Max's standard): slowmo N gives an N× longer video, not a smoother clip of the same length. `validate_render_form()` holds these rules for the TUI, `run_pipeline` and `render_dumps.py`.
  - `resume` checkbox — keep `frame_*.png` already in the output folder and render only the missing ones (carry on a stopped render; `DEMHeadlessRender.py --resume`).
  - Render fields are validated when you click, before the sim starts.
- **Render paths:**
  - Four checkboxes, any combination: `per_sphere` (legacy, parity reference), `composite`, `instance_grains` (ticked by default since 2026-09-29, so a default launch takes the dump route with no convert), `instance_static`.
  - `envelope_method` (`hull` | `sdf`, default `hull`) — path-specific field for `composite`.
  - `placeholder_obj` (Windows copy of `apophis_v233s7.obj`) and `n_points` (default `1000000`) — path-specific fields for `instance_static`.
  - Path-specific fields are validated only when their path is ticked.
  - `compare` checkbox (off by default): side-by-side of the ticked paths; validated at click (≥ 2 paths).
  - `.venv` python existence is checked at click time when `composite` or `instance_grains` is ticked (each runs its preprocess script in the Windows `.venv`).

**Behaviour.**
- At startup it checks `BLENDER_EXE` exists (from `Code/honours_paths.py`; default `/mnt/c/Program Files/Blender Foundation/Blender 5.2/blender.exe`; a bare name on `PATH` is accepted). Tool path args are Windows-form only when the tool is a `.exe` (`blender_arg` / `python_arg`).
- The status bar updates about every 2 s. During sim it counts `<prefix>_[0-9]*` dump files (`Running sim...`). When PHANTOM exits it flips to `Converting... N/M npz`, then cycles per ticked path (dump route instead: `Preprocessing dumps... k/N dump(s)`, counting composite's camera files, then renders only): `Preprocessing [<path> i/n]...` (composite/instance_grains/instance_static only) and `Rendering [<path> i/n]... k/N frame(s)` (`frame_*.png` under that path's Windows render dir), then `Comparing... k/N frame(s)` (`compare_*.png`) when `compare` is ticked. The last dump count no longer freezes the bar.
- The final status is one of: `[Ns] Done (done): converted N frame(s); rendered M frame(s) to …` (dump route: `read N dump(s) directly (convert skipped); …`), `[Ns] Finished with failures: …` (`partial` — one or more ticked render paths failed while others succeeded), `Failed at sim|convert: …`, or `Failed at error: <Type>: …` for an unexpected exception.
- Only one pipeline can run at a time: the buttons are disabled and a second launch is refused.
- `np_apophis > 2000` prepends a non-blocking warning only when `per_sphere` is ticked, because that path is meant for ≤~2000 grains.

**Limits (v1).**
- No mid-run cancel. Quitting mid-run leaves PHANTOM or Blender child processes running.
- No streamed Blender log.
- No `OMP_NUM_THREADS` field (TODO, Max 2026-10-07): PHANTOM (settle + flyby) inherits the launching shell's env; unset → all 16 logical CPUs. Launch as `OMP_NUM_THREADS=4 python3 sobol/tui_sim_render.py` until the form has it.
- The `instance_static` camera tracks the origin (static placeholder); `instance_grains` tracks the inner-80% grain core like `per_sphere`.

**Headless drivers.** Scripts can call the stages without the TUI (examples in `Honours/sobol_mass_runs/`, 2026-10-08/09): full pipeline `t.run_pipeline(SimParams(..., keep_dumps=True), RenderFormValues(samples=500, encode_video=True, paths=("instance_grains",)), t.DATA_DIR / "DEMCSVs", on_stage=...)` (`run_breakup_np10k_fric_boulders.py`); sim only `t.run_sim_stage(sim)` (`run_ablation_np10k_rp8000.py`; `run_pipeline` needs at least one render path); render only `t.run_render_stage(RenderParams(grains_dir, bodies_dir, output_dir, samples, encode_video=True, viz_path="instance", manifest=<run>/run_0001_viz_instance/manifest.json))` after a render crash (`rerender_fric_boulders.py`). Launch long ones in tmux with `OMP_NUM_THREADS=6`.

**Tests.** `sobol/tests/test_tui_sim_render.py` (mocked) mocks PHANTOM, the converter and Blender, and includes Textual pilot tests run via `asyncio.run` (no pytest-asyncio needed). `sobol/tests/test_dem_headless_world.py` checks the TUI World HDRI (including a blender.exe smoke). `sobol/tests/test_dem_headless_gpu.py` checks Cycles OPTIX/CUDA GPU-only (CPU hybrid off). `sobol/tests/test_dem_headless_video.py` checks the optional `--encode-video`/`encode_video()` stage (arg parsing, frame-count-mismatch guard, and a blender.exe smoke that encodes 2 dummy frames to mp4). `sobol/tests/test_dem_headless_paths.py` covers `DEMHeadlessRender.py`'s `--viz-path composite|instance` patching and the "Procedural rock" append — Blender integration tests, skip without `blender.exe`. `sobol/tests/test_viz_preprocess_grains_instance.py` covers the `instance_grains` preprocess writer. `test_dump_frames.py`, `test_viz_preprocess_dump_mode.py` and `test_viz_preprocess_dump.py` cover the dump reader, composite dump mode and the single-pass driver. `sobol/tests/test_render_compare.py` covers pairing, stitching, stale-frame cleanup, ffmpeg encode (real encode skips without ffmpeg) and the CLI.

