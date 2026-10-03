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
- **Render:**
  - `resolution` (1920x1080), `samples` (500), `fps` (24).
  - `max_frames` (blank means all).
  - `camera_mode` (`auto` | `grain_only`, which patches `CAMERA_MODE`).
  - `encode_video` checkbox (off by default) + `video_fps` (blank = same as `fps`) — optional final MP4-encode stage, applied per ticked render path.
  - Render fields are validated when you click, before the sim starts.
- **Render paths:**
  - Four checkboxes, any combination: `per_sphere` (legacy, parity reference), `composite`, `instance_grains` (ticked by default since 2026-09-29, so a default launch takes the dump route with no convert), `instance_static`.
  - `envelope_method` (`hull` | `sdf`, default `hull`) — path-specific field for `composite`.
  - `placeholder_obj` (Windows copy of `apophis_v233s7.obj`) and `n_points` (default `1000000`) — path-specific fields for `instance_static`.
  - Path-specific fields are validated only when their path is ticked.
  - `compare` checkbox (off by default): side-by-side of the ticked paths; validated at click (≥ 2 paths).
  - `.venv` python existence is checked at click time when `composite` or `instance_grains` is ticked (each runs its preprocess script in the Windows `.venv`).

**Behaviour.**
- At startup it checks `blender.exe` exists at `/mnt/c/Program Files/Blender Foundation/Blender 5.2/blender.exe`.
- The status bar updates about every 2 s. During sim it counts `<prefix>_[0-9]*` dump files (`Running sim...`). When PHANTOM exits it flips to `Converting... N/M npz`, then cycles per ticked path (dump route instead: `Preprocessing dumps... k/N dump(s)`, counting composite's camera files, then renders only): `Preprocessing [<path> i/n]...` (composite/instance_grains/instance_static only) and `Rendering [<path> i/n]... k/N frame(s)` (`frame_*.png` under that path's Windows render dir), then `Comparing... k/N frame(s)` (`compare_*.png`) when `compare` is ticked. The last dump count no longer freezes the bar.
- The final status is one of: `[Ns] Done (done): converted N frame(s); rendered M frame(s) to …` (dump route: `read N dump(s) directly (convert skipped); …`), `[Ns] Finished with failures: …` (`partial` — one or more ticked render paths failed while others succeeded), `Failed at sim|convert: …`, or `Failed at error: <Type>: …` for an unexpected exception.
- Only one pipeline can run at a time: the buttons are disabled and a second launch is refused.
- `np_apophis > 2000` prepends a non-blocking warning only when `per_sphere` is ticked, because that path is meant for ≤~2000 grains.

**Limits (v1).**
- No mid-run cancel. Quitting mid-run leaves PHANTOM or Blender child processes running.
- No streamed Blender log.
- The `instance_static` camera tracks the origin (static placeholder); `instance_grains` tracks the inner-80% grain core like `per_sphere`.

**Tests.** `sobol/tests/test_tui_sim_render.py` (mocked) mocks PHANTOM, the converter and Blender, and includes Textual pilot tests run via `asyncio.run` (no pytest-asyncio needed). `sobol/tests/test_dem_headless_world.py` checks the TUI World HDRI (including a blender.exe smoke). `sobol/tests/test_dem_headless_gpu.py` checks Cycles OPTIX/CUDA GPU-only (CPU hybrid off). `sobol/tests/test_dem_headless_video.py` checks the optional `--encode-video`/`encode_video()` stage (arg parsing, frame-count-mismatch guard, and a blender.exe smoke that encodes 2 dummy frames to mp4). `sobol/tests/test_dem_headless_paths.py` covers `DEMHeadlessRender.py`'s `--viz-path composite|instance` patching and the "Procedural rock" append — Blender integration tests, skip without `blender.exe`. `sobol/tests/test_viz_preprocess_grains_instance.py` covers the `instance_grains` preprocess writer. `test_dump_frames.py`, `test_viz_preprocess_dump_mode.py` and `test_viz_preprocess_dump.py` cover the dump reader, composite dump mode and the single-pass driver. `sobol/tests/test_render_compare.py` covers pairing, stitching, stale-frame cleanup, ffmpeg encode (real encode skips without ffmpeg) and the CLI.

