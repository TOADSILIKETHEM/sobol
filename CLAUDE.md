# Sobol sweep runner (`sobol/`)

Parent context: `../CLAUDE.md`. Metrics: `../docs/METRICS.md`. Batches: `../docs/BATCHES.md`.

## Architecture: sweep runner

`sobol/run_mass_sobol_phantom.py` is the canonical runner. Its design:

- **`build_parser()`** registers all CLI args. The argparse registration order is also the interactive prompt order — moving an `add_argument` call changes where that question appears in the wizard.
- **`validate_args()`** enforces the unified optional-dimension rule: a dimension is active only if **both** its min and max bounds are set; one without the other is an error. No active dimensions at all is also an error.
- **`build_run_samples()` / `run_sample_from_salib_row()`** map Sobol unit samples `[0,1)` to physical parameter spaces. Dimension ordering in the Sobol matrix must match `count_dimensions()` and the CSV column order from `sample_column_order()`.
- **`run_one_case()`** copies template `.in`/`.setup` into `run_XXXX/`, patches the setup file via `apply_run_sample_to_setup()`, optionally copies ephemeris `.txt` files, runs `phantomsetup` then `phantom`, and extracts closest-approach distance from sink `.ev` files. For DEM runs it additionally (a) sets `nfulldump=1` in the run's `.in` after `phantomsetup` (every dump becomes a full dump — Blender export needs this; non-DEM runs are left untouched), (b) calls `_use_fast_metrics_defaults()` then computes breakup + spin in one pass via `_extract_dem_metrics_bundle()` (reads each Apophis `.ev` once, **dump-time rows**, spin window gating), and (c) for multi-sink Apophis (≥2 grains) writes **`intrinsic_spin_period_hr`**, **`approach_spin_period_hr`**, and **`post_flyby_spin_period_hr`** (see `../docs/METRICS.md`).
- **`main()`** orchestrates: validate → build samples → preflight → dispatch via `ProcessPoolExecutor` (when `--jobs > 1`) → write summary CSV.

Output per batch: `sobol_mass_samples.csv` (input parameters), `sobol_mass_outputs.csv` (results). Saltelli mode additionally writes `saltelli_problem.json`, `saltelli_meta.json`, `saltelli_Y.csv`, `saltelli_eval_manifest.csv`.

### Sweep runner features (May 2026)

- **Incremental CSV:** After each run completes, `sobol_mass_outputs.csv` is rewritten (`--jobs 1` and `--jobs > 1`). Mid-batch WSL kills retain finished rows.
- **`--no-cleanup`:** Default deletes binary dumps, `.ev`, and `phantom.log` after metrics extraction. Pass `--no-cleanup` to keep files for Blender / `sarracen`.
- **Parallelism on dev laptop (i7-12650H, WSL2):** 6 P-cores + 4 E-cores, 16 logical CPUs. **`--jobs 2`** with `OMP_NUM_THREADS=1` is the recommended sweet spot (~2× throughput, avoids thermal throttling seen at 3–4 jobs). Optional: `OMP_NUM_THREADS=2` with `--jobs 2` (4 threads total) for modest extra gain. Do not default to `--jobs 4` on this machine.

## Architecture: interactive wizard

`sobol/interactive_run_mass_sobol.py` — imported by the runner when `-i` is passed.

- **`run_interactive_wizard(parser, initial_args)`** iterates `parser._actions` and prompts for each. The section header `=== X ===` is printed on section change; since prompt order = registration order in `build_parser()`, these headers appear exactly once per section.
- **Gates:** `_prompt_mass_vary_selection()` fires before `mass_min_kg`; if declined, all of `MASS_GATE_DESTS` are skipped. `_prompt_scale_vary_selection()` fires before the first scale bound dest and gates all five scale/DEM parameters at once.
- **Custom handlers** (`WIZARD_CUSTOM_HANDLERS` dict): override the default prompt for a dest. Currently: `vary_use_dem` (ask vary-first, then fixed preset only if not varying) and `saltelli_calc_second_order` (skipped entirely when `saltelli_n` is None).
- **Extending:** when adding a new CLI flag, add its dest to `DEST_TO_SECTION` (controls heading) and `INTERACTIVE_BRIEF` (controls the two-line help shown at the prompt). Add a `WIZARD_CUSTOM_HANDLERS` entry only for flags that need non-standard prompt logic.
- **`WIZARD_SKIP_DESTS`:** dests handled entirely by a custom handler on another dest (currently `use_dem_fixed`, managed inside the `vary_use_dem` handler).

## Architecture: DEM multi-count wrapper

`sobol/inter_DEM_run_mass_sobol.py` — meta-runner for DEM sweeps that need multiple `np_apophis` values.

Runs the shared wizard once, then for each requested particle count: stages a temp `--base-dir` with `np_apophis` patched in the `.setup` copy, and invokes `run_mass_sobol_phantom.py` as a subprocess. Does **not** re-patch `use_dem` in staging (manage that via wizard flags). Pass `--dem-np N1 N2 ...` before other flags to skip the interactive particle-count prompt.

## Architecture: sim + render pipeline TUI

Parent: `../CLAUDE.md`. Shared facts: `../docs/SHARED_FACTS.md`. Merge/rename caveats: `../docs/CONTEXT_CHANGELOG.md`.

`sobol/tui_sim_render.py` is a Textual TUI that takes **one** DEM run from configure → PHANTOM → dump conversion → headless Blender render (any combination of render paths), with no manual steps. It handles one run per launch, not a sweep; `tui_run.py` is the sweep TUI. Windows render-script how-to: `Code/CLAUDE.md`.

**Stages.** `run_pipeline()` calls each stage in order in one worker thread and stops at the first failure. Each stage reuses existing code:
1. **Sim.** `run_sim_stage()` calls `preflight()` and `run_one_case(run_id=1, ...)` from `run_mass_sobol_phantom.py`. That means DEM `nfulldump=1`, ephemeris copy, and metrics extraction all behave exactly as in a sweep. Any `record.status != "ok"` stops the pipeline; `prepared_only` is reported as a dry-run success.
2. **Convert.** `run_convert_stage()` calls `Analysis/run_demtocsv_batch.run_dem_dump_convert()` with `min_dem_grains` read automatically from the run's `sobol.setup`, writing Windows `DEMCSVs/<batch>/run_0001_{bodies,grains}_output/`. `DEMDumpConvert.py` never raises on bad or missing dumps, so the pipeline then counts `*.npz` and fails at `convert` if there are 0. The converter hardcodes the `sobol_[0-9]*` dump prefix, so keep `prefix = sobol`.
3. **Render paths.** Tick any combination on the form; one sim + one convert feed all of them, run in fixed order `per_sphere`, `composite`, `instance_grains`, `instance_static`. Each path is `[preprocess] → DEMHeadlessRender.py --viz-path …`:
   - `per_sphere` — no preprocess; `DEMGrainsBlenderEarthCam.py` (≤~2000 grains).
   - `composite` (D3) — `viz/viz_preprocess.py --envelope-method hull|sdf` in the Windows `.venv` python → `DEMBulkVizBlender.py`.
   - `instance_grains` — `viz/viz_preprocess_grains_instance.py` in the Windows `.venv` python (real grains, per-grain radius, sorted by `grain_id`) → `DEMBulkInstanceBlender.py` (`static: false`).
   - `instance_static` — `viz/viz_preprocess_lite.py --shape-obj --n-points` under `blender.exe --background` → `DEMBulkInstanceBlender.py` (`static: true`; placeholder, not sim motion).
   A failed path is recorded and the remaining paths still run; result `done` only when every ticked path succeeds, else `partial`. Optional video encode (`encode_video`/`video_fps`) runs per path, inside `run_render_path()`, writing each path's `animation.mp4` next to its own `frame_*.png` sequence.
   Every preprocess/render subprocess runs through `_run_logged()`: stdout+stderr merged, streamed line by line to `preprocess.log` / `render.log` next to that stage's output, with only a bounded tail kept for the error message (which names the full log). Output streams live because `DEMHeadlessRender.py` line-buffers stdout (Blender's Python block-buffers into a pipe and ignores `PYTHONUNBUFFERED`) and the `.venv` preprocess commands run `python.exe -u`. `instance_static`'s `viz_preprocess_lite.py` under `blender.exe` is not line-buffered, so its log arrives at exit.
4. **Timing.** After any non-dry run whose `run_dir` exists (including `partial` and convert failures), `run_pipeline()` writes `<run_dir>/pipeline_timing.json` (`build_timing_report()`): wall-clock `total`/`sim`/`convert`; per path `preprocess_s`, `render_s` (includes video encode), `total_s`, `n_frames`, `render_s_per_frame`, `failed_stage`; `np_apophis` read from the run's own `sobol.setup` (not the form, which may be blank); dump/npz counts; the full sim sample and render form; hostname and `OMP_NUM_THREADS`; `sobol` and Windows `Code` git SHA with a `dirty` flag (tracked files only). A write failure is appended to the status message and never fails the run. This is the per-run evidence for the scope's measured wall-clock vs particle count (Obj 2) and D3's unattended render.

**Output layout.** Each launch creates `<output_root>/<prefix>_<YYYYmmdd_HHMMSS>_<batch_label>/run_0001/`; the label defaults to `sim_render` and is sanitised with `sanitize_batch_label()`. Windows `Code/DEMCSVs/<batch>/`: `run_0001_{grains,bodies}_output/`, `run_0001_viz/` (composite), `run_0001_viz_instance/` (instance_grains), `run_0001_viz_instance_static/` (instance_static), and one `run_0001_render_<path>/` per ticked path with `frame_####.png` + `render.log` (+ `animation.mp4` if `encode_video`); each `run_0001_viz*/` also holds `preprocess.log`. WSL `run_0001/pipeline_timing.json` holds the run's timings. Rename from v1 `run_0001_render/`: `../docs/CONTEXT_CHANGELOG.md`.

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
  - `dn_cohes_factor` (0.1). It is converted with `coh_gap_max_cgs_from_dn()`, but only when `kt_cgs > 0`.
  - `tmax`/`dtmax (hr)`.
  - `sink_earth_id` (4) and `sink_apophis_id` (11).
  - `shape_file`: a non-blank value turns on `use_shape_crop`.
  - `use_dem` is hardcoded to `T`, because the convert and render stages need DEM grains.
- **Render:**
  - `resolution` (1920x1080), `samples` (500), `fps` (24).
  - `max_frames` (blank means all). For `per_sphere` it is patched into `DEMGrainsBlenderEarthCam.py`'s `MAX_FRAMES` so only those frames are keyframed (8-frame fixture, `max_frames 2`: ~72 s → ~26 s); composite/instance already load frames lazily.
  - `camera_mode` (`auto` | `grain_only`, which patches `CAMERA_MODE`).
  - `encode_video` checkbox (off by default) + `video_fps` (blank = same as `fps`) — optional final MP4-encode stage, applied per ticked render path.
  - Render fields are validated when you click, before the sim starts.
- **Render paths:**
  - Four checkboxes, any combination: `per_sphere` (ticked by default), `composite`, `instance_grains`, `instance_static`.
  - `envelope_method` (`hull` | `sdf`, default `hull`) — path-specific field for `composite`.
  - `placeholder_obj` (Windows copy of `apophis_v233s7.obj`) and `n_points` (default `1000000`) — path-specific fields for `instance_static`.
  - Path-specific fields are validated only when their path is ticked.
  - `.venv` python existence is checked at click time when `composite` or `instance_grains` is ticked (each runs its preprocess script in the Windows `.venv`).

**Behaviour.**
- At startup it checks `blender.exe` exists at `/mnt/c/Program Files/Blender Foundation/Blender 5.2/blender.exe`.
- The status bar updates about every 2 s. During sim it counts `<prefix>_[0-9]*` dump files (`Running sim...`). When PHANTOM exits it flips to `Converting... N/M npz`, then cycles per ticked path: `Preprocessing [<path> i/n]...` (composite/instance_grains/instance_static only) and `Rendering [<path> i/n]... k/N frame(s)` (`frame_*.png` under that path's Windows render dir). The last dump count no longer freezes the bar. During preprocess/render the latest subprocess line is appended (`| … Saved: '…frame_0003.png'`, truncated to 100 chars) and cleared at each stage change.
- The final status is one of: `[Ns] Done (done): converted N frame(s); rendered M frame(s) to …`, `[Ns] Finished with failures: …` (`partial` — one or more ticked render paths failed while others succeeded), `Failed at sim|convert: …`, or `Failed at error: <Type>: …` for an unexpected exception. When the timing file was written, ` | timing: <path>` is appended.
- Only one pipeline can run at a time: the buttons are disabled and a second launch is refused.
- `np_apophis > 2000` prepends a non-blocking warning only when `per_sphere` is ticked, because that path is meant for ≤~2000 grains.

**Limits (v1).**
- No mid-run cancel. Quitting mid-run leaves PHANTOM or Blender child processes running. Spike (2026-09-23): SIGTERM or SIGKILL on the WSL-side `Popen` of `blender.exe` does end the Windows process, so render/preprocess cancel only needs the TUI to hold the running `Popen`; the sim stage is harder because PHANTOM is spawned inside `run_one_case()`.
- The `instance_grains`/`instance_static` camera tracks the grain CoM (origin), not the densest core — can sit between fragments on a breakup.

**Tests.** `sobol/tests/test_tui_sim_render.py` (131; mocked apart from `_run_logged`/`_git_state` tests that spawn real `python`/`git` children) mocks PHANTOM, the converter and Blender, and includes Textual pilot tests run via `asyncio.run` (no pytest-asyncio needed). `sobol/tests/test_dem_headless_world.py` checks the TUI World HDRI (including a blender.exe smoke). `sobol/tests/test_dem_headless_gpu.py` checks Cycles OPTIX/CUDA GPU-only (CPU hybrid off). `sobol/tests/test_dem_headless_video.py` checks the optional `--encode-video`/`encode_video()` stage (arg parsing, frame-count-mismatch guard, and a blender.exe smoke that encodes 2 dummy frames to mp4). `sobol/tests/test_dem_headless_paths.py` (21) covers `DEMHeadlessRender.py`'s `--viz-path composite|instance` patching, the per-sphere `MAX_FRAMES` patch, and the "Procedural rock" append — Blender integration tests, skip without `blender.exe`. Set `HONOURS_BLENDERCONVERT_DIR` to run them against another `Code` checkout (fixtures still come from the main `Code/DEMCSVs/`); the instance tests then need `Blenders/11.blend`, which is untracked and absent from a fresh worktree. `sobol/tests/test_viz_preprocess_grains_instance.py` (6) covers the `instance_grains` preprocess writer. A real sim→render run through the UI was done 2026-09-16 (composite + per_sphere).

## Key commands

Install Python dependencies (run once, from repo root):
```bash
pip install -r sobol/requirements.txt   # numpy, scipy (multi-d Sobol), SALib (Saltelli)
```

Run a sweep (from repo root):
```bash
python3 sobol/run_mass_sobol_phantom.py \
  --base-dir sobol --prefix sobol \
  --num-samples 50 \
  --mass-min-kg 1e10 --mass-max-kg 1e11 \
  --scale-vel-min 0.9 --scale-vel-max 1.1 \
  --jobs 4
```

Interactive mode (prompts for all parameters; CLI flags set defaults):
```bash
python3 sobol/run_mass_sobol_phantom.py -i
```

DEM multi-particle-count sweep (runs wizard once, then one subprocess per `np_apophis` value):
```bash
cd sobol && python3 inter_DEM_run_mass_sobol.py --dem-np 20 30 64 [wizard-seed-flags...]
```

Dry run (prepares directories and patched `.setup` files, does not execute PHANTOM):
```bash
python3 sobol/run_mass_sobol_phantom.py --dry-run [other flags]
```

Torque-align Blender re-runs (two cases, dumps kept; add `--use-shape-crop` for OBJ):
```bash
bash sobol/campaigns/run_torque_align_blender_vis.sh
bash sobol/campaigns/run_torque_align_blender_vis_obj.sh
```

DEM dumps → Blender inputs, bodies CSV + grains `.npz` (WSL; adjust `run_dirs` and `--base-output-dir`):
```bash
python3 sobol/Analysis/run_demtocsv_batch.py --min-dem-grains 450 \
  --base-output-dir "/mnt/c/Users/22boy/OneDrive/Documents/GC-Max_desktop/Honours/Code/DEMCSVs/torque_align_obj" \
  sobol_mass_runs/<batch>/run_0001 sobol_mass_runs/<batch>/run_0002
```

Single run end-to-end: sim → convert → headless Blender render (Textual TUI, one run per launch; see **Architecture: sim + render pipeline TUI**):
```bash
cd /home/mboyle/Honours && python3 sobol/tui_sim_render.py
# quick real test: np_apophis 500, tmax 1, dtmax 0.5, ephemeris_cache_dir /home/mboyle/Honours/sobol,
#                  resolution 640x360, samples 16, max_frames 2  (Dry Run first)
# all four paths: tick every render path box, max_frames 3 → four run_0001_render_<path>/ folders
cd sobol && python3 -m pytest tests/test_tui_sim_render.py   # 131 tests
```

Sensitivity analysis — classic (correlation stats from a completed sweep):
```bash
python3 sobol/Analysis/Analysis.py --method classic \
  --csv sobol/sobol_mass_runs/<batch>/sobol_mass_outputs.csv \
  --response closest_approach_au
# DEM spin (tidal attribution): --response intrinsic_spin_period_hr, approach_spin_period_hr, or post_flyby_spin_period_hr
```

Sensitivity analysis — Saltelli (variance-based Sobol indices; requires `--saltelli-n` sweep):
```bash
python3 sobol/Analysis/Analysis.py --method saltelli \
  --sobol-problem-json <batch>/saltelli_problem.json \
  --saltelli-meta-json <batch>/saltelli_meta.json \
  --saltelli-y-csv <batch>/saltelli_Y.csv \
  --saltelli-y-column closest_approach_au
```

Parallel jobs with OpenMP PHANTOM (prevent thread oversubscription on shared laptops):
```bash
OMP_NUM_THREADS=1 python3 sobol/run_mass_sobol_phantom.py --jobs 2 [flags]
```

Keep dumps for visualisation (default auto-deletes heavy files after each run):
```bash
python3 sobol/run_mass_sobol_phantom.py ... --no-cleanup
```

Metricfix rerun (re-run all thesis spin-sensitive campaigns after intrinsic-window fix; see `../docs/BATCHES.md`):
```bash
OMP_NUM_THREADS=2 JOBS=2 bash sobol/campaigns/run_metricfix_reruns.sh
```

Re-extract spin/breakup columns when `.ev` files remain:
```bash
python3 sobol/reextract_spin_metrics.py --batch-dir sobol_mass_runs/<batch>
```

Regenerate thesis plots (scripts pick latest batch per suffix):
```bash
python3 sobol/Analysis/plot_np_sensitivity.py
python3 sobol/Analysis/plot_np_spin_grid.py
python3 sobol/Analysis/plot_flyby_spin30_torque_align.py
python3 sobol/Analysis/plot_spin_disruption_threshold.py
```

## Legacy / variant files

- `sobol/run_mass_sobol_phantomThisWorks.py` — snapshot kept as a reference; keep in sync with `run_mass_sobol_phantom.py` if changing behavior.
- `sobol/run_mass_sobol_phantomGIT.py` — another variant; treat similarly.
- `solarsystem/run_mass_sobol_phantom.py` — independent runner for the solarsystem setup; different CLI/behavior from the sobol one.
