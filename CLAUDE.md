# Sobol sweep runner (`sobol/`)

Parent context: `../CLAUDE.md`. Metrics: `../docs/METRICS.md`. Batches: `../docs/BATCHES.md`.

## Architecture: sweep runner

`sobol/run_mass_sobol_phantom.py` is the canonical runner. Its design:

- **`build_parser()`** registers all CLI args. The argparse registration order is also the interactive prompt order — moving an `add_argument` call changes where that question appears in the wizard.
- **`validate_args()`** enforces the unified optional-dimension rule: a dimension is active only if **both** its min and max bounds are set; one without the other is an error. No active dimensions at all is also an error.
- **`build_run_samples()` / `run_sample_from_salib_row()`** map Sobol unit samples `[0,1)` to physical parameter spaces. Dimension ordering in the Sobol matrix must match `count_dimensions()` and the CSV column order from `sample_column_order()`.
- **`run_one_case()`** copies template `.in`/`.setup` into `run_XXXX/`, patches the setup file via `apply_run_sample_to_setup()`, optionally copies ephemeris `.txt` files, runs `phantomsetup` then `phantom`, and computes metrics via `compute_run_metrics()` (grains from sink `.ev` or, for particle DEM, from dumps via `particle_dem.py`; see `../docs/METRICS.md`). `--dem-model particle|sink` (default particle) writes `use_dem` / `use_dem_as_sinks`; `check_setup_log()` refuses runs whose setup skipped spin; cohesion gap = `dn × 2 × R_grain` from the setup dump. For DEM runs it additionally (a) sets `nfulldump=1` in the run's `.in` after `phantomsetup` (every dump becomes a full dump — Blender export needs this; non-DEM runs are left untouched), (b) calls `_use_fast_metrics_defaults()` then computes breakup + spin in one pass via `_extract_dem_metrics_bundle()` (reads each Apophis `.ev` once, **dump-time rows**, spin window gating), and (c) for multi-sink Apophis (≥2 grains) writes **`intrinsic_spin_period_hr`**, **`approach_spin_period_hr`**, and **`post_flyby_spin_period_hr`** (see `../docs/METRICS.md`).
- **`main()`** orchestrates: validate → build samples → preflight → dispatch via `ProcessPoolExecutor` (when `--jobs > 1`) → write summary CSV.

Output per batch: `sobol_mass_samples.csv` (input parameters), `sobol_mass_outputs.csv` (results). Saltelli mode additionally writes `saltelli_problem.json`, `saltelli_meta.json`, `saltelli_Y.csv`, `saltelli_eval_manifest.csv`.

### Sweep runner features (May 2026)

- **Incremental CSV:** After each run completes, `sobol_mass_outputs.csv` is rewritten (`--jobs 1` and `--jobs > 1`). Mid-batch WSL kills retain finished rows.
- **`--no-cleanup`:** Default deletes binary dumps, `.ev`, and `phantom.log` after metrics extraction. Pass `--no-cleanup` to keep files for Blender / `sarracen`. **Cleanup is default ON everywhere**: runner, torque-align reruns, and `tui_sim_render.py` (raw dumps deleted once npz convert is verified; tick `keep_dumps` to keep).
- **Parallelism on dev laptop (i7-12650H, WSL2):** 6 P-cores + 4 E-cores, 16 logical CPUs. **`--jobs 2`** with `OMP_NUM_THREADS=1` is the recommended sweet spot (~2× throughput, avoids thermal throttling seen at 3–4 jobs). Optional: `OMP_NUM_THREADS=2` with `--jobs 2` (4 threads total) for modest extra gain. Do not default to `--jobs 4` on this machine.

### Settled bodies and hyperbola encounters (Sep 2026)

- **`--body-source {lattice,settled}`** (default `lattice`). `settled`: `settled_body.py` settles (`pack_settle=T`,
  `apophis_only=T`; the settle `.setup` sets `idamp=2`), crops (`phantommoddump`, default/crop build) and relaxes
  (`--relax-tdyn`, default 0.5 t_dyn) one body per `(np_apophis, scale_rho, shape_file, binaries)` into
  `<cache-root>/<key>/` (`body.json`; `<cache-root>` = `--body-cache-dir` or `<output-root>/settled_bodies`). Built
  serially in `attach_settled_bodies()` (called from `main()`, before workers start) so two workers never settle the
  same body at once; the cache key is content-hashed (spec + shape/mesh + binary SHA256s), so a rebuilt
  `phantom`/`phantomsetup`/`phantommoddump` or an edited shape file invalidates it automatically. A `<key>.partial`
  dir (crash/Ctrl-C, or an unconverged settle) is discarded and rebuilt on the next call.
  `build_settled_body()` refuses to cache a settle with `packing_fraction < MIN_SETTLED_PACKING = 0.5` (raises
  `RuntimeError`, tells you to raise `--settle-tdyn`) — a loose cloud needs its full free-fall time to collapse.
  **Measured 2026-09-27 at np=300:** `--settle-tdyn 5` (the default) converges; a real build kept 391/923 grains at
  `packing_fraction 0.834` in ~13 s. `kn_cgs=1e7` (Mia's default) under self-gravity at this N gives ~13% grain
  overlap, so `packing_phi` runs above Mia's ~0.66 at 10k grains (overlap falls as N⁻¹/³, worse at small N) — expect
  `packing_phi` and kept-grain counts above nominal at low `np_apophis`.
- **`--encounter {ephemeris,hyperbola}`** (default `ephemeris`; `hyperbola` needs `--body-source settled`):
  `encounter.py` runs `phantomflyby` on the cached body with `--flyby-rp-km`/`--flyby-vinf-kms` (fixed), or swept as
  an optional Sobol/Saltelli dimension pair via `--flyby-rp-km-min/-max` and `--flyby-vinf-kms-min/-max` (both bounds
  required together, mutually exclusive with the fixed flag per parameter), and `--flyby-start-sep-km` (default
  4e5). `patch_encounter_in()` forces `idamp = 0` in the flyby `.in` — load-bearing: `phantomflyby` copies the
  settle's `idamp = 2` forward verbatim, and without this the settle's damping would leak into every hyperbola run.
  Writes `encounter.json` and a record `sobol.setup` (never read by `phantomsetup`, just a manifest for the
  converter). Earth = sink 1 (`--sink-earth-id` auto-set). No spin (the flyby moddump sets none).
  `--tmax-hours` must be at least 2× the **true max** time to pericentre over the whole swept rp/v_inf range —
  `encounter._max_time_to_pericentre_hr()` grids both dimensions (65×9) because `time_to_pericentre_hr(rp)` has an
  interior maximum and falls to 0 as `rp` approaches `--flyby-start-sep-km`, so the worst case is not at a sweep
  corner. Pericentre must satisfy `R_EARTH_KM` (6371) `< rp < --flyby-start-sep-km` over the whole swept range (CLI
  and TUI).
- **Rejected with settled/hyperbola** (`_validate_body_encounter()`): `--dem-model sink`, varied `use_dem`,
  `--mass-min/max-kg`, `--scale-r-apophis-*`, a kn sweep, `--scale-rho-*` sweeps (one settle per sample; a fixed
  template `scale_rho` is fine), `use_shape_crop` flags (the body carries its shape). **Hyperbola only:**
  `--scale-vel-*`/`--scale-pos-*` (the hyperbola path never runs `phantomsetup`), spin flags, `apophis_only` flags.
  `--flyby-rp-km`/`--flyby-vinf-kms` without `--encounter hyperbola` are rejected.
- **Outputs:** hyperbola rows fill `np_apophis`/`scale_rho` from the sample (written when those columns are in
  `sample_column_order()`) and every settled-body batch (hyperbola or ephemeris) adds **`np_kept`** (grains kept after the crop, `body.json`
  `n_kept` — not `np_apophis`; np=300 kept 391 of 923 settled), blank on dry-run rows. Column detail:
  `../docs/METRICS.md`. Batch slug gains `hyp` / `settled` tokens.
- **Resume:** `resume_batch.py` calls the same `prepare_settled_bodies_and_dem_tools()` as `main()` (settled bodies
  + `phantomflyby` for hyperbola only / `phantomanalysis`). Pass the original `--output-root`/`--body-cache-dir`, else the cache misses and
  the body is rebuilt.
- **Known gaps (Sep 2026):** CLI + TUI building the same new cache key at the same time is unguarded; the cache is
  never pruned; a settle failure during resume prints a traceback; `Analysis/Analysis.py` classic mode has no
  `flyby_rp_km`/`flyby_vinf_kms` in `INPUT_CANDIDATES`.
- **TUI:** `tui_sim_render.py` has the same `body_source`/`encounter`/flyby fields — `docs/tui_sim_render.md`.
- **`--body-source settled --encounter ephemeris`** (Mia's Option 2 step 4: settled body on the real 2029 flyby;
  wired 2026-09-27). `run_one_case` → `encounter.apply_settled_body_to_setup()` copies the cached dump to
  `run_XXXX/settled_body`, stages the body's own shape (`body.json` `spec.shape_file`; body mass = shape volume ×
  `rho_0*scale_rho`), patches `packing_file = settled_body`, `pack_settle = F`, `apophis_shape_file`, `np_apophis =
  n_kept`, then the normal `phantomsetup` → `phantom` path. Earth stays sink 4. Spin and `--scale-vel-*`/`--scale-pos-*`
  apply here. `check_packing_setup_log()` fails the run unless `setup.log` says `placed <n_kept> pre-built grains`
  and has no sphere-volume fallback. Preflight refuses a `phantomsetup` without the `packing_file` string. Needs
  phantom `DEMsync-mia` ≥ `464ea28fa` (Mia `853b86818` + `get_conserv` reset; not pushed) — without the reset,
  `phantom` dies at step 2 with "Large error in angular momentum conservation". Semantics:
  `../docs/MIA_PACKING_WORKFLOW.md`. Example (np300, 30 hr spin, 12 hr):
  `OMP_NUM_THREADS=1 python3 sobol/run_mass_sobol_phantom.py --base-dir sobol --prefix sobol --phantom-dir
  /home/mboyle/Honours/sobol --ephemeris-cache-dir sobol --use-dem-fixed true --np-apophis-list 300 --body-source
  settled --spin-period-fixed 30 --tmax-hours 12 --dtmax-hours 1 --num-samples 1 --jobs 1`
- **Template `packing_file =` (blank)** in `sobol/sobol.setup` is required: `phantomsetup` since `853b86818` counts a
  missing key as a read error, rewrites the `.setup` and stops. Any other `.setup` template (campaign `--base-dir`,
  old run dirs) needs the line too. Pre-merge binaries (`bin_demsync_7a243de`) ignore it.
- **Tools:** `cd sobol && make demtools` builds `phantommoddump` (crop, default build), `phantomflyby`
  (`moddump_earthflyby.f90`), `phantomanalysis` (`analysis_demshape.f90`); binaries land at `sobol/<name>` (repo
  root) — `resolve_dem_tools()` prefers a `bin/<name>` copy if one exists. `phantomflyby` needs phantom
  `DEMsync-mia` commit `0c3f4b06d` (orbit fix, not pushed); see `../docs/CONTEXT_CHANGELOG.md` 2026-09-27.
  Rebuild with `make demtools` after pulling it.
- With `phantomanalysis` present, every particle-DEM run (ephemeris or hyperbola) also gets **`shape_b_on_a`**,
  **`shape_c_on_a`**, **`packing_phi`**, **`f_unbound_energy`** in the CSV (see `../docs/METRICS.md`).

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

## Sim + render pipeline TUI (summary)

`tui_sim_render.py`: **one** DEM run per launch, configure → PHANTOM (`run_one_case`) → convert (`run_demtocsv_batch`) → headless Blender, any mix of render paths `per_sphere`, `composite`, `instance_grains`, `instance_static` (a failed path doesn't stop the rest; result `done` or `partial`). Sweep TUI is `tui_run.py`. Code in WSL `CODE_DIR=/home/mboyle/Honours/Code`; outputs in OneDrive `DATA_DIR/DEMCSVs/<batch>/run_0001_*`. Keep `prefix = sobol` (converter hardcodes `sobol_[0-9]*`). **Set `ephemeris_cache_dir` = `/home/mboyle/Honours/sobol`** — blank triggers a live Horizons download. No mid-run cancel (quitting leaves PHANTOM/Blender running).

Form fields, stages, status strings, output layout, limits, tests: `docs/tui_sim_render.md`.

## Key commands

Install Python dependencies (run once, from repo root):
```bash
pip install --user --break-system-packages -r sobol/requirements.txt   # numpy, scipy, SALib, textual, matplotlib, pytest (Ubuntu 24.04: PEP 668 needs these flags)
```

Run a sweep (from repo root):
```bash
python3 sobol/run_mass_sobol_phantom.py \
  --base-dir sobol --prefix sobol \
  --num-samples 50 \
  --mass-min-kg 1e10 --mass-max-kg 1e11 \
  --scale-vel-min 0.9 --scale-vel-max 1.1 \
  --jobs 2          # with OMP_NUM_THREADS=1; not 4 on this laptop
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

Torque-align re-runs: `run_torque_align_blender_reruns.py` cleans up after metrics by default (`--no-cleanup` keeps dumps). These campaign wrappers pass `--no-cleanup` (Blender needs dumps; add `--use-shape-crop` for OBJ):
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
cd sobol && python3 -m pytest tests   # full suite (TUI tests mocked; headless tests launch blender.exe)
```

DEM test fixtures (tiny real runs, particle committed, sink twin gitignored/regenerable): `tests/fixtures/README.md`, `tests/fixtures/dem_run_dir.sh <particle|sink> <dir> <np> <tmax_in> <dtmax_in> <torque_deg> [run]`.

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

Build the DEM crop/flyby/shape tools once (needed for `--body-source settled`):
```bash
cd sobol && make demtools
```

Hyperbola sweep over pericentre and v_inf (settled np=300 body, built once, reused across the sweep):
```bash
OMP_NUM_THREADS=1 python3 sobol/run_mass_sobol_phantom.py --phantom-dir /home/mboyle/Honours/sobol \
  --base-dir sobol --prefix sobol \
  --ephemeris-cache-dir sobol --use-dem-fixed true --np-apophis 300 --body-source settled \
  --encounter hyperbola --flyby-rp-km-min 20000 --flyby-rp-km-max 40000 \
  --flyby-vinf-kms-min 5 --flyby-vinf-kms-max 7 --tmax-hours 48 --dtmax-hours 0.5 \
  --num-samples 16 --jobs 2 --no-cleanup
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
