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

DEM dumps → Blender CSVs (WSL; adjust `run_dirs` and `--base-output-dir`):
```bash
python3 sobol/Analysis/run_demtocsv_batch.py --min-dem-grains 450 \
  --base-output-dir "/mnt/c/Users/22boy/OneDrive/Documents/GC-Max_desktop/Honours/Code/DEMCSVs/torque_align_obj" \
  sobol_mass_runs/<batch>/run_0001 sobol_mass_runs/<batch>/run_0002
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
