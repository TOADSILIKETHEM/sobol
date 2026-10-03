# Sobol sweep runner (`sobol/`)

Parent context: `../CLAUDE.md`. Metrics: `../docs/METRICS.md`. Batches: `../docs/BATCHES.md`. Dated notes: `../docs/CONTEXT_CHANGELOG.md`.

## Architecture: sweep runner

`sobol/run_mass_sobol_phantom.py` is the only runner.

- **`build_parser()`** registers all CLI args. Registration order = interactive prompt order.
- **`validate_args()`**: a dimension is active only if **both** min and max are set; one without the other is an error; no active dimension is an error.
- **`build_run_samples()` / `run_sample_from_salib_row()`** map `[0,1)` samples to physical values. Dimension order must match `count_dimensions()` and `sample_column_order()`.
- **`run_one_case()`** copies template `.in`/`.setup` into `run_XXXX/`, patches via `apply_run_sample_to_setup()`, copies ephemeris `.txt`, runs `phantomsetup` → `phantom`, then `compute_run_metrics()` (sink DEM: `.ev`; particle DEM: dumps via `particle_dem.py`). `--dem-model particle|sink` (default particle) writes `use_dem` / `use_dem_as_sinks`. `check_setup_log()` refuses runs whose setup skipped spin. Cohesion gap = `dn × 2 × R_grain` from the setup dump. DEM runs: `nfulldump=1` (Blender needs full dumps); breakup + spin in one pass (`_extract_dem_metrics_bundle()`); spin columns `intrinsic_/approach_/post_flyby_spin_period_hr` (`../docs/METRICS.md`).
- **`main()`**: validate → build samples → `prepare_settled_bodies_and_dem_tools()` → preflight → `ProcessPoolExecutor` (`--jobs > 1`) → summary CSV.

Output per batch: `sobol_mass_samples.csv`, `sobol_mass_outputs.csv` (rewritten after every run, so a killed batch keeps finished rows). Saltelli adds `saltelli_problem.json`, `saltelli_meta.json`, `saltelli_Y.csv`, `saltelli_eval_manifest.csv`.

- **Cleanup default ON** everywhere (runner, torque-align reruns, `tui_sim_render.py`): dumps, `.ev`, `phantom.log` deleted after metrics. `--no-cleanup` (TUI: `keep_dumps`) keeps them.
- **Template `sobol.setup`** is point-mass (`use_dem = F`, `np_apophis = 1`); DEM comes from flags. `apophis_spin_period` is in **seconds**. Spin axis keys are `apophis_spin_axis_x/y/z` (obliquity/azimuth keys are gone). `packing_file =` (blank) must be present in every `.setup` template: current `phantomsetup` treats a missing key as a read error, rewrites the file and stops (the runner retries once).

### Settled bodies and hyperbola encounters

Both need `--use-dem-fixed true` and `--np-apophis` / `--np-apophis-list` (grains kept after the crop), and `make demtools`.

- **`--body-source {lattice,settled}`** (default `lattice`). `settled`: `settled_body.py` settles (`pack_settle=T`, `apophis_only=T`, `idamp=2`, `--settle-tdyn` default 5), crops (`phantommoddump`) and relaxes (`--relax-tdyn` default 0.5) one body per `(np_apophis, scale_rho, shape_file, binaries)` into `<cache-root>/<key>/` (`body.json`; cache root = `--body-cache-dir` or `<output-root>/settled_bodies`). Built serially before workers start. The key hashes spec + shape/mesh + binary SHA256s, so a rebuilt binary or edited shape resettles. `<key>.partial` dirs are discarded. A settle with `packing_fraction < 0.5` is refused (raise `--settle-tdyn`). Small N overlaps more, so `packing_phi` and kept counts run above nominal at low `np_apophis`.
- **`--body-source settled --encounter ephemeris`**: settled body on the real 2029 flyby. `encounter.apply_settled_body_to_setup()` stages the cached dump as `run_XXXX/settled_body` plus the body's shape, patches `packing_file`, `pack_settle = F`, `apophis_shape_file`, `np_apophis = n_kept`, then the normal `phantomsetup` → `phantom` path (Earth = sink 4; spin and `--scale-vel-*`/`--scale-pos-*` apply). `check_packing_setup_log()` fails the run unless `setup.log` says `placed <n_kept> pre-built grains`. Needs phantom `DEMsync-mia` with the `get_conserv` reset (else "Large error in angular momentum conservation" at step 2). Semantics: `../docs/MIA_PACKING_WORKFLOW.md`.
- **`--encounter hyperbola`** (needs `--body-source settled`): `encounter.py` runs `phantomflyby` on the cached body. Required: `--flyby-rp-km`/`--flyby-vinf-kms` (fixed) or `-min/-max` pairs (swept; no silent default), `--tmax-hours`, `--dtmax-hours`. Optional: `--flyby-start-sep-km` (4e5), `--flyby-perturber-earth-masses` (1). `encounter.check_flyby_geometry()` (shared by CLI and TUI) requires `R_EARTH_KM < rp < start_sep` and `tmax >= 2 ×` the longest time to pericentre over the swept range (gridded, because it peaks inside the rp range). `patch_encounter_in()` forces `idamp = 0` (load-bearing: `phantomflyby` copies the settle's `idamp = 2`). Writes `encounter.json` and a record `sobol.setup` (converter manifest only). Earth = sink 1 (`--sink-earth-id` auto-set). No spin. `--maxp = max(2000, 4·n_kept)`.
- **Rejected with settled** (`_validate_body_encounter()`): `--dem-model sink`, varied `use_dem`, mass sweeps, `--scale-r-apophis-*`, kn sweeps, `--scale-rho-*` sweeps, `use_shape_crop` flags. **Hyperbola also rejects** `--scale-vel-*`/`--scale-pos-*`, spin flags, `apophis_only` flags.
- **Outputs:** settled batches add `np_kept`; hyperbola rows carry `np_apophis`/`scale_rho`. With `phantomanalysis` built, particle-DEM runs add `shape_b_on_a`, `shape_c_on_a`, `packing_phi`, `f_unbound_energy`. Slug gains `settled` / `hyp`. Columns: `../docs/METRICS.md`.
- **Resume:** `resume_batch.py` calls the same `prepare_settled_bodies_and_dem_tools()`; pass the original `--output-root`/`--body-cache-dir` or the body is rebuilt.
- **Tools:** `make demtools` builds `phantommoddump` (crop), `phantomflyby` (`moddump_earthflyby.f90`), `phantomanalysis` (`analysis_demshape.f90`) into `sobol/`; `resolve_dem_tools()` prefers `bin/<name>` if present.
- **Known gaps:** CLI + TUI building the same new cache key at once is unguarded; the cache is never pruned; `Analysis/Analysis.py` classic mode lacks `flyby_rp_km`/`flyby_vinf_kms` in `INPUT_CANDIDATES`.

## Architecture: interactive wizard

`sobol/interactive_run_mass_sobol.py`, used by `-i` and by `inter_DEM_run_mass_sobol.py`.

- **`run_interactive_wizard(parser, initial_args)`** prompts once per `parser._actions` entry. Section headers print on each section change, so a section can appear more than once.
- **Gates** skip whole groups when declined: mass (`MASS_GATE_DESTS`), scale bounds, DEM contact bounds (`_prompt_dem_contact_vary_selection`), time (`_prompt_time_vary_selection`).
- `nargs='+'` options (`--np-apophis-list`, `--spin-period-list`) are prompted as space-separated lists of their argparse `type`.
- **Extending:** add a new flag's dest to `DEST_TO_SECTION` (heading) and `INTERACTIVE_BRIEF` (help text).

`inter_DEM_run_mass_sobol.py`: runs the wizard once, then one subprocess per `np_apophis` (temp `--base-dir` with `np_apophis` patched). `--dem-np N1 N2 ...` skips the count prompt.

## Sim + render pipeline TUI (summary)

`tui_sim_render.py`: **one** DEM run per launch: PHANTOM (`run_one_case`) → convert (`run_demtocsv_batch`) → headless Blender, any mix of `per_sphere`, `composite`, `instance_grains`, `instance_static`, optional side-by-side compare (`render_compare.py` → `run_0001_compare/`) (result `done` or `partial`). Sweep TUI is `tui_run.py`. Keep `prefix = sobol` (converter globs `sobol_[0-9]*`). Set `ephemeris_cache_dir = /home/mboyle/Honours/sobol` (blank = live Horizons download). No mid-run cancel. Detail: `docs/tui_sim_render.md`.

## Key commands

Laptop: sweeps `OMP_NUM_THREADS=1`, `--jobs 2`; single long run `--jobs 1 OMP_NUM_THREADS=4..6` (hub rule). Batches live in `Honours/sobol_mass_runs/`.

```bash
python3 sobol/run_mass_sobol_phantom.py --base-dir sobol --prefix sobol --num-samples 50 \
  --mass-min-kg 1e10 --mass-max-kg 1e11 --scale-vel-min 0.9 --scale-vel-max 1.1 --jobs 2
python3 sobol/run_mass_sobol_phantom.py -i                 # wizard
python3 sobol/run_mass_sobol_phantom.py --dry-run [flags]  # stage dirs + patched .setup only
cd sobol && python3 inter_DEM_run_mass_sobol.py --dem-np 20 30 64 [flags]
```

Settled body on the ephemeris (np300, 30 hr spin, 12 hr):
```bash
OMP_NUM_THREADS=1 python3 sobol/run_mass_sobol_phantom.py --base-dir sobol --prefix sobol \
  --phantom-dir /home/mboyle/Honours/sobol --ephemeris-cache-dir sobol --use-dem-fixed true \
  --np-apophis-list 300 --body-source settled --spin-period-fixed 30 --tmax-hours 12 --dtmax-hours 1 \
  --num-samples 1 --jobs 1
```

Hyperbola sweep over rp and v_inf (one settled body reused):
```bash
OMP_NUM_THREADS=1 python3 sobol/run_mass_sobol_phantom.py --phantom-dir /home/mboyle/Honours/sobol \
  --base-dir sobol --prefix sobol --ephemeris-cache-dir sobol --use-dem-fixed true --np-apophis 300 \
  --body-source settled --encounter hyperbola --flyby-rp-km-min 20000 --flyby-rp-km-max 40000 \
  --flyby-vinf-kms-min 5 --flyby-vinf-kms-max 7 --tmax-hours 48 --dtmax-hours 0.5 --num-samples 16 --jobs 2
```

Torque-align reruns (`run_torque_align_blender_reruns.py`, cleanup on; these wrappers pass `--no-cleanup` for Blender):
```bash
bash sobol/campaigns/run_torque_align_blender_vis.sh
bash sobol/campaigns/run_torque_align_blender_vis_obj.sh
OMP_NUM_THREADS=2 JOBS=2 bash sobol/campaigns/run_metricfix_reruns.sh   # thesis spin campaigns, ../docs/BATCHES.md
```

Dumps → Blender inputs (bodies CSV + grains `.npz`):
```bash
python3 sobol/Analysis/run_demtocsv_batch.py --min-dem-grains 450 \
  --base-output-dir "/mnt/c/Users/22boy/OneDrive/Documents/GC-Max_desktop/Honours/Code/DEMCSVs/torque_align_obj" \
  sobol_mass_runs/<batch>/run_0001 sobol_mass_runs/<batch>/run_0002
```

Sensitivity analysis:
```bash
python3 sobol/Analysis/Analysis.py --method classic --csv sobol_mass_runs/<batch>/sobol_mass_outputs.csv \
  --response closest_approach_au   # or intrinsic_/approach_/post_flyby_spin_period_hr
python3 sobol/Analysis/Analysis.py --method saltelli --sobol-problem-json <batch>/saltelli_problem.json \
  --saltelli-meta-json <batch>/saltelli_meta.json --saltelli-y-csv <batch>/saltelli_Y.csv \
  --saltelli-y-column closest_approach_au
```

Tests: `cd sobol && python3 -m pytest tests` (~2.5 min; Blender tests launch `blender.exe`). Fixtures: `tests/fixtures/README.md`, `tests/fixtures/dem_run_dir.sh <particle|sink> <dir> <np> <tmax_in> <dtmax_in> <torque_deg> [run]`.
