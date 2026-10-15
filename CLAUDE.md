# Sobol sweep runner (`sobol/`)

Parent context: `../CLAUDE.md`. Metrics: `../docs/METRICS.md`. Batches: `../docs/BATCHES.md`. Dated notes: `../docs/CONTEXT_CHANGELOG.md`.

## Architecture: sweep runner

`sobol/run_mass_sobol_phantom.py` is the only runner.

- **`build_parser()`** registers all CLI args. Registration order = interactive prompt order.
- **`validate_args()`**: a dimension is active only if **both** min and max are set; one without the other is an error; no active dimension is an error unless `--np-apophis-list` or `--num-samples 1` without Saltelli (one fixed run; the unified TUI's Single run).
- **`build_run_samples()` / `run_sample_from_salib_row()`** map `[0,1)` samples to physical values. Dimension order must match `count_dimensions()` and `sample_column_order()`.
- **`run_one_case()`** copies template `.in`/`.setup` into `run_XXXX/`, patches via `apply_run_sample_to_setup()`, copies ephemeris `.txt`, runs `phantomsetup` → `phantom`, then `compute_run_metrics()` (sink DEM: `.ev`; particle DEM: dumps via `particle_dem.py`). `--dem-model particle|sink` (default particle) writes `use_dem` / `use_dem_as_sinks`. `check_setup_log()` refuses runs whose setup skipped spin. Cohesion gap = `dn × 2 × R_grain` from the setup dump. Friction: every new sample gets `mu_s = DEFAULT_MU_S` (0, frictionless) via `apply_contact_defaults()` unless `--mu-s X` or `--mu-s-min/max` (binary default is 0.5); other Mia contact axes `--mu-r-*`, `--mu-t-*`, `--eps-t-*`, `--coh-pa-*`; fixed `--mu-r`/`--mu-t`/`--eps-t` (need `mu_s > 0`, exclusive with their axes). Keys in `_OPTIONAL_IN_KEYS` missing from a `.in` are inserted (phantom only warns on unknown keys); `check_binary_supports_contact_keys()` refuses `--mu-s > 0` or a Mia contact sweep on a binary without friction (`--phantom-dir sobol/bin_demsync-mia_464ea28fa` reproduces pre-merge runs). `validate_args` rejects `mu_r`/`mu_t`/`eps_t` axes without `mu_s > 0`, friction axes with `--dem-model sink`, and a `cohesion_pa` axis with any `kt_cgs` (`dem.f90` ignores kt when `cohesion_pa > 0`). Settled bodies settle and relax with `SettleSpec.mu_s` (0; in the cache key) regardless of the flyby's `mu_s`, unless `--settle-with-friction` (TUI/driver: `SimParams.settle_friction`), which settles with the run's `mu_s`/`mu_r`/`mu_t` under its own key (refused with a swept friction axis: one body per sample). `mu_r`/`mu_t` enter the key only when set, so older cached bodies still hit. DEM runs: `nfulldump=1` (Blender needs full dumps); breakup + spin in one pass (`_extract_dem_metrics_bundle()`); spin columns `intrinsic_/approach_/post_flyby_spin_period_hr` (`../docs/METRICS.md`).
- **`main()`**: validate → build samples → `prepare_settled_bodies_and_dem_tools()` → preflight → `ProcessPoolExecutor` (`--jobs > 1`) → summary CSV.

Output per batch: `sobol_mass_samples.csv`, `sobol_mass_outputs.csv` (rewritten after every run, so a killed batch keeps finished rows). Saltelli adds `saltelli_problem.json`, `saltelli_meta.json`, `saltelli_Y.csv`, `saltelli_eval_manifest.csv`.

- **Cleanup default ON** everywhere (runner, torque-align reruns, `tui_sim_render.py`): dumps, `.ev`, `phantom.log` deleted after metrics. `--no-cleanup` (TUI: `keep_dumps`) keeps them.
- **Template `sobol.setup`** is point-mass (`use_dem = F`, `np_apophis = 1`); DEM comes from flags. `apophis_spin_period` is in **seconds**. Spin axis keys are `apophis_spin_axis_x/y/z` (obliquity/azimuth keys are gone). `packing_file =` (blank) must be present in every `.setup` template: current `phantomsetup` treats a missing key as a read error, rewrites the file and stops (the runner retries once).

### Settled bodies and hyperbola encounters

Both need `--use-dem-fixed true` and `--np-apophis` / `--np-apophis-list` (grains kept after the crop), and `make demtools`.

- **`--body-source {lattice,settled}`** (default `lattice`). `settled`: `settled_body.py` settles (`pack_settle=T`, `apophis_only=T`, `idamp=2`, `--settle-tdyn` default 5), crops (`phantommoddump`) and relaxes (`--relax-tdyn` default 0.5) one body per `(np_apophis, scale_rho, shape_file, binaries)` into `<cache-root>/<key>/` (`body.json`; cache root = `--body-cache-dir` or `<output-root>/settled_bodies`). Built before workers start; a per-key `flock` (`<cache-root>/<key>.lock`) makes a second launch of the same key (CLI or TUI job) wait and reuse the body (no lock, with a warning, where the filesystem lacks flock). The key hashes spec + shape/mesh + binary SHA256s, so a rebuilt binary or edited shape resettles. `<key>.partial` dirs are discarded. A settle with `packing_fraction < 0.5` is refused (raise `--settle-tdyn`). Small N overlaps more, so `packing_phi` and kept counts run above nominal at low `np_apophis`.
- **`--body-source settled --encounter ephemeris`**: settled body on the real 2029 flyby. `encounter.apply_settled_body_to_setup()` stages the cached dump as `run_XXXX/settled_body` plus the body's shape, patches `packing_file`, `pack_settle = F`, `apophis_shape_file`, `np_apophis = n_kept`, then the normal `phantomsetup` → `phantom` path (Earth = sink 4; spin and `--scale-vel-*`/`--scale-pos-*` apply). `check_packing_setup_log()` fails the run unless `setup.log` says `placed <n_kept> pre-built grains`. Needs phantom `DEMsync-mia` with the `get_conserv` reset (else "Large error in angular momentum conservation" at step 2). Semantics: `../docs/MIA_PACKING_WORKFLOW.md`.
- **`--encounter hyperbola`** (needs `--body-source settled`): `encounter.py` runs `phantomflyby` on the cached body. Required: `--flyby-rp-km`/`--flyby-vinf-kms` (fixed) or `-min/-max` pairs (swept; no silent default), `--tmax-hours`, `--dtmax-hours`. Optional: `--flyby-start-sep-km` (4e5), `--flyby-perturber-earth-masses` (1). `encounter.check_flyby_geometry()` (shared by CLI and TUI) requires `R_EARTH_KM < rp < start_sep` and `tmax >= 2 ×` the longest time to pericentre over the swept range (gridded, because it peaks inside the rp range). `patch_encounter_in()` forces `idamp = 0` (load-bearing: `phantomflyby` copies the settle's `idamp = 2`). Writes `encounter.json` and a record `sobol.setup` (converter manifest only). Earth = sink 1 (`--sink-earth-id` auto-set). No spin. `--maxp = max(2000, 4·n_kept)`.
- **`--boulders`** (needs `--body-source settled`, `--bond-sigma-pa`): `boulders.py` copies the cached body into each run dir and runs `phantomclumps` (stdin from `clumps_stdin`; `clumps.log`, `boulders.json`), then stages `clumped_00000` instead of the cached dump on either encounter. Bond keys (`kb_cgs` default = `kn_cgs`, `bond_reach` 0.1, `bond_sigma_pa`, `bond_tau_pa` = `--bond-tau-ratio` × σ) go in through `apply_run_sample_to_in()`; on hyperbola this is load-bearing, `fly.in` comes out with `kb_cgs = 0`. Axes: `--f-boulder-min/max`, `--rb-max-min/max` (not with `--np-apophis-list`); everything else fixed per batch (`--f-boulder` 0.3, `--rb-max` 4, `--rb-min` 1.5, `--boulder-q` 3.5, `--boulder-shape`, `--boulder-placed`, `--lobe-ratio`, `--boulder-seed` default `--seed`, ...; boulder flags without `--boulders` are an error). Run fails if boulders were asked for but none made (asked for = placed != none, or the Fortran target `nint(f_boulder × n_grains) > 0`, so a tiny `f_boulder` on a small body is an all-dust run, not an error), or `phantom.log` lacks ` DEM clump bonds enabled` (ephemeris also checks `setup.log` for `boulders in body`). A run that made no boulders (`boulders.json` `n_boulders = 0`) gets no bond keys in its `.in` (`boulders.bonds_wanted()`): `kb_cgs > 0` shrinks the DEM step by √2 even with no clumps, so `f_boulder = 0` + `--boulder-placed none` (all-dust baseline) reproduces a non-boulder settled run exactly. Slug gains `bld<placed>_sig<σ>` plus `fb<v>` / `rbx<v>` for fixed `--f-boulder` / `--rb-max` (swept axes appear as `fb<lo>-<hi>` / `rbx<lo>-<hi>`). Columns: `f_boulder_actual` (placed + random grains / n_kept), `n_boulders` (`../docs/METRICS.md`). Cleanup removes `clumped_*`/`unclumped_*`; `clumps.log` and `boulders.json` stay. Spec: `docs/superpowers/specs/2026-10-06-boulders-in-sweeps-design.md`.
- **Rejected with settled** (`_validate_body_encounter()`): `--dem-model sink`, varied `use_dem`, mass sweeps, `--scale-r-apophis-*`, kn sweeps, `--scale-rho-*` sweeps, `use_shape_crop` flags. **Hyperbola also rejects** `--scale-vel-*`/`--scale-pos-*`, spin flags, `apophis_only` flags.
- **Outputs:** settled batches add `np_kept`; hyperbola rows carry `np_apophis`/`scale_rho`. With `phantomanalysis` built, particle-DEM runs add `shape_b_on_a`, `shape_c_on_a`, `packing_phi`, `f_unbound_energy`. Slug gains `settled` / `hyp`. Columns: `../docs/METRICS.md`.
- **Resume:** `resume_batch.py` calls the same `prepare_settled_bodies_and_dem_tools()`; pass the original `--output-root`/`--body-cache-dir` or the body is rebuilt. `check_resume_boulders()` acts only on boulder batches (`--boulders` set or a `run_*/boulders.json` on disk; other batches resume exactly as before) and refuses (exit 1) when the samples-CSV header differs from `run_id` + `sample_column_order(args)`, when a `boulders.json` exists but `--boulders` is not set, or when the first `boulders.json` `spec` differs from the flags on any field except the axis values `f_boulder` / `rb_max`. Boulder batches need every boulder flag passed again.
- **Tools:** `make demtools` builds `phantommoddump` (crop), `phantomflyby` (`moddump_earthflyby.f90`), `phantomclumps` (`moddump_clumps.f90`, boulders; `--boulders`) and `phantomanalysis` (`analysis_demshape.f90`) into `sobol/`; `resolve_dem_tools()` prefers `bin/<name>` if present.
- **Known gaps:** the cache is never pruned; `Analysis/Analysis.py` classic mode lacks `flyby_rp_km`/`flyby_vinf_kms` in `INPUT_CANDIDATES`.

## Architecture: interactive wizard

`sobol/interactive_run_mass_sobol.py`, used by `-i` and by `inter_DEM_run_mass_sobol.py`.

- **`run_interactive_wizard(parser, initial_args)`** prompts once per `parser._actions` entry. Section headers print on each section change, so a section can appear more than once.
- **Gates** skip whole groups when declined: mass (`MASS_GATE_DESTS`), scale bounds, DEM contact bounds (`_prompt_dem_contact_vary_selection`), time (`_prompt_time_vary_selection`). Boulder flags (`BOULDER_GATE_DESTS`) are skipped unless the body is settled and `--boulders` is on.
- `nargs='+'` options (`--np-apophis-list`, `--spin-period-list`) are prompted as space-separated lists of their argparse `type`.
- **Extending:** add a new flag's dest to `DEST_TO_SECTION` (heading) and `INTERACTIVE_BRIEF` (help text).

`inter_DEM_run_mass_sobol.py`: runs the wizard once, then one subprocess per `np_apophis` (temp `--base-dir` with `np_apophis` patched). `--dem-np N1 N2 ...` skips the count prompt.

## Sim + render pipeline TUI (summary)

`tui_sim_render.py`: **one** DEM run per launch: PHANTOM (`run_one_case`) → convert (`run_demtocsv_batch`) → headless Blender, any mix of `composite`, `instance_grains`, `instance_static`, optional side-by-side compare (`render_compare.py` → `run_0001_compare/`) (result `done` or `partial`); convert is skipped when only composite/instance_grains are ticked (one `viz_preprocess_dump.py` call reads the dumps). Sweep TUI is `tui_run.py` (DEM Contact section: vary rows for every `_IN_VARIATION_SPEC` axis incl. `mu_s`/`mu_r`/`mu_t`/`eps_t`/`coh_pa`, plus `mu_s_fixed` → `--mu-s`, exclusive with vary `mu_s`; "Body / encounter" section: `body_source`, `encounter`, flyby rp/v_inf fixed or min/max, `flyby_start_sep_km`, `settle_tdyn`, `relax_tdyn`, `body_cache_dir`; "Boulders" section: `boulders` checkbox, shape/placed selects, `f_boulder`/`rb_max` fixed or min/max, bond keys; boulder fields filled without the checkbox are refused; hyperbola and boulder fields are greyed out unless `encounter = hyperbola` / `body_source = settled` + ticked, visual only). `tui_sim_render.py` has a "Sim — boulders (settled only)" block, same fields fixed only; `bond_sigma_pa` required; blank seed = `DEFAULT_SEED` (42, as the CLI). Keep `prefix = sobol` (converter globs `sobol_[0-9]*`). Set `ephemeris_cache_dir = /home/mboyle/Honours/sobol` (blank = live Horizons download). Each preprocess/render writes `preprocess.log` / `render.log` (latest line in the status bar); every non-dry run writes `run_0001/pipeline_timing.json` (per-stage wall-clock, `np_apophis`, git SHAs). No mid-run cancel. Detail: `docs/tui_sim_render.md`.

**Unified TUI (new, Oct 2026; old TUIs unchanged until retirement):** `python3 sobol/tui.py`: Simulate (single run or sweep, form built from `params_schema.py`, validated by the runner's own `validate_args`) / Render (sweep runs or any run dir) / Jobs. Every launch is a detached job (`job.py`, `<output_root>/_jobs/<id>/`, survives SSH drops; Cancel, Resume). New runner flags must be classified in `params_schema.py` (test fails otherwise). Detail: `docs/tui.md`. Spec/plan: `docs/superpowers/{specs,plans}/2026-10-13-unified-tui*`.

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

Boulders on the settled body, f_boulder swept, contact binary (np300, 12 hr):
```bash
OMP_NUM_THREADS=1 python3 sobol/run_mass_sobol_phantom.py --phantom-dir /home/mboyle/Honours/sobol \
  --base-dir sobol --prefix sobol --ephemeris-cache-dir sobol --use-dem-fixed true --np-apophis 300 \
  --body-source settled --spin-period-fixed 30 --tmax-hours 12 --dtmax-hours 1 --num-samples 8 --jobs 2 \
  --boulders --f-boulder-min 0 --f-boulder-max 0.5 --boulder-placed binary --bond-sigma-pa 10
```
(σ = 10 Pa is an example only; the batch value comes from the σ trial in the spec's Open items.)

Torque-align reruns (`run_torque_align_blender_reruns.py`, cleanup on; these wrappers pass `--no-cleanup` for Blender):
```bash
bash sobol/campaigns/run_torque_align_blender_vis.sh
bash sobol/campaigns/run_torque_align_blender_vis_obj.sh
OMP_NUM_THREADS=2 JOBS=2 bash sobol/campaigns/run_metricfix_reruns.sh   # thesis spin campaigns, ../docs/BATCHES.md
```

Breakup clump analysis (FoF clumps, bonds/boulders intact; needs kept dumps; `../docs/BATCHES.md` § Breakup ablation):
```bash
python3 sobol/Analysis/plot_breakup_clumps.py --run-dir <batch>/run_0001   # one run
python3 sobol/Analysis/plot_breakup_compare.py                             # frictionless vs friction + boulders
python3 sobol/Analysis/plot_breakup_ablation.py                            # baseline, A, B, C, realistic
```

Render existing dumps without the sim (dump route; any prefix with `--dem-model`/`--np-apophis`; other-machine setup `Code/README.md`):
```bash
python3 sobol/render_dumps.py sobol_mass_runs/<batch>/run_0001 --name <out> [--samples 124 --fps 24 --max-frames N --video]
#   [--slowmo 4 --resume]: instance x4 frames per dump into run_0001_render_instance_grains_slowmo4/; --resume skips frames on disk
python3 sobol/render_dumps.py <fly dir> --prefix fly --dem-model particle --np-apophis N --apophis-only \
  --encounter hyperbola   # someone else's phantomflyby run (lone Earth sink, no Sun)
python3 sobol/setup_paths.py --check   # paths, assets, Blender + pandas, preprocess deps
python3 sobol/render_dumps.py ... --queue   # preprocess only; jobs -> <data>/DEMCSVs/_render_queue/ for the desktop (Code/render_queue.py, Code/docs/DESKTOP_RENDER.md)
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

Tests: `cd sobol && python3 -m pytest tests` (~5.5 min, 2026-10-14; Blender tests launch `blender.exe`). Fixtures: `tests/fixtures/README.md`, `tests/fixtures/dem_run_dir.sh <particle|sink> <dir> <np> <tmax_in> <dtmax_in> <torque_deg> [run]`.
