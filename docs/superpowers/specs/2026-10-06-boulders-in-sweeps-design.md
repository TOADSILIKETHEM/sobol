# Boulders in sweeps and TUIs — design

Date: 2026-10-06. Status: approved in chat, awaiting spec review.

## Goal

Run Mia's bonded-boulder bodies (`phantomclumps`, `moddump_clumps.f90`) through the sobol pipeline, with boulder
structure as Sobol axes, from the CLI, `tui_run.py` and `tui_sim_render.py`.

Serves SCI4502 RQ1 (how much grain structure shifts the disruption threshold) and RQ3 (bi-lobed vs single-lobed at
matched mass, cohesive strength, spin and flyby geometry). Scope context: boulders replace the PDF's polydisperse /
polyhedral grains (scope change 2026-10-06, not yet agreed with supervisors).

## Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Where clumping happens | Per sample, in the run dir, from the cached settled body | Clumping takes seconds; continuous axes make every sample unique, so a clump cache never hits. All samples share one packing, so only boulder parameters differ. |
| Re-settle after clumping | No. Bonds form on the first force evaluation of the flyby run | Body is already settled; clumping moves no grains. Cost per run unchanged. |
| Sobol axes | `f_boulder`, `rb_max` | Boulder structure is the RQ1 variable. |
| Bond strength | Fixed per batch, as `bond_sigma_pa` (Pa) | RQ1/RQ3 hold cohesive strength matched; sweeping it would mix strength with structure in the indices. Pa matches `--coh-pa` and the literature. |
| `kb_cgs` | Fixed, default = `kn_cgs` | With σ set, `kb` is stiffness only. `kn_cgs + kb_cgs` sets the contact timestep (`dem.f90` `get_dem_dt`), so sweeping it varies cost. |
| `bond_tau_pa` | Fixed as `bond_tau_ratio × bond_sigma_pa` (default ratio 1) | One interpretable strength value. |
| `bond_reach` | Fixed, default 0.1 | Sets bond topology, break strain and the `phantomclumps` connectivity walk at once; must match between `phantomclumps` and `.in`. |
| `lobe_ratio` / RQ3 | Fixed. RQ3 = two batches with identical samples, `--boulder-placed none` vs `binary` | Paired comparison answers RQ3 directly. |
| Random seed | Fixed per batch | Keeps clump randomness out of the indices. |
| `-i` wizard | Gated boulder section | New parser flags are prompted automatically (one prompt per action), so without a gate every wizard user would get ~17 boulder prompts. |

## Inputs

All require `--boulders`, which requires `--body-source settled`, `--dem-model particle`, `--use-dem-fixed true`.

| Input | CLI | Default | Axis? |
|-------|-----|---------|-------|
| `f_boulder` | `--f-boulder` or `--f-boulder-min/--f-boulder-max` | 0.3 | yes |
| `rb_max` (grain radii) | `--rb-max` or `--rb-max-min/--rb-max-max` | 4 | yes |
| `rb_min` (grain radii) | `--rb-min` | 1.5 | no |
| `q_slope` | `--boulder-q` | 3.5 | no |
| shape | `--boulder-shape {round,ellipsoid,grown}` | round | no |
| `axis_min` | `--boulder-axis-min` (ellipsoid only) | binary default | no |
| `jagged` | `--boulder-jagged` (grown only) | binary default | no |
| placed | `--boulder-placed {none,binary,file}` | none | no |
| `lobe_ratio` | `--lobe-ratio` (binary only) | binary default | no |
| placed file | `--boulder-file PATH` (file only) | — | no |
| `nmin_clump` | `--boulder-nmin` | binary default | no |
| seed | `--boulder-seed` | `--seed` | no |
| `bond_sigma_pa` | `--bond-sigma-pa` | required (no default) | no |
| `bond_tau_pa` | `--bond-tau-ratio` | 1.0 | no |
| `kb_cgs` | `--kb-cgs` | `kn_cgs` of the run `.in` | no |
| `bond_reach` | `--bond-reach` | 0.1 | no |

"binary default" = press Enter at that `phantomclumps` prompt (blank stdin line), so the Fortran default applies. The
numeric defaults above (0.3, 4, 1.5, 3.5) are starting points, to be revised after the σ trial (Open items).

`bond_sigma_pa` has no default: no tested value exists yet, and a silent default would hide that.

`f_boulder` is the fraction of the **whole body** put into **random** boulders; placed boulders (binary lobes, file
boulders) come on top. Spike: `f_boulder = 0.3` + contact binary on a 391-grain body gave 33% in lobes, 36% in random
boulders, 31% dust.

## Components

### `sobol/boulders.py` (new)

Pure functions, in the `encounter.py` style.

- `BoulderSpec` (dataclass): all inputs above, resolved per sample (axis values filled in).
- `clumps_stdin(spec) -> str`: stdin answers in `moddump_clumps.f90` prompt order:
  `f_boulder`, `rb_min`, `rb_max`, `q_slope`, `ishape`, [`axis_min` if ellipsoid | `jagged` if grown], `iplaced`,
  [`lobe_ratio` if binary | file name if file], `bond_reach`, `nmin_clump`, seed. Order is checked against
  `moddump_clumps.f90:110-122`; conditional lines are the off-by-one risk.
- `ClumpResult` (dataclass) + `parse_clumps_log(text) -> ClumpResult`: `n_grains`, `n_placed`, `n_random`,
  `n_in_boulders`, `f_dust`, `n_dropped`, from the summary block (`moddump_clumps.f90:217-230`). Flags the
  "ran out of free grains" warning.
- `clump_body(run_dir, body, spec, clumps_bin) -> ClumpResult`: copies the cached dump to `run_dir/body_00000`
  (+ `body.in`), runs `phantomclumps body_00000 clumped 0 --maxp=max(2000, 4·n_kept)` with `clumps_stdin(spec)`
  (moddump appends the dump number: output is `clumped_00000` + `clumped.in`), logs to `run_dir/clumps.log`, writes
  `run_dir/boulders.json` (spec + result). Never writes to the cache.
- `bond_in_keys(spec, kn_cgs) -> dict[str, float]`: `kb_cgs`, `bond_reach`, `bond_sigma_pa`, `bond_tau_pa`.
- `check_bonds_on(log_text)`: raises unless ` DEM clump bonds enabled` (`dem.f90` `dem_cohesion_summary`) is present.

### `run_mass_sobol_phantom.py`

- Flags as in Inputs; `f_boulder` and `rb_max` join the Sobol problem when given as min/max, through a new
  `_BOULDER_VARIATION_SPEC` (same shape as `_FLYBY_VARIATION_SPEC`). It is appended **last** in
  `count_dimensions()`, `sample_column_order()`, `build_run_samples()`, `run_sample_from_salib_row()` and the slug,
  so dimension order (and resume) of existing batches is unchanged.
- `RunSample` gains `f_boulder`, `rb_max` (per sample) and `boulders: Optional[BoulderSpec]` (fixed part; `None` =
  boulders off). Fixed values are filled by `_apply_fixed_run_sample_overrides()`, which resume also calls.
- `canonical_sweep_descriptor()` and the slug include every boulder input (fixed ones too). Without this, the RQ3
  pair (`--boulder-placed none` vs `binary`, otherwise identical) would hash to the same batch folder.
- `interactive_run_mass_sobol.py`: boulder dests in `DEST_TO_SECTION` / `INTERACTIVE_BRIEF`, plus a gate (like the
  mass gate) that skips the whole boulder section unless the user turns boulders on; asked only when
  `body_source = settled`.
- `_validate_boulders(args)`, called from `validate_args`:
  - `--boulders` needs settled + particle DEM; boulder flags without `--boulders` are rejected.
  - `0 ≤ f_boulder ≤ 1`; `rb_min ≥ 1`; `rb_max` (lower bound if swept) `≥ rb_min`.
  - `--lobe-ratio` only with `binary`; `--boulder-file` only with `file`, and the file must exist.
  - `--boulder-axis-min` only with ellipsoid; `--boulder-jagged` only with grown.
  - `--bond-sigma-pa` required; `= 0` allowed with a warning ("stretch-only breaking").
- `_OPTIONAL_IN_KEYS` gains `kb_cgs`, `bond_reach`, `bond_sigma_pa`, `bond_tau_pa` (template `sobol.in` has none of
  them; checked 2026-10-06).
- `phantomclumps` resolved with `resolve_phantom_executable(..., "phantomclumps", must_exist=True)` by
  `prepare_settled_bodies_and_dem_tools()` (and `tui_sim_render.run_sim_stage`) when boulders are on, and stored
  on each sample as `RunSample.phantomclumps_bin` (same pattern as `settled_body_dir`). `DemTools` unchanged.
- A placed-boulder file is copied into each run dir as `boulders.txt` and passed by that name (the Fortran
  `placed_file` is `character(len=256)`).
- Binary probe (like `check_binary_supports_contact_keys`): refuse if `phantom` lacks the write_inopt text
  `clump bond tensile strength` (present in current `sobol/phantom`).
- Sample CSV gains `f_boulder`, `rb_max`. Resume rebuilds samples from the same flags (deterministic, as now), so
  the boulder flags must be passed again. `resume_batch.check_resume_boulders()` acts only on boulder batches
  (`--boulders` set or a `run_*/boulders.json` on disk; other batches resume as before) and refuses to resume when the
  samples-CSV header differs from `run_id` + `sample_column_order(args)`, when a run dir has `boulders.json` but
  `--boulders` is unset, or when the first `boulders.json` `spec` differs from `spec_from_args(args)` on any field
  except the axis values `f_boulder` and `rb_max`. (The batch hash is used only for truncated slugs, so it is not
  a mismatch check.)
- `_cleanup_run_dir()` also removes `clumped_[0-9]*` and `clumped.in` (the flyby dumps already carry `iclump`).
  Existing patterns unchanged.
- `Analysis/Analysis.py` `INPUT_CANDIDATES` gains `f_boulder`, `rb_max` (classic mode).

### `run_one_case()` flow

After the settled body loads, when boulders are on:

1. `clump_body(...)`.
2. **Ephemeris:** `apply_settled_body_to_setup` stages `clumped_00000` as `settled_body` (new optional `dump`
   argument; default unchanged).
3. **Hyperbola:** `prepare_hyperbola_run` takes an optional source dump and runs `phantomflyby` on
   `clumped_00000`. Spike: the `.in` phantomflyby writes has the bond keys but `kb_cgs = 0`, so without step 4
   the flyby runs unbonded.
4. **Both paths:** `apply_run_sample_to_in()` (already called after `phantomsetup` on ephemeris and after
   `phantomflyby` on hyperbola) also writes `bond_in_keys(...)` when `sample.boulders` is set and the run made
   boulders (`boulders.json` `n_boulders > 0`; a run with none gets no bond keys, so its `.in` and timestep match
   a non-boulder run: `get_dem_dt` uses `kn_cgs + kb_cgs`, so `kb_cgs > 0` shrinks dt by √2 even with no clumps); missing keys are
   inserted (template `sobol.in`), present ones replaced (`fly.in`). `patch_encounter_in` is not changed.
5. Fail the run (no silent no-op) if:
   - boulders were asked for and `n_placed + n_random == 0`. Asked for means placed ≠ none, or the Fortran
     random target `nint(f_boulder × n_grains) > 0` (`moddump_clumps.f90`: `ntarget = nint(f_boulder*npart)`,
     round half away from zero). A smaller `f_boulder` on a small body makes nothing and is not an error, so
     a sweep from `f_boulder = 0` leaves no failed rows;
   - ephemeris and `setup.log` lacks `boulders in body =`;
   - boulders exist but `phantom.log` lacks ` DEM clump bonds enabled` (`dem_cohesion_summary`, printed when
     `kb_cgs > 0`), checked right after `phantom` returns, before metrics and cleanup.

`f_boulder = 0` with placed = none still runs `phantomclumps` (all-dust body, bonds off). This is the RQ1 baseline
and must reproduce a non-boulder settled run.

### Output columns

`f_boulder` (requested), `rb_max`, `f_boulder_actual` (`n_in_boulders / n_grains`; carving makes it differ),
`n_boulders`, `boulder_placed`, `bond_sigma_pa`. Fixed values also in each `boulders.json`. Add to `docs/METRICS.md`.

### `tui_run.py` (sweep TUI, builds CLI argv)

1. **New section "Body / encounter"** (prerequisite, currently missing): `body_source`, `encounter`,
   `flyby_rp_km` / `flyby_vinf_kms` (fixed or min/max), `flyby_start_sep_km`, `settle_tdyn`, `relax_tdyn`,
   `body_cache_dir`. Maps 1:1 to existing flags; hyperbola fields disabled unless `encounter = hyperbola`.
2. **New section "Boulders"**, disabled unless `body_source = settled`: on/off checkbox; `f_boulder` and `rb_max`
   each fixed or min/max; fixed fields for the rest of Inputs. Conditional fields (`lobe_ratio`, file,
   `axis_min`, `jagged`) enabled by the shape/placed selects.
3. `_build_argv` emits the flags; validation errors come from `validate_args` as now.

### `tui_sim_render.py` (one run)

"Boulders" block under the `body_source` rows, enabled only when `settled`: same fields as above, fixed values
only. Builds a `BoulderSpec` and passes it through the existing `attach_settled_bodies` → `run_one_case` path.
Update `sobol/docs/tui_sim_render.md`.

## Verified by spike (2026-10-06, scratchpad, cached body `np300_srho1_712bb7806311`)

- `phantomclumps` on a cached body with piped stdin (`f_boulder = 0.3`, binary): rc 0, 2 placed + 8 random boulders.
- `sarracen.read_phantom` (used by `particle_dem.py`, `viz_preprocess_dump.py`, `dump_frames.py`,
  `ToCSVFirst.py`) reads the clumped dump; `iclump` is a normal column.
- `phantomflyby` on the clumped dump keeps `iclump` (271 bonded-ID grains in `fly_00000`).
- Not checked in the spike, checked in the Task 11 integration runs:
  - `ibond01`… arrays: `sarracen.read_phantom` reads bonded flyby dumps (`iclump`, `ibond01`..`ibond10`) without
    error on both encounters; `viz_preprocess_dump.py` also runs on them.
  - Bond pre-load: no jump at step 1 (internal velocity grows about linearly in time). The bonded run's initial
    internal acceleration is about 8x the no-boulder run's; internal KE peaks near 14% of rotational KE at about
    1 h, then decays within a few hours. The cause (bond rest length vs staged gap) is unproven.
  - All-dust baseline vs a non-boulder run: identical metrics and step count only once the bond keys are skipped
    for runs with no boulders (before that, dt differed by √2 and approach spin by 0.4%).
  - The plan's Step 4 command (hyperbola, fixed everything, `--np-apophis 300`) needed `--np-apophis-list 300`:
    a batch with no swept dimension is refused.

## Testing

Unit (`sobol/tests/test_boulders.py`, plus additions to existing files):

- `clumps_stdin` for every shape × placed combination: line count and position of each answer.
- `parse_clumps_log` on a captured real log (fixture), including the "ran out" warning.
- `bond_in_keys` (`kb_cgs` default = `kn_cgs`, τ = ratio × σ).
- `_validate_boulders` rejects (each rule above).
- `apply_run_sample_to_in` with boulders inserts missing bond keys (template `.in`) and replaces present ones (`fly.in`).
- TUI argv round-trips for both new `tui_run.py` sections (`test_tui_*` pattern).

Integration (manual, small body, `np_apophis` ~300):

- Ephemeris and hyperbola each: `clumps.log`, `setup.log` (ephemeris) and `phantom.log` show boulders and bonds on.
- `f_boulder = 0`, placed none: metrics match a non-boulder settled run with the same spec and seed.
- Contact binary: two placed boulders, `f_boulder_actual` near the lobe fraction.
- On a bonded flyby run: `particle_dem` metrics and `viz_preprocess_dump.py` read dumps carrying `ibond01`…
- No-boulder regression: the existing lattice, settled-ephemeris and hyperbola example commands in
  `sobol/CLAUDE.md` give the same slug/hash and outputs as before; full `pytest tests` passes.

## Open items (outside this spec)

- **σ trial before any campaign:** one body, σ ladder (e.g. 1, 10, 100 Pa) on the ephemeris flyby. If boulders all
  survive or all break across the ladder, the `f_boulder` axis shows nothing. Pick the batch σ and revise the
  numeric defaults from this.
- **Bond pre-load:** bond rest length is "touching" but settled grains overlap slightly; check for a start-up
  transient in the integration run.
- **Supervisors:** the boulder scope change is not yet agreed; check with Daniel before locking the campaign.
- Boulder survival metrics (e.g. fraction of boulders intact after flyby) are not in this spec.
- **RQ3 matching:** with the same `f_boulder`, the binary body has more bonded grains than the single-lobe body
  (lobes are extra). Decide before the RQ3 campaign whether "matched cohesive strength" needs matched bonded
  fraction (e.g. lower `f_boulder` for the binary batch).
