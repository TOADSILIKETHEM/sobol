# Unified TUI (sweep + single run + render) — design

Date: 2026-10-13. Status: approved; revised 2026-10-13 (built alongside the old TUIs, see "Coexistence").

## Goal

Replace the two Textual front-ends (`tui_run.py`, sweep config; `tui_sim_render.py`, one run → convert → render) with
one app, `sobol/tui.py`, in which every sim parameter is defined once. Fixes three problems at once:

- **Maintenance.** A new runner flag today touches `build_parser()`, the wizard's `DEST_TO_SECTION` /
  `INTERACTIVE_BRIEF`, `tui_run.py` and `tui_sim_render.py` (`SimParams` + its own validation).
  `tests/test_tui_contact_keys.py` exists to keep the two forms in sync.
- **Workflow.** One place to go for a sweep, a single run, rendering runs of a sweep, or rendering existing dumps,
  with live progress. Today `tui_run.py` exits and shells out (no progress); `tui_sim_render.py` runs in a thread,
  so an SSH drop kills it.
- **Other users.** Mia and future students get one discoverable tool, in a terminal or a browser.

## Requirements

- Runs on laptop WSL over Remote-SSH, on an OzSTAR login node (terminal only, no Blender), and on other students'
  PCs (portable-viz setup, `setup_paths.py`).
- On OzSTAR it configures and launches only: runs on whatever node it is started on. No Slurm logic.
- No change to runner or render-stage behaviour. Front-end and plumbing only, apart from the three small changes
  under "Changes outside the UI".
- `render_dumps.py` stays as a CLI (students use it); its flags do not change.
- Jobs longer than a few minutes survive the TUI closing and SSH drops.

## Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Toolkit | Textual (already used by both TUIs, 8.2.6 installed) | Terminal on OzSTAR; `textual serve` gives the same app in a browser for students and via VS Code port forwarding. One codebase. |
| Desktop GUI (Qt/Tk) | No | No display on OzSTAR or over Remote-SSH. |
| Web GUI (NiceGUI/FastAPI) | No | Rewrites ~2,900 lines of UI; OzSTAR needs an `ssh -L` tunnel; not worth it before D1. |
| Launcher shell over the old TUIs | No | Fixes "which tool" only; duplication stays. |
| Single run | A runner run with `--num-samples 1` and every axis fixed | No separate sim form; same code path as sweeps. |
| Validation | The runner's own `parse_args` + `validate_args`, called in-process on the form's argv | One source of truth; the form holds no rules of its own beyond widget types. |
| Execution | Detached headless job (`setsid nohup python3 sobol/job.py <dir>`); the TUI only views status files | Survives SSH drops (long-jobs rule); reattach after reconnect; cancel by process group. |
| `-i` wizard | Kept, rebased on the schema | Cheap after phase 1; `inter_DEM_run_mass_sobol.py` uses it. |
| Old TUIs | Untouched until `tui.py` is verified; then aliases to `tui.py` on the matching tab, then deleted | Max: new program first, no replacement until it works. |

## Architecture

| Unit | Job | Depends on |
|------|-----|------------|
| `sobol/params_schema.py` (new) | Per-flag metadata keyed by argparse dest: section, brief help, axis grouping (stem → fixed flag + `_min`/`_max`), widget hints, gate predicates (e.g. "only when `body_source == settled`"). Reads section and help from `DEST_TO_SECTION` / `INTERACTIVE_BRIEF` in `interactive_run_mass_sobol.py` (imported, not moved). | `run_mass_sobol_phantom.build_parser()`, `interactive_run_mass_sobol.py` |
| `sobol/render_backend.py` (new, small) | Lazy loader for the render stages. `load()` imports `tui_sim_render` and returns the module, or a reason string if the import fails (it `sys.exit`s when the per-machine paths config is missing, e.g. on OzSTAR; `SystemExit` and `ImportError` are caught) or `check_blender_exe()` fails. Phase 5 swaps its target to `pipeline.py`. | `tui_sim_render.py` (imported, unchanged) |
| `sobol/job.py` (new) | Headless job runner. Reads `job.json`, runs the runner as a subprocess, then renders the selected runs through `render_backend`; writes `job_status.json` after every stage transition. | runner (subprocess), `render_backend.py` |
| `sobol/tui.py` (new) | Textual app: builds forms from the schema, validates via the runner, writes `job.json`, launches `job.py` detached, polls status files. No sim or render logic. | the three above |

### Coexistence

Until phase 5, `tui_run.py`, `tui_sim_render.py`, `render_dumps.py`, `interactive_run_mass_sobol.py` and their
tests are not edited. New code imports from them. The runner and `settled_body.py` changes are additive (a new flag,
a relaxed 0-dim rule, a lock), so the old tools behave exactly as before. Phase 5 extracts `pipeline.py` from
`tui_sim_render.py` and retires the old TUIs, only after the new app passes its manual checks.

### Schema

- One entry per `parser._actions` dest (117 today). Entries carry: `section`, `brief`, optional `axis` stem, optional
  `gate` (a predicate over current form values), optional `hidden_in_single` (Sampling fields).
- Axis = a fixed flag plus a `_min`/`_max` pair (25 pairs today, e.g. `--mu-s` / `--mu-s-min` / `--mu-s-max`,
  `--flyby-rp-km` / `-min` / `-max`). Flags with only min/max (e.g. mass) are axes with no fixed form; the fixed side of
  the toggle is then "not varied".
- Widget type comes from argparse: `choices` → Select, `store_true` → Checkbox, `nargs='+'` → space-separated Input,
  otherwise Input typed by `type`.
- Section order: Paths, Sampling, Body / encounter, DEM contact, Boulders, Spin, Time, Setup, Execution.
- Gates replace the hand-written handlers in both TUIs: boulder fields greyed unless settled + `boulders` ticked;
  hyperbola fields unless `encounter = hyperbola`. Greying is visual; the runner still decides validity.

## Tabs

### Simulate

- **Mode switch** at the top:
  - **Single run**: hides axis toggles and Sampling, sets `--num-samples 1`, `--jobs 1`, hints
    `OMP_NUM_THREADS=4`, pre-ticks Render after.
  - **Sweep**: shows axis toggles, hints `--jobs 2`, `OMP_NUM_THREADS=1`.
- **Rows:** each axis is one row with a `fixed | sweep` toggle (fixed: one Input; sweep: min + max). Plain flags get
  one widget.
- **Validate** (also run before launch) shows the runner's error text unchanged. **Show command** prints the
  equivalent CLI (with `OMP_NUM_THREADS=` prefix) for scripts or OzSTAR.
- **Render after** (collapsed): reuses the Render tab's settings; run selection = all, first N, or a list of run ids.
  Ticking it adds `--no-cleanup` (renders need dumps).
- **Clone**: prefill the form from a past job's `job.json`.

### Render

- **Source:** pick a batch under `<output_root>` (newest first), then a table of its runs with key columns from
  `sobol_mass_outputs.csv`. Runs with no dumps on disk are marked and not selectable. Or type any run-dir path, with
  `prefix`, `dem_model`, `np_apophis`, `apophis_only` and `encounter` fields as in `render_dumps.py` (for e.g. someone
  else's `fly` run).
- **Settings:** today's `tui_sim_render.py` Render and Render-paths fields unchanged (`resolution`, `samples`, `fps`,
  `max_frames`, `camera_mode`, `encode_video`, `video_fps`, `slowmo`, `resume`, the four path checkboxes,
  `envelope_method`, `placeholder_obj`, `n_points`, `compare`), validated by `validate_render_form()`.
- **Render** launches a render-only job.
- Greyed out with the reason shown when the startup check (`setup_paths.py --check` logic, `check_blender_exe`)
  fails, e.g. on OzSTAR.

### Jobs

- List of jobs under `<output_root>/_jobs/`, newest first: label, state, elapsed.
- Selected job: stage list with states and times; per-run status (rows of `sobol_mass_outputs.csv` plus `run_*` dirs
  present); latest `job.log` line; `pipeline_timing.json` summary per rendered run.
- Buttons: **Cancel** (`os.killpg` on the job's pgid, state → `cancelled`), **Resume** (`resume_batch.py` with the
  job's argv and `--output-root`/`--body-cache-dir`), **Open log**.
- Polls every 2 s. Reopening the app shows all jobs again.

## Job model

Directory: `<output_root>/_jobs/<job_id>/`, `job_id = YYYYmmdd_HHMMSS_<label>`.

- `job.json` (written once by the TUI): `sim_argv` (list, or null for render-only), `render` (render form values, or
  null), `run_selection` (`all` | `{"first": N}` | `{"ids": [...]}` | `{"paths": [...]}` for render-only),
  `omp_num_threads`, `host`, `created`.
- `job_status.json` (rewritten atomically by `job.py`, temp file + `os.replace`): `pid`, `pgid`, `host`, `state`
  (`running` | `done` | `partial` | `failed` | `cancelled`), `batch_dir`, `stages` (list of `{name, state, started,
  ended, message}`).
- `job.log`: merged stdout/stderr of the runner and every pipeline stage (per-path `preprocess.log` / `render.log`
  are still written where they are today).

Flow: TUI validates → writes `job.json` → `setsid nohup python3 sobol/job.py <dir>` with `OMP_NUM_THREADS` set →
`job.py` runs the runner with env `SOBOL_REPORT_BATCH_DIR=<dir>/batch_dir.txt` → reads the batch path → renders selected ok
runs via `render_backend` → final state.

## Changes outside the UI

1. **Runner, single run:** `validate_args` allows `count_dimensions(args) == 0` when `num_samples == 1` (today only
   with `--np-apophis-list`, `run_mass_sobol_phantom.py` ~line 1191). Saltelli with 0 dims stays an error.
2. **Runner, `SOBOL_REPORT_BATCH_DIR`:** when this env var names a file, the runner writes the batch directory path
   to it as soon as the path is known. Needed because diffing `output_root` breaks with concurrent jobs. An env var,
   not a flag: the `-i` wizard prompts once per parser flag, so a flag would change the old tool.
3. **Settled-body cache lock:** `settled_body.py` takes an exclusive `fcntl.flock` on `<cache-root>/<key>.lock` around
   the build; a second process waits, then finds the finished body and reuses it. Closes the known "CLI + TUI building
   the same new cache key at once is unguarded" gap, which concurrent jobs make likely.

## Error handling

| Case | Behaviour |
|------|-----------|
| Form invalid | Runner's message shown inline; no job created. |
| Runner exits non-zero, no ok rows | Job `failed`; no render. |
| Some runs failed | Render only selected ok runs; job `partial`, skipped runs and reasons listed. |
| A render path fails | Today's semantics: others still run; job `partial`. |
| `running`, same host, pid gone | Jobs tab shows **died**, offers Resume, shows `job.log` tail. |
| `running`, different host | Shown as "running on `<host>`" (pid not checkable); Cancel disabled. |
| No Blender / setup check fails | Render tab and Render after disabled with the reason; Simulate works. |
| Launch while another job runs on the same host | Warning (laptop throttles above `--jobs 2`); not blocked. |
| Same settled-body key from two jobs | Lock: second waits and reuses. |
| `job_status.json` write fails | Logged to `job.log`; job continues (status is a view, not control). |

## Testing

- **Schema completeness:** every runner dest has a schema entry; fails when a flag is added without one.
- **Argv equivalence:** for the `sobol/CLAUDE.md` example commands plus representative real configs (np300 settled
  ephemeris; hyperbola rp/v_inf sweep; boulders `f_boulder` sweep; contact `mu_s` sweep; mass + scale sweep), prefill
  the form, emit argv, assert the parsed `Namespace` equals the hand-written command's.
- **Old tools unchanged:** the full existing suite (`tests/test_tui_sim_render.py`, `test_tui_run*`,
  `test_tui_contact_keys.py`, `test_tui_boulders.py`, `test_render_dumps.py`) passes at the end of every phase.
- **`render_backend`:** returns the module when config exists; returns a reason (no exception) when the import
  `sys.exit`s or Blender is missing.
- **Phase 5 only:** stage tests in `tests/test_tui_sim_render.py` move to `tests/test_pipeline.py` with imports
  changed only, and must pass unchanged.
- **`job.py`:** against a stub runner script: `done`, `partial`, `failed`, `cancelled` (process-group kill), died
  detection, `SOBOL_REPORT_BATCH_DIR` read.
- **Runner changes:** 0-dim single run accepted with `--num-samples 1`, rejected otherwise; `SOBOL_REPORT_BATCH_DIR`
  written; Saltelli 0-dim still rejected.
- **Cache lock:** two processes on one key → one settle, second reuses (stub build).
- **Textual pilot (`App.run_test`):** mode switch hides/shows Sampling and toggles; gates; Render disabled without
  Blender; Clone round-trips a `job.json`.
- **Manual, before calling done:**
  1. Real np300 settled single run → `instance_grains`, compared with the old TUI on the same inputs (metrics row,
     frame count).
  2. Sweep dry run (`--dry-run`) from the Simulate tab.
  3. `textual serve` through VS Code port forwarding from the desktop.
  4. Terminal launch on OzSTAR (Textual in the user env; ask before installing anything on the cluster).

## Phases

Each phase ships and works on its own; the old TUIs stay untouched through phase 4.

1. **Schema.** `params_schema.py` + completeness test + argv-equivalence helpers (form values → argv).
2. **Runner additions.** Single-run 0-dim rule, `SOBOL_REPORT_BATCH_DIR`, settled-body cache lock.
3. **Job layer.** `render_backend.py`, `job.py`. CLI users can already launch detached, resumable jobs from a
   hand-written `job.json`.
4. **`tui.py`.** Simulate, Render, Jobs tabs. Then the manual checks under Testing.
5. **Retirement (only after phase 4 passes the manual checks and Max agrees).** Extract `pipeline.py` from
   `tui_sim_render.py`; point `render_dumps.py` and `render_backend.py` at it; `tui_run.py` and `tui_sim_render.py`
   become aliases opening `tui.py` on Simulate (Sweep) and Simulate (Single run), then are deleted; wizard maps move
   into the schema; `docs/tui_sim_render.md` → `docs/tui.md`; update `sobol/CLAUDE.md`; rewrite
   `tests/test_tui_contact_keys.py` / `tests/test_tui_boulders.py` against the schema form.

Timing against D1 (2 Nov 2026): phases 1–3 are low risk and useful before D1; 4–5 after D1 unless time allows. Max's
call.

## Out of scope

- Slurm submission or monitoring.
- Analysis/plot launching (`Analysis/*.py`) from the app.
- Inline frame previews or metric plots.
- Any change to sim, metric or render behaviour.
