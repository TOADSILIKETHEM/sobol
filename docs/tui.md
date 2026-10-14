# Unified TUI (`sobol/tui.py`)

## What it is

`sobol/tui.py` is one Textual app with three tabs: **Simulate** (a single run or a sweep), **Render** (render runs that already exist) and **Jobs** (everything launched, with progress). It is new and sits alongside `tui_run.py` and `tui_sim_render.py`, which are unchanged until they are retired.

Every launch is a detached job, so closing the TUI or losing SSH never stops a run.

## Launch

```bash
python3 sobol/tui.py [runner flags]                 # flags prefill the Simulate form
textual serve "python3 sobol/tui.py"                # same app in a browser (needs: pip install textual-dev)
```

For the browser version from the desktop, forward the port `textual serve` prints through VS Code (Ports tab) and open it there.

Keys (as `tui_run.py`): `r` launches (Simulate) or renders (Render), `d` launches a dry run, `q` quits; `ctrl+l` also launches. In a text box the letters are typed instead, and on the Jobs tab `r` and `d` do nothing.

Unless the flag is given on the command line, `DEFAULT_OVERRIDES` sets `base_dir` and `ephemeris_cache_dir` to `sobol/`, `output_root` to `Honours/sobol_mass_runs` and `body_source` to `settled` (so the form opens with `use_dem_fixed true` and `np_apophis 2000` filled), wherever the TUI is started from. The runner's own CLI default stays `lattice`. Relative paths typed in the form are resolved against the directory the TUI was started from.

## Simulate

- **Single run vs Sweep** (the bar above the form; it stays in view while the form scrolls). Single run sets `--num-samples 1` and `--jobs 1`, hides the Sampling section and the sweep toggles, and sets `OMP_NUM_THREADS` to 4. Sweep sets `OMP_NUM_THREADS` to 1 (laptop rule: `--jobs 2`, OMP 1). The OMP box can be edited. The launch button says what it starts (`Launch sweep` / `Launch single run`). A single run refuses an `np_apophis_list` or `spin_period_list` with more than one value (each value is a run).
- **Rows.** Every runner flag appears once, in sections taken from the `-i` wizard. A quantity that can be swept (an *axis*) has a `sweep` box: ticked, it takes min and max; unticked, a fixed value. Axes without a fixed flag in the runner (e.g. `spin_torque_align`, `scale_vel`) are fixed by writing min = max. `mass` has no fixed form: unticked means the template value.
- **Gates.** Boulder rows are greyed out unless `body_source = settled` and `boulders` is ticked; flyby rows unless `encounter = hyperbola`; settle rows unless `body_source = settled`. Greying is visual only.
- **Defaults on a mode choice** (`params_schema.fill_defaults`). Picking a mode fills the blank fields it needs, and the status line lists them (e.g. `defaults filled: np_apophis 2000, tmax_hours 40 (auto)`). A typed value is never overwritten, and a swept axis is never filled. The runner and CLI still refuse to guess these values.

  | Choice | Fills |
  |---|---|
  | `body_source = settled` | `use_dem_fixed true`, `np_apophis 2000` (not with `np_apophis_list`) |
  | `encounter = hyperbola` | `body_source settled`, `flyby rp 38000` km (real 2029, geocentric), `v_inf 5.9`, `dtmax 0.5`, `tmax` auto |
  | `boulders` ticked | `body_source settled`, `bond_sigma_pa 1e5` (breakup campaign value; no σ trial yet), `f_boulder 0.3` |

  Auto `tmax` is 2 × the longest time to pericentre (the runner's own geometry check) × 1.1, rounded up to a whole hour. It is recomputed when rp, v_inf, `flyby_start_sep_km` or the perturber mass change, until you type your own `tmax`. A clone also gets its blank fields filled. The `f_boulder` box says "fraction 0-1".
- **Clone** gives back the form as it was typed (`form_argv` in `job.json`), not the launched argv: the `--no-cleanup` that Render after forces and the `--dry-run` of the `d` key are not carried into the next launch. Jobs from before `form_argv` drop both from `sim_argv` instead.
- **Validate** runs the runner's own `parse_args` and `validate_args` on the form and shows its error or warnings. **Show command** prints the equivalent shell command.
- **Render after** (collapsed panel) renders the chosen runs (all ok runs, the first N, or run ids) when the sim finishes, with the settings on the Render tab. It forces `--no-cleanup` so the dumps survive until the render; the `no_cleanup` box then decides whether the dumps are deleted after the render (unticked: the launch status line warns that they will be). Unticked, the dumps of ok runs that were not chosen for rendering are deleted once the renders finish (a `cleanup` stage), as the runner would have done; failed runs keep theirs, as with the runner. It is ticked by default in Single run mode and unticked in Sweep mode (switching the mode resets it).

## Render

- **Batch picker.** Choose a batch, then press Enter on each run to render. Runs that failed, or whose dumps were cleaned up, are refused. Ticked runs survive switching tabs (the batch list is reloaded each time the tab opens).
- **Other run dir.** Any run dir, e.g. someone else's `phantomflyby` run, rendered through `render_dumps.py` (`instance_grains` and `composite` only, no compare). `camera_mode`, `envelope_method` and `video_fps` are passed on (`--camera-mode`, `--envelope-method`, `--video-fps`).
- **Settings** match `tui_sim_render.py` (resolution, samples, fps, paths, slowmo, resume, ...).
- **Delete dumps after a successful render** is off by default: the dumps may be the only copy of a run.
- **Grain-count warning.** Launching a render with `per_sphere` ticked on a run of more than 2000 grains adds `tui_sim_render.py`'s warning to the status line. It does not stop the launch.
- **Timing.** Each rendered run gets `<run_dir>/pipeline_timing.json` in `tui_sim_render.py`'s layout (stage and per-path wall-clock, frame counts, `np_apophis`, render settings, git SHAs). The sim time is null for a sweep run; a re-render keeps the sim time from an earlier file.
- Where rendering is impossible (OzSTAR: no Blender, no `Code/local_paths.json`) the tab shows the reason instead, and Render after is disabled.

## Jobs

Each job lives in `<output_root>/_jobs/<job_id>/`:

| File | Content |
|------|---------|
| `job.json` | what to run (written once by the TUI) |
| `job_status.json` | state and stages (rewritten by `job.py`) |
| `job.log` | everything the job printed |
| `batch_dir.txt` | the batch dir the runner created |

States: `running`, `done`, `partial` (some runs or renders failed), `failed`, `cancelled`. The list also shows `died` (the process is gone without a final state), `running on <host>` (started on another machine; cannot be cancelled here) and `unknown` (status file unreadable).

The detail view lists the stages, batch progress (`N/M runs finished`) and the last log line. While a render runs, its stage shows a live count that updates every 2 s, e.g. `rendering instance_grains (path 1/2): 37/120 frames | Saved: 'frame_0037.png'`: the phase (converting, preprocessing, rendering, comparing), the files written so far against the number expected, and the last line of Blender's `render.log`. Frames left in the folder by an earlier render are not counted unless `resume` is ticked (then they really are done). An Other run dir job counts the same way: `render_dumps.py` writes its phase to `<job dir>/progress.json` (path from `SOBOL_PROGRESS_FILE`). Each `render run_NNNN` stage ends with the frame count and seconds per path.

- **Cancel** sends SIGTERM to the job's process group (runner, PHANTOM, render). A Windows `blender.exe` started by the job may keep running; the stage message says so. The job records its process start time (`start_time` in `job_status.json`), so a job that died and whose pid was later reused shows `died`, and Cancel never signals the new process.
- **Resume** launches a new job that runs `resume_batch.py` on the job's batch with the same flags. Runs the original job (or the jobs it resumed) already rendered are listed in the new job's `skip_runs` and not rendered again.
- **Show log** shows the last 400 lines of `job.log`.

Cancel is enabled only for a job running on this machine (a job that ends just as Cancel is pressed is reported as already ended); Resume only for a finished sim job that has a batch (`partial`, `failed`, `cancelled` or `died`).

## Runner additions

- `--num-samples 1` with no swept axis is allowed (one all-fixed run). Sweeps and Saltelli still need an axis.
- `SOBOL_REPORT_BATCH_DIR=<file>`: the runner writes the batch dir to that file. An environment variable, not a flag, so the `-i` wizard is unchanged.
- The settled-body cache has a per-key lock (`<cache-root>/<key>.lock`): two launches that need the same body build it once. On a filesystem without `flock` the build runs unlocked, with a warning.

## Extending

A new runner flag fails `tests/test_params_schema.py::test_every_runner_dest_is_classified` until it is classified in `params_schema.py`: an axis in `AXES`, a plain field (it needs a section: the wizard's `DEST_TO_SECTION` or `_EXTRA_SECTION`), or `HIDDEN`. Gates live in `params_schema.GATES`.

## Known limits

- A render-only job renders one run at a time.
- On OzSTAR only Simulate and Jobs work.
