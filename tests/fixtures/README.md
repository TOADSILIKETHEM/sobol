# DEM test fixtures

Tiny real runs made by `dem_run_dir.sh` (built from phantom `DEMsync-mia`):
np_apophis=300, mesh apophis_v233s7.obj 0.170, epoch 2029-04-10, tmax 1 hr,
dumps every 20 min, P=2 hr spin, nfulldump=1. Ephemeris .txt and the OBJ
were deleted after the run.

- `particle_np300/` (committed, ~2 MB): particle DEM (use_dem=T, idem grains).
- `sink_np300/` (gitignored, ~26 MB of per-grain sink .ev files): sink DEM twin.
  Tests that need it skip when it is absent. Regenerate (about 1 s each):

      F=/home/mboyle/Honours/sobol/tests/fixtures
      $F/dem_run_dir.sh particle $F/particle_np300 300 "1 hr" "20 min" -1 run
      $F/dem_run_dir.sh sink     $F/sink_np300     300 "1 hr" "20 min" -1 run
      # then delete the copied *.txt ephemeris files and apophis_v233s7.obj in each

PHANTOM deletes `sobol_00000.tmp` when the run starts; use `sobol_00000` (t=0).
Grain mass is `params['massoftype_2']` in sarracen (`params['mass']` is the gas type, 0).

- `particle_flyby_np300/` (committed): settled np=300 body (settle 5 t_dyn, relax 0.25 t_dyn) on a
  phantomflyby hyperbola, rp 38000 km, v_inf 5.9 km/s, start 1e5 km, 8 hr, dumps every 2 hr.
  Earth is the only sink (sink 1). Regenerate with the runner command in
  `docs/superpowers/plans/2026-09-26-settled-body-pipeline.md` Task 5 Step 7.
