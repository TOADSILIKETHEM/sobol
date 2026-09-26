#!/usr/bin/env bash
# Stage (and optionally run) a tiny Apophis DEM run for either DEM model.
# usage: dem_run_dir.sh <particle|sink> <outdir> <np> <tmax_in> <dtmax_in> <torque_deg> [run]
set -euo pipefail
model=$1; out=$2; np=$3; tmax=$4; dtmax=$5; torque=$6; do_run=${7:-}
root=/home/mboyle/Honours
case "$model" in
  particle) use_dem=T; use_sinks=F ;;
  sink)     use_dem=F; use_sinks=T ;;
  *) echo "model must be particle|sink" >&2; exit 2 ;;
esac
mkdir -p "$out"; cd "$out"
rm -f sobol.setup sobol.in sobol_0* ./*.ev ./*.log
for f in apophis earth jupiter mars mercury moon neptune saturn uranus venus Distant; do
  cp "$root/sobol/$f.txt" .
done
cp "$root/Shapes/apophis.shape" "$root/Shapes/apophis_v233s7.obj" .
cat > sobol.setup <<EOF
             tmax_in =  $tmax
            dtmax_in =  $dtmax
           asteroids =           F
          np_apophis =  $np
               epoch =  2029-04-10
             use_dem =  $use_dem
    use_dem_as_sinks =  $use_sinks
        apophis_only =  ${APOPHIS_ONLY:-F}
      add_mars_moons =           F
           scale_vel =       1.000
           scale_pos =       1.000
     scale_earth_sep =       1.000
     scale_r_apophis =       1.000
           scale_rho =       1.000
        mass_apophis =       0.000
  apophis_shape_file =  apophis.shape
         pack_settle =           F
         pack_expand =       1.800
            pack_phi =       0.640
 apophis_spin_period =    7200.000
 apophis_spin_axis_x =       0.276
 apophis_spin_axis_y =      -0.116
 apophis_spin_axis_z =       0.954
apophis_spin_torque_align_deg =  $torque
EOF
maxp=$(( np * 4 > 4000 ? np * 4 : 4000 ))
"$root/sobol/phantomsetup" sobol --maxp=$maxp > setup.log 2>&1
sed -i 's/^\( *nfulldump *= *\)[0-9]*/\1         1/' sobol.in
if [ "$do_run" = run ]; then
  OMP_NUM_THREADS=1 "$root/sobol/phantom" sobol.in --maxp=$maxp > phantom.log 2>&1
fi
echo "staged $model in $out"
