import csv
import math
import stat
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import particle_dem as pdem
import run_mass_sobol_phantom as runner
import resume_batch

DAT = """#[01      time_s]  [02       npart]  [03         a_m]  [04         b_m]  [05         c_m]  [06      b_on_a]  [07      c_on_a]  [08      rmax_m]  [09    rgrain_m]  [10         phi]  [11   f_unbound]
  1.0000000000E+03         300  2.0000000000E+02  1.5000000000E+02  1.0000000000E+02  7.5000000000E-01  5.0000000000E-01  2.2000000000E+02  1.0000000000E+01  6.4000000000E-01  0.0000000000E+00
  2.0000000000E+03         300  2.0000000000E+02  1.6000000000E+02  1.2000000000E+02  8.0000000000E-01  6.0000000000E-01  2.2000000000E+02  1.0000000000E+01  6.2000000000E-01  1.0000000000E-02
"""

FAKE_ANALYSIS = """#!/bin/sh
echo "analysis $*" >> calls.log
cat > sobol_shape.dat <<'EOF'
""" + DAT + """EOF
"""


def test_parse_demshape_dat_last_row(tmp_path):
    p = tmp_path / "sobol_shape.dat"
    p.write_text(DAT)
    assert pdem.parse_demshape_dat(p) == pdem.ShapeMetrics(0.8, 0.6, 0.62, 0.01)


def test_run_demshape_analysis_uses_last_dump(tmp_path):
    for n in (0, 1, 2):
        (tmp_path / f"sobol_{n:05d}").write_text("")
    exe = tmp_path / "phantomanalysis"
    exe.write_text(FAKE_ANALYSIS)
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    m = pdem.run_demshape_analysis(tmp_path, "sobol", exe)
    assert m.b_on_a == 0.8
    assert (tmp_path / "calls.log").read_text().strip() == "analysis sobol_00002"


def test_summary_csv_adds_shape_columns_before_error(tmp_path):
    rec = runner.RunRecord(run_id=1, mass_input_kg=float("nan"), run_dir="r", status="ok",
                           closest_approach_km=1.0, closest_approach_au=1.0, error="",
                           shape_b_on_a=0.8, shape_c_on_a=0.6, packing_phi=0.62, f_unbound_energy=0.01)
    out = tmp_path / "o.csv"
    runner.write_summary_csv(out, [rec], [])
    header = next(csv.reader(out.open()))
    i = header.index("post_flyby_spin_period_hr")
    assert header[i + 1:] == ["shape_b_on_a", "shape_c_on_a", "packing_phi", "f_unbound_energy", "error"]
    row = next(csv.DictReader(out.open()))
    assert float(row["shape_b_on_a"]) == 0.8


def test_resume_batch_roundtrip_preserves_shape_metrics(tmp_path):
    """Verify shape metrics survive a write-then-load cycle via resume_batch._load_records."""
    # Write a summary CSV with shape metrics
    rec = runner.RunRecord(run_id=1, mass_input_kg=float("nan"), run_dir="run_0001", status="ok",
                           closest_approach_km=0.1, closest_approach_au=0.001, error="",
                           shape_b_on_a=0.8, shape_c_on_a=0.6, packing_phi=0.62, f_unbound_energy=0.01)
    csv_path = tmp_path / "summary.csv"
    runner.write_summary_csv(csv_path, [rec], [])

    # Load it back via resume_batch._load_records
    loaded = resume_batch._load_records(csv_path, [])
    assert len(loaded) == 1
    loaded_rec = loaded[0]

    # Verify shape metrics survived the round-trip
    assert loaded_rec.shape_b_on_a == 0.8
    assert loaded_rec.shape_c_on_a == 0.6
    assert loaded_rec.packing_phi == 0.62
    assert loaded_rec.f_unbound_energy == 0.01


def test_resume_batch_loads_old_csv_without_shape_columns(tmp_path):
    """Verify old CSVs without shape columns load with NaN shape metrics."""
    # Write a CSV without the shape columns (simulating an old batch)
    csv_path = tmp_path / "old_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run_id", "mass_input_kg", "run_dir", "status",
            "closest_approach_km", "closest_approach_au", "error"
        ])
        writer.writerow([
            "1", "", "run_0001", "ok",
            "0.1", "0.001", ""
        ])

    # Load it back via resume_batch._load_records
    loaded = resume_batch._load_records(csv_path, [])
    assert len(loaded) == 1
    loaded_rec = loaded[0]

    # Verify shape metrics are NaN (backward compatibility)
    assert math.isnan(loaded_rec.shape_b_on_a)
    assert math.isnan(loaded_rec.shape_c_on_a)
    assert math.isnan(loaded_rec.packing_phi)
    assert math.isnan(loaded_rec.f_unbound_energy)
