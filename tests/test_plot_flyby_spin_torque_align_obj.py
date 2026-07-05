"""Tests for OBJ torque-align tidal spin plot loader."""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "Analysis"))
import plot_flyby_spin_torque_align_obj as plot_mod


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "status",
        "apophis_spin_torque_align_deg",
        "intrinsic_spin_period_hr",
        "approach_spin_period_hr",
        "post_flyby_spin_period_hr",
        "dispersion_ratio",
        "unbound_fraction",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def test_load_batch_sorts_by_angle(tmp_path: Path) -> None:
    csv_path = tmp_path / "sobol_mass_outputs.csv"
    _write_csv(
        csv_path,
        [
            {
                "status": "ok",
                "apophis_spin_torque_align_deg": "90",
                "intrinsic_spin_period_hr": "2.0",
                "approach_spin_period_hr": "2.0",
                "post_flyby_spin_period_hr": "2.01",
                "dispersion_ratio": "1.02",
                "unbound_fraction": "0",
            },
            {
                "status": "ok",
                "apophis_spin_torque_align_deg": "10",
                "intrinsic_spin_period_hr": "2.0",
                "approach_spin_period_hr": "2.0",
                "post_flyby_spin_period_hr": "1.99",
                "dispersion_ratio": "1.01",
                "unbound_fraction": "0",
            },
        ],
    )
    d = plot_mod.load_batch(csv_path)
    assert list(d["align"]) == [10.0, 90.0]
    assert list(d["post"]) == [1.99, 2.01]


def test_post_approach_delta_percent() -> None:
    app, post = 2.0, 1.98
    delta = plot_mod.delta_percent(post, app)
    assert np.isclose(delta, -1.0)
