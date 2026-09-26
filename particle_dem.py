"""Particle-DEM (Mia's idem grains) readers for sweep metrics.

Grains are SPH-type particles in full dumps, not sinks, so they are read
from ``{prefix}_NNNNN`` dumps via sarracen. Output matches the sink ``.ev``
reader contract in ``run_mass_sobol_phantom._apophis_time_groups``:
``groups[key]`` is ``(N, 7)`` ``[x, y, z, m, vx, vy, vz]`` in code units.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

IDEM = 2  # part.F90: integer, parameter :: idem = 2
_DUMP_RE = re.compile(r"_(\d{5})$")


def _setup_value(text: str, key: str) -> Optional[str]:
    m = re.search(rf"^\s*{re.escape(key)}\s*=\s*([^!\n]*)", text, re.MULTILINE)
    return m.group(1).strip() if m else None


def _is_true(val: Optional[str]) -> bool:
    return bool(val) and val.strip().upper().startswith("T")


def dem_model_from_setup(setup_path: Path) -> str:
    """'particle', 'sink' or 'none' for a run's .setup.

    Setups without a ``use_dem_as_sinks`` key predate the DEMsync-mia merge;
    there ``use_dem = T`` meant sink DEM, so they map to 'sink'.
    """
    text = Path(setup_path).read_text(encoding="utf-8", errors="replace")
    np_val = _setup_value(text, "np_apophis")
    if np_val is None or int(float(np_val)) <= 1:
        return "none"
    use_dem = _is_true(_setup_value(text, "use_dem"))
    sinks_raw = _setup_value(text, "use_dem_as_sinks")
    if sinks_raw is None:
        return "sink" if use_dem else "none"
    if _is_true(sinks_raw):
        return "sink"
    return "particle" if use_dem else "none"


def list_full_dumps(run_dir: Path, prefix: str) -> List[Path]:
    """Dump files ``{prefix}_NNNNN`` (no extension), sorted by number."""
    out = [p for p in Path(run_dir).glob(f"{prefix}_[0-9][0-9][0-9][0-9][0-9]")
           if _DUMP_RE.search(p.name)]
    return sorted(out, key=lambda p: int(_DUMP_RE.search(p.name).group(1)))


def _read_dump(path: Path):
    import sarracen
    res = sarracen.read_phantom(str(path))
    if isinstance(res, (list, tuple)):
        return res[0], (res[-1] if len(res) > 1 else None)
    return res, None


def _grain_mass(sdf) -> np.ndarray:
    """Per-particle mass: ``m`` column if present, else ``massoftype`` of the DEM type."""
    if "m" in sdf.columns:
        return sdf["m"].to_numpy()
    # sarracen: params['mass'] / 'massoftype' is the gas type; type n>1 is 'massoftype_n'
    return np.full(len(sdf), float(sdf.params[f"massoftype_{IDEM}"]))


def read_particle_frame(dump: Path) -> Tuple[float, Optional[np.ndarray]]:
    """``(time, arr)`` for DEM grains in one dump; ``arr`` None if velocities are missing."""
    sdf, _ = _read_dump(Path(dump))
    t = float(sdf.params.get("time", float("nan")))
    if len(sdf) == 0:
        return t, np.empty((0, 7))
    if not {"vx", "vy", "vz"}.issubset(sdf.columns):
        print(f"[WARN] {Path(dump).name}: no velocities (small dump?); skipped for DEM metrics",
              file=sys.stderr, flush=True)
        return t, None
    keep = sdf["h"].to_numpy() > 0.0  # dead/accreted have h <= 0
    if "itype" in sdf.columns:
        keep &= sdf["itype"].to_numpy() == IDEM
    cols = [sdf["x"].to_numpy(), sdf["y"].to_numpy(), sdf["z"].to_numpy(), _grain_mass(sdf),
            sdf["vx"].to_numpy(), sdf["vy"].to_numpy(), sdf["vz"].to_numpy()]
    return t, np.column_stack(cols).astype(np.float64)[keep]


def apophis_time_groups_from_dumps(
    run_dir: Path, prefix: str
) -> Tuple[Dict[str, np.ndarray], Dict[str, float], int]:
    """Grain groups keyed by dump time; same contract as the sink ``.ev`` reader."""
    groups: Dict[str, np.ndarray] = {}
    time_of_key: Dict[str, float] = {}
    n_max = 0
    for dump in list_full_dumps(run_dir, prefix):
        t, arr = read_particle_frame(dump)
        if arr is None or len(arr) == 0:
            continue
        key = f"{t:.9g}"
        groups[key] = arr
        time_of_key[key] = t
        n_max = max(n_max, len(arr))
    return groups, time_of_key, n_max
