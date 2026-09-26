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


def closest_approach_from_series(t_earth, xyz_earth, t_apo, xyz_apo) -> Tuple[float, float]:
    """Min Earth–Apophis distance with Earth interpolated to Apophis sample times.

    Apophis CoM exists only at dump times, so the raw minimum is refined by a
    parabola through the three samples around it (interior minima only).
    Returns ``(d_min, t_min)`` in code units.
    """
    te = np.asarray(t_earth, float)
    order = np.argsort(te)
    te = te[order]
    xe = np.asarray(xyz_earth, float)[order]
    ta = np.asarray(t_apo, float)
    xa = np.asarray(xyz_apo, float)
    inside = (ta >= te[0]) & (ta <= te[-1])
    ta, xa = ta[inside], xa[inside]
    if ta.size == 0:
        raise RuntimeError("No Apophis samples inside the Earth .ev time range")
    earth_at = np.column_stack([np.interp(ta, te, xe[:, k]) for k in range(3)])
    d = np.linalg.norm(xa - earth_at, axis=1)
    i = int(np.argmin(d))
    d_min, t_min = float(d[i]), float(ta[i])
    if 0 < i < d.size - 1:
        t0, t1, t2 = ta[i - 1:i + 2]
        d0, d1, d2 = d[i - 1:i + 2]
        den = (t0 - t1) * (t0 - t2) * (t1 - t2)
        a = (t2 * (d1 - d0) + t1 * (d0 - d2) + t0 * (d2 - d1)) / den
        b = (t2 * t2 * (d0 - d1) + t1 * t1 * (d2 - d0) + t0 * t0 * (d1 - d2)) / den
        if a > 0.0:
            tv = -b / (2.0 * a)
            if t0 <= tv <= t2:
                c = d1 - a * t1 * t1 - b * t1
                dv = a * tv * tv + b * tv + c
                if dv < d_min:
                    d_min, t_min = float(dv), float(tv)
    return d_min, t_min


def grain_radius_cm_from_dump(dump: Path, model: str) -> float:
    """Median DEM grain radius in cm: particle R = h (Mia fbc0e4e63); sink R = Reff."""
    sdf, sinks = _read_dump(Path(dump))
    udist = float(sdf.params.get("udist", 1.0))
    if model == "particle":
        h = sdf["h"].to_numpy()
        return float(np.median(h[h > 0])) * udist
    if model == "sink":
        if sinks is None or "Reff" not in sinks.columns:
            raise RuntimeError(f"{dump}: no sink Reff column")
        reff = sinks["Reff"].to_numpy()
        reff = reff[reff > 0]
        if reff.size == 0:
            raise RuntimeError(f"{dump}: no DEM grain sinks (Reff > 0)")
        return float(np.median(reff)) * udist
    raise ValueError(f"model must be 'particle' or 'sink', got {model!r}")
