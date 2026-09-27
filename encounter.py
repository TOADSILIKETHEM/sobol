"""Hyperbolic flyby encounter from a settled body (Mia's Option 3, moddump_earthflyby.f90)."""
from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict

GM_EARTH_KM3_S2 = 3.986004e5  # G * M_earth


def time_to_pericentre_hr(rp_km: float, vinf_kms: float, start_sep_km: float,
                          perturber_earth_masses: float) -> float:
    """Time from separation start_sep_km (incoming) to pericentre on the hyperbola
    moddump_earthflyby builds (e = 1 + rp v_inf^2 / mu, a = -mu / v_inf^2)."""
    mu = GM_EARTH_KM3_S2 * perturber_earth_masses
    a_abs = mu / vinf_kms ** 2
    ecc = 1.0 + rp_km * vinf_kms ** 2 / mu
    cosh_f = max((1.0 + start_sep_km / a_abs) / ecc, 1.0)
    f = math.acosh(cosh_f)
    mean_anom = ecc * math.sinh(f) - f
    return mean_anom / math.sqrt(mu / a_abs ** 3) / 3600.0


def _max_time_to_pericentre_hr(rp_lo_km: float, rp_hi_km: float, vinf_lo_kms: float, vinf_hi_kms: float,
                               start_sep_km: float, perturber_earth_masses: float,
                               n_rp: int = 65, n_vinf: int = 9) -> float:
    """Worst-case (longest) time to pericentre over a swept rp x v_inf range.

    time_to_pericentre_hr(rp) is NOT monotonic in rp: it has an interior maximum and falls back
    to 0 as rp approaches start_sep_km, so the (rp_min, vinf_min) corner can understate the true
    worst case. Grid both dimensions (endpoints included; a fixed value collapses to a single
    point) and take the max, rather than trusting a corner.
    """
    def _grid(lo: float, hi: float, n: int):
        if lo == hi:
            return [lo]
        return [lo + i * (hi - lo) / (n - 1) for i in range(n)]

    rp_vals = _grid(rp_lo_km, rp_hi_km, n_rp)
    vinf_vals = _grid(vinf_lo_kms, vinf_hi_kms, n_vinf)
    return max(
        time_to_pericentre_hr(rp, vinf, start_sep_km, perturber_earth_masses)
        for rp in rp_vals for vinf in vinf_vals
    )


def _runner():
    try:
        import run_mass_sobol_phantom as mod
    except ImportError:
        from sobol import run_mass_sobol_phantom as mod
    return mod


def flyby_stdin(rp_km: float, vinf_kms: float, start_sep_km: float, perturber_earth_masses: float) -> str:
    """Answers to moddump_earthflyby.f90's prompts, in its order."""
    return "".join(f"{v:.10g}\n" for v in (rp_km, vinf_kms, start_sep_km, perturber_earth_masses))


def patch_encounter_in(in_path: Path, *, tmax_code: float, dtmax_code: float) -> None:
    """The flyby .in inherits the body's relax .in: set the encounter window, full dumps, and turn the
    settle damping (idamp=2, setup_solarsystem.f90 settle mode) OFF, or the flyby is damped."""
    r = _runner()
    text = Path(in_path).read_text(encoding="utf-8")
    for key, val in (("tmax", r.format_real_token(tmax_code)),
                     ("dtmax", r.format_real_token(dtmax_code)),
                     ("nfulldump", f"{1:>10}")):
        text = r.replace_setup_assignment(text, key, val)
    # phantomflyby (moddump) copies the settle .in forward verbatim, so the freshly-written
    # sobol.in still carries the settle's idamp=2 at this point; this line is what zeroes it.
    # Do not "simplify" this away because committed run dirs show no idamp line at all — that
    # is PHANTOM's own .in rewrite on exit (damping.f90: `if (idamp <= 0) return`) dropping the
    # whole damping block *after* this guard already zeroed it, not evidence the block was
    # already absent.
    if re.search(r"^\s*idamp\s*=", text, re.MULTILINE):
        text = r.replace_setup_assignment(text, "idamp", f"{0:>10}")
    Path(in_path).write_text(text, encoding="utf-8")


def write_record_setup(template_setup: Path, dst: Path, n_kept: int) -> None:
    """A .setup copy phantomsetup never reads: tells particle_dem.dem_model_from_setup and
    DEMDumpConvert this is a particle-DEM run with the Earth present and n_kept grains."""
    r = _runner()
    text = Path(template_setup).read_text(encoding="utf-8")
    for key, val in (("use_dem", True), ("use_dem_as_sinks", False),
                     ("apophis_only", False), ("pack_settle", False)):
        text = r.replace_setup_assignment(text, key, r.format_logical_token(val))
    text = r._NP_APOPHIS_RE.sub(lambda m: m.group(1) + str(int(n_kept)), text, count=1)
    Path(dst).write_text(text, encoding="utf-8")


def prepare_hyperbola_run(run_dir: Path, prefix: str, sample, body, phantomflyby_bin: Path,
                          template_setup: Path) -> Dict[str, str]:
    """Put the cached body on a hyperbola in run_dir: <prefix>_00000 + <prefix>.in ready for phantom."""
    run_dir = Path(run_dir)
    stale_in = run_dir / f"{prefix}.in"
    if stale_in.exists():
        stale_in.unlink()  # moddump reads <out>.in as revised defaults
    shutil.copy2(body.dir / body.dump, run_dir / "body_00000")
    shutil.copy2(body.dir / body.infile, run_dir / "body.in")  # moddump reads <in>.in as defaults
    log = run_dir / "flyby.log"
    with open(log, "w", encoding="utf-8") as fh:
        # --maxp: without it phantom_moddump allocates maxp_alloc = 5.2M particles (~6 GB; WSL has 7)
        proc = subprocess.run(
            [str(phantomflyby_bin), "body_00000", prefix, "0", f"--maxp={max(2000, 4 * body.n_kept)}"],
            cwd=str(run_dir), text=True,
            input=flyby_stdin(sample.flyby_rp_km, sample.flyby_vinf_kms, sample.flyby_start_sep_km,
                              sample.flyby_perturber_earth_masses),
            stdout=fh, stderr=subprocess.STDOUT)
    if proc.returncode != 0 or not (run_dir / f"{prefix}_00000").is_file():
        raise RuntimeError(f"phantomflyby failed ({proc.returncode}); see {log}")
    patch_encounter_in(run_dir / f"{prefix}.in",
                       tmax_code=sample.tmax_hours * 3600.0 / body.utime_s,
                       dtmax_code=sample.dtmax_hours * 3600.0 / body.utime_s)
    write_record_setup(template_setup, run_dir / f"{prefix}.setup", body.n_kept)
    (run_dir / "encounter.json").write_text(json.dumps({
        "mode": "hyperbola", "rp_km": sample.flyby_rp_km, "vinf_kms": sample.flyby_vinf_kms,
        "start_sep_km": sample.flyby_start_sep_km,
        "perturber_earth_masses": sample.flyby_perturber_earth_masses,
        "body_dir": str(body.dir), "body_key": body.key, "n_kept": body.n_kept,
    }, indent=2), encoding="utf-8")
    return {"body_source": "settled", "encounter": "hyperbola",
            "flyby_rp_km": f"{sample.flyby_rp_km:.12g}", "flyby_vinf_kms": f"{sample.flyby_vinf_kms:.12g}",
            "np_kept": str(body.n_kept)}


PACKING_FILE_NAME = "settled_body"
_PACKING_FILE_LINE = (f"{'packing_file':>20} = {PACKING_FILE_NAME:<12}"
                      "! dump holding a pre-built body (settled+cropped); blank = build one\n")


def _set_packing_file(text: str) -> str:
    """Point packing_file at the staged body; insert the key after pack_settle when absent
    (kept out of the template so pre-merge binaries never see it)."""
    r = _runner()
    if re.search(r"^\s*packing_file\s*=", text, re.M):
        return r.replace_setup_assignment(text, "packing_file", PACKING_FILE_NAME)
    m = re.search(r"^\s*pack_settle\s*=.*\n", text, re.M)
    at = m.end() if m else len(text)
    return text[:at] + _PACKING_FILE_LINE + text[at:]


def apply_settled_body_to_setup(run_dir: Path, prefix: str, body) -> Dict[str, str]:
    """Stage the cached body and its shape, and point the run's .setup at them (Mia Option 2 step 4).

    phantomsetup takes the body volume, hence its mass, from apophis_shape_file; a blank one only
    warns and falls back to a sphere, so the body's own shape is always staged.
    """
    r = _runner()
    run_dir = Path(run_dir)
    if not body.shape_file:
        raise RuntimeError(f"settled body {body.dir} has no shape_file in body.json; delete it and rebuild")
    shutil.copy2(Path(body.dir) / body.dump, run_dir / PACKING_FILE_NAME)
    shape_name = r.stage_shape_assets(Path(body.shape_file), run_dir)
    setup = run_dir / f"{prefix}.setup"
    text = _set_packing_file(setup.read_text(encoding="utf-8"))
    for key, val in (("pack_settle", r.format_logical_token(False)), ("apophis_shape_file", shape_name)):
        text = r.replace_setup_assignment(text, key, val)
        r.validate_assignment(text, key, val)
    text = r._NP_APOPHIS_RE.sub(lambda m: m.group(1) + str(int(body.n_kept)), text, count=1)
    setup.write_text(text, encoding="utf-8")
    return {"body_source": "settled", "np_kept": str(body.n_kept)}


def check_packing_setup_log(setup_log: Path, n_kept: int) -> None:
    """Refuse a settled-ephemeris run whose phantomsetup did not load the body as asked."""
    text = Path(setup_log).read_bytes().replace(b"\x00", b"").decode("utf-8", errors="replace")
    if "could not read shape for packing_file volume" in text:
        raise RuntimeError(f"phantomsetup sized the packing_file body as a sphere (shape unreadable); "
                           f"body mass is wrong. See {setup_log}")
    m = re.search(r"placed\s+(\d+)\s+pre-built grains", text)
    if m is None:
        raise RuntimeError(f"phantomsetup never loaded packing_file (binary without phantom 853b86818? "
                           f"cd sobol && make setup && make). See {setup_log}")
    if int(m.group(1)) != int(n_kept):
        raise RuntimeError(f"phantomsetup placed {m.group(1)} grains but the body has {n_kept}. See {setup_log}")
