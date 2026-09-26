"""Settled-body cache: Mia's settle -> crop -> relax packing, built once per spec.

Stages (docs/MIA_PACKING_WORKFLOW.md, Option 2 steps 1-3):
  1. phantomsetup settle      (pack_settle=T, apophis_only=T; setup sets idamp=2)
  2. phantom settle.in        -> settle_NNNNN
  3. phantommoddump <last> cropped 0   (moddump_cropshape.f90) -> cropped_00000, cropped.in
  4. phantom cropped.in       (relax_tdyn t_dyn, optional) -> cropped_NNNNN
The last dump of the last stage and its .in are the body; body.json records them.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

DEM_TOOL_NAMES = ("phantommoddump", "phantomflyby", "phantomanalysis")

G_CGS = 6.674e-8
RHO_0_CGS = 2.7  # eos_tillotson.f90:34 rho_0; setup bulk density = rho_0 * scale_rho
SETTLE_PREFIX = "settle"
CROP_PREFIX = "cropped"
# a settled random packing is ~0.6-0.66; a loose cloud cut by the crop (settle_tdyn too short) is ~0.1-0.3
MIN_SETTLED_PACKING = 0.5


@dataclass(frozen=True)
class SettleSpec:
    np_apophis: int          # grains wanted AFTER the crop (setup settles ~1.1*np*V_circ/V_shape)
    scale_rho: float         # bulk density = RHO_0_CGS * scale_rho
    shape_file: str          # .shape config (or bare .obj) to crop to
    pack_phi: float = 0.64   # setup_solarsystem default
    pack_expand: float = 1.8  # setup_solarsystem default
    settle_tdyn: float = 5.0  # settle length in t_dyn (assumption A1)
    relax_tdyn: float = 0.5   # Mia: 0.5 t_dyn relax after the crop; 0 = skip


def tdyn_seconds(scale_rho: float) -> float:
    """t_dyn = 1/sqrt(G rho): what setup_solarsystem.f90 writes as tdyn_s for idamp=2."""
    return 1.0 / math.sqrt(G_CGS * RHO_0_CGS * scale_rho)


def _file_sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _shape_asset_paths(shape_file: Path) -> list:
    """The .shape config plus the OBJ its 'mesh' line points at (same rule as stage_shape_assets)."""
    src = Path(shape_file).resolve()
    out = [src]
    if src.suffix.lower() == ".obj":
        return out
    for line in src.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].lower() == "mesh":
            obj = Path(parts[1])
            out.append(obj if obj.is_absolute() else (src.parent / obj).resolve())
            break
    return out


def settle_key(spec: SettleSpec, binaries: Sequence[Path]) -> str:
    """Cache dir name: readable prefix + hash of spec, shape/mesh bytes and binary bytes."""
    payload = asdict(spec)
    payload["shape_file"] = [_file_sha(p) for p in _shape_asset_paths(Path(spec.shape_file))]
    payload["binaries"] = [_file_sha(Path(b).resolve()) for b in binaries]
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
    return f"np{spec.np_apophis}_srho{spec.scale_rho:g}_{digest}"


def write_settle_setup(template_setup: Path, dst: Path, spec: SettleSpec, shape_basename: str) -> None:
    """Settle .setup: isolated body (apophis_only=T), loose cloud, no spin, dumps every 0.25 t_dyn."""
    r = _runner()
    text = Path(template_setup).read_text(encoding="utf-8")
    tdyn_hr = tdyn_seconds(spec.scale_rho) / 3600.0
    keys = {
        "pack_settle": r.format_logical_token(True),
        "apophis_only": r.format_logical_token(True),
        "use_dem": r.format_logical_token(True),
        "use_dem_as_sinks": r.format_logical_token(False),
        "scale_rho": r.format_real_token(spec.scale_rho),
        "mass_apophis": r.format_real_token(0.0),
        "pack_phi": r.format_real_token(spec.pack_phi),
        "pack_expand": r.format_real_token(spec.pack_expand),
        "apophis_shape_file": shape_basename,
        "apophis_spin_period": r.format_real_token(0.0),
        "apophis_spin_torque_align_deg": r.format_real_token(-1.0),
        "tmax_in": r.hours_to_phantom_time_string(spec.settle_tdyn * tdyn_hr),
        "dtmax_in": r.hours_to_phantom_time_string(0.25 * tdyn_hr),
    }
    for key, val in keys.items():
        text = r.replace_setup_assignment(text, key, val)
        r.validate_assignment(text, key, val)
    if not r._NP_APOPHIS_RE.search(text):
        raise RuntimeError("np_apophis assignment not found in template .setup")
    text = r._NP_APOPHIS_RE.sub(lambda m: m.group(1) + str(spec.np_apophis), text, count=1)
    Path(dst).write_text(text, encoding="utf-8")


_KEPT_RE = re.compile(r"grains kept\s*=\s*(\d+)\s+of\s+(\d+)")
_PHI_RE = re.compile(r"packing fraction\s*=\s*([-+\d.Ee]+)")


def parse_crop_log(text: str) -> Tuple[int, int, float]:
    """(n_kept, n_settled, packing_fraction) from moddump_cropshape.f90 output."""
    k, p = _KEPT_RE.search(text), _PHI_RE.search(text)
    if not k or not p:
        raise RuntimeError("crop moddump log has no 'grains kept' / 'packing fraction' lines")
    return int(k.group(1)), int(k.group(2)), float(p.group(1))


def _runner():
    """run_mass_sobol_phantom, imported lazily (it imports this module lazily too)."""
    try:
        import run_mass_sobol_phantom as mod
    except ImportError:
        from sobol import run_mass_sobol_phantom as mod
    return mod


def _pdem():
    try:
        import particle_dem as mod
    except ImportError:
        from sobol import particle_dem as mod
    return mod


@dataclass(frozen=True)
class DemTools:
    moddump: Path   # crop build (moddump_cropshape.f90, the solarsystem MODFILE default)
    flyby: Path     # moddump_earthflyby.f90 build
    analysis: Path  # analysis_demshape.f90 build


def resolve_dem_tools(phantom_dir: Path) -> DemTools:
    r = _runner()
    try:
        paths = [r.resolve_phantom_executable(Path(phantom_dir), n, must_exist=True)
                 for n in DEM_TOOL_NAMES]
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{exc}. Build them with: cd sobol && make demtools") from None
    return DemTools(*paths)


@dataclass(frozen=True)
class SettledBody:
    dir: Path
    dump: str               # final dump basename inside dir
    infile: str             # .in matching that dump (carries idamp=2 from the settle!)
    n_kept: int
    n_settled: int
    packing_fraction: float
    utime_s: float          # seconds per code time unit
    key: str

    @classmethod
    def load(cls, body_dir: Path) -> "SettledBody":
        m = json.loads((Path(body_dir) / "body.json").read_text(encoding="utf-8"))
        return cls(dir=Path(body_dir), dump=m["dump"], infile=m["infile"], n_kept=int(m["n_kept"]),
                   n_settled=int(m["n_settled"]), packing_fraction=float(m["packing_fraction"]),
                   utime_s=float(m["utime_s"]), key=m["key"])


def settle_maxp(np_apophis: int) -> int:
    """The settle cloud holds ~1.1*np*V_sphere(r_circ)/V_shape grains (~2.6x np for Apophis)."""
    return max(4000, 8 * int(np_apophis))


def set_in_keys(in_path: Path, keys: Dict[str, str]) -> None:
    r = _runner()
    text = Path(in_path).read_text(encoding="utf-8")
    for key, val in keys.items():
        text = r.replace_setup_assignment(text, key, val)
    Path(in_path).write_text(text, encoding="utf-8")


def _run(cmd, cwd: Path, log: Path, stdin_text: Optional[str] = None) -> None:
    with open(log, "w", encoding="utf-8") as fh:
        proc = subprocess.run([str(c) for c in cmd], cwd=str(cwd), input=stdin_text, text=True,
                              stdout=fh, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"{Path(str(cmd[0])).name} failed ({proc.returncode}); see {log}")


def build_settled_body(
    spec: SettleSpec,
    cache_root: Path,
    template_setup: Path,
    phantomsetup_bin: Path,
    phantom_bin: Path,
    tools: DemTools,
    ephemeris_cache_dir: Optional[Path],
) -> SettledBody:
    """Settle, crop and relax one body, or return the cached one. Not safe to call concurrently
    for the same spec: callers build serially before dispatching workers."""
    r = _runner()
    key = settle_key(spec, (phantomsetup_bin, phantom_bin, tools.moddump))
    final = Path(cache_root) / key
    if (final / "body.json").is_file():
        return SettledBody.load(final)
    work = Path(cache_root) / f"{key}.partial"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    maxp = [f"--maxp={settle_maxp(spec.np_apophis)}"]

    shape_name = r.stage_shape_assets(Path(spec.shape_file), work)
    if ephemeris_cache_dir is not None:
        r.copy_ephemeris_txt_cache(Path(ephemeris_cache_dir), work)  # apophis_only still queries Horizons
    write_settle_setup(template_setup, work / f"{SETTLE_PREFIX}.setup", spec, shape_name)
    r.run_phantomsetup(Path(phantomsetup_bin), SETTLE_PREFIX, maxp, work, work / "settle_setup.log")
    # every dump full: the crop needs velocities in the last settle dump
    set_in_keys(work / f"{SETTLE_PREFIX}.in", {"nfulldump": f"{1:>10}"})
    _run([phantom_bin, f"{SETTLE_PREFIX}.in", *maxp], work, work / "settle_phantom.log")
    utime = r._parse_utime_from_phantom_log(work / "settle_phantom.log")
    if utime is None:
        raise RuntimeError(f"no 'Time: ... s' unit line in {work / 'settle_phantom.log'}")
    settle_dumps = _pdem().list_full_dumps(work, SETTLE_PREFIX)
    if not settle_dumps:
        raise RuntimeError(f"settle wrote no {SETTLE_PREFIX}_NNNNN dumps in {work}")

    rho = RHO_0_CGS * spec.scale_rho
    _run([tools.moddump, settle_dumps[-1].name, CROP_PREFIX, "0", *maxp], work, work / "crop.log",
         stdin_text=f"{shape_name}\n{rho:.6g}\n")  # prompts: shape file, bulk density g/cm^3
    n_kept, n_settled, phi = parse_crop_log((work / "crop.log").read_text(errors="replace"))
    if phi < MIN_SETTLED_PACKING:
        raise RuntimeError(
            f"settle did not converge: n_kept={n_kept}, n_settled={n_settled}, packing_fraction={phi} "
            f"< {MIN_SETTLED_PACKING} (loose cloud never collapsed); raise --settle-tdyn"
        )
    dump, infile = f"{CROP_PREFIX}_00000", f"{CROP_PREFIX}.in"

    if spec.relax_tdyn > 0.0:
        tdyn_code = tdyn_seconds(spec.scale_rho) / utime
        set_in_keys(work / infile, {
            "tmax": r.format_real_token(spec.relax_tdyn * tdyn_code),
            "dtmax": r.format_real_token(0.25 * spec.relax_tdyn * tdyn_code),
            "nfulldump": f"{1:>10}",
        })
        _run([phantom_bin, infile, *maxp], work, work / "relax_phantom.log")
        dump = _pdem().list_full_dumps(work, CROP_PREFIX)[-1].name

    manifest = {
        "key": key, "spec": asdict(spec), "dump": dump, "infile": infile,
        "n_kept": n_kept, "n_settled": n_settled, "packing_fraction": phi, "utime_s": utime,
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    (work / "body.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(work, final)  # atomic: a dir without body.json is never loaded
    return SettledBody.load(final)
