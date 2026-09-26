"""Settled-body cache: Mia's settle -> crop -> relax packing, built once per spec.

Stages (docs/MIA_PACKING_WORKFLOW.md, Option 2 steps 1-3):
  1. phantomsetup settle      (pack_settle=T, apophis_only=T; setup sets idamp=2)
  2. phantom settle.in        -> settle_NNNNN
  3. phantommoddump <last> cropped 0   (moddump_cropshape.f90) -> cropped_00000, cropped.in
  4. phantom cropped.in       (relax_tdyn t_dyn, optional) -> cropped_NNNNN
The last dump of the last stage and its .in are the body; body.json records them.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEM_TOOL_NAMES = ("phantommoddump", "phantomflyby", "phantomanalysis")


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
