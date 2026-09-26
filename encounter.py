"""Hyperbolic flyby encounter from a settled body (Mia's Option 3, moddump_earthflyby.f90)."""
from __future__ import annotations

import math

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
