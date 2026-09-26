import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import encounter as enc
import run_mass_sobol_phantom as runner
from run_mass_sobol_phantom import RunSample

HYPER = ["--body-source", "settled", "--encounter", "hyperbola",
         "--flyby-rp-km", "38000", "--flyby-vinf-kms", "5.9",
         "--flyby-start-sep-km", "1e5", "--tmax-hours", "8", "--dtmax-hours", "0.25"]


def _args(*extra):
    return runner.parse_args(["--np-apophis-list", "300", "--use-dem-fixed", "true", *extra])


def test_defaults_are_lattice_ephemeris():
    a = _args()
    runner.validate_args(a)
    assert (a.body_source, a.encounter) == ("lattice", "ephemeris")
    s = RunSample()
    assert (s.body_source, s.encounter, s.settled_body_dir) == ("lattice", "ephemeris", None)


def test_settled_ephemeris_blocked_until_packing_file():
    with pytest.raises(ValueError, match="packing_file"):
        runner.validate_args(_args("--body-source", "settled"))


def test_hyperbola_needs_settled_body():
    with pytest.raises(ValueError, match="--body-source settled"):
        runner.validate_args(_args("--encounter", "hyperbola", "--flyby-rp-km", "38000",
                                   "--flyby-vinf-kms", "5.9", "--tmax-hours", "8", "--dtmax-hours", "0.25"))


def test_hyperbola_needs_rp_and_vinf():
    argv = [a for a in HYPER if a not in ("--flyby-vinf-kms", "5.9")]
    with pytest.raises(ValueError, match="flyby-vinf-kms"):
        runner.validate_args(_args(*argv))


def test_hyperbola_rejects_spin():
    with pytest.raises(ValueError, match="no spin"):
        runner.validate_args(_args(*HYPER, "--spin-period-fixed", "2.0"))


def test_hyperbola_rejects_sink_dem():
    with pytest.raises(ValueError, match="particle"):
        runner.validate_args(_args(*HYPER, "--dem-model", "sink"))


def test_hyperbola_rejects_too_short_tmax():
    argv = list(HYPER)
    argv[argv.index("--tmax-hours") + 1] = "1"
    with pytest.raises(ValueError, match="tmax"):
        runner.validate_args(_args(*argv))


def test_hyperbola_sets_earth_sink_1():
    a = _args(*HYPER)
    runner.validate_args(a)
    assert a.sink_earth_id == 1


def test_fixed_flyby_values_reach_samples():
    a = _args(*HYPER)
    runner.validate_args(a)
    s = runner.build_np_list_samples(a)[0]
    assert (s.body_source, s.encounter) == ("settled", "hyperbola")
    assert (s.flyby_rp_km, s.flyby_vinf_kms, s.flyby_start_sep_km) == (38000.0, 5.9, 1e5)
    assert s.flyby_perturber_earth_masses == 1.0


def test_time_to_pericentre_straight_line_limit():
    # huge v_inf: path is a straight line, t = sqrt(d^2 - rp^2) / v
    t = enc.time_to_pericentre_hr(1.0e4, 1000.0, 1.0e5, 1.0)
    assert t == pytest.approx((1.0e10 - 1.0e8) ** 0.5 / 1000.0 / 3600.0, rel=0.01)
    assert enc.time_to_pericentre_hr(1.0e4, 5.9, 1.0e4, 1.0) == pytest.approx(0.0, abs=1e-9)


def test_attach_settled_bodies_builds_each_spec_once(tmp_path, monkeypatch):
    import settled_body as sb
    built = []

    def fake_build(spec, cache_root, *a, **k):
        built.append(spec)
        return sb.SettledBody(dir=tmp_path / f"b{len(built)}", dump="cropped_00000", infile="cropped.in",
                              n_kept=spec.np_apophis, n_settled=0, packing_fraction=0.64,
                              utime_s=2.745e-6, key=f"k{len(built)}")

    monkeypatch.setattr(sb, "build_settled_body", fake_build)
    monkeypatch.setattr(sb, "resolve_dem_tools", lambda d: None)
    samples = [RunSample(np_apophis=300, body_source="settled"),
               RunSample(np_apophis=300, body_source="settled"),
               RunSample(np_apophis=500, body_source="settled"),
               RunSample(np_apophis=300)]  # lattice: untouched
    n = runner.attach_settled_bodies(
        samples, template_setup=Path(runner.__file__).parent / "sobol.setup",
        shape_file=runner.resolve_default_shape_file(), settle_tdyn=5.0, relax_tdyn=0.5,
        cache_root=tmp_path, phantomsetup_bin=Path("x"), phantom_bin=Path("y"),
        phantom_dir=tmp_path, ephemeris_cache_dir=None, dry_run=False)
    assert n == 2 and len(built) == 2
    assert samples[0].settled_body_dir == samples[1].settled_body_dir != samples[2].settled_body_dir
    assert samples[3].settled_body_dir is None
