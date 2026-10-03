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


EPHEM = ["--body-source", "settled"]


def test_settled_ephemeris_allowed_with_packing_file():
    a = _args(*EPHEM)
    runner.validate_args(a)
    assert a.sink_earth_id == runner.EARTH_SINK_ID_DEFAULT


@pytest.mark.parametrize("extra", [
    ("--spin-period-fixed", "30"),
    ("--scale-vel-min", "0.9", "--scale-vel-max", "1.1"),
    ("--scale-pos-min", "0.9", "--scale-pos-max", "1.1"),
])
def test_settled_ephemeris_allows_spin_and_setup_sweeps(extra):
    runner.validate_args(_args(*EPHEM, *extra))


@pytest.mark.parametrize("extra,match", [
    (("--scale-rho-min", "0.9", "--scale-rho-max", "1.1"), "scale_rho"),
    (("--use-shape-crop-fixed", "true"), "use_shape_crop"),
    (("--flyby-rp-km", "38000"), "--encounter hyperbola"),
    (("--flyby-vinf-kms", "5.9"), "--encounter hyperbola"),
])
def test_settled_ephemeris_rejects(extra, match):
    with pytest.raises(ValueError, match=match):
        runner.validate_args(_args(*EPHEM, *extra))


def test_settled_ephemeris_column_order_has_np_kept():
    a = _args(*EPHEM)
    runner.validate_args(a)
    assert "np_kept" in runner.sample_column_order(a)


def test_prepare_tools_ephemeris_checks_packing_key_and_skips_flyby(tmp_path, monkeypatch):
    import settled_body as sb
    a = _args(*EPHEM, "--phantom-dir", str(tmp_path))
    runner.validate_args(a)
    monkeypatch.setattr(runner, "attach_settled_bodies", lambda samples, **k: 1)
    monkeypatch.setattr(sb, "resolve_dem_tools", lambda d: pytest.fail("phantomflyby not needed for ephemeris"))
    setup_bin = tmp_path / "phantomsetup"
    setup_bin.write_bytes(b"\x00np_apophis\x00")
    with pytest.raises(RuntimeError, match="packing_file"):
        runner.prepare_settled_bodies_and_dem_tools(a, [], tmp_path / "s.setup", setup_bin,
                                                    tmp_path / "phantom", None)
    setup_bin.write_bytes(b"\x00packing_file\x00")
    flyby, _ = runner.prepare_settled_bodies_and_dem_tools(a, [], tmp_path / "s.setup", setup_bin,
                                                           tmp_path / "phantom", None)
    assert flyby is None


def test_run_one_case_settled_ephemeris_stages_body(tmp_path, monkeypatch):
    import json
    body_dir = tmp_path / "body"
    body_dir.mkdir()
    (body_dir / "cropped_00002").write_text("dump")
    (body_dir / "body.json").write_text(json.dumps({
        "key": "np300_srho1_abc", "dump": "cropped_00002", "infile": "cropped.in", "n_kept": 310,
        "n_settled": 812, "packing_fraction": 0.651, "utime_s": 2.745e-6,
        "spec": {"shape_file": str(runner.resolve_default_shape_file())}}))
    seen = {}

    def fake_setup(bin_, prefix, maxp_flag, run_dir, log):
        seen["maxp"] = maxp_flag
        seen["setup"] = (run_dir / f"{prefix}.setup").read_text()
        (run_dir / "setup.log").write_text(" placed 310 pre-built grains on the ephemeris orbit\n")

    monkeypatch.setattr(runner, "run_phantomsetup", fake_setup)
    monkeypatch.setattr(runner, "run_command", lambda *a, **k: None)
    monkeypatch.setattr(runner, "compute_run_metrics",
                        lambda *a, **k: runner.RunMetrics(38000.0, 0.0, float("nan"), float("nan"),
                                                          float("nan"), float("nan"), float("nan")))
    base_in = tmp_path / "sobol.in"
    base_in.write_text("           nfulldump =           5    ! full dump every n dumps\n")
    sample = runner.RunSample(use_dem=True, dem_model="particle", np_apophis=300,
                              body_source="settled", settled_body_dir=str(body_dir))
    rec = runner.run_one_case(
        run_id=1, sample=sample, base_setup=Path(runner.__file__).parent / "sobol.setup",
        base_input=base_in, output_root=tmp_path / "out", prefix="sobol",
        phantomsetup_bin=Path("x"), phantom_bin=Path("y"), ref_mass_kg=None, dry_run=False,
        earth_sink_id=4, apophis_sink_id=11)
    assert rec.status == "ok", rec.error
    assert rec.param_columns["np_kept"] == "310"
    assert "settled_body" in seen["setup"]
    assert seen["maxp"] == ["--maxp=2000"]
    assert (tmp_path / "out" / "run_0001" / "settled_body").read_text() == "dump"


def test_run_one_case_settled_ephemeris_fails_on_stale_binary(tmp_path, monkeypatch):
    import json
    body_dir = tmp_path / "body"
    body_dir.mkdir()
    (body_dir / "cropped_00002").write_text("dump")
    (body_dir / "body.json").write_text(json.dumps({
        "key": "k", "dump": "cropped_00002", "infile": "cropped.in", "n_kept": 310,
        "n_settled": 812, "packing_fraction": 0.651, "utime_s": 2.745e-6,
        "spec": {"shape_file": str(runner.resolve_default_shape_file())}}))
    monkeypatch.setattr(runner, "run_phantomsetup",
                        lambda b, p, m, d, log: (d / "setup.log").write_text(" particles kept: 300\n"))
    monkeypatch.setattr(runner, "run_command", lambda *a, **k: pytest.fail("phantom must not run"))
    base_in = tmp_path / "sobol.in"
    base_in.write_text("           nfulldump =           5    ! full dump every n dumps\n")
    sample = runner.RunSample(use_dem=True, dem_model="particle", np_apophis=300,
                              body_source="settled", settled_body_dir=str(body_dir))
    rec = runner.run_one_case(
        run_id=1, sample=sample, base_setup=Path(runner.__file__).parent / "sobol.setup",
        base_input=base_in, output_root=tmp_path / "out", prefix="sobol",
        phantomsetup_bin=Path("x"), phantom_bin=Path("y"), ref_mass_kg=None, dry_run=False,
        earth_sink_id=4, apophis_sink_id=11)
    assert rec.status == "failed"
    assert "packing_file" in rec.error


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
    monkeypatch.setattr(sb, "resolve_dem_tools", lambda d, **k: None)
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


SWEEP = ["--body-source", "settled", "--encounter", "hyperbola",
         "--flyby-rp-km-min", "20000", "--flyby-rp-km-max", "40000",
         "--flyby-vinf-kms-min", "5", "--flyby-vinf-kms-max", "7",
         "--flyby-start-sep-km", "1e5", "--tmax-hours", "12", "--dtmax-hours", "0.25",
         "--np-apophis", "300", "--use-dem-fixed", "true", "--num-samples", "4"]


def test_flyby_sweep_dims_fill_samples():
    a = runner.parse_args(SWEEP)
    runner.validate_args(a)
    assert runner.count_dimensions(a) == 2
    assert runner.sample_column_order(a) == ["flyby_rp_km", "flyby_vinf_kms", "np_kept"]
    ss = runner.build_run_samples(4, a)
    assert all(20000 <= s.flyby_rp_km <= 40000 and 5 <= s.flyby_vinf_kms <= 7 for s in ss)
    assert len({s.flyby_rp_km for s in ss}) == 4


def test_flyby_sweep_saltelli_roundtrip():
    a = runner.parse_args(SWEEP)
    runner.validate_args(a)
    prob = runner.build_salib_problem(a)
    assert prob["names"] == ["flyby_rp_km", "flyby_vinf_kms"]
    s = runner.run_sample_from_salib_row([25000.0, 6.0], a)
    assert (s.flyby_rp_km, s.flyby_vinf_kms) == (25000.0, 6.0)


def test_flyby_fixed_and_sweep_are_exclusive():
    with pytest.raises(ValueError, match="flyby-rp-km"):
        runner.validate_args(runner.parse_args([*SWEEP, "--flyby-rp-km", "30000"]))


def test_flyby_sweep_needs_hyperbola():
    argv = [a for a in SWEEP if a not in ("--encounter", "hyperbola", "--body-source", "settled")]
    with pytest.raises(ValueError, match="hyperbola"):
        runner.validate_args(runner.parse_args(argv))


def test_flyby_sweep_tmax_checked_at_slowest_corner():
    argv = list(SWEEP)
    argv[argv.index("--tmax-hours") + 1] = "3"
    with pytest.raises(ValueError, match="tmax"):
        runner.validate_args(runner.parse_args(argv))


def test_flyby_sweep_tmax_checked_at_true_max_not_corner():
    # t_peri(rp) has an interior max: at v_inf=5, start_sep=1e5, the (rp_min=20000) corner gives
    # 4.4947 hr (2x = 8.99, passes tmax=9.0), but the true max over [20000, 40000] is 4.5215 hr
    # at rp~27000 (2x = 9.04, fails tmax=9.0). This must be caught, not just the corner.
    argv = ["--body-source", "settled", "--encounter", "hyperbola",
            "--flyby-rp-km-min", "20000", "--flyby-rp-km-max", "40000",
            "--flyby-vinf-kms", "5",
            "--flyby-start-sep-km", "1e5", "--tmax-hours", "9.0", "--dtmax-hours", "0.25",
            "--np-apophis", "300", "--use-dem-fixed", "true", "--num-samples", "4"]
    with pytest.raises(ValueError, match="tmax"):
        runner.validate_args(runner.parse_args(argv))


# --- Fix 1: hyperbola CSV rows carry np_apophis / scale_rho / np_kept ------


def test_hyperbola_dry_run_fills_np_apophis(tmp_path):
    a = runner.parse_args(["--np-apophis-list", "300", "500", "--use-dem-fixed", "true", *HYPER])
    runner.validate_args(a)
    for np_val, sample in zip((300, 500), runner.build_np_list_samples(a)):
        run_dir = tmp_path / f"run_{np_val}"
        run_dir.mkdir()
        rec = runner._run_hyperbola_case(
            run_id=1, sample=sample, base_setup=Path(runner.__file__).parent / "sobol.setup",
            run_dir=run_dir, prefix="sobol", phantom_bin=Path("phantom"), dry_run=True,
            earth_sink_id=1, apophis_sink_id=2, phantomflyby_bin=None, phantomanalysis_bin=None)
        assert rec.param_columns["np_apophis"] == str(np_val)


def test_sample_column_order_always_has_np_kept_for_hyperbola():
    a = _args(*HYPER)
    runner.validate_args(a)
    assert "np_kept" in runner.sample_column_order(a)


def test_hyperbola_real_run_fills_np_kept(tmp_path, monkeypatch):
    import json
    import stat
    import settled_body as sb

    body_dir = tmp_path / "body"
    body_dir.mkdir()
    (body_dir / "cropped_00002").write_text("dump")
    (body_dir / "cropped.in").write_text(
        "                tmax =   1.000E+09    ! end time\n"
        "               dtmax =   1.000E+08    ! time between dumps\n"
        "           nfulldump =           1    ! full dump every n dumps\n"
        "              kn_cgs =   1.000E+07    ! DEM normal spring constant\n"
        "               idamp =           2    ! artificial damping of velocities\n"
    )
    (body_dir / "body.json").write_text(json.dumps({
        "key": "np300_srho1_abc", "dump": "cropped_00002", "infile": "cropped.in", "n_kept": 310,
        "n_settled": 812, "packing_fraction": 0.651, "utime_s": 2.745e-6}))

    flyby_bin = tmp_path / "phantomflyby"
    flyby_bin.write_text(
        "#!/bin/sh\ncat > flyby_stdin.txt\n: > \"$2_00000\"\ncp body.in \"$2.in\"\n"
    )
    flyby_bin.chmod(flyby_bin.stat().st_mode | stat.S_IXUSR)

    phantom_bin = tmp_path / "phantom"
    phantom_bin.write_text("#!/bin/sh\nexit 0\n")
    phantom_bin.chmod(phantom_bin.stat().st_mode | stat.S_IXUSR)

    run_dir = tmp_path / "run_0001"
    run_dir.mkdir()

    sample = runner.RunSample(use_dem=True, dem_model="particle", np_apophis=300,
                              body_source="settled", encounter="hyperbola",
                              flyby_rp_km=38000.0, flyby_vinf_kms=5.9, flyby_start_sep_km=4.0e5,
                              flyby_perturber_earth_masses=1.0, tmax_hours=48.0, dtmax_hours=0.5,
                              settled_body_dir=str(body_dir))
    monkeypatch.setattr(runner, "compute_run_metrics",
                        lambda *a, **k: runner.RunMetrics(38000.0, 0.0, float("nan"), float("nan"),
                                                          float("nan"), float("nan"), float("nan")))
    rec = runner._run_hyperbola_case(
        run_id=1, sample=sample, base_setup=Path(runner.__file__).parent / "sobol.setup",
        run_dir=run_dir, prefix="sobol", phantom_bin=phantom_bin, dry_run=False,
        earth_sink_id=1, apophis_sink_id=2, phantomflyby_bin=flyby_bin, phantomanalysis_bin=None)
    assert rec.status == "ok", rec.error
    assert rec.param_columns["np_kept"] == "310"
    assert rec.param_columns["np_apophis"] == "300"


# --- Fix 3: reject sweeps the hyperbola path never applies -----------------


@pytest.mark.parametrize("extra,match", [
    (("--scale-vel-min", "0.9", "--scale-vel-max", "1.1"), "scale_vel"),
    (("--scale-pos-min", "0.9", "--scale-pos-max", "1.1"), "scale_pos"),
    (("--vary-use-shape-crop",), "use_shape_crop"),
    (("--use-shape-crop-fixed", "true"), "use_shape_crop"),
    (("--scale-rho-min", "0.9", "--scale-rho-max", "1.1"), "scale_rho"),
])
def test_hyperbola_rejects_setup_only_sweeps(extra, match):
    with pytest.raises(ValueError, match=match):
        runner.validate_args(_args(*HYPER, *extra))


# --- Fix 4: pericentre must sit strictly between Earth's radius and start_sep


def test_hyperbola_rejects_rp_inside_earth():
    argv = list(HYPER)
    argv[argv.index("--flyby-rp-km") + 1] = "5000"  # < R_EARTH_KM (6371)
    with pytest.raises(ValueError, match="R_EARTH_KM|Earth"):
        runner.validate_args(_args(*argv))


def test_hyperbola_rejects_rp_past_start_sep():
    argv = list(HYPER)
    argv[argv.index("--flyby-rp-km") + 1] = "150000"
    argv[argv.index("--flyby-start-sep-km") + 1] = "100000"
    with pytest.raises(ValueError, match="start_sep|start-sep"):
        runner.validate_args(_args(*argv))


def test_hyperbola_sweep_rejects_rp_bounds_inside_earth():
    argv = list(SWEEP)
    argv[argv.index("--flyby-rp-km-min") + 1] = "5000"
    with pytest.raises(ValueError, match="R_EARTH_KM|Earth"):
        runner.validate_args(runner.parse_args(argv))


def test_hyperbola_sweep_rejects_rp_bounds_past_start_sep():
    argv = list(SWEEP)
    argv[argv.index("--flyby-rp-km-max") + 1] = "150000"
    argv[argv.index("--flyby-start-sep-km") + 1] = "100000"
    with pytest.raises(ValueError, match="start_sep|start-sep"):
        runner.validate_args(runner.parse_args(argv))


# --- Fix 5: batch slug carries hyp / settled tokens -------------------------


def test_slug_lattice_ephemeris_unchanged():
    a = runner.parse_args(["--mass-min-kg", "1e10", "--mass-max-kg", "1e11",
                           "--num-samples", "8", "--seed", "42"])
    runner.validate_args(a)
    assert runner.build_auto_batch_sweep_slug(a, max_len=200) == "n8_s42_m1e10-1e11"


def test_slug_has_hyp_token_for_hyperbola():
    a = _args(*HYPER)
    runner.validate_args(a)
    slug = runner.build_auto_batch_sweep_slug(a, max_len=200)
    assert "hyp" in slug.split("_")


def test_slug_has_settled_token_for_settled_body_source():
    a = _args(*HYPER)
    runner.validate_args(a)
    slug = runner.build_auto_batch_sweep_slug(a, max_len=200)
    assert "settled" in slug.split("_")


# --- final-review fixes ---------------------------------------------------


def test_attach_settled_bodies_needs_only_moddump(tmp_path, monkeypatch):
    # ephemeris/resume batches must not demand phantomflyby (or phantomanalysis) to settle a body
    import settled_body as sb
    tool = tmp_path / "phantommoddump"
    tool.write_text("#!/bin/sh\n")
    tool.chmod(0o755)
    seen = {}

    def fake_build(spec, cache_root, template_setup, psetup, phantom, tools, eph):
        seen["tools"] = tools
        return sb.SettledBody(dir=tmp_path / "body", dump="d", infile="i", n_kept=310, n_settled=800,
                              packing_fraction=0.65, utime_s=1.0, key="k")

    monkeypatch.setattr(sb, "build_settled_body", fake_build)
    samples = [RunSample(use_dem=True, np_apophis=300, body_source="settled")]
    n = runner.attach_settled_bodies(
        samples, template_setup=Path(runner.__file__).parent / "sobol.setup",
        shape_file=runner.resolve_default_shape_file(), settle_tdyn=5.0, relax_tdyn=0.5,
        cache_root=tmp_path / "cache", phantomsetup_bin=tmp_path / "ps", phantom_bin=tmp_path / "p",
        phantom_dir=tmp_path, ephemeris_cache_dir=None, dry_run=False)
    assert n == 1
    assert seen["tools"].moddump.name == "phantommoddump"
    assert samples[0].settled_body_dir == str(tmp_path / "body")


def test_run_phantomsetup_retry_judges_second_pass_only(tmp_path, monkeypatch):
    # a template missing a new key: pass 1 rewrites .setup and STOPs, pass 2 succeeds
    calls = []

    def fake_run(cmd, cwd, log_path, append=False):
        calls.append(append)
        with open(log_path, "a" if append else "w") as f:
            f.write(" ERROR: packing_file not found\n STOP rerun phantomsetup after editing .setup file\n"
                    if len(calls) == 1 else " writing sobol_00000.tmp\n")

    monkeypatch.setattr(runner, "run_command", fake_run)
    runner.run_phantomsetup(Path("ps"), "sobol", [], tmp_path, tmp_path / "setup.log")
    assert calls == [False, True]


def test_run_phantomsetup_retry_still_fails_when_second_pass_stops(tmp_path, monkeypatch):
    def fake_run(cmd, cwd, log_path, append=False):
        with open(log_path, "a" if append else "w") as f:
            f.write(" STOP rerun phantomsetup after editing .setup file\n")

    monkeypatch.setattr(runner, "run_command", fake_run)
    with pytest.raises(RuntimeError, match="second pass"):
        runner.run_phantomsetup(Path("ps"), "sobol", [], tmp_path, tmp_path / "setup.log")


BLANK_NO_COMMENT = ("         pack_settle =           F    ! settle\n"
                    "        packing_file =\n"
                    "         pack_expand =       1.800    ! initial cloud radius\n")


def test_replace_setup_assignment_blank_value_without_comment_keeps_next_line():
    out = runner.replace_setup_assignment(BLANK_NO_COMMENT, "packing_file", "settled_body")
    assert "packing_file = settled_body\n" in out
    assert "         pack_expand =       1.800    ! initial cloud radius\n" in out


def test_validate_assignment_blank_value_does_not_read_next_line():
    runner.validate_assignment(BLANK_NO_COMMENT, "packing_file", "")
