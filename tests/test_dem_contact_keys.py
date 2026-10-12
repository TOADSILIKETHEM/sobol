"""Mia's 1fd4ceb88 DEM contact keys (mu_s, mu_r, mu_t, epsilon_t_dem, cohesion_pa) in the runner."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import run_mass_sobol_phantom as runner
from run_mass_sobol_phantom import RunSample, apply_run_sample_to_in

NEW_IN = """\
              kn_cgs =   1.000E+07    ! DEM normal spring constant
       epsilon_n_dem =       0.500    ! DEM normal coefficient of restitution [0,1]
                mu_s =       0.500    ! DEM static (sliding) friction coefficient (0=frictionless)
       epsilon_t_dem =       0.500    ! DEM tangential coefficient of restitution (0,1]
                mu_r =       0.000    ! DEM rolling friction coefficient
"""

# .in written by a binary from before 1fd4ceb88: has ct_dem, no friction keys
OLD_IN = """\
              kn_cgs =   1.000E+07    ! DEM normal spring constant
       epsilon_n_dem =       0.500    ! DEM normal coefficient of restitution [0,1]
              ct_dem =       0.000    ! DEM tangential damping coefficient
"""


def _val(text, key):
    for line in text.splitlines():
        if line.split("=")[0].strip() == key:
            return float(line.split("=")[1].split("!")[0])
    return None


def test_new_keys_in_variation_spec():
    keys = {p for p, *_ in runner._IN_VARIATION_SPEC}
    assert {"mu_s", "mu_r", "mu_t", "epsilon_t_dem", "cohesion_pa"} <= keys


def test_mu_s_patched_when_set(tmp_path):
    p = tmp_path / "sobol.in"
    p.write_text(NEW_IN)
    cols = apply_run_sample_to_in(p, RunSample(mu_s=0.3))
    assert _val(p.read_text(), "mu_s") == 0.3
    assert cols["mu_s"] == "0.3"


def test_mu_s_pinned_when_unset(tmp_path):
    # a sweep that never mentions friction must not inherit the binary default
    p = tmp_path / "sobol.in"
    p.write_text(NEW_IN)
    s = RunSample(kn_cgs=1e7)
    runner.apply_contact_defaults(s)
    apply_run_sample_to_in(p, s)
    if runner.DEFAULT_MU_S is None:
        assert _val(p.read_text(), "mu_s") == 0.5
    else:
        assert _val(p.read_text(), "mu_s") == runner.DEFAULT_MU_S


def test_np_list_samples_pin_mu_s():
    args = runner.parse_args(["--np-apophis-list", "300", "--use-dem-fixed", "true"])
    (s,) = runner.build_np_list_samples(args)
    assert s.mu_s == runner.DEFAULT_MU_S


def test_in_patch_inserts_missing_new_keys(tmp_path):
    # .in without mu_s (old binary, or a stale phantomsetup next to a new phantom): insert the
    # line so a new phantom cannot fall back to its 0.5 default; old phantom only warns on it
    p = tmp_path / "sobol.in"
    p.write_text(OLD_IN)
    cols = apply_run_sample_to_in(p, RunSample(kn_cgs=2e7, mu_s=0.0))
    text = p.read_text()
    assert _val(text, "kn_cgs") == 2e7
    assert _val(text, "mu_s") == 0.0
    assert cols["mu_s"] == "0"
    assert "ct_dem" in text


def test_in_patch_still_fails_on_missing_core_key(tmp_path):
    p = tmp_path / "sobol.in"
    p.write_text("              kn_cgs =   1.000E+07\n")
    with pytest.raises(RuntimeError):
        apply_run_sample_to_in(p, RunSample(epsilon_n_dem=0.2))


def test_cli_fixed_mu_s_reaches_sample():
    args = runner.parse_args(["--np-apophis-list", "300", "--use-dem-fixed", "true", "--mu-s", "0.7"])
    assert args.mu_s == 0.7
    (s,) = runner.build_np_list_samples(args)
    assert s.mu_s == 0.7


def test_cli_fixed_mu_s_excludes_sweep():
    args = runner.parse_args(["--np-apophis-list", "300", "--use-dem-fixed", "true",
                              "--mu-s", "0.7", "--mu-s-min", "0", "--mu-s-max", "0.6"])
    with pytest.raises(ValueError):
        runner.validate_args(args)


def test_torque_reruns_pin_mu_s(tmp_path, monkeypatch):
    # thesis torque-align reruns must reproduce the frictionless S1 physics
    import run_torque_align_blender_reruns as tr
    monkeypatch.setattr(tr, "preflight", lambda *a, **k: (None, None, None, None))
    seen = []

    def fake_run_one_case(*, run_id, output_root, sample, **kw):
        seen.append(sample)
        run_dir = output_root / f"run_{run_id:04d}"
        run_dir.mkdir(parents=True)
        return tr.RunRecord(run_id=run_id, mass_input_kg=float("nan"), run_dir=str(run_dir),
                            status="error", closest_approach_km=float("nan"),
                            closest_approach_au=float("nan"), error="fake")
    monkeypatch.setattr(tr, "run_one_case", fake_run_one_case)
    tr.main(["--output-root", str(tmp_path), "--batch-label", "t", "--case", "aligned"])
    assert seen and all(s.mu_s == runner.DEFAULT_MU_S for s in seen)


def test_tui_sim_render_pins_mu_s():
    import asyncio
    import tui_sim_render as tsr

    async def _scenario():
        app = tsr.SimRenderTUIApp()
        async with app.run_test():
            return app._build_sim_params(False)

    assert asyncio.run(_scenario()).sample.mu_s == runner.DEFAULT_MU_S


def test_wizard_gates_every_in_dimension():
    # a new .in sweep flag missing here would be prompted ungated under "Other"
    import interactive_run_mass_sobol as wiz
    spec = {(p, lo, hi) for p, lo, hi, _ in runner._IN_VARIATION_SPEC}
    assert spec == set(wiz.IN_BOUND_PAIRS)
    for p, lo, hi in wiz.IN_BOUND_PAIRS:
        assert p in wiz.IN_PARAM_LABELS
        assert wiz.DEST_TO_SECTION[lo] == wiz.DEST_TO_SECTION[hi] == "DEM Contact (optional dimensions)"
        assert lo in wiz.INTERACTIVE_BRIEF and hi in wiz.INTERACTIVE_BRIEF


def test_wizard_knows_fixed_mu_s():
    import interactive_run_mass_sobol as wiz
    assert wiz.DEST_TO_SECTION["mu_s"] == "DEM Contact (optional dimensions)"
    assert "mu_s" in wiz.INTERACTIVE_BRIEF


def _fake_bin(tmp_path, with_friction):
    b = tmp_path / ("phantom_new" if with_friction else "phantom_old")
    b.write_bytes(b"\x00" + (runner.FRICTION_BINARY_MARKER if with_friction else b"old") + b"\x00")
    return b


def test_explicit_friction_needs_friction_binary(tmp_path):
    args = runner.parse_args(["--np-apophis-list", "300", "--use-dem-fixed", "true", "--mu-s", "0.4"])
    with pytest.raises(RuntimeError, match="friction"):
        runner.check_binary_supports_contact_keys(args, _fake_bin(tmp_path, False))
    runner.check_binary_supports_contact_keys(args, _fake_bin(tmp_path, True))


def test_default_pin_ok_on_old_binary(tmp_path):
    args = runner.parse_args(["--np-apophis-list", "300", "--use-dem-fixed", "true"])
    runner.check_binary_supports_contact_keys(args, _fake_bin(tmp_path, False))


def test_swept_friction_axis_needs_friction_binary(tmp_path):
    args = runner.parse_args(["--np-apophis-list", "300", "--use-dem-fixed", "true",
                              "--mu-s-min", "0", "--mu-s-max", "0.6"])
    with pytest.raises(RuntimeError, match="friction"):
        runner.check_binary_supports_contact_keys(args, _fake_bin(tmp_path, False))


@pytest.mark.parametrize("axis", [["--mu-r-min", "0", "--mu-r-max", "1"],
                                  ["--mu-t-min", "0", "--mu-t-max", "1"],
                                  ["--eps-t-min", "0.2", "--eps-t-max", "0.8"]])
def test_friction_only_axis_without_mu_s_rejected(axis):
    args = runner.parse_args(["--np-apophis-list", "300", "--use-dem-fixed", "true", *axis])
    with pytest.raises(ValueError, match="mu_s"):
        runner.validate_args(args)


def test_friction_only_axis_with_fixed_mu_s_ok():
    args = runner.parse_args(["--np-apophis-list", "300", "--use-dem-fixed", "true",
                              "--mu-s", "0.5", "--mu-r-min", "0", "--mu-r-max", "1"])
    runner.validate_args(args)


def test_friction_axis_with_sink_dem_rejected():
    args = runner.parse_args(["--np-apophis-list", "300", "--use-dem-fixed", "true", "--dem-model", "sink",
                              "--mu-s-min", "0", "--mu-s-max", "0.6"])
    with pytest.raises(ValueError, match="sink"):
        runner.validate_args(args)


def test_contact_key_check_skips_missing_binary(tmp_path):
    # --dry-run resolves phantom without requiring it to exist
    args = runner.parse_args(["--np-apophis-list", "300", "--use-dem-fixed", "true", "--mu-s", "0.4"])
    runner.check_binary_supports_contact_keys(args, tmp_path / "no_such_phantom")


@pytest.mark.parametrize("kt", [["--kt-fixed", "1e5"], ["--kc-fixed", "1e5"], ["--kt-min", "1e4", "--kt-max", "1e5"]])
def test_coh_pa_axis_excludes_kt(kt):
    a = runner.parse_args(["--use-dem-fixed", "true", "--np-apophis", "300", "--coh-pa-min", "1",
                           "--coh-pa-max", "10"] + kt)
    with pytest.raises(ValueError, match="one cohesion model"):
        runner.validate_args(a)


def test_coh_pa_axis_alone_is_fine():
    a = runner.parse_args(["--use-dem-fixed", "true", "--np-apophis", "300", "--coh-pa-min", "1",
                           "--coh-pa-max", "10"])
    runner.validate_args(a)


def test_cli_fixed_rolling_twisting_eps_t_reach_sample():
    args = runner.parse_args(["--np-apophis-list", "300", "--use-dem-fixed", "true", "--mu-s", "1",
                              "--mu-r", "1.05", "--mu-t", "1.3", "--eps-t", "0.55"])
    runner.validate_args(args)
    (s,) = runner.build_np_list_samples(args)
    assert (s.mu_s, s.mu_r, s.mu_t, s.epsilon_t_dem) == (1.0, 1.05, 1.3, 0.55)


@pytest.mark.parametrize("extra", [["--mu-r", "1"], ["--mu-t", "1"], ["--eps-t", "0.5"]])
def test_fixed_friction_only_flag_needs_mu_s(extra):
    args = runner.parse_args(["--np-apophis-list", "300", "--use-dem-fixed", "true", *extra])
    with pytest.raises(ValueError, match="mu_s"):
        runner.validate_args(args)


def test_fixed_mu_r_excludes_sweep():
    args = runner.parse_args(["--np-apophis-list", "300", "--use-dem-fixed", "true", "--mu-s", "1",
                              "--mu-r", "1", "--mu-r-min", "0", "--mu-r-max", "2"])
    with pytest.raises(ValueError, match="mutually exclusive"):
        runner.validate_args(args)


def test_settle_with_friction_rejects_swept_friction():
    args = runner.parse_args(["--np-apophis", "300", "--use-dem-fixed", "true", "--body-source", "settled",
                              "--settle-with-friction", "--mu-s-min", "0", "--mu-s-max", "1",
                              "--num-samples", "2"])
    with pytest.raises(ValueError, match="settle-with-friction"):
        runner.validate_args(args)


def test_wizard_knows_fixed_rolling_twisting():
    import interactive_run_mass_sobol as wiz
    for dest in ("mu_r", "mu_t", "epsilon_t_dem", "settle_with_friction"):
        assert dest in wiz.DEST_TO_SECTION and dest in wiz.INTERACTIVE_BRIEF
