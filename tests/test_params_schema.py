"""params_schema: every runner flag described once (completeness, sections, gates)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import params_schema as ps
import run_mass_sobol_phantom as runner


def _dests():
    return {a.dest for a in runner.build_parser()._actions if a.dest != "help"}


def test_every_runner_dest_is_classified():
    # Fails when a flag is added to build_parser() without a schema decision.
    plain = {f.dest for f in ps.PLAIN}
    assert _dests() == plain | ps.AXIS_DESTS | ps.HIDDEN
    assert not plain & ps.AXIS_DESTS
    assert not plain & ps.HIDDEN


def test_axis_dests_exist_and_are_unique():
    seen = []
    for ax in ps.AXES:
        for d in (ax.lo, ax.hi, ax.fixed):
            if d:
                assert d in _dests(), d
                seen.append(d)
    assert len(seen) == len(set(seen))


def test_every_item_has_a_known_section_and_appears_once():
    for f in ps.PLAIN:
        assert f.section in ps.SECTION_ORDER, f.dest
    for ax in ps.AXES:
        assert ps.section_of(ax.lo) in ps.SECTION_ORDER, ax.stem
    items = [i for s in ps.SECTION_ORDER for i in ps.items_in(s)]
    assert len(items) == len(ps.PLAIN) + len(ps.AXES)


def test_widget_kinds_follow_argparse():
    kinds = {f.dest: f.kind for f in ps.PLAIN}
    assert kinds["boulders"] == "check"
    assert kinds["body_source"] == "select"
    assert kinds["np_apophis_list"] == "list"
    assert kinds["prefix"] == "text"


def test_gates():
    fv = ps.FormValues(plain={"body_source": "lattice", "encounter": "ephemeris", "boulders": False}, axes={})
    assert not ps.enabled("boulders", fv)
    assert not ps.enabled("f_boulder", fv)
    assert not ps.enabled("flyby_rp_km", fv)
    assert not ps.enabled("settle_tdyn", fv)
    assert ps.enabled("prefix", fv)
    fv.plain.update(body_source="settled", boulders=True, encounter="hyperbola")
    assert ps.enabled("boulders", fv)
    assert ps.enabled("bond_sigma_pa", fv)
    assert ps.enabled("f_boulder", fv)
    assert ps.enabled("flyby_rp_km", fv)
    assert ps.enabled("settle_tdyn", fv)


import shlex

import pytest

CASES = {
    "mass_scale": "--base-dir sobol --prefix sobol --num-samples 50 --mass-min-kg 1e10 --mass-max-kg 1e11 "
                  "--scale-vel-min 0.9 --scale-vel-max 1.1 --jobs 2",
    "settled_ephemeris": "--base-dir sobol --prefix sobol --phantom-dir /home/mboyle/Honours/sobol "
                         "--ephemeris-cache-dir sobol --use-dem-fixed true --np-apophis-list 300 "
                         "--body-source settled --spin-period-fixed 30 --tmax-hours 12 --dtmax-hours 1 "
                         "--num-samples 1 --jobs 1",
    "hyperbola": "--use-dem-fixed true --np-apophis 300 --body-source settled --encounter hyperbola "
                 "--flyby-rp-km-min 20000 --flyby-rp-km-max 40000 --flyby-vinf-kms-min 5 --flyby-vinf-kms-max 7 "
                 "--tmax-hours 48 --dtmax-hours 0.5 --num-samples 16 --jobs 2",
    "boulders": "--use-dem-fixed true --np-apophis 300 --body-source settled --spin-period-fixed 30 "
                "--tmax-hours 12 --dtmax-hours 1 --num-samples 8 --jobs 2 --boulders --f-boulder-min 0 "
                "--f-boulder-max 0.5 --boulder-placed binary --bond-sigma-pa 10",
    "contact": "--use-dem-fixed true --np-apophis 300 --mu-s-min 0.2 --mu-s-max 0.8 --mu-r 0.1 --eps-t 0.5 "
               "--coh-pa-min 1 --coh-pa-max 10",
    "pinned": "--use-dem-fixed true --np-apophis 300 --spin-torque-align-min 45 --spin-torque-align-max 45",
    "lists": "--use-dem-fixed true --np-apophis-list 100 300 --spin-period-list 3 30 --no-cleanup",
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_round_trip_matches_hand_written_command(name):
    argv = shlex.split(CASES[name])
    assert vars(ps.parse(ps.to_argv(ps.from_argv(argv)))) == vars(ps.parse(argv))


def test_pinned_axis_reads_back_as_fixed():
    fv = ps.from_argv(shlex.split(CASES["pinned"]))
    assert fv.axes["spin_torque_align"] == ps.AxisValue(sweep=False, fixed="45.0")
    argv = ps.to_argv(fv)
    assert argv[argv.index("--spin-torque-align-min") + 1] == "45.0"
    assert argv[argv.index("--spin-torque-align-max") + 1] == "45.0"


def test_defaults_are_not_emitted():
    assert ps.to_argv(ps.from_argv([])) == []


def test_apply_mode_single_forces_one_run_and_no_sweeps():
    fv = ps.from_argv(shlex.split(CASES["hyperbola"]))
    single = ps.apply_mode(fv, "single")
    assert single.plain["num_samples"] == "1" and single.plain["jobs"] == "1"
    assert not any(av.sweep for av in single.axes.values())
    assert fv.axes["flyby_rp_km"].sweep  # input not mutated


def test_check_reports_runner_errors_without_exiting():
    assert "mass bounds" in ps.check(["--mass-min-kg", "1e10"]).error
    assert "invalid int value" in ps.check(["--num-samples", "x"]).error
    assert ps.check(shlex.split(CASES["mass_scale"])).error is None


def test_show_command():
    cmd = ps.show_command(["--jobs", "2", "--batch-label", "a b"], 1)
    assert cmd == "OMP_NUM_THREADS=1 python3 sobol/run_mass_sobol_phantom.py --jobs 2 --batch-label 'a b'"


def test_relative_paths_become_absolute(tmp_path):
    fv = ps.from_argv(["--output-root", "runs", "--base-dir", "/abs/sobol"])
    out = ps.absolutise(fv, tmp_path)
    assert out.plain["output_root"] == str(tmp_path / "runs")
    assert out.plain["base_dir"] == "/abs/sobol"
    assert fv.plain["output_root"] == "runs"


def test_single_mode_refuses_lists_that_make_several_runs():
    fv = ps.from_argv(["--use-dem-fixed", "true", "--np-apophis-list", "300", "600"])
    assert "one value" in ps.single_mode_error(fv)
    assert ps.single_mode_error(ps.from_argv(["--use-dem-fixed", "true", "--np-apophis-list", "300"])) is None
    assert "spin_period_list" in ps.single_mode_error(ps.from_argv(["--spin-period-list", "10", "30"]))
    assert ps.single_mode_error(ps.from_argv(["--np-apophis", "300"])) is None


# --- defaults filled when a mode is picked (fill_defaults) ---

import math

import encounter


def _expected_tmax(rp, vinf, start_sep=4e5, pert=1.0):
    return str(math.ceil(2 * encounter.time_to_pericentre_hr(rp, vinf, start_sep, pert) * ps.TMAX_MARGIN))


def test_fill_defaults_does_nothing_for_the_default_form():
    res = ps.fill_defaults(ps.from_argv([]))
    assert res.notes == [] and ps.to_argv(res.fv) == []


def test_fill_defaults_settled_body():
    res = ps.fill_defaults(ps.from_argv(["--body-source", "settled"]))
    assert res.fv.plain["use_dem_fixed"] == "true"
    assert res.fv.plain["np_apophis"] == "2000"
    assert "np_apophis 2000" in res.notes


def test_fill_defaults_never_overwrites_typed_values():
    fv = ps.from_argv(["--body-source", "settled", "--np-apophis", "500", "--use-dem-fixed", "false"])
    res = ps.fill_defaults(fv)
    assert res.fv.plain["np_apophis"] == "500" and res.fv.plain["use_dem_fixed"] == "false"
    listed = ps.fill_defaults(ps.from_argv(["--body-source", "settled", "--np-apophis-list", "300", "600"]))
    assert listed.fv.plain["np_apophis"] == ""


def test_fill_defaults_hyperbola_switches_to_settled_and_fills_the_flyby():
    fv = ps.from_argv(["--encounter", "hyperbola"])
    res = ps.fill_defaults(fv)
    assert res.fv.plain["body_source"] == "settled"
    ax = res.fv.axes
    assert (ax["flyby_rp_km"].fixed, ax["flyby_vinf_kms"].fixed, ax["dtmax_hours"].fixed) == ("38000", "5.9", "0.5")
    assert ax["tmax_hours"].fixed == res.auto_tmax == _expected_tmax(38000, 5.9)
    assert fv.plain["body_source"] == "lattice"  # input not mutated


def test_auto_tmax_follows_rp_until_typed_over():
    first = ps.fill_defaults(ps.from_argv(["--encounter", "hyperbola"]))
    first.fv.axes["flyby_rp_km"].fixed = "8000"
    again = ps.fill_defaults(first.fv, last_auto_tmax=first.auto_tmax)
    assert again.fv.axes["tmax_hours"].fixed == _expected_tmax(8000, 5.9)
    again.fv.axes["tmax_hours"].fixed = "50"
    typed = ps.fill_defaults(again.fv, last_auto_tmax=again.auto_tmax)
    assert typed.fv.axes["tmax_hours"].fixed == "50" and typed.notes == []


def test_auto_tmax_uses_the_swept_range_and_skips_a_swept_tmax():
    fv = ps.from_argv(["--body-source", "settled", "--encounter", "hyperbola", "--flyby-rp-km-min", "8000",
                       "--flyby-rp-km-max", "30000", "--flyby-vinf-kms", "5"])
    res = ps.fill_defaults(fv)
    assert res.fv.axes["flyby_rp_km"].fixed == ""  # swept: not filled
    t = encounter._max_time_to_pericentre_hr(8000, 30000, 5, 5, 4e5, 1.0)
    assert res.fv.axes["tmax_hours"].fixed == str(math.ceil(2 * t * ps.TMAX_MARGIN))
    swept = ps.from_argv(["--body-source", "settled", "--encounter", "hyperbola", "--tmax-hours-min", "40",
                          "--tmax-hours-max", "50"])
    assert ps.fill_defaults(swept).fv.axes["tmax_hours"].fixed == ""


def test_auto_tmax_waits_for_numbers():
    fv = ps.from_argv(["--encounter", "hyperbola"])
    fv.axes["flyby_rp_km"].fixed = "8e"
    res = ps.fill_defaults(fv)
    assert res.fv.axes["tmax_hours"].fixed == "" and res.auto_tmax == ""


def test_fill_defaults_boulders_only_when_ticked():
    res = ps.fill_defaults(ps.from_argv(["--body-source", "settled", "--boulders"]))
    assert res.fv.plain["bond_sigma_pa"] == "1e5" and res.fv.axes["f_boulder"].fixed == "0.3"
    off = ps.fill_defaults(ps.from_argv(["--body-source", "settled"]))
    assert off.fv.plain["bond_sigma_pa"] == "" and off.fv.axes["f_boulder"].fixed == ""


def test_boulders_switch_a_lattice_body_to_settled():
    res = ps.fill_defaults(ps.from_argv(["--boulders"]))
    assert res.fv.plain["body_source"] == "settled" and res.fv.plain["bond_sigma_pa"] == "1e5"


def test_three_mode_choices_give_a_valid_single_run():
    res = ps.fill_defaults(ps.from_argv(["--encounter", "hyperbola", "--boulders"]))
    argv = ps.to_argv(ps.apply_mode(res.fv, "single"))
    assert ps.check(argv).error is None, ps.check(argv).error
