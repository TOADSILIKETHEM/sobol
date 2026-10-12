import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import boulders as bd
import run_mass_sobol_phantom as runner

# scale_vel axis: the runner rejects a batch with no Sobol dimension before boulder validation runs
SETTLED = ["--use-dem-fixed", "true", "--np-apophis", "300", "--body-source", "settled",
           "--tmax-hours", "12", "--dtmax-hours", "1", "--spin-period-fixed", "30",
           "--scale-vel-min", "0.9", "--scale-vel-max", "1.1"]


def _args(*extra):
    a = runner.parse_args(SETTLED + list(extra))
    runner.validate_args(a)
    return a


def _err(*extra, base=SETTLED):
    a = runner.parse_args(list(base) + list(extra))
    with pytest.raises(ValueError) as ei:
        runner.validate_args(a)
    return str(ei.value)


def test_fixed_boulders_validate_and_reach_samples():
    a = _args("--boulders", "--f-boulder", "0.2", "--rb-max", "5", "--bond-sigma-pa", "10",
              "--boulder-placed", "binary", "--lobe-ratio", "0.6", "--num-samples", "2")
    s = runner.RunSample()
    runner._apply_fixed_run_sample_overrides(s, a)
    assert s.boulders == bd.BoulderSpec(f_boulder=0.2, rb_max=5.0, bond_sigma_pa=10.0, placed="binary",
                                        lobe_ratio=0.6, seed=a.seed)
    assert (s.f_boulder, s.rb_max) == (0.2, 5.0)


def test_no_boulders_leaves_sample_untouched():
    a = _args("--num-samples", "1")
    s = runner.RunSample()
    runner._apply_fixed_run_sample_overrides(s, a)
    assert (s.boulders, s.f_boulder, s.rb_max, s.phantomclumps_bin) == (None, None, None, None)


def test_boulder_flags_need_boulders():
    assert "--boulders" in _err("--f-boulder", "0.2", "--num-samples", "1")


def test_boulders_need_settled():
    lattice = ["--use-dem-fixed", "true", "--np-apophis", "300", "--num-samples", "1",
               "--spin-period-fixed", "30", "--scale-vel-min", "0.9", "--scale-vel-max", "1.1"]
    assert "settled" in _err("--boulders", "--bond-sigma-pa", "10", base=lattice)


def test_boulders_need_sigma():
    assert "--bond-sigma-pa" in _err("--boulders", "--num-samples", "1")


@pytest.mark.parametrize("extra,match", [
    (["--f-boulder-min", "0.1"], "--f-boulder"),                              # one bound only
    (["--f-boulder", "0.2", "--f-boulder-min", "0.1", "--f-boulder-max", "0.3"], "mutually exclusive"),
    (["--rb-max-min", "6", "--rb-max-max", "5"], "min < max"),
    (["--f-boulder-min", "0.1", "--f-boulder-max", "1.5"], "f_boulder"),      # hi bound out of range
    (["--rb-min", "3", "--rb-max-min", "2", "--rb-max-max", "5"], "rb_max"),  # lo bound < rb_min
    (["--lobe-ratio", "0.5"], "binary"),
    (["--boulder-placed", "file"], "file"),
])
def test_bad_boulder_args(extra, match):
    assert match in _err("--boulders", "--bond-sigma-pa", "10", "--num-samples", "4", *extra)


def test_axes_rejected_in_np_list_mode():
    base = ["--use-dem-fixed", "true", "--np-apophis-list", "300", "--body-source", "settled",
            "--tmax-hours", "12", "--dtmax-hours", "1", "--spin-period-fixed", "30"]
    msg = _err("--boulders", "--bond-sigma-pa", "10", "--f-boulder-min", "0.1", "--f-boulder-max", "0.4",
               base=base)
    assert "np-apophis-list" in msg


def test_sigma_zero_warns(capsys):
    _args("--boulders", "--bond-sigma-pa", "0", "--num-samples", "1")
    assert "stretch" in capsys.readouterr().err


def test_boulder_seed_defaults_to_batch_seed():
    a = _args("--boulders", "--bond-sigma-pa", "10", "--num-samples", "1", "--seed", "7")
    assert bd.spec_from_args(a).seed == 7
    a = _args("--boulders", "--bond-sigma-pa", "10", "--num-samples", "1", "--boulder-seed", "99")
    assert bd.spec_from_args(a).seed == 99


def test_boulder_file_is_made_absolute(tmp_path, monkeypatch):
    f = tmp_path / "b.txt"
    f.write_text("0 0 0 0.05 0.05 0.05\n")
    monkeypatch.chdir(tmp_path)
    a = _args("--boulders", "--bond-sigma-pa", "10", "--num-samples", "1", "--boulder-placed", "file",
              "--boulder-file", "b.txt")
    assert bd.spec_from_args(a).placed_file == str(f)


NP_LIST_SETTLED = ["--use-dem-fixed", "true", "--np-apophis-list", "300", "--body-source", "settled",
                   "--spin-period-fixed", "30", "--tmax-hours", "12", "--dtmax-hours", "1", "--num-samples", "1"]
HYP = ["--use-dem-fixed", "true", "--np-apophis", "300", "--body-source", "settled", "--encounter",
       "hyperbola", "--flyby-rp-km-min", "20000", "--flyby-rp-km-max", "40000", "--flyby-vinf-kms-min", "5",
       "--flyby-vinf-kms-max", "7", "--tmax-hours", "48", "--dtmax-hours", "0.5", "--num-samples", "16"]


def _parsed(argv):
    a = runner.parse_args(argv)
    runner.validate_args(a)
    return a


def test_no_boulder_regression():
    # values captured 2026-10-06 before this change
    a = _parsed(NP_LIST_SETTLED)
    assert runner.build_auto_batch_sweep_slug(a, 200) == "np300_tmx12_dtmx1_settled"
    assert (runner.count_dimensions(a), runner.sample_column_order(a)) == (0, ["np_apophis", "np_kept"])
    assert "boulder" not in runner.canonical_sweep_descriptor(a)
    a = _parsed(HYP)
    assert runner.build_auto_batch_sweep_slug(a, 200) == "n16_s42_rp20000-40000_vinf5-7_tmx48_dtmx0.5_hyp_settled"
    assert (runner.count_dimensions(a), runner.sample_column_order(a)) == (
        2, ["flyby_rp_km", "flyby_vinf_kms", "np_kept"])


def test_boulder_axes_are_dimensions():
    a = _parsed(HYP + ["--boulders", "--bond-sigma-pa", "10", "--f-boulder-min", "0.1", "--f-boulder-max",
                       "0.5", "--rb-max-min", "2", "--rb-max-max", "6"])
    assert runner.count_dimensions(a) == 4
    order = runner.sample_column_order(a)
    assert order[:4] == ["flyby_rp_km", "flyby_vinf_kms", "f_boulder", "rb_max"]
    assert order[-5:] == ["np_kept", "f_boulder_actual", "n_boulders", "boulder_placed", "bond_sigma_pa"]
    samples = runner.build_run_samples(16, a)
    assert all(0.1 <= s.f_boulder <= 0.5 and 2 <= s.rb_max <= 6 for s in samples)
    assert all(s.boulders is not None for s in samples)
    prob = runner.build_salib_problem(a)
    assert prob["names"][2:] == ["f_boulder", "rb_max"] and prob["bounds"][2:] == [[0.1, 0.5], [2.0, 6.0]]
    s = runner.run_sample_from_salib_row([30000, 6, 0.3, 4], a)
    assert (s.f_boulder, s.rb_max) == (0.3, 4.0)


def test_fixed_boulders_add_columns_not_dimensions():
    a = _parsed(NP_LIST_SETTLED + ["--boulders", "--bond-sigma-pa", "10", "--f-boulder", "0.2"])
    assert runner.count_dimensions(a) == 0
    assert runner.sample_column_order(a) == ["np_apophis", "np_kept", "f_boulder", "rb_max", "f_boulder_actual",
                                             "n_boulders", "boulder_placed", "bond_sigma_pa"]


def test_slug_distinguishes_placed():
    common = NP_LIST_SETTLED + ["--boulders", "--bond-sigma-pa", "10"]
    none = runner.build_auto_batch_sweep_slug(_parsed(common), 200)
    binary = runner.build_auto_batch_sweep_slug(_parsed(common + ["--boulder-placed", "binary"]), 200)
    assert none != binary and none.endswith("bldnone_sig10") and binary.endswith("bldbinary_sig10")
    d1 = runner.canonical_sweep_descriptor(_parsed(common))
    d2 = runner.canonical_sweep_descriptor(_parsed(common + ["--boulder-placed", "binary"]))
    assert d1 != d2


def test_slug_has_fixed_boulder_values():
    common = NP_LIST_SETTLED + ["--boulders", "--bond-sigma-pa", "10"]
    base = runner.build_auto_batch_sweep_slug(_parsed(common), 200)
    fixed = runner.build_auto_batch_sweep_slug(
        _parsed(common + ["--f-boulder", "0.2", "--rb-max", "5"]), 200)
    assert fixed == base + "_fb0.2_rbx5"
    other = runner.build_auto_batch_sweep_slug(_parsed(common + ["--f-boulder", "0.3", "--rb-max", "5"]), 200)
    assert other != fixed
    swept = runner.build_auto_batch_sweep_slug(_parsed(HYP + common[len(NP_LIST_SETTLED):]
                                                       + ["--f-boulder-min", "0.1", "--f-boulder-max", "0.5"]), 200)
    assert "fb0.1-0.5" in swept and "_fb0.1_" not in swept


# Task 6: wizard gate and Analysis inputs
import argparse
import interactive_run_mass_sobol as wiz


def test_wizard_knows_every_boulder_dest():
    for d in bd.BOULDER_ARG_DESTS | {"boulders"}:
        assert wiz.DEST_TO_SECTION[d] == "Boulders"
        assert d in wiz.INTERACTIVE_BRIEF


def test_wizard_boulder_gate():
    lattice = argparse.Namespace(body_source="lattice", boulders=False)
    settled_off = argparse.Namespace(body_source="settled", boulders=False)
    settled_on = argparse.Namespace(body_source="settled", boulders=True)
    assert wiz._skip_boulder_dest("boulders", lattice)
    assert not wiz._skip_boulder_dest("boulders", settled_off)
    assert wiz._skip_boulder_dest("f_boulder", settled_off)
    assert not wiz._skip_boulder_dest("f_boulder", settled_on)
    assert not wiz._skip_boulder_dest("np_apophis", lattice)


def test_analysis_knows_boulder_inputs():
    sys.path.insert(0, str(Path(__file__).parent.parent / "Analysis"))
    import Analysis
    assert {"f_boulder", "rb_max"} <= set(Analysis.INPUT_CANDIDATES)


def test_samples_csv_fills_known_boulder_columns(tmp_path):
    a = _parsed(NP_LIST_SETTLED + ["--boulders", "--bond-sigma-pa", "10", "--boulder-placed", "binary"])
    s = runner.RunSample(boulders=bd.spec_from_args(a), rb_max=5.0)
    path = tmp_path / "samples.csv"
    runner.write_samples_csv(path, [s], runner.sample_column_order(a))
    import csv
    row = next(csv.DictReader(path.open()))
    assert (row["f_boulder"], row["rb_max"], row["boulder_placed"], row["bond_sigma_pa"]) == ("0.3", "5", "binary", "10")
    assert row["f_boulder_actual"] == row["n_boulders"] == ""  # results, filled in the outputs CSV
