import dataclasses
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import boulders as bd
import resume_batch as rb
import run_mass_sobol_phantom as runner

BASE = ["--use-dem-fixed", "true", "--np-apophis-list", "300", "--body-source", "settled",
        "--spin-period-fixed", "30", "--tmax-hours", "12", "--dtmax-hours", "1", "--num-samples", "1"]
BOULDERS = ["--boulders", "--bond-sigma-pa", "10", "--boulder-placed", "binary"]


def _args(*extra):
    a = runner.parse_args(BASE + list(extra))
    runner.validate_args(a)
    return a


def _batch(tmp_path, args, spec_args=None):
    """Batch dir whose samples CSV matches `args`; boulders.json from `spec_args` when given."""
    (tmp_path / "sobol_mass_samples.csv").write_text(
        ",".join(["run_id"] + runner.sample_column_order(args)) + "\n1,300,,,,,,,\n")
    if spec_args is not None:
        run = tmp_path / "run_0001"
        run.mkdir()
        spec = dataclasses.asdict(bd.spec_from_args(spec_args))
        (run / "boulders.json").write_text(json.dumps({"spec": spec, "result": {"n_boulders": 3}}))
    return tmp_path / "sobol_mass_samples.csv"


def test_matching_boulder_batch_passes(tmp_path):
    a = _args(*BOULDERS)
    rb.check_resume_boulders(tmp_path, a, _batch(tmp_path, a, a))


def test_matching_plain_batch_passes(tmp_path):
    a = _args()
    rb.check_resume_boulders(tmp_path, a, _batch(tmp_path, a))


def test_boulder_batch_without_json_yet_passes(tmp_path):
    a = _args(*BOULDERS)
    rb.check_resume_boulders(tmp_path, a, _batch(tmp_path, a))


def test_non_boulder_batch_header_not_checked(tmp_path):
    a = _args()
    csv_path = tmp_path / "sobol_mass_samples.csv"
    csv_path.write_text("run_id,kc_cgs\n1,1e7\n")   # legacy header, differs from sample_column_order(a)
    rb.check_resume_boulders(tmp_path, a, csv_path)


def test_header_mismatch_refused(tmp_path):
    csv_path = _batch(tmp_path, _args())   # plain header, but the flags turn boulders on
    with pytest.raises(RuntimeError, match="columns differ"):
        rb.check_resume_boulders(tmp_path, _args(*BOULDERS), csv_path)


def test_boulders_off_but_json_present_refused(tmp_path):
    plain = _args()
    csv_path = _batch(tmp_path, plain, _args(*BOULDERS))   # header plain, but a run made boulders
    with pytest.raises(RuntimeError, match="--boulders is not set"):
        rb.check_resume_boulders(tmp_path, plain, csv_path)


@pytest.mark.parametrize("changed", [
    ["--boulder-placed", "none"],
    ["--bond-sigma-pa", "20"],
    ["--boulder-seed", "7"],
])
def test_changed_boulder_flag_refused(tmp_path, changed):
    orig = _args(*BOULDERS)
    csv_path = _batch(tmp_path, orig, orig)
    new_flags = ["--boulders", "--bond-sigma-pa", "10", "--boulder-placed", "binary"]
    for i in range(0, len(changed), 2):
        if changed[i] in new_flags:
            j = new_flags.index(changed[i])
            new_flags[j + 1] = changed[i + 1]
        else:
            new_flags += changed[i:i + 2]
    with pytest.raises(RuntimeError, match="boulder flags differ"):
        rb.check_resume_boulders(tmp_path, _args(*new_flags), csv_path)


def test_axis_values_may_differ(tmp_path):
    orig = _args(*BOULDERS, "--f-boulder", "0.2", "--rb-max", "5")
    csv_path = _batch(tmp_path, orig, orig)
    again = _args(*BOULDERS, "--f-boulder", "0.2", "--rb-max", "5")
    again.f_boulder, again.rb_max = 0.4, 6.0   # axis fields are per-run, so ignored by the guard
    rb.check_resume_boulders(tmp_path, again, csv_path)


def test_main_exits_nonzero_on_mismatch(tmp_path, monkeypatch, capsys):
    orig = _args(*BOULDERS)
    _batch(tmp_path, orig, orig)
    monkeypatch.setattr(sys, "argv", ["resume_batch.py", "--batch-dir", str(tmp_path)] + BASE)  # no --boulders
    assert rb.main() == 1
    assert "--boulders is not set" in capsys.readouterr().err


def test_main_exits_nonzero_on_header_mismatch(tmp_path, monkeypatch, capsys):
    _batch(tmp_path, _args())
    monkeypatch.setattr(sys, "argv", ["resume_batch.py", "--batch-dir", str(tmp_path)] + BASE + BOULDERS)
    assert rb.main() == 1
    assert "columns differ" in capsys.readouterr().err
