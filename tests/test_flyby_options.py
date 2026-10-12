"""render_dumps --encounter: render other people's phantomflyby runs (lone Earth
sink, no Sun) with Earth placed and spun, as sobol's own hyperbola runs are."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import render_dumps as rd  # noqa: E402


def _dumps(d: Path, prefix="sobol", n=3):
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (d / f"{prefix}_{i:05d}").write_bytes(b"x")
    return d


# -- render_dumps --encounter --

def test_encounter_hyperbola_writes_encounter_json(tmp_path):
    run = _dumps(tmp_path / "fly", prefix="fly")
    out = rd.stage_run(run, tmp_path / "stage", "fly", "particle", True, 10, encounter="hyperbola")
    assert json.loads((out / "encounter.json").read_text())["mode"] == "hyperbola"


def test_encounter_hyperbola_overrides_a_carried_ephemeris_file(tmp_path):
    run = _dumps(tmp_path / "run")
    (run / "encounter.json").write_text('{"mode": "ephemeris"}')
    out = rd.stage_run(run, tmp_path / "stage", "sobol", "particle", True, 10, encounter="hyperbola")
    assert json.loads((out / "encounter.json").read_text())["mode"] == "hyperbola"
    assert json.loads((run / "encounter.json").read_text())["mode"] == "ephemeris"  # source untouched


def test_encounter_on_a_ready_run_stages_it_and_keeps_its_setup(tmp_path):
    run = _dumps(tmp_path / "run")
    (run / "sobol.setup").write_text("apophis_only = F\nnp_apophis = 7\nuse_dem = T\nuse_dem_as_sinks = F\n")
    out = rd.stage_run(run, tmp_path / "stage", "sobol", None, None, None, encounter="hyperbola")
    assert out != run
    assert "np_apophis = 7" in (out / "sobol.setup").read_text()
    assert json.loads((out / "encounter.json").read_text())["mode"] == "hyperbola"
    assert not (run / "encounter.json").exists()


def test_ready_run_without_encounter_still_used_in_place(tmp_path):
    run = _dumps(tmp_path / "run")
    (run / "sobol.setup").write_text("use_dem = T\nuse_dem_as_sinks = F\n")
    assert rd.stage_run(run, tmp_path / "stage", "sobol", None, None, None) == run


def _fake(monkeypatch, seen, rows):
    """Preprocess fake that writes one bodies CSV with these body names."""
    monkeypatch.setattr(rd.t, "check_blender_exe", lambda *a: None)

    def pre(cmd, targets, on_line=None):
        bodies = Path(seen["bodies_dir"]())
        bodies.mkdir(parents=True, exist_ok=True)
        (bodies / "sobol_00000.csv").write_text(
            "name,time_s,x_km,y_km,z_km\n" + "".join(f"{n},0,0,0,0\n" for n in rows))
        return dict(targets)

    class Ok:
        ok = True

    def ren(spec, ctx, on_line=None):
        seen["ctx"] = ctx
        return Ok

    monkeypatch.setattr(rd.t, "run_dump_preprocess_stage", pre)
    monkeypatch.setattr(rd.t, "run_render_path", ren)
    monkeypatch.setattr(rd.t, "format_path_result", lambda r: "ok")


def _main(tmp_path, extra):
    run = _dumps(tmp_path / "fly", prefix="fly")
    return rd.main([str(run), "--prefix", "fly", "--dem-model", "particle", "--np-apophis", "10",
                    "--apophis-only", "--name", "demo", "--out-root", str(tmp_path / "out"), *extra])


def _bodies(tmp_path):
    return lambda: tmp_path / "out" / "demo" / "run_0001_bodies_output"


def test_lone_sink_named_sun_stops_before_rendering(tmp_path, monkeypatch, capsys):
    seen = {"bodies_dir": _bodies(tmp_path)}
    _fake(monkeypatch, seen, ["Sun", "Apophis"])
    assert _main(tmp_path, []) == 1
    assert "ctx" not in seen
    assert "--encounter hyperbola" in capsys.readouterr().err


def test_lone_sun_sink_allowed_with_explicit_ephemeris(tmp_path, monkeypatch):
    seen = {"bodies_dir": _bodies(tmp_path)}
    _fake(monkeypatch, seen, ["Sun", "Apophis"])
    assert _main(tmp_path, ["--encounter", "ephemeris"]) == 0
    assert "ctx" in seen


def test_solar_system_rows_pass_the_guard(tmp_path, monkeypatch):
    seen = {"bodies_dir": _bodies(tmp_path)}
    _fake(monkeypatch, seen, ["Sun", "Earth", "Moon", "Apophis"])
    assert _main(tmp_path, []) == 0


def test_apophis_only_rows_pass_the_guard(tmp_path, monkeypatch):
    seen = {"bodies_dir": _bodies(tmp_path)}
    _fake(monkeypatch, seen, ["Apophis"])
    assert _main(tmp_path, []) == 0


def test_hyperbola_run_renders(tmp_path, monkeypatch):
    seen = {"bodies_dir": _bodies(tmp_path)}
    _fake(monkeypatch, seen, ["Earth", "Apophis"])
    assert _main(tmp_path, ["--encounter", "hyperbola"]) == 0
    assert "ctx" in seen
