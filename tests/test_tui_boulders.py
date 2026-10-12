import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import run_mass_sobol_phantom as runner
import tui_run as tr
from textual.widgets import Checkbox, Input, Select


def _sweep(cli=(), actions=None):
    async def _scenario():
        app = tr.SobolTUIApp(runner.build_parser().parse_args(list(cli)))
        async with app.run_test() as pilot:
            if actions:
                actions(app)
            await pilot.pause()
            try:
                return app._build_argv(force_dry=True)
            except ValueError as exc:
                return exc
    return asyncio.run(_scenario())


def _arg(argv, flag):
    return argv[argv.index(flag) + 1] if flag in argv else None


def test_default_body_section_emits_nothing():
    argv = _sweep()
    for flag in ("--body-source", "--encounter", "--flyby-rp-km", "--settle-tdyn", "--relax-tdyn"):
        assert flag not in argv


def test_body_section_reaches_argv():
    def act(app):
        app.query_one("#body-source", Select).value = "settled"
        app.query_one("#encounter", Select).value = "hyperbola"
        app.query_one("#flyby-rp-km-min", Input).value = "20000"
        app.query_one("#flyby-rp-km-max", Input).value = "40000"
        app.query_one("#flyby-vinf-kms", Input).value = "6"
        app.query_one("#settle-tdyn", Input).value = "8"
    argv = _sweep(actions=act)
    assert _arg(argv, "--body-source") == "settled" and _arg(argv, "--encounter") == "hyperbola"
    assert _arg(argv, "--flyby-rp-km-min") == "20000" and _arg(argv, "--flyby-rp-km-max") == "40000"
    assert _arg(argv, "--flyby-vinf-kms") == "6" and _arg(argv, "--settle-tdyn") == "8"


def test_body_section_prefills_from_cli():
    cli = ["--body-source", "settled", "--relax-tdyn", "0.3"]
    argv = _sweep(cli=cli)
    assert _arg(argv, "--body-source") == "settled" and _arg(argv, "--relax-tdyn") == "0.3"


def test_bad_number_raises():
    def act(app):
        app.query_one("#flyby-rp-km", Input).value = "far"
    assert isinstance(_sweep(actions=act), ValueError)


def _boulder_act(**fields):
    def act(app):
        app.query_one("#boulders", Checkbox).value = True
        for wid, val in fields.items():
            w = app.query_one(f"#{wid}")
            w.value = val
    return act


def test_boulders_reach_argv():
    argv = _sweep(actions=_boulder_act(**{"f-boulder-min": "0.1", "f-boulder-max": "0.5", "rb-max": "5",
                                          "bond-sigma-pa": "10", "boulder-placed": "binary", "lobe-ratio": "0.6"}))
    assert "--boulders" in argv
    assert _arg(argv, "--f-boulder-min") == "0.1" and _arg(argv, "--f-boulder-max") == "0.5"
    assert _arg(argv, "--rb-max") == "5" and _arg(argv, "--bond-sigma-pa") == "10"
    assert _arg(argv, "--boulder-placed") == "binary" and _arg(argv, "--lobe-ratio") == "0.6"


def test_boulders_off_emits_nothing():
    argv = _sweep()
    assert not any(a.startswith(("--boulder", "--f-boulder", "--rb-", "--bond-", "--kb-cgs", "--lobe"))
                   for a in argv)


def test_sweep_tui_boulder_fields_need_checkbox():
    def act(app):
        app.query_one("#f-boulder", Input).value = "0.2"
    out = _sweep(actions=act)
    assert isinstance(out, ValueError) and "boulders" in str(out)


def test_boulder_prefill_from_cli():
    cli = ["--boulders", "--bond-sigma-pa", "10", "--boulder-shape", "grown"]
    argv = _sweep(cli=cli)
    assert "--boulders" in argv and _arg(argv, "--bond-sigma-pa") == "10"
    assert _arg(argv, "--boulder-shape") == "grown"


def test_boulder_prefill_keeps_precision():
    cli = ["--boulders", "--bond-sigma-pa", "1234567", "--kb-cgs", "123456789"]
    argv = _sweep(cli=cli)
    assert _arg(argv, "--bond-sigma-pa") == "1234567"
    assert _arg(argv, "--kb-cgs") == "123456789"


def _sweep_disabled(cli=(), actions=None):
    async def _scenario():
        app = tr.SobolTUIApp(runner.build_parser().parse_args(list(cli)))
        async with app.run_test() as pilot:
            if actions:
                actions(app)
            await pilot.pause()
            ids = ("flyby-rp-km", "boulders", "f-boulder", "lobe-ratio", "boulder-jagged")
            return {w: app.query_one(f"#{w}").disabled for w in ids}
    return asyncio.run(_scenario())


def test_sweep_gates_follow_body_source():
    d = _sweep_disabled()
    assert d["flyby-rp-km"] and d["boulders"] and d["f-boulder"]
    d = _sweep_disabled(cli=["--body-source", "settled", "--encounter", "hyperbola"])
    assert not d["flyby-rp-km"] and not d["boulders"] and d["f-boulder"]
    d = _sweep_disabled(cli=["--body-source", "settled", "--boulders", "--bond-sigma-pa", "10",
                             "--boulder-placed", "binary"])
    assert not d["f-boulder"] and not d["lobe-ratio"] and d["boulder-jagged"]


import boulders as bd
import tui_sim_render as tsr


def _sim(selects=None, checks=None, **fields):
    async def _scenario():
        app = tsr.SimRenderTUIApp()
        async with app.run_test() as pilot:
            for wid, val in (selects or {}).items():
                app.query_one(f"#{wid}", Select).value = val
            for wid, val in (checks or {}).items():
                app.query_one(f"#{wid}", Checkbox).value = val
            for wid, val in fields.items():
                app.query_one(f"#{wid}", Input).value = val
            await pilot.pause()
            try:
                return app._build_sim_params(False)
            except ValueError as exc:
                return exc
    return asyncio.run(_scenario())


def test_sim_boulders_off_by_default():
    assert _sim().sample.boulders is None


def test_sim_boulders_reach_sample():
    p = _sim(selects={"body-source": "settled", "boulder-placed": "binary"}, checks={"boulders": True},
             **{"f-boulder": "0.2", "bond-sigma-pa": "10", "lobe-ratio": "0.6"})
    assert p.sample.boulders == bd.BoulderSpec(f_boulder=0.2, bond_sigma_pa=10.0, placed="binary", lobe_ratio=0.6,
                                               seed=runner.DEFAULT_SEED)


def test_sim_blank_seed_matches_cli_default():
    p = _sim(selects={"body-source": "settled"}, checks={"boulders": True}, **{"bond-sigma-pa": "10"})
    cli = runner.parse_args(["--boulders", "--bond-sigma-pa", "10"])
    assert p.sample.boulders.seed == bd.spec_from_args(cli).seed == runner.DEFAULT_SEED
    p = _sim(selects={"body-source": "settled"}, checks={"boulders": True},
             **{"bond-sigma-pa": "10", "boulder-seed": "7"})
    assert p.sample.boulders.seed == 7


def _sim_disabled(selects=None, checks=None):
    """Which sim-TUI body/boulder widgets are disabled after the given selections."""
    async def _scenario():
        app = tsr.SimRenderTUIApp()
        async with app.run_test() as pilot:
            for wid, val in (selects or {}).items():
                app.query_one(f"#{wid}", Select).value = val
            await pilot.pause()
            for wid, val in (checks or {}).items():
                app.query_one(f"#{wid}", Checkbox).value = val
            await pilot.pause()
            ids = ("flyby-rp", "boulders", "f-boulder", "lobe-ratio", "boulder-axis-min")
            return {w: app.query_one(f"#{w}").disabled for w in ids}
    return asyncio.run(_scenario())


def test_sim_gates_follow_body_source():
    d = _sim_disabled()
    assert d["flyby-rp"] and d["boulders"] and d["f-boulder"]
    d = _sim_disabled(selects={"body-source": "settled", "encounter": "hyperbola"})
    assert not d["flyby-rp"] and not d["boulders"] and d["f-boulder"]
    d = _sim_disabled(selects={"body-source": "settled", "boulder-placed": "binary"}, checks={"boulders": True})
    assert not d["f-boulder"] and not d["lobe-ratio"] and d["boulder-axis-min"]


def _sim_switch_body(after, untick=False):
    """Tick boulders on a settled body, switch body_source to `after`, optionally untick."""
    async def _scenario():
        app = tsr.SimRenderTUIApp()
        async with app.run_test() as pilot:
            app.query_one("#body-source", Select).value = "settled"
            await pilot.pause()
            app.query_one("#boulders", Checkbox).value = True
            app.query_one("#bond-sigma-pa", Input).value = "10"
            await pilot.pause()
            app.query_one("#body-source", Select).value = after
            await pilot.pause()
            box = app.query_one("#boulders", Checkbox)
            if untick:
                box.toggle()
                await pilot.pause()
            try:
                built = app._build_sim_params(False)
            except ValueError as exc:
                built = exc
            return box.disabled, box.value, built
    return asyncio.run(_scenario())


def test_sim_ticked_boulders_stay_untickable_off_settled():
    # Greyed-out + ticked could not be unticked by hand; now the box stays enabled until unticked.
    disabled, ticked, built = _sim_switch_body("lattice")
    assert not disabled and ticked and isinstance(built, ValueError) and "settled" in str(built)
    disabled, ticked, built = _sim_switch_body("lattice", untick=True)
    assert disabled and not ticked and built.sample.boulders is None


def test_sweep_ticked_boulders_stay_untickable_off_settled():
    async def _scenario():
        app = tr.SobolTUIApp(runner.build_parser().parse_args([]))
        async with app.run_test() as pilot:
            app.query_one("#body-source", Select).value = "settled"
            await pilot.pause()
            app.query_one("#boulders", Checkbox).value = True
            await pilot.pause()
            app.query_one("#body-source", Select).value = "lattice"
            await pilot.pause()
            box = app.query_one("#boulders", Checkbox)
            before = box.disabled
            box.toggle()
            await pilot.pause()
            return before, box.disabled, app._build_argv(force_dry=True)
    before, after, argv = asyncio.run(_scenario())
    assert not before and after and "--boulders" not in argv


@pytest.mark.parametrize("selects,fields,match", [
    ({}, {"bond-sigma-pa": "10"}, "settled"),                                     # lattice body
    ({"body-source": "settled"}, {}, "bond_sigma_pa"),                            # sigma missing
    ({"body-source": "settled"}, {"bond-sigma-pa": "10", "f-boulder": "2"}, "f_boulder"),
])
def test_sim_boulders_rejects(selects, fields, match):
    out = _sim(selects=selects, checks={"boulders": True}, **fields)
    assert isinstance(out, ValueError) and match in str(out)
