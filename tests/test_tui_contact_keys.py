"""Mia's 1fd4ceb88 contact keys (mu_s, mu_r, mu_t, epsilon_t_dem, cohesion_pa) in both TUIs."""
import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import run_mass_sobol_phantom as runner
import tui_run as tr
import tui_sim_render as tsr
from textual.widgets import Checkbox, Input

NEW_AXES = [("mu-s", "--mu-s"), ("mu-r", "--mu-r"), ("mu-t", "--mu-t"),
            ("eps-t", "--eps-t"), ("coh-pa", "--coh-pa")]


def _sweep(cli=(), actions=None):
    """Run the sweep TUI headless with defaults parsed from cli; return (app, argv or exc)."""
    async def _scenario():
        app = tr.SobolTUIApp(runner.build_parser().parse_args(list(cli)))
        async with app.run_test() as pilot:
            if actions:
                actions(app)
            await pilot.pause()
            try:
                return app, app._build_argv(force_dry=True)
            except ValueError as exc:
                return app, exc
    return asyncio.run(_scenario())


def _arg(argv, flag):
    return argv[argv.index(flag) + 1] if flag in argv else None


@pytest.mark.parametrize("css,flag", NEW_AXES)
def test_sweep_tui_new_axis_reaches_argv(css, flag):
    def act(app):
        app.query_one(f"#vary-{css}", Checkbox).value = True
        app.query_one(f"#{css}-min", Input).value = "0.1"
        app.query_one(f"#{css}-max", Input).value = "0.9"
    _, argv = _sweep(actions=act)
    assert _arg(argv, f"{flag}-min") == "0.1" and _arg(argv, f"{flag}-max") == "0.9"


def test_sweep_tui_fixed_mu_s_reaches_argv():
    def act(app):
        app.query_one("#mu-s-fixed", Input).value = "0.4"
    _, argv = _sweep(actions=act)
    assert _arg(argv, "--mu-s") == "0.4"


def test_sweep_tui_blank_mu_s_emits_nothing():
    # runner then pins DEFAULT_MU_S
    _, argv = _sweep()
    assert "--mu-s" not in argv and "--mu-s-min" not in argv


def test_sweep_tui_prefill_round_trips_cli_mu_s():
    app, argv = _sweep(cli=["--mu-s", "0.7"])
    assert app is not None and _arg(argv, "--mu-s") == "0.7"


def test_sweep_tui_fixed_and_vary_mu_s_are_exclusive():
    async def _scenario():
        app = tr.SobolTUIApp(runner.build_parser().parse_args([]))
        async with app.run_test() as pilot:
            app.query_one("#vary-mu-s", Checkbox).value = True
            await pilot.pause()
            fixed_off = app.query_one("#mu-s-fixed", Input).disabled
            app.query_one("#vary-mu-s", Checkbox).value = False
            app.query_one("#mu-s-fixed", Input).value = "0.3"
            await pilot.pause()
            vary_off = app.query_one("#vary-mu-s", Checkbox).disabled
            return fixed_off, vary_off
    assert asyncio.run(_scenario()) == (True, True)


def _sim(**fields):
    async def _scenario():
        app = tsr.SimRenderTUIApp()
        async with app.run_test():
            for wid, val in fields.items():
                app.query_one(f"#{wid}", tsr.Input).value = val
            try:
                return app._build_sim_params(False)
            except ValueError as exc:
                return exc
    return asyncio.run(_scenario())


def test_sim_tui_blank_contact_fields_pin_mu_s_only():
    s = _sim().sample
    assert s.mu_s == runner.DEFAULT_MU_S
    assert (s.mu_r, s.mu_t, s.epsilon_t_dem, s.cohesion_pa) == (None, None, None, None)


def test_sim_tui_contact_fields_reach_sample():
    s = _sim(**{"mu-s": "0.5", "mu-r": "1.05", "mu-t": "1.3", "eps-t": "0.4", "cohesion-pa": "25"}).sample
    assert (s.mu_s, s.mu_r, s.mu_t, s.epsilon_t_dem, s.cohesion_pa) == (0.5, 1.05, 1.3, 0.4, 25.0)


@pytest.mark.parametrize("fields,match", [
    ({"mu-r": "1.05"}, "mu_s"),                       # rolling friction needs sliding friction
    ({"mu-s": "0", "mu-t": "1.3"}, "mu_s"),
    ({"cohesion-pa": "25", "kt-cgs": "1e7"}, "cohesion"),  # phantom fatal: one cohesion model
    ({"eps-t": "0"}, "epsilon_t_dem"),                # read_inopt min=tiny(0.)
    ({"mu-s": "-0.1"}, "mu_s"),
])
def test_sim_tui_rejects_bad_contact_fields(fields, match):
    out = _sim(**fields)
    assert isinstance(out, ValueError) and match in str(out)


def test_sample_contact_check_needs_friction_binary(tmp_path):
    old = tmp_path / "old"
    old.write_bytes(b"no friction here")
    new = tmp_path / "new"
    new.write_bytes(runner.FRICTION_BINARY_MARKER)
    with pytest.raises(RuntimeError, match="friction"):
        runner.check_sample_supports_contact_keys(runner.RunSample(mu_s=0.5), old)
    runner.check_sample_supports_contact_keys(runner.RunSample(mu_s=0.5), new)
    runner.check_sample_supports_contact_keys(runner.RunSample(mu_s=0.0), old)  # default pin is fine
