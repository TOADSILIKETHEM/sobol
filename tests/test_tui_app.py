"""tui_app widgets driven through Textual's pilot."""
import asyncio
import os
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from textual.app import App
from textual.widgets import Button, Checkbox, Input, RadioButton, Select

import params_schema as ps
from tui_app.simulate import SimulateForm, axis_id, field_id, section_id


class _Host(App):
    def __init__(self, fv, mode="sweep"):
        super().__init__()
        self.fv, self.mode, self.notes = fv, mode, []

    def compose(self):
        yield SimulateForm(self.fv, mode=self.mode, id="sim")

    def on_simulate_form_defaults_filled(self, message) -> None:
        self.notes.extend(message.notes)


def _drive(fv, actions, mode="sweep"):
    async def scenario():
        app = _Host(fv, mode)
        async with app.run_test(size=(160, 60)) as pilot:
            form = app.query_one(SimulateForm)
            out = await actions(app, form, pilot)
            await pilot.pause()
            return out
    return asyncio.run(scenario())


def test_form_reads_back_what_it_was_given():
    fv = ps.from_argv(["--use-dem-fixed", "true", "--np-apophis", "300", "--mu-s-min", "0.2", "--mu-s-max", "0.8",
                       "--spin-torque-align-min", "45", "--spin-torque-align-max", "45"])

    async def act(app, form, pilot):
        return form.read()

    out = _drive(fv, act)
    assert ps.to_argv(out) == ps.to_argv(fv)


def test_single_mode_hides_sampling_and_sweep_toggles():
    async def act(app, form, pilot):
        app.query_one("#mode-single", RadioButton).value = True
        await pilot.pause()
        return (app.query_one("#" + section_id("Sampling")).display,
                app.query_one("#row-" + field_id("num_samples")).display,
                app.query_one("#" + axis_id("mu_s", "sweep")).display,
                app.query_one("#omp", Input).value, form.mode)

    assert _drive(ps.from_argv([]), act) == (False, False, False, "4", "single")


def test_boulder_rows_follow_gates():
    async def act(app, form, pilot):
        before = app.query_one("#" + field_id("bond_sigma_pa")).disabled
        app.query_one("#" + field_id("body_source"), Select).value = "settled"
        app.query_one("#" + field_id("boulders"), Checkbox).value = True
        await pilot.pause()
        return before, app.query_one("#" + field_id("bond_sigma_pa")).disabled

    assert _drive(ps.from_argv([]), act) == (True, False)


def test_mode_choices_fill_defaults_into_the_form():
    async def act(app, form, pilot):
        app.query_one("#mode-single", RadioButton).value = True
        app.query_one("#" + field_id("encounter"), Select).value = "hyperbola"
        await pilot.pause()
        app.query_one("#" + field_id("boulders"), Checkbox).value = True
        await pilot.pause()
        return ps.to_argv(ps.apply_mode(form.read(), "single")), list(app.notes)

    argv, notes = _drive(ps.from_argv([]), act)
    assert ps.check(argv).error is None, ps.check(argv).error
    assert "np_apophis 2000" in notes and "bond_sigma_pa 1e5" in notes
    assert any(n.startswith("tmax_hours ") and n.endswith("(auto)") for n in notes)


def test_auto_tmax_follows_a_typed_rp_but_not_a_typed_tmax():
    import math
    import encounter

    async def act(app, form, pilot):
        app.query_one("#" + field_id("encounter"), Select).value = "hyperbola"
        await pilot.pause()
        app.query_one("#" + axis_id("flyby_rp_km", "fixed"), Input).value = "8000"
        await pilot.pause()
        followed = app.query_one("#" + axis_id("tmax_hours", "fixed"), Input).value
        app.query_one("#" + axis_id("tmax_hours", "fixed"), Input).value = "60"
        await pilot.pause()
        app.query_one("#" + axis_id("flyby_rp_km", "fixed"), Input).value = "20000"
        await pilot.pause()
        return followed, app.query_one("#" + axis_id("tmax_hours", "fixed"), Input).value

    want = str(math.ceil(2 * encounter.time_to_pericentre_hr(8000, 5.9, 4e5, 1.0) * ps.TMAX_MARGIN))
    assert _drive(ps.from_argv([]), act) == (want, "60")


def test_typed_values_survive_a_mode_choice():
    async def act(app, form, pilot):
        app.query_one("#" + field_id("body_source"), Select).value = "settled"
        await pilot.pause()
        return form.read().plain["np_apophis"]

    assert _drive(ps.from_argv(["--np-apophis", "500"]), act) == "500"


def test_f_boulder_hint_says_fraction():
    async def act(app, form, pilot):
        return app.query_one("#" + axis_id("f_boulder", "fixed"), Input).placeholder

    assert "0-1" in _drive(ps.from_argv([]), act)


def test_sweep_toggle_swaps_inputs():
    async def act(app, form, pilot):
        app.query_one("#" + axis_id("tmax_hours", "sweep"), Checkbox).value = True
        await pilot.pause()
        return (app.query_one("#" + axis_id("tmax_hours", "fixed")).display,
                app.query_one("#" + axis_id("tmax_hours", "lo")).display)

    assert _drive(ps.from_argv([]), act) == (False, True)


def test_write_replaces_values():
    target = ps.from_argv(["--body-source", "settled", "--use-dem-fixed", "true", "--np-apophis", "500",
                           "--tmax-hours-min", "10",
                           "--tmax-hours-max", "20"])

    async def act(app, form, pilot):
        form.write(target)
        await pilot.pause()
        return form.read()

    assert ps.to_argv(_drive(ps.from_argv([]), act)) == ps.to_argv(target)


import pytest

import render_backend
from tui_app import render as tr_render

T = render_backend.load()
needs_t = pytest.mark.skipif(isinstance(T, str), reason="render stages unavailable here")


def test_parse_selection():
    assert tr_render.parse_selection("all", "") == "all"
    assert tr_render.parse_selection("first", "3") == {"first": 3}
    assert tr_render.parse_selection("ids", "1, 4 7") == {"ids": [1, 4, 7]}
    with pytest.raises(ValueError):
        tr_render.parse_selection("first", "0")
    with pytest.raises(ValueError):
        tr_render.parse_selection("ids", "")


def test_batches_and_runs(tmp_path):
    b = tmp_path / "sobol_20261013_120000_n2"
    (b / "run_0001").mkdir(parents=True)
    (b / "run_0001" / "sobol_00000").write_text("")
    (b / "sobol_mass_outputs.csv").write_text(
        f"run_id,run_dir,status\n1,{b}/run_0001,ok\n2,{b}/run_0002,failed\n")
    (tmp_path / "_jobs").mkdir()
    assert tr_render.batches(tmp_path) == [b]
    rows = tr_render.runs_of(b)
    assert [(r.run_id, r.status, r.has_dumps) for r in rows] == [(1, "ok", True), (2, "failed", False)]


class _RenderHost(App):
    def compose(self):
        yield tr_render.RenderSettings(T, id="settings")


@needs_t
def test_render_settings_defaults_and_errors():
    async def scenario():
        app = _RenderHost()
        async with app.run_test(size=(160, 60)) as pilot:
            s = app.query_one(tr_render.RenderSettings)
            form = s.read()
            app.query_one("#r-fps", Input).value = "x"
            await pilot.pause()
            try:
                s.read()
            except ValueError as exc:
                return form, str(exc)
    form, err = asyncio.run(scenario())
    assert form.paths == ("instance_grains",) and form.fps == 24 and form.samples == T.DEFAULT_SAMPLES
    assert "fps" in err


import jobstore as js
import tui
from textual.widgets import Static  # noqa: F401  (used by later tests in this file)
from tui_app.app import UnifiedApp


def _app_run(app, actions):
    async def scenario():
        async with app.run_test(size=(180, 60)) as pilot:
            await pilot.pause()
            return await actions(app, pilot)
    return asyncio.run(scenario())


def test_initial_form_overrides_only_unset_paths():
    fv = tui.initial_form(["--output-root", "/x/runs"])
    assert fv.plain["output_root"] == "/x/runs"
    assert fv.plain["base_dir"] == tui.DEFAULT_OVERRIDES["base_dir"]


def test_launch_sim_writes_job_and_launches(tmp_path, monkeypatch):
    launched = []
    monkeypatch.setattr(js, "launch", lambda d, spec, script=js.JOB_PY: launched.append((d, spec)) or 4242)
    fv = tui.initial_form(["--output-root", str(tmp_path), "--use-dem-fixed", "true", "--np-apophis", "300",
                           "--spin-period-fixed", "30"])

    async def act(app, pilot):
        app.query_one("#mode-single", RadioButton).value = True
        await pilot.pause()
        app.query_one("#launch").press()
        await pilot.pause()
        return launched

    out = _app_run(UnifiedApp(fv, "render unavailable: test"), act)
    (job_dir, spec), = out
    assert job_dir.parent == tmp_path / "_jobs"
    assert js.read_spec(job_dir) == spec
    assert spec.omp_num_threads == 4 and spec.render is None
    assert spec.sim_argv[spec.sim_argv.index("--num-samples") + 1] == "1"


def test_invalid_form_creates_no_job(tmp_path, monkeypatch):
    monkeypatch.setattr(js, "launch", lambda *a, **k: pytest.fail("launched"))
    fv = tui.initial_form(["--output-root", str(tmp_path), "--mass-min-kg", "1e10"])

    async def act(app, pilot):
        app.query_one("#launch").press()
        await pilot.pause()
        return app.last_status

    assert "mass bounds" in _app_run(UnifiedApp(fv, "render unavailable: test"), act)
    assert not (tmp_path / "_jobs").exists()


def test_relative_paths_become_absolute_in_job(tmp_path, monkeypatch):
    launched = []
    monkeypatch.setattr(js, "launch", lambda d, spec, script=js.JOB_PY: launched.append(spec) or 1)
    fv = tui.initial_form(["--output-root", "runs", "--use-dem-fixed", "true", "--np-apophis-list", "300"])

    async def act(app, pilot):
        app.query_one("#launch").press()
        await pilot.pause()

    _app_run(UnifiedApp(fv, "render unavailable: test", cwd=tmp_path), act)
    argv = launched[0].sim_argv
    assert argv[argv.index("--output-root") + 1] == str(tmp_path / "runs")
    assert (tmp_path / "runs" / "_jobs").is_dir()


def test_render_controls_disabled_without_backend():
    async def act(app, pilot):
        return (len(app.query("#render-unavailable")), len(app.query("#render-settings")),
                app.query_one("#render-after", Checkbox).disabled)

    assert _app_run(UnifiedApp(ps.from_argv([]), "render unavailable: no Blender"), act) == (1, 0, True)


def test_clone_fills_form_from_past_job(tmp_path):
    d = js.new_job_dir(tmp_path, "old")
    js.write_spec(d, js.JobSpec(label="old", sim_argv=["--np-apophis", "777", "--tmax-hours-min", "10",
                                                       "--tmax-hours-max", "20"]))
    fv = tui.initial_form(["--output-root", str(tmp_path)])

    async def act(app, pilot):
        app._refresh_clone_list()
        app.query_one("#clone-from", Select).value = str(d)
        app.query_one("#clone").press()
        await pilot.pause()
        return app.query_one(SimulateForm).read()

    out = _app_run(UnifiedApp(fv, "render unavailable: test"), act)
    assert out.plain["np_apophis"] == "777" and out.axes["tmax_hours"].sweep


@needs_t
@pytest.mark.parametrize("ticked", [False, True])
def test_render_after_forces_no_cleanup_but_keeps_choice(tmp_path, monkeypatch, ticked):
    launched = []
    monkeypatch.setattr(js, "launch", lambda d, spec, script=js.JOB_PY: launched.append(spec) or 1)
    fv = tui.initial_form(["--output-root", str(tmp_path), "--use-dem-fixed", "true", "--np-apophis-list", "300"]
                          + (["--no-cleanup"] if ticked else []))

    async def act(app, pilot):
        app.query_one("#render-after", Checkbox).value = True
        await pilot.pause()
        app.query_one("#launch").press()
        await pilot.pause()
        return app.last_status

    status = _app_run(UnifiedApp(fv, T), act)
    spec = launched[0]
    assert "--no-cleanup" in spec.sim_argv and spec.keep_dumps is ticked
    assert ("dumps will be deleted after the render" in status) is not ticked
    assert spec.render["paths"] == ["instance_grains"] and spec.run_selection == "all"


from textual.widgets import DataTable, TabbedContent
from tui_app import jobs_pane


def _fake_job(root, label, state, pid=0, stages=()):
    d = js.new_job_dir(root, label)
    js.write_spec(d, js.JobSpec(label=label, sim_argv=["--jobs", "1"], created="2026-10-13T12:00:00"))
    js.write_status(d, js.JobStatus(state=state, pid=pid, host=js.socket.gethostname(), stages=list(stages)))
    return d


def test_describe_lists_stages_and_progress(tmp_path):
    batch = tmp_path / "b"
    batch.mkdir()
    (batch / "sobol_mass_samples.csv").write_text("run_id\n1\n2\n")
    (batch / "sobol_mass_outputs.csv").write_text("run_id,run_dir,status\n1,r,ok\n")
    d = _fake_job(tmp_path, "a", "running", stages=[js.Stage("sim", "running", "2026-10-13T12:00:00")])
    st = js.read_status(d)
    st.batch_dir = str(batch)
    js.write_status(d, st)
    (d / "job.log").write_text("first\nlast line\n")
    text = jobs_pane.describe(d)
    assert "sim" in text and "1/2 runs finished" in text and "last line" in text


def test_jobs_tab_lists_jobs_including_corrupt_status(tmp_path):
    good = _fake_job(tmp_path, "good", "done")
    bad = _fake_job(tmp_path, "bad", "done")
    (bad / "job_status.json").write_text("{not json")
    fv = tui.initial_form(["--output-root", str(tmp_path)])

    async def act(app, pilot):
        app.query_one("#tabs", TabbedContent).active = "tab-jobs"
        pane = app.query_one(jobs_pane.JobsPane)
        pane.refresh_jobs()
        await pilot.pause()
        table = app.query_one("#jobs", DataTable)
        return {table.get_row(k)[0]: table.get_row(k)[1] for k in table.rows}

    rows = _app_run(UnifiedApp(fv, "render unavailable: test"), act)
    assert rows == {good.name: "done", bad.name: "unknown"}


def test_resume_launches_resume_job(tmp_path, monkeypatch):
    launched = []
    monkeypatch.setattr(js, "launch", lambda d, spec, script=js.JOB_PY: launched.append(spec) or 1)
    d = _fake_job(tmp_path, "sweep", "partial")
    st = js.read_status(d)
    st.batch_dir = str(tmp_path / "batch")
    js.write_status(d, st)
    fv = tui.initial_form(["--output-root", str(tmp_path)])

    async def act(app, pilot):
        pane = app.query_one(jobs_pane.JobsPane)
        pane.current = d
        app.query_one("#job-resume").press()
        await pilot.pause()

    _app_run(UnifiedApp(fv, "render unavailable: test"), act)
    assert launched[0].resume_batch_dir == str(tmp_path / "batch")
    assert launched[0].sim_argv == ["--jobs", "1"]


def test_jobs_refresh_keeps_selected_job(tmp_path):
    _fake_job(tmp_path, "a", "done")
    _fake_job(tmp_path, "b", "done")
    fv = tui.initial_form(["--output-root", str(tmp_path)])

    async def act(app, pilot):
        app.query_one("#tabs", TabbedContent).active = "tab-jobs"
        await pilot.pause()
        pane = app.query_one(jobs_pane.JobsPane)
        table = app.query_one("#jobs", DataTable)
        table.move_cursor(row=1)
        await pilot.pause()
        assert pane.current == Path(table.coordinate_to_cell_key((1, 0)).row_key.value)
        picked = pane.current
        pane.refresh_jobs()  # the 2 s poll
        await pilot.pause()
        return picked, pane.current

    picked, after = _app_run(UnifiedApp(fv, "render unavailable: test"), act)
    assert after == picked


@needs_t
def test_render_tab_sees_batches_made_after_start(tmp_path):
    fv = tui.initial_form(["--output-root", str(tmp_path)])

    async def act(app, pilot):
        b = tmp_path / "sobol_new_batch"
        b.mkdir()
        (b / "sobol_mass_outputs.csv").write_text("run_id,run_dir,status\n")
        app.query_one("#tabs", TabbedContent).active = "tab-render"
        await pilot.pause()
        return [str(v) for _, v in app.query_one("#r-batch", Select)._options if v is not Select.NULL]

    assert _app_run(UnifiedApp(fv, T), act) == [str(tmp_path / "sobol_new_batch")]


def test_status_line_is_on_screen_above_footer():
    async def act(app, pilot):
        status, footer = app.query_one("#status").region, app.query_one("Footer").region
        return status.bottom <= footer.y and status.height >= 1

    assert _app_run(UnifiedApp(ps.from_argv([]), "render unavailable: test"), act)


def test_mode_bar_controls_are_visible():
    async def act(app, pilot):
        screen_w = app.size.width
        single, sweep, omp = (app.query_one(s).region for s in ("#mode-single", "#mode-sweep", "#omp"))
        return single.width >= len("Single run"), sweep.width >= len("Sweep"), omp.right <= screen_w

    assert _app_run(UnifiedApp(ps.from_argv([]), "render unavailable: test"), act) == (True, True, True)


def test_tab_content_keeps_off_column_zero():
    # textual serve (xterm.js in the browser) clips column 0: labels lost their first letter
    async def act(app, pilot):
        return min(w.region.x for w in app.query("#tab-simulate .rl, #tab-simulate .sec") if w.region.width)

    assert _app_run(UnifiedApp(ps.from_argv([]), "render unavailable: test"), act) >= 1


def test_mouse_click_on_launch_shows_jobs_tab(tmp_path, monkeypatch):
    monkeypatch.setattr(js, "launch", lambda d, spec, script=js.JOB_PY: 1)
    fv = tui.initial_form(["--output-root", str(tmp_path), "--use-dem-fixed", "true", "--np-apophis-list", "300"])

    async def act(app, pilot):
        await pilot.click("#launch")  # a real click also focuses the button, unlike .press()
        await pilot.pause()
        await pilot.pause()
        return app.query_one("#tabs", TabbedContent).active

    assert _app_run(UnifiedApp(fv, "render unavailable: test"), act) == "tab-jobs"


def test_form_rows_are_one_line():
    async def act(app, pilot):
        heights = {app.query_one(s).region.height for s in (
            "#row-" + field_id("prefix"), "#row-" + field_id("body_source"), "#row-" + field_id("boulders"),
            "#" + axis_id("mass", "row"))}
        return heights, app.query_one("#mode").region.height

    assert _app_run(UnifiedApp(ps.from_argv([]), "render unavailable: test"), act) == ({1}, 1)


@needs_t
def test_render_rows_are_one_line():
    async def act(app, pilot):
        app.query_one("#tabs", TabbedContent).active = "tab-render"
        await pilot.pause()
        return {r.region.height for r in app.query("#render-settings Row, #picker Row") if r.region.height}

    assert _app_run(UnifiedApp(ps.from_argv([]), T), act) == {1}


# ── layout / clarity review ─────────────────────────────────────────────────

def test_mode_bar_stays_visible_when_form_scrolls():
    async def act(app, pilot):
        app.query_one("#sim-body").scroll_end(animate=False)
        await pilot.pause()
        r = app.query_one("#mode").region
        return r.height > 0 and r.bottom <= app.size.height

    assert _app_run(UnifiedApp(ps.from_argv([]), "render unavailable: test"), act)


def test_launch_label_follows_mode():
    async def act(app, pilot):
        before = str(app.query_one("#launch", Button).label)
        app.query_one("#mode-single", RadioButton).value = True
        await pilot.pause()
        return before, str(app.query_one("#launch", Button).label)

    assert _app_run(UnifiedApp(ps.from_argv([]), "render unavailable: test"), act) == ("Launch sweep",
                                                                                         "Launch single run")


def test_bottom_bar_puts_launch_right_and_keeps_clone_narrow():
    async def act(app, pilot):
        return (app.size.width - app.query_one("#launch").region.right,
                app.query_one("#clone-from").region.width)

    gap, clone_w = _app_run(UnifiedApp(ps.from_argv([]), "render unavailable: test"), act)
    assert gap <= 4 and clone_w <= 50


def test_checkbox_labels_do_not_repeat_row_label():
    async def act(app, pilot):
        return [str(app.query_one("#" + field_id(f.dest), Checkbox).label) for f in ps.PLAIN if f.kind == "check"]

    labels = _app_run(UnifiedApp(ps.from_argv([]), "render unavailable: test"), act)
    assert labels and not {f.dest for f in ps.PLAIN if f.kind == "check"} & set(labels)


def test_cancel_and_resume_enabled_only_when_they_apply(tmp_path):
    done = _fake_job(tmp_path, "done_job", "done")
    partial = _fake_job(tmp_path, "partial_job", "partial")
    st = js.read_status(partial)
    st.batch_dir = str(tmp_path / "b")
    js.write_status(partial, st)
    fv = tui.initial_form(["--output-root", str(tmp_path)])

    async def act(app, pilot):
        pane = app.query_one(jobs_pane.JobsPane)
        out = []
        for d in (done, partial):
            pane.current = d
            pane.refresh_jobs()
            await pilot.pause()
            out.append((app.query_one("#job-cancel").disabled, app.query_one("#job-resume").disabled))
        return out

    assert _app_run(UnifiedApp(fv, "render unavailable: test"), act) == [(True, True), (True, False)]


def test_status_line_starts_with_a_hint():
    async def act(app, pilot):
        return app.last_status

    assert "Launch" in _app_run(UnifiedApp(ps.from_argv([]), "render unavailable: test"), act)


def test_elapsed_stops_for_finished_job_without_stage_end(tmp_path):
    spec = js.JobSpec(label="x", created="2026-10-13T12:00:00")
    st = js.JobStatus(state="cancelled", stages=[js.Stage("sim", "running", "2026-10-13T12:00:00")])
    d = tmp_path / "j"
    d.mkdir()
    js.write_status(d, st)
    os.utime(d / "job_status.json", (0, datetime(2026, 10, 13, 12, 5, 0).timestamp()))
    assert jobs_pane.elapsed(spec, st, d) == "5m00s"


def test_describe_shows_live_frame_count(tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    (frames / "frame_0001.png").write_text("")
    d = _fake_job(tmp_path, "a", "running", stages=[js.Stage(
        "render run_0001", "running", "2026-10-13T12:00:00",
        progress={"phase": "render", "path": "instance_grains", "index": 1, "total": 1, "dir": str(frames),
                  "glob": "frame_*.png", "n_expected": 4})])
    assert "rendering instance_grains (path 1/1): 1/4 frames" in jobs_pane.describe(d)
    (frames / "frame_0002.png").write_text("")  # the next poll ticks
    assert "2/4 frames" in jobs_pane.describe(d)


def test_cancel_of_a_job_that_just_ended_is_reported_not_raised(tmp_path, monkeypatch):
    d = _fake_job(tmp_path, "ending", "running", pid=os.getpid())
    monkeypatch.setattr(js, "cancel", lambda st: (_ for _ in ()).throw(ProcessLookupError()))
    fv = tui.initial_form(["--output-root", str(tmp_path)])

    async def act(app, pilot):
        pane = app.query_one(jobs_pane.JobsPane)
        pane.current = d
        pane._cancel()
        await pilot.pause()
        return app.last_status

    assert "already ended" in _app_run(UnifiedApp(fv, "render unavailable: test"), act)


@needs_t
def test_render_after_default_follows_mode_switch():
    async def act(app, pilot):
        box = app.query_one("#render-after", Checkbox)
        out = [box.value]
        app.query_one("#mode-single", RadioButton).value = True
        await pilot.pause()
        out.append(box.value)
        app.query_one("#mode-sweep", RadioButton).value = True
        await pilot.pause()
        return out + [box.value]

    assert _app_run(UnifiedApp(ps.from_argv([]), T, mode="sweep"), act) == [False, True, False]


def test_single_run_refuses_a_list_of_several_counts(tmp_path, monkeypatch):
    monkeypatch.setattr(js, "launch", lambda *a, **k: pytest.fail("launched"))
    fv = tui.initial_form(["--output-root", str(tmp_path), "--use-dem-fixed", "true",
                           "--np-apophis-list", "300", "600"])

    async def act(app, pilot):
        app.query_one("#mode-single", RadioButton).value = True
        await pilot.pause()
        app.query_one("#launch").press()
        await pilot.pause()
        return app.last_status

    assert "one value" in _app_run(UnifiedApp(fv, "render unavailable: test"), act)
    assert not (tmp_path / "_jobs").exists()


def test_old_single_keys_launch_dry_run_and_quit(tmp_path, monkeypatch):
    launched = []
    monkeypatch.setattr(js, "launch", lambda d, spec, script=js.JOB_PY: launched.append(spec) or 1)
    fv = tui.initial_form(["--output-root", str(tmp_path), "--use-dem-fixed", "true", "--np-apophis-list", "300"])

    async def act(app, pilot):
        app.query_one("#" + field_id("no_cleanup")).focus()  # a checkbox: letters are not typed anywhere
        await pilot.press("d")
        await pilot.pause()
        n_after_d = len(launched)
        await pilot.pause()
        app.query_one("#tabs", TabbedContent).active = "tab-simulate"
        await pilot.pause()
        app.query_one("#" + field_id("batch_label"), Input).focus()
        await pilot.press("r")  # typed into the input, not a launch
        await pilot.pause()
        typed = app.query_one("#" + field_id("batch_label"), Input).value
        app.query_one("#tabs", TabbedContent).active = "tab-jobs"
        await pilot.pause()
        app.query_one("#jobs").focus()
        await pilot.press("r")  # no launch from the Jobs tab
        await pilot.pause()
        n_jobs_tab = len(launched)
        app.query_one("#tabs", TabbedContent).active = "tab-simulate"
        await pilot.pause()
        app.query_one("#" + field_id("no_cleanup")).focus()
        await pilot.press("r")
        await pilot.pause()
        n_after_r = len(launched)
        await pilot.press("q")
        await pilot.pause()
        return n_after_d, typed, n_jobs_tab, n_after_r, app.is_running

    n_d, typed, n_jobs, n_r, running = _app_run(UnifiedApp(fv, "render unavailable: test"), act)
    assert (n_d, typed, n_jobs, n_r, running) == (1, "r", 1, 2, False)
    assert "--dry-run" in launched[0].sim_argv and "--dry-run" not in launched[1].sim_argv


def test_launch_button_is_never_a_dry_run(tmp_path, monkeypatch):
    launched = []
    monkeypatch.setattr(js, "launch", lambda d, spec, script=js.JOB_PY: launched.append(spec) or 1)
    fv = tui.initial_form(["--output-root", str(tmp_path), "--use-dem-fixed", "true", "--np-apophis-list", "300"])

    async def act(app, pilot):
        await pilot.click("#launch")
        await pilot.pause()

    _app_run(UnifiedApp(fv, "render unavailable: test"), act)
    assert len(launched) == 1 and "--dry-run" not in launched[0].sim_argv


def test_status_line_lists_filled_defaults():
    async def act(app, pilot):
        app.query_one("#" + field_id("body_source"), Select).value = "settled"
        await pilot.pause()
        await pilot.pause()
        return app.last_status

    status = _app_run(UnifiedApp(ps.from_argv([]), "render unavailable: test"), act)
    assert status.startswith("defaults filled:") and "np_apophis 2000" in status


def test_settled_is_the_tui_default_and_fills_its_fields():
    assert tui.initial_form([]).plain["body_source"] == "settled"
    assert tui.initial_form(["--body-source", "lattice"]).plain["body_source"] == "lattice"

    async def act(app, pilot):
        await pilot.pause()
        return app.query_one(SimulateForm).read()

    fv = _app_run(UnifiedApp(tui.initial_form([]), "render unavailable: test"), act)
    assert (fv.plain["use_dem_fixed"], fv.plain["np_apophis"]) == ("true", "2000")
    assert ps.check(ps.to_argv(ps.apply_mode(fv, "single"))).error is None


@needs_t
def test_clone_gives_back_the_form_not_the_forced_flags(tmp_path, monkeypatch):
    launched = []
    monkeypatch.setattr(js, "launch", lambda d, spec, script=js.JOB_PY: launched.append(spec) or 1)
    fv = tui.initial_form(["--output-root", str(tmp_path), "--use-dem-fixed", "true", "--np-apophis-list", "300"])

    async def act(app, pilot):
        app.query_one("#render-after", Checkbox).value = True
        await pilot.pause()
        app.action_dry_run()  # the 'd' key
        await pilot.pause()
        app._refresh_clone_list()
        app.query_one("#clone-from", Select).value = str(next(iter(js.list_jobs(tmp_path))))
        app.query_one("#clone").press()
        await pilot.pause()
        return app.query_one(SimulateForm).read()

    out = _app_run(UnifiedApp(fv, T), act)
    spec = launched[0]
    assert "--no-cleanup" in spec.sim_argv and "--dry-run" in spec.sim_argv  # what ran
    assert not out.plain.get("no_cleanup") and not out.plain.get("dry_run")  # what Clone gives back


def test_clone_argv_of_a_job_from_before_form_argv():
    from tui_app.app import clone_argv
    old = js.JobSpec(label="o", sim_argv=["--np-apophis", "300", "--no-cleanup", "--dry-run"],
                     render={"paths": ["instance_grains"]}, keep_dumps=False)
    assert clone_argv(old) == ["--np-apophis", "300"]
    kept = js.JobSpec(label="o", sim_argv=["--no-cleanup"], render={"paths": ["x"]}, keep_dumps=True)
    assert clone_argv(kept) == ["--no-cleanup"]  # the user ticked no_cleanup


def test_resume_skips_runs_the_job_rendered(tmp_path, monkeypatch):
    launched = []
    monkeypatch.setattr(js, "launch", lambda d, spec, script=js.JOB_PY: launched.append(spec) or 1)
    d = _fake_job(tmp_path, "sweep", "cancelled", stages=[
        js.Stage("sim", state="done"), js.Stage("render run_0001", state="done"),
        js.Stage("render run_0002", state="failed", message="cancelled")])
    st = js.read_status(d)
    st.batch_dir = str(tmp_path / "batch")
    js.write_status(d, st)
    spec = js.read_spec(d)
    spec.render, spec.skip_runs = {"paths": ["instance_grains"]}, ["run_0000"]  # from an earlier resume
    js.write_spec(d, spec)
    fv = tui.initial_form(["--output-root", str(tmp_path)])

    async def act(app, pilot):
        app.query_one(jobs_pane.JobsPane).current = d
        app.query_one("#job-resume").press()
        await pilot.pause()

    _app_run(UnifiedApp(fv, "render unavailable: test"), act)
    assert launched[0].skip_runs == ["run_0000", "run_0001"]  # run_0002 failed: rendered again


@needs_t
def test_render_tab_keeps_ticked_runs_across_tab_switches(tmp_path):
    b = tmp_path / "sobol_20261013_120000_n3"
    for n in (1, 2, 3):
        (b / f"run_{n:04d}").mkdir(parents=True)
        (b / f"run_{n:04d}" / "sobol.setup").write_text("")
        (b / f"run_{n:04d}" / "sobol_00000").write_text("")
    (b / "sobol_mass_outputs.csv").write_text(
        "run_id,run_dir,status\n" + "".join(f"{n},{b}/run_{n:04d},ok\n" for n in (1, 2, 3)))
    fv = tui.initial_form(["--output-root", str(tmp_path)])

    async def act(app, pilot):
        tabs = app.query_one("#tabs", TabbedContent)
        tabs.active = "tab-render"
        await pilot.pause()
        app.query_one("#r-batch", Select).value = str(b)
        await pilot.pause()
        table = app.query_one("#r-runs", DataTable)
        for row in (0, 2):
            table.move_cursor(row=row)
            table.action_select_cursor()
            await pilot.pause()
        tabs.active = "tab-jobs"
        await pilot.pause()
        tabs.active = "tab-render"
        await pilot.pause()
        return [p.name for p in app.query_one(tr_render.RunPicker).chosen_runs()], table.get_cell_at((0, 0))

    chosen, tick = _app_run(UnifiedApp(fv, T), act)
    assert chosen == ["run_0001", "run_0003"] and tick == "✓"
