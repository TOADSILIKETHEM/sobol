"""UnifiedApp: Simulate / Render / Jobs tabs. Builds job.json files and launches job.py detached."""
from __future__ import annotations

import os
from pathlib import Path
from types import ModuleType
from typing import Optional, Tuple, Union

from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import (Button, Checkbox, Collapsible, Footer, Header, Input, RadioSet, Select, Static,
                             TabbedContent, TabPane)

import jobstore as js
import params_schema as ps
import render_backend
from tui_app.jobs_pane import JobsPane
from tui_app.render import RenderSettings, RunPicker, parse_selection
from tui_app.simulate import Row, SimulateForm


def clone_argv(spec: js.JobSpec) -> list:
    """The form a job was launched from. Jobs from before form_argv: sim_argv without the 'd' key's
    --dry-run and without the --no-cleanup that Render after forced (keep_dumps records the real choice)."""
    if spec.form_argv is not None:
        return list(spec.form_argv)
    drop = {"--dry-run"} | ({"--no-cleanup"} if spec.render and not spec.keep_dumps else set())
    return [a for a in spec.sim_argv if a not in drop]


class UnifiedApp(App[None]):
    TITLE = "PHANTOM runs"
    # r / d / q as in tui_run.py; a focused text box takes the letter instead, so they never fire while typing
    BINDINGS = [Binding("ctrl+l", "launch", "Launch"), Binding("r", "launch", "Launch / Render"),
                Binding("d", "dry_run", "Dry run"), Binding("q", "quit", "Quit"),
                Binding("ctrl+q", "quit", "Quit", show=False)]
    DEFAULT_CSS = """
    #tabs { height: 1fr; }
    #tabs TabPane { padding: 0 1; }
    #status { height: auto; min-height: 3; max-height: 8; padding: 0 2; background: $surface-darken-1; }
    .err { color: $error; }
    .bar { height: auto; }
    .bar .spacer { width: 1fr; }
    #clone-from { width: 40; }
    Button { margin: 0 1; }
    """

    def __init__(self, initial: ps.FormValues, backend: Union[ModuleType, str], mode: str = "sweep",
                 cwd: Optional[Path] = None) -> None:
        super().__init__()
        self._initial, self._mode = initial, mode
        self.backend = backend
        self.cwd = Path(cwd) if cwd is not None else Path.cwd()
        self.last_status = ""

    @property
    def render_ok(self) -> bool:
        return not isinstance(self.backend, str)

    def compose(self) -> ComposeResult:
        yield Header(icon="*")  # the default glyph is missing from browser fonts (textual serve)
        with TabbedContent(initial="tab-simulate", id="tabs"):
            with TabPane("Simulate", id="tab-simulate"):
                with Vertical():
                    yield SimulateForm(self._initial, mode=self._mode, id="sim")
                    with Collapsible(title="Render after the sim (settings on the Render tab)", collapsed=True):
                        yield Row("render after", Checkbox("render the selected runs when the sim finishes",
                                                           value=self._mode == "single" and self.render_ok,
                                                           disabled=not self.render_ok, id="render-after",
                                                           compact=True))
                        yield Row("runs", Select([("all ok runs", "all"), ("first N", "first"), ("run ids", "ids")],
                                                 value="all", allow_blank=False, id="sel-kind", compact=True),
                                  Input("", placeholder="N, or ids like 1 4 7", id="sel-value", compact=True))
                    with Horizontal(classes="bar"):
                        yield Button("Validate", id="validate")
                        yield Button("Show command", id="show-command")
                        yield Static("", classes="spacer")
                        yield Select([], allow_blank=True, prompt="clone a past job", id="clone-from")
                        yield Button("Clone", id="clone")
                        yield Button(self._launch_label(self._mode), variant="primary", id="launch")
            with TabPane("Render", id="tab-render"):
                if self.render_ok:
                    with Vertical():
                        yield RunPicker(self._output_root_from(self._initial), id="picker")
                        yield RenderSettings(self.backend, id="render-settings")
                        yield Button("Render", variant="primary", id="render")
                else:
                    yield Static(Text(self.backend), id="render-unavailable")
            with TabPane("Jobs", id="tab-jobs"):
                yield JobsPane(id="jobs-pane")
        yield Static("", id="status")
        yield Footer()

    def on_mount(self) -> None:
        self._refresh_clone_list()
        self.set_status("Fill in the form (hover a label for help), Validate, then Launch. "
                        "Jobs run detached: closing this window does not stop them.")

    @staticmethod
    def _launch_label(mode: str) -> str:
        return "Launch single run" if mode == "single" else "Launch sweep"

    @on(RadioSet.Changed, "#mode")
    def _mode_changed(self) -> None:
        mode = self.query_one(SimulateForm).mode
        self.query_one("#launch", Button).label = self._launch_label(mode)
        self.query_one("#render-after", Checkbox).value = mode == "single" and self.render_ok

    def _active_tab(self) -> str:
        try:
            return str(self.query_one("#tabs", TabbedContent).active)
        except Exception:  # bindings are checked during start-up and shutdown too, when there is no screen
            return ""

    def check_action(self, action: str, parameters) -> Optional[bool]:
        """r launches from Simulate and renders from Render; d is Simulate only; neither acts on Jobs."""
        if action == "launch" and self._active_tab() not in ("tab-simulate", "tab-render"):
            return False
        if action == "dry_run" and self._active_tab() != "tab-simulate":
            return False
        return True

    @on(TabbedContent.TabActivated, "#tabs")
    def _tab_activated(self, event: TabbedContent.TabActivated) -> None:
        if self.screen_stack:  # the switch can arrive while the app shuts down
            self.refresh_bindings()
        if event.pane.id == "tab-render" and self.render_ok:
            # batches made since start (or a changed output_root) show up without restarting the TUI
            self.query_one(RunPicker).reload(self._output_root_from(self.query_one(SimulateForm).read()))

    # ── helpers ──────────────────────────────────────────────────────────────

    def on_simulate_form_defaults_filled(self, message: SimulateForm.DefaultsFilled) -> None:
        self.set_status("defaults filled: " + ", ".join(message.notes))

    def set_status(self, msg: str, error: bool = False) -> None:
        self.last_status = msg  # tests read this instead of the widget's rendering
        status = self.query_one("#status", Static)
        status.set_class(error, "err")
        status.update(Text(msg))

    def _output_root_from(self, fv: ps.FormValues) -> Path:
        text = str(fv.plain.get("output_root") or "sobol_mass_runs")
        return Path(ps.absolutise(ps.FormValues({"output_root": text}, {}), self.cwd).plain["output_root"])

    def _sim_values(self) -> Tuple[ps.FormValues, str]:
        form = self.query_one(SimulateForm)
        return ps.absolutise(ps.apply_mode(form.read(), form.mode), self.cwd), form.mode

    def _refresh_clone_list(self) -> None:
        root = self._output_root_from(self.query_one(SimulateForm).read())
        options = []
        for d in js.list_jobs(root):
            try:
                if js.read_spec(d).sim_argv is not None:
                    options.append((d.name, str(d)))
            except (OSError, ValueError, TypeError):
                continue
        self.query_one("#clone-from", Select).set_options(options)

    # ── job building ─────────────────────────────────────────────────────────

    def build_sim_job(self, dry_run: bool = False) -> Tuple[js.JobSpec, Path]:
        fv, mode = self._sim_values()
        if mode == "single":
            error = ps.single_mode_error(fv)
            if error:
                raise ValueError(error)
        form_argv = ps.to_argv(fv)  # before the forced flags below: Clone gives back what was typed
        if dry_run:
            fv.plain["dry_run"] = True
        keep_dumps = bool(fv.plain.get("no_cleanup"))
        render = selection = None
        if self.query_one("#render-after", Checkbox).value:
            if not self.render_ok:
                raise ValueError(self.backend)
            render = render_backend.form_to_dict(self.query_one(RenderSettings).read())
            selection = parse_selection(str(self.query_one("#sel-kind", Select).value),
                                        self.query_one("#sel-value", Input).value)
            fv.plain["no_cleanup"] = True  # renders need the dumps; keep_dumps decides after the render
        argv = ps.to_argv(fv)
        result = ps.check(argv)
        if result.error:
            raise ValueError(result.error)
        label = str(fv.plain.get("batch_label") or ("single" if mode == "single" else "sweep"))
        spec = js.JobSpec(label=label, sim_argv=argv, render=render, run_selection=selection or "all",
                          keep_dumps=keep_dumps, omp_num_threads=self.query_one(SimulateForm).omp,
                          form_argv=form_argv)
        return spec, self._output_root_from(fv)

    def build_render_job(self) -> Tuple[js.JobSpec, Path]:
        form = self.query_one(RenderSettings).read()
        picker = self.query_one(RunPicker)
        root = self._output_root_from(self.query_one(SimulateForm).read())
        other = picker.other_run_argv(form)
        if other is not None:
            return js.JobSpec(label="render_dumps", render_dumps_argv=other, omp_num_threads=1), root
        runs = picker.chosen_runs()
        if not runs:
            raise ValueError("pick a batch and press Enter on the runs to render, or fill in Other run dir")
        return js.JobSpec(label="render", render=render_backend.form_to_dict(form),
                          run_selection={"paths": [str(p) for p in runs]}, keep_dumps=not picker.delete_dumps,
                          omp_num_threads=1), root

    def launch_job(self, spec: js.JobSpec, output_root: Path) -> Path:
        busy = js.running_on_host(output_root)
        job_dir = js.new_job_dir(output_root, spec.label)
        js.write_spec(job_dir, spec)
        pid = js.launch(job_dir, spec)
        warn = (f"  WARNING: {len(busy)} other job(s) running on this host (laptop throttles above 2)"
                if busy else "")
        if spec.sim_argv is not None and spec.render and not spec.keep_dumps and "--dry-run" not in spec.sim_argv:
            # Render after forces --no-cleanup into sim_argv, so the job record alone hides this
            warn += "\n  WARNING: dumps will be deleted after the render (tick no_cleanup to keep them)"
        self.set_status(f"launched {job_dir.name} (pid {pid}); log: {job_dir / 'job.log'}{warn}")
        self._refresh_clone_list()
        pane = self.query_one(JobsPane)
        pane.current = job_dir
        pane.refresh_jobs()
        # after the click settles: a clicked Launch button holds focus and would pull its tab back
        self.call_after_refresh(self._show_jobs)
        return job_dir

    def _show_jobs(self) -> None:
        self.query_one("#tabs", TabbedContent).active = "tab-jobs"
        self.query_one("#jobs").focus()

    # ── buttons ──────────────────────────────────────────────────────────────

    @on(Button.Pressed, "#validate")
    def _validate(self) -> None:
        fv, mode = self._sim_values()
        error = ps.single_mode_error(fv) if mode == "single" else None
        if error:
            self.set_status(error, error=True)
            return
        result = ps.check(ps.to_argv(fv))
        if result.error:
            self.set_status(result.error, error=True)
        else:
            self.set_status("OK" + (f"\n{result.warnings}" if result.warnings else ""))

    @on(Button.Pressed, "#show-command")
    def _show_command(self) -> None:
        try:
            spec, _ = self.build_sim_job()
        except ValueError as exc:
            self.set_status(str(exc), error=True)
            return
        self.set_status(ps.show_command(spec.sim_argv, spec.omp_num_threads))

    @on(Button.Pressed, "#clone")
    def _clone(self) -> None:
        value = self.query_one("#clone-from", Select).value
        if value is Select.NULL:
            self.set_status("choose a past job to clone", error=True)
            return
        spec = js.read_spec(Path(str(value)))
        self.query_one(SimulateForm).write(ps.from_argv(clone_argv(spec)))
        self.set_status(f"form filled from {Path(str(value)).name}")

    def action_launch(self) -> None:
        if self._active_tab() == "tab-render":
            self._render()
        else:
            self._launch_sim()

    def action_dry_run(self) -> None:
        self._launch_sim(dry_run=True)

    @on(Button.Pressed, "#launch")
    def _launch_pressed(self) -> None:  # no parameters: Textual would pass the event into one
        self._launch_sim()

    def _launch_sim(self, dry_run: bool = False) -> None:
        try:
            spec, root = self.build_sim_job(dry_run=dry_run)
        except ValueError as exc:
            self.set_status(str(exc), error=True)
            return
        self.launch_job(spec, root)

    @on(Button.Pressed, "#render")
    def _render(self) -> None:
        try:
            spec, root = self.build_render_job()
        except ValueError as exc:
            self.set_status(str(exc), error=True)
            return
        self.launch_job(spec, root)
