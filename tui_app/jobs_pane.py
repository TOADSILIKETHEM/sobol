"""Jobs tab: every job under <output_root>/_jobs, its stages, progress and log; cancel and resume."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, DataTable, Log, Static

import jobstore as js


def rendered_runs(spec: js.JobSpec, status: js.JobStatus) -> list:
    """Run dir names the job (and any job it resumed) rendered, so Resume does not render them again."""
    done = {s.name[len("render "):] for s in status.stages if s.name.startswith("render ") and s.state == "done"}
    return sorted(done | set(spec.skip_runs))


def elapsed(spec: js.JobSpec, status: Optional[js.JobStatus], job_dir: Optional[Path] = None) -> str:
    try:
        start = datetime.fromisoformat(spec.created)
    except ValueError:
        return "?"
    end = datetime.now()
    if status is not None and status.state != "running":
        if status.stages and status.stages[-1].ended:
            end = datetime.fromisoformat(status.stages[-1].ended)
        elif job_dir is not None:  # finished without a stage end (older job.py): last status write
            try:
                end = datetime.fromtimestamp((Path(job_dir) / "job_status.json").stat().st_mtime)
            except OSError:
                pass
    secs = int((end - start).total_seconds())
    return f"{secs // 3600}h{secs % 3600 // 60:02d}m" if secs >= 3600 else f"{secs // 60}m{secs % 60:02d}s"


def describe(job_dir: Path) -> str:
    status = js.read_status(job_dir)
    lines = [str(job_dir)]
    if status is None:
        lines.append("status: unknown (job_status.json missing or unreadable)")
    else:
        lines.append(f"state: {js.liveness(status)}   pid {status.pid} on {status.host}")
        for s in status.stages:
            lines.append(f"  {s.name:<18} {s.state:<8} {s.message}")
            live = js.stage_progress(s) if s.state == "running" else ""
            if live:
                lines.append(f"    {live}")  # recounted on each 2 s refresh, so it ticks
        if status.batch_dir:
            lines.append(f"batch: {status.batch_dir}")
            lines.append("  " + js.batch_progress(Path(status.batch_dir)))
    last = js.tail(job_dir / "job.log", 1)
    if last:
        lines.append(f"log: {last[0]}")
    return "\n".join(lines)


class JobsPane(Vertical):
    DEFAULT_CSS = """
    JobsPane DataTable { height: 12; }
    JobsPane #job-detail { height: auto; padding: 1 0; }
    JobsPane Log { height: 1fr; border: round $panel; }
    """

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self.current: Optional[Path] = None

    def compose(self) -> ComposeResult:
        yield DataTable(id="jobs", cursor_type="row")
        yield Static("", id="job-detail")
        with Horizontal(classes="bar"):
            yield Button("Cancel", variant="error", id="job-cancel")
            yield Button("Resume", id="job-resume")
            yield Button("Show log", id="job-log")
        log = Log(id="job-log-view", max_lines=400)
        log.border_title = "job.log (Show log)"
        yield log

    def on_mount(self) -> None:
        self.query_one("#jobs", DataTable).add_columns("job", "state", "elapsed", "label")
        self.refresh_jobs()
        self.set_interval(2, self.refresh_jobs)

    def _root(self) -> Path:
        return self.app._output_root_from(self.app.query_one("#sim").read())

    def refresh_jobs(self) -> None:
        table = self.query_one("#jobs", DataTable)
        table.clear()
        for d in js.list_jobs(self._root()):
            try:
                spec = js.read_spec(d)
            except (OSError, ValueError, TypeError):
                continue
            status = js.read_status(d)
            state = js.liveness(status) if status is not None else "unknown"
            table.add_row(d.name, state, elapsed(spec, status, d), spec.label, key=str(d))
        if self.current is not None and str(self.current) in table.rows:
            # clear() put the cursor on row 0; put it back so the poll never changes the selected job
            table.move_cursor(row=table.get_row_index(str(self.current)))
        if self.current is not None:
            self.query_one("#job-detail", Static).update(Text(describe(self.current)))
        self._update_buttons()

    def _update_buttons(self) -> None:
        """Cancel only for a job running here; Resume only for a finished sim job with a batch."""
        status = js.read_status(self.current) if self.current else None
        live = js.liveness(status) if status is not None else "unknown"
        try:
            is_sim = self.current is not None and js.read_spec(self.current).sim_argv is not None
        except (OSError, ValueError, TypeError):
            is_sim = False
        self.query_one("#job-cancel", Button).disabled = live != "running"
        self.query_one("#job-resume", Button).disabled = not (
            is_sim and status is not None and status.batch_dir and live in ("partial", "failed", "cancelled", "died"))
        self.query_one("#job-log", Button).disabled = self.current is None

    @on(DataTable.RowHighlighted, "#jobs")
    def _highlight(self, event: DataTable.RowHighlighted) -> None:
        if event.row_key is not None and event.row_key.value:
            self.current = Path(event.row_key.value)
            self.query_one("#job-detail", Static).update(Text(describe(self.current)))
            self._update_buttons()

    @on(Button.Pressed, "#job-cancel")
    def _cancel(self) -> None:
        status = js.read_status(self.current) if self.current else None
        if status is None or js.liveness(status) != "running":
            self.app.set_status("only a job running on this host can be cancelled", error=True)
            return
        try:
            js.cancel(status)
        except ProcessLookupError:  # it ended between the check above and the signal
            self.app.set_status(f"{self.current.name} already ended; nothing to cancel")
            return
        except PermissionError as exc:
            self.app.set_status(f"cannot cancel {self.current.name}: {exc}", error=True)
            return
        self.app.set_status(f"cancel sent to {self.current.name}")

    @on(Button.Pressed, "#job-resume")
    def _resume(self) -> None:
        if self.current is None:
            self.app.set_status("select a job first", error=True)
            return
        spec, status = js.read_spec(self.current), js.read_status(self.current)
        if spec.sim_argv is None or status is None or not status.batch_dir:
            self.app.set_status("only a sim job with a batch dir can be resumed", error=True)
            return
        if js.liveness(status) == "running":
            self.app.set_status("job is still running; cancel it first", error=True)
            return
        resumed = js.JobSpec(label=spec.label + "_resume", sim_argv=spec.sim_argv, resume_batch_dir=status.batch_dir,
                             render=spec.render, run_selection=spec.run_selection, keep_dumps=spec.keep_dumps,
                             omp_num_threads=spec.omp_num_threads, skip_runs=rendered_runs(spec, status),
                             form_argv=spec.form_argv)
        self.app.launch_job(resumed, self._root())

    @on(Button.Pressed, "#job-log")
    def _log(self) -> None:
        if self.current is None:
            return
        view = self.query_one("#job-log-view", Log)
        view.clear()
        view.write_lines(js.tail(self.current / "job.log", 400))
