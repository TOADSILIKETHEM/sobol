"""Render tab body: choose runs (from a batch, or any run dir) and render settings.

`t` is the tui_sim_render module from render_backend.load(); this file never imports it, so the
app still starts where rendering is unavailable (OzSTAR).
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import List, Optional, Set

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Checkbox, Collapsible, DataTable, Input, Select, Static

import render_backend
from tui_app.simulate import Row


@dataclass
class RunRow:
    run_id: int
    status: str
    run_dir: Path
    has_dumps: bool


def batches(output_root: Path) -> List[Path]:
    root = Path(output_root)
    if not root.is_dir():
        return []
    found = [d for d in root.iterdir() if d.is_dir() and (d / "sobol_mass_outputs.csv").is_file()]
    return sorted(found, key=lambda d: d.stat().st_mtime, reverse=True)


def runs_of(batch: Path) -> List[RunRow]:
    with open(Path(batch) / "sobol_mass_outputs.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        run_dir = Path(r["run_dir"])
        out.append(RunRow(int(r["run_id"]), r.get("status", ""), run_dir, any(run_dir.glob("sobol_[0-9]*"))))
    return sorted(out, key=lambda r: r.run_id)


def parse_selection(kind: str, value: str) -> object:
    if kind == "all":
        return "all"
    if kind == "first":
        try:
            n = int(value)
        except ValueError:
            raise ValueError(f"first N needs a whole number (got {value!r})") from None
        if n < 1:
            raise ValueError("first N must be >= 1")
        return {"first": n}
    if kind == "ids":
        try:
            ids = [int(x) for x in value.replace(",", " ").split()]
        except ValueError:
            raise ValueError(f"run ids must be whole numbers (got {value!r})") from None
        if not ids:
            raise ValueError("list at least one run id")
        return {"ids": ids}
    raise ValueError(f"unknown selection {kind!r}")


_RES_RE = re.compile(r"^\d+x\d+$")


def _int(name: str, text: str, optional: bool = False) -> Optional[int]:
    text = text.strip()
    if not text and optional:
        return None
    try:
        return int(text)
    except ValueError:
        raise ValueError(f"{name} must be a whole number (got {text!r})") from None


def _float(name: str, text: str) -> float:
    try:
        return float(text.strip())
    except ValueError:
        raise ValueError(f"{name} must be a number (got {text.strip()!r})") from None


class RenderSettings(Vertical):
    DEFAULT_CSS = "RenderSettings { height: auto; }"

    def __init__(self, t: ModuleType, **kw) -> None:
        super().__init__(**kw)
        self.t = t

    def compose(self) -> ComposeResult:
        t = self.t
        yield Row("resolution", Input("1920x1080", id="r-resolution", compact=True))
        yield Row("samples", Input(str(t.DEFAULT_SAMPLES), id="r-samples", compact=True))
        yield Row("fps", Input("24", id="r-fps", compact=True))
        yield Row("max_frames", Input("", placeholder="blank = all", id="r-max-frames", compact=True))
        yield Row("cameras", *[Checkbox(m, value=(m == "auto"), id=f"r-cam-{m.replace('_', '-')}", compact=True)
                               for m in t.CAMERA_MODES],
                  tip="one render per path per camera; earth_chase = behind Apophis, aimed at Earth")
        yield Row("camera_fov", Input(f"{t.DEFAULT_CAMERA_FOV:g}", id="r-camera-fov", compact=True),
                  tip="degrees; earth_chase 20-30 = telephoto (same Apophis, bigger Earth)")
        yield Row("paths", *[Checkbox(n, value=(n == "instance_grains"), id=f"r-path-{n.replace('_', '-')}",
                                      compact=True)
                             for n in t.RENDER_PATH_NAMES])
        yield Row("envelope_method", Select([("hull", "hull"), ("sdf", "sdf")], value="hull", allow_blank=False,
                                            id="r-envelope", compact=True), tip="composite only")
        yield Row("placeholder_obj", Input(str(t.DEFAULT_PLACEHOLDER_OBJ), id="r-placeholder", compact=True),
                  tip="instance_static only")
        yield Row("n_points", Input("1000000", id="r-n-points", compact=True), tip="instance_static only")
        yield Row("options", Checkbox("compare", id="r-compare", compact=True),
                  Checkbox("encode_video", id="r-video", compact=True),
                  Checkbox("resume", id="r-resume", compact=True))
        yield Row("video_fps", Input("", placeholder="blank = fps", id="r-video-fps", compact=True))
        yield Row("slowmo", Input("1", id="r-slowmo", compact=True), tip="instance paths only; keep video_fps 24")

    def _v(self, wid: str) -> str:
        return self.query_one("#" + wid, Input).value

    def _c(self, wid: str) -> bool:
        return self.query_one("#" + wid, Checkbox).value

    def read(self):
        t = self.t
        resolution = self._v("r-resolution").strip()
        if not _RES_RE.match(resolution):
            raise ValueError(f"resolution must look like 1920x1080 (got {resolution!r})")
        paths = tuple(n for n in t.RENDER_PATH_NAMES if self._c(f"r-path-{n.replace('_', '-')}"))
        if not paths:
            raise ValueError("tick at least one render path")
        form = t.RenderFormValues(
            resolution=resolution,
            samples=_int("samples", self._v("r-samples")),
            fps=_int("fps", self._v("r-fps")),
            max_frames=_int("max_frames", self._v("r-max-frames"), optional=True),
            camera_modes=tuple(m for m in t.CAMERA_MODES if self._c(f"r-cam-{m.replace('_', '-')}")),
            camera_fov=_float("camera_fov", self._v("r-camera-fov")),
            encode_video=self._c("r-video"),
            video_fps=_int("video_fps", self._v("r-video-fps"), optional=True),
            paths=paths,
            envelope_method=str(self.query_one("#r-envelope", Select).value),
            placeholder_obj=self._v("r-placeholder").strip(),
            n_points=_int("n_points", self._v("r-n-points")),
            compare=self._c("r-compare"),
            slowmo=_int("slowmo", self._v("r-slowmo")),
            resume=self._c("r-resume"),
        )
        if form.compare and len(paths) < 2:
            raise ValueError("compare needs at least two render paths")
        t.validate_render_form(form)
        return form


class RunPicker(Vertical):
    """A batch's runs (Enter toggles a row), or one other run dir rendered via render_dumps.py."""

    DEFAULT_CSS = "RunPicker { height: auto; } RunPicker DataTable { height: 12; }"

    def __init__(self, output_root: Path, **kw) -> None:
        super().__init__(**kw)
        self._root = Path(output_root)
        self._rows: List[RunRow] = []
        self._chosen: Set[int] = set()

    def compose(self) -> ComposeResult:
        yield Row("batch", Select([], allow_blank=True, prompt="choose a batch", id="r-batch", compact=True))
        yield DataTable(id="r-runs", cursor_type="row")
        yield Static("Enter toggles a run; runs without dumps (cleaned up) cannot be rendered.", classes="rl")
        with Collapsible(title="Other run dir (render_dumps.py: instance_grains / composite)", collapsed=True):
            yield Row("run_dir", Input("", placeholder="blank = use the batch above", id="r-other-dir", compact=True))
            yield Row("name", Input("", placeholder="output folder name", id="r-other-name", compact=True))
            yield Row("prefix", Input("sobol", id="r-other-prefix", compact=True))
            yield Row("dem_model", Select([("particle", "particle"), ("sink", "sink")], allow_blank=True,
                                          id="r-other-dem", compact=True))
            yield Row("np_apophis", Input("", id="r-other-np", compact=True))
            yield Row("apophis_only", Checkbox("dumps hold no Earth/Moon/Sun sinks", id="r-other-apophis-only",
                                               compact=True))
            yield Row("encounter", Select([("hyperbola", "hyperbola"), ("ephemeris", "ephemeris")],
                                          allow_blank=True, id="r-other-encounter", compact=True))
        yield Row("cleanup", Checkbox("delete dumps after a successful render", value=False, id="r-delete-dumps",
                                      compact=True))

    def on_mount(self) -> None:
        table = self.query_one("#r-runs", DataTable)
        table.add_columns("✓", "run", "status", "dumps", "run_dir")
        self.reload(self._root)

    def reload(self, output_root: Path) -> None:
        """Re-list the batches; a batch still listed keeps its ticked runs (Render tab opened again)."""
        self._root = Path(output_root)
        sel = self.query_one("#r-batch", Select)
        keep, chosen = sel.value, set(self._chosen)
        listed = batches(self._root)
        found = {str(b) for b in listed}
        # set_options posts Changed(NULL) and setting the value posts Changed(batch); either would clear
        # the ticks in _batch_changed, so both are suppressed and the table is redrawn here
        with sel.prevent(Select.Changed):
            sel.set_options([(b.name, str(b)) for b in listed])
            if keep is not Select.NULL and str(keep) in found:
                sel.value = keep
        if keep is not Select.NULL and str(keep) in found:
            self._show(Path(str(keep)), chosen)
        else:
            self._show(None, set())

    @on(Select.Changed, "#r-batch")
    def _batch_changed(self, event: Select.Changed) -> None:
        self._show(None if event.value is Select.NULL else Path(str(event.value)), set())

    def _show(self, batch: Optional[Path], chosen: Set[int]) -> None:
        self._rows = [] if batch is None else runs_of(batch)
        renderable = {r.run_id for r in self._rows if r.status == "ok" and r.has_dumps}
        self._chosen = chosen & renderable  # a run cleaned up since it was ticked drops out
        table = self.query_one("#r-runs", DataTable)
        table.clear()
        for r in self._rows:
            table.add_row("✓" if r.run_id in self._chosen else "", str(r.run_id), r.status,
                          "yes" if r.has_dumps else "no", str(r.run_dir), key=str(r.run_id))

    @on(DataTable.RowSelected, "#r-runs")
    def _toggle(self, event: DataTable.RowSelected) -> None:
        run_id = int(event.row_key.value)
        row = next(r for r in self._rows if r.run_id == run_id)
        if row.status != "ok" or not row.has_dumps:
            self.app.bell()
            return
        self._chosen ^= {run_id}
        self.query_one("#r-runs", DataTable).update_cell(event.row_key, self.query_one(
            "#r-runs", DataTable).ordered_columns[0].key, "✓" if run_id in self._chosen else "")

    def chosen_runs(self) -> List[Path]:
        return [r.run_dir for r in self._rows if r.run_id in self._chosen]

    @property
    def delete_dumps(self) -> bool:
        return self.query_one("#r-delete-dumps", Checkbox).value

    def other_run_argv(self, form) -> Optional[List[str]]:
        run_dir = self.query_one("#r-other-dir", Input).value.strip()
        if not run_dir:
            return None
        np_text = self.query_one("#r-other-np", Input).value.strip()
        dem = self.query_one("#r-other-dem", Select).value
        enc = self.query_one("#r-other-encounter", Select).value
        return render_backend.render_dumps_argv(
            Path(run_dir).expanduser().resolve(), form,
            name=self.query_one("#r-other-name", Input).value.strip() or None,
            prefix=self.query_one("#r-other-prefix", Input).value.strip() or "sobol",
            dem_model=None if dem is Select.NULL else str(dem),
            np_apophis=_int("np_apophis", np_text) if np_text else None,
            apophis_only=self.query_one("#r-other-apophis-only", Checkbox).value,
            encounter=None if enc is Select.NULL else str(enc),
        )
