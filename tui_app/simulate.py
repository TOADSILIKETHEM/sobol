"""Simulate tab body: every runner flag as a row, built from params_schema.

Rows: plain flags get one widget; axes get a sweep checkbox, a fixed input and min/max inputs.
Gates (params_schema.enabled) only grey rows out; Validate / Launch ask the runner itself.
"""
from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.message import Message
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Checkbox, Input, RadioButton, RadioSet, Select, Static

import params_schema as ps


def field_id(dest: str) -> str:
    return "f-" + dest.replace("_", "-")


def axis_id(stem: str, part: str) -> str:
    return f"ax-{stem.replace('_', '-')}-{part}"


def section_id(section: str) -> str:
    return "sec-" + "".join(c if c.isalnum() else "-" for c in section.lower())


class Row(Horizontal):
    DEFAULT_CSS = """
    Row { height: auto; }
    Row .rl { width: 30; padding: 0 1 0 0; color: $text-muted; }
    Row Input, Row Select { width: 1fr; margin: 0 1 0 0; }
    Row Checkbox { width: auto; margin: 0 2 0 0; }
    """

    def __init__(self, label: str, *widgets, tip: str = "", **kw) -> None:
        super().__init__(**kw)
        self._label, self._widgets, self._tip = label, widgets, tip

    def compose(self) -> ComposeResult:
        yield Static(self._label, classes="rl")
        yield from self._widgets

    def on_mount(self) -> None:
        if self._tip:
            self.tooltip = self._tip


def _plain_widget(f: ps.PlainField, value):
    wid = field_id(f.dest)
    if f.kind == "check":
        return Checkbox("on", value=bool(value), id=wid, compact=True)  # the row label already names it
    if f.kind == "select":
        return Select([(c, c) for c in f.choices], value=value if value else Select.NULL, allow_blank=True, id=wid,
                      compact=True)
    hint = f"default {f.default}" if f.default not in (None, False) else "blank = template / unset"
    if f.kind == "list":
        hint = "space-separated list"
    return Input(str(value or ""), placeholder=hint, id=wid, compact=True)


AXIS_HINTS = {"f_boulder": "fraction 0-1 (0.4 = 40%)"}

# Changes to these can need new defaults (fill_defaults); typing elsewhere never does.
_FILL_TRIGGER_IDS = frozenset(
    {field_id(d) for d in ("body_source", "encounter", "boulders", "flyby_start_sep_km",
                           "flyby_perturber_earth_masses")}
    | {axis_id(stem, part) for stem in ("flyby_rp_km", "flyby_vinf_kms") for part in ("fixed", "lo", "hi", "sweep")}
)


def _axis_row(ax: ps.Axis, av: ps.AxisValue) -> Row:
    fixed_hint = AXIS_HINTS.get(ax.stem) or (
        "fixed value" if (ax.fixed or ax.pin) else "(template value; tick sweep to vary)")
    return Row(
        ax.label,
        Checkbox("sweep", value=av.sweep, id=axis_id(ax.stem, "sweep"), compact=True),
        Input(av.fixed, placeholder=fixed_hint, id=axis_id(ax.stem, "fixed"), compact=True),
        Input(av.lo, placeholder="min", id=axis_id(ax.stem, "lo"), compact=True),
        Input(av.hi, placeholder="max", id=axis_id(ax.stem, "hi"), compact=True),
        id=axis_id(ax.stem, "row"),
    )


class SimulateForm(Vertical):
    DEFAULT_CSS = """
    SimulateForm { height: 1fr; }
    SimulateForm #sim-body { height: 1fr; }
    SimulateForm .sec { color: $accent; text-style: bold; padding: 1 0 0 0; }
    SimulateForm #mode-bar { height: auto; padding: 0 0 1 0; border-bottom: hkey $panel; }
    /* the radio cursor (blue) is only drawn while the set has focus; the green dot shows the mode */
    SimulateForm #mode:blur > RadioButton.-selected > .toggle--label { background: transparent; }
    SimulateForm #mode { width: auto; height: auto; layout: horizontal; }
    SimulateForm #mode RadioButton { width: auto; }
    SimulateForm #mode-bar Static { width: auto; padding: 0 1 0 2; color: $text-muted; }
    SimulateForm #omp { width: 10; }
    """

    class DefaultsFilled(Message):
        """Blank fields were filled because of a mode choice; notes are "field value" strings."""

        def __init__(self, notes) -> None:
            super().__init__()
            self.notes = list(notes)

    def __init__(self, initial: ps.FormValues, mode: str = "sweep", **kw) -> None:
        super().__init__(**kw)
        self._initial, self._initial_mode = initial, mode
        self._auto_tmax = ""  # tmax the last fill wrote; it keeps following rp / v_inf until typed over

    def compose(self) -> ComposeResult:
        with Horizontal(id="mode-bar"):
            with RadioSet(id="mode", compact=True):
                yield RadioButton("Single run", value=self._initial_mode == "single", id="mode-single", compact=True)
                yield RadioButton("Sweep", value=self._initial_mode != "single", id="mode-sweep", compact=True)
            yield Static("  OMP_NUM_THREADS ", classes="rl")
            yield Input("4" if self._initial_mode == "single" else "1", id="omp", compact=True)
        with VerticalScroll(id="sim-body"):  # the mode bar above stays put while this scrolls
            for section in ps.SECTION_ORDER:
                yield Static(section, classes="sec", id=section_id(section))
                for item in ps.items_in(section):
                    if isinstance(item, ps.Axis):
                        yield _axis_row(item, self._initial.axes.get(item.stem, ps.AxisValue()))
                    else:
                        yield Row(item.dest, _plain_widget(item, self._initial.plain.get(item.dest)),
                                  tip=item.brief, id="row-" + field_id(item.dest))

    def on_mount(self) -> None:
        self.sync()
        self.fill_defaults()

    @property
    def mode(self) -> str:
        return "single" if self.query_one("#mode-single", RadioButton).value else "sweep"

    @property
    def omp(self) -> int:
        text = self.query_one("#omp", Input).value.strip()
        try:
            n = int(text)
        except ValueError:
            raise ValueError(f"OMP_NUM_THREADS must be an integer (got {text!r})") from None
        if n < 1:
            raise ValueError("OMP_NUM_THREADS must be >= 1")
        return n

    def read(self) -> ps.FormValues:
        plain = {}
        for f in ps.PLAIN:
            w = self.query_one("#" + field_id(f.dest))
            if f.kind == "check":
                plain[f.dest] = w.value
            elif f.kind == "select":
                plain[f.dest] = None if w.value is Select.NULL else str(w.value)
            else:
                plain[f.dest] = w.value.strip()
        axes = {
            ax.stem: ps.AxisValue(
                sweep=self.query_one("#" + axis_id(ax.stem, "sweep"), Checkbox).value,
                fixed=self.query_one("#" + axis_id(ax.stem, "fixed"), Input).value.strip(),
                lo=self.query_one("#" + axis_id(ax.stem, "lo"), Input).value.strip(),
                hi=self.query_one("#" + axis_id(ax.stem, "hi"), Input).value.strip(),
            )
            for ax in ps.AXES
        }
        return ps.FormValues(plain, axes)

    def write(self, fv: ps.FormValues) -> None:
        for f in ps.PLAIN:
            w = self.query_one("#" + field_id(f.dest))
            v = fv.plain.get(f.dest)
            if f.kind == "check":
                w.value = bool(v)
            elif f.kind == "select":
                if v:
                    w.value = v
                else:
                    w.clear()
            else:
                w.value = str(v or "")
        for ax in ps.AXES:
            av = fv.axes.get(ax.stem, ps.AxisValue())
            self.query_one("#" + axis_id(ax.stem, "sweep"), Checkbox).value = av.sweep
            self.query_one("#" + axis_id(ax.stem, "fixed"), Input).value = av.fixed
            self.query_one("#" + axis_id(ax.stem, "lo"), Input).value = av.lo
            self.query_one("#" + axis_id(ax.stem, "hi"), Input).value = av.hi
        self.sync()

    def sync(self) -> None:
        fv = self.read()
        single = self.mode == "single"
        self.query_one("#" + section_id("Sampling")).display = not single
        for f in ps.PLAIN:
            self.query_one("#row-" + field_id(f.dest)).display = not (
                single and (f.section == "Sampling" or f.dest in ps.SINGLE_HIDDEN))
            self.query_one("#" + field_id(f.dest)).disabled = not ps.enabled(f.dest, fv)
        for ax in ps.AXES:
            on_ = ps.enabled(ax.stem, fv)
            sweep = fv.axes[ax.stem].sweep and not single
            toggle = self.query_one("#" + axis_id(ax.stem, "sweep"))
            toggle.display, toggle.disabled = not single, not on_
            fixed = self.query_one("#" + axis_id(ax.stem, "fixed"))
            fixed.display = not sweep
            fixed.disabled = not on_ or not (ax.fixed or ax.pin)
            for part in ("lo", "hi"):
                w = self.query_one("#" + axis_id(ax.stem, part))
                w.display, w.disabled = sweep, not on_

    def fill_defaults(self) -> None:
        """Fill blank fields the chosen modes need; only changed widgets are written."""
        before = self.read()
        res = ps.fill_defaults(before, self._auto_tmax)
        self._auto_tmax = res.auto_tmax
        if not res.notes and ps.to_argv(res.fv) == ps.to_argv(before):
            return
        for f in ps.PLAIN:
            new = res.fv.plain.get(f.dest)
            if new != before.plain.get(f.dest):
                self.query_one("#" + field_id(f.dest)).value = new
        for ax in ps.AXES:
            new = res.fv.axes[ax.stem].fixed
            if new != before.axes[ax.stem].fixed:
                self.query_one("#" + axis_id(ax.stem, "fixed"), Input).value = new
        self.sync()
        if res.notes:
            self.post_message(self.DefaultsFilled(res.notes))

    @on(RadioSet.Changed, "#mode")
    def _mode_changed(self) -> None:
        self.query_one("#omp", Input).value = "4" if self.mode == "single" else "1"
        self.sync()

    @on(Checkbox.Changed)
    def _checkbox_changed(self, event: Checkbox.Changed) -> None:
        self.sync()
        if event.checkbox.id in _FILL_TRIGGER_IDS:
            self.fill_defaults()

    @on(Select.Changed)
    def _select_changed(self, event: Select.Changed) -> None:
        self.sync()
        if event.select.id in _FILL_TRIGGER_IDS:
            self.fill_defaults()

    @on(Input.Changed)
    def _input_changed(self, event: Input.Changed) -> None:
        if event.input.id in _FILL_TRIGGER_IDS:
            self.fill_defaults()
