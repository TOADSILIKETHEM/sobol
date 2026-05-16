#!/usr/bin/env python3
"""Textual TUI for configuring and launching the PHANTOM Sobol sweep.

Launch:
    python3 sobol/tui_run.py [any flags from run_mass_sobol_phantom.py]

CLI flags pre-fill form defaults.  Navigate with Tab / Shift-Tab, Space to
toggle checkboxes, then press [Run Sweep], [Dry Run], or [Quit] (also bound
to R, D, Q).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent))

try:
    from textual import on
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, ScrollableContainer
    from textual.widgets import Button, Checkbox, Footer, Header, Input, Select, Static
except ImportError:
    sys.exit(
        "textual is not installed.\n"
        "Run:  pip install textual\n"
        "Or add it to sobol/requirements.txt and run:  pip install -r sobol/requirements.txt"
    )

import run_mass_sobol_phantom as _runner

# ── constants ────────────────────────────────────────────────────────────────

# (CSS-id-stem, attr-name, checkbox label, is_integer)
_SCALE_DIMS = [
    ("scale-vel",       "scale_vel",       "scale_vel  — Apophis velocity scale",     False),
    ("scale-pos",       "scale_pos",       "scale_pos  — Apophis position scale",     False),
    ("scale-r-apophis", "scale_r_apophis", "scale_r_apophis  — Apophis radius scale", False),
    ("scale-rho",       "scale_rho",       "scale_rho  — bulk density scale",         False),
]

_MASS_GATE_IDS = ("mass-min-kg", "mass-max-kg", "mass-unit")


# ── helper: one labelled row ─────────────────────────────────────────────────

class _Row(Horizontal):
    """Label + interactive widget on a single row, with optional right-side hint."""

    DEFAULT_CSS = """
    _Row {
        height: 3;
    }
    _Row .rl {
        width: 26;
        content-align: right middle;
        padding-right: 1;
        color: $text-muted;
    }
    _Row .rw {
        width: 1fr;
    }
    _Row .rh {
        width: 15;
        content-align: left middle;
        padding-left: 1;
        color: $text-muted;
        text-style: italic;
    }
    """

    def __init__(self, label: str, widget, hint: str = "", **kw) -> None:
        super().__init__(**kw)
        self._label = label
        self._widget = widget
        self._hint = hint

    def compose(self) -> ComposeResult:
        yield Static(self._label, classes="rl")
        self._widget.add_class("rw")
        yield self._widget
        if self._hint:
            yield Static(self._hint, classes="rh")


# ── main app ─────────────────────────────────────────────────────────────────

class SobolTUIApp(App[Optional[List[str]]]):
    """Full-screen parameter form for the PHANTOM Sobol sweep."""

    TITLE = "PHANTOM · Sobol Sweep"
    SUB_TITLE = "edit parameters → launch"

    BINDINGS = [
        Binding("r", "run_sweep", "Run"),
        Binding("d", "do_dry_run", "Dry-run"),
        Binding("q", "app_quit", "Quit"),
    ]

    DEFAULT_CSS = """
    Screen { background: $surface; }

    #scroll {
        height: 1fr;
        border: solid $primary;
        margin: 0 1;
        padding: 0 2;
    }

    .sec {
        color: $accent;
        text-style: bold;
        padding: 1 0 0 0;
    }

    #bar {
        height: 5;
        padding: 1 2;
        background: $surface-darken-1;
        border-top: solid $primary;
        align: center middle;
    }

    #status {
        width: 1fr;
        content-align: left middle;
        padding-left: 2;
    }

    .err { color: $error; }
    .ok  { color: $success; }

    Button { margin: 0 1; }
    """

    def __init__(self, defaults) -> None:
        super().__init__()
        self._d = defaults

    # ── compose ──────────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:  # noqa: C901
        d = self._d

        yield Header()

        with ScrollableContainer(id="scroll"):

            # ── Paths ──────────────────────────────────────────────────────
            yield Static("Paths", classes="sec")
            yield _Row("prefix",
                       Input(str(d.prefix or "sobol"), id="prefix"))
            yield _Row("base_dir",
                       Input(str(d.base_dir or "."), id="base-dir"), "dir")
            yield _Row("ephemeris_cache_dir",
                       Input(str(d.ephemeris_cache_dir or ""), id="eph-cache",
                             placeholder="(none — skips Horizons download)"),
                       "optional dir")
            yield _Row("phantom_dir",
                       Input(str(d.phantom_dir or ""), id="phantom-dir"), "dir")
            yield _Row("output_root",
                       Input(str(d.output_root or "sobol_mass_runs"), id="output-root"),
                       "dir")

            # ── Sampling ───────────────────────────────────────────────────
            yield Static("Sampling", classes="sec")
            yield _Row("num_samples",
                       Input(str(d.num_samples if d.num_samples is not None else 8),
                             id="num-samples"),
                       "int ≥ 1")
            yield _Row("seed",
                       Input(str(d.seed if d.seed is not None else 42), id="seed"),
                       "int")
            yield _Row("saltelli_n",
                       Input(str(d.saltelli_n or ""), id="saltelli-n",
                             placeholder="(blank → use num_samples)"),
                       "int ≥ 2 or blank")
            yield _Row("saltelli_2nd_order",
                       Checkbox("also estimate 2nd-order Sobol indices",
                                value=bool(getattr(d, "saltelli_calc_second_order", False)),
                                id="saltelli-2nd"))

            # ── Mass ───────────────────────────────────────────────────────
            yield Static("Mass", classes="sec")
            mass_on = d.mass_min_kg is not None and d.mass_max_kg is not None
            yield _Row("vary mass",
                       Checkbox("vary Apophis mass as a Sobol dimension",
                                value=mass_on, id="vary-mass"))
            yield _Row("mass_min_kg",
                       Input(str(d.mass_min_kg or ""), id="mass-min-kg",
                             disabled=not mass_on, placeholder="e.g. 1e10"),
                       "kg")
            yield _Row("mass_max_kg",
                       Input(str(d.mass_max_kg or ""), id="mass-max-kg",
                             disabled=not mass_on, placeholder="e.g. 1e11"),
                       "kg")
            yield _Row("mass_unit",
                       Select([("kg", "kg"), ("g", "g"), ("msun", "msun")],
                              value=(d.mass_unit or "kg"),
                              id="mass-unit", disabled=not mass_on))

            # ── Scale bounds ───────────────────────────────────────────────
            yield Static("Scale bounds  (optional Sobol dimensions)", classes="sec")
            for css, attr, label, is_int in _SCALE_DIMS:
                lo = getattr(d, attr + "_min", None)
                hi = getattr(d, attr + "_max", None)
                active = lo is not None and hi is not None
                hint = "int" if is_int else "float"
                yield _Row(f"vary {attr}",
                           Checkbox(label, value=active, id=f"vary-{css}"))
                yield _Row(f"  {attr}_min",
                           Input(str(lo or ""), id=f"{css}-min",
                                 disabled=not active, placeholder="min"),
                           hint)
                yield _Row(f"  {attr}_max",
                           Input(str(hi or ""), id=f"{css}-max",
                                 disabled=not active, placeholder="max"),
                           hint)

            # ── Setup toggles ──────────────────────────────────────────────
            yield Static("Setup toggles", classes="sec")
            vary_dem = bool(getattr(d, "vary_use_dem", False))
            dem_fixed_raw = getattr(d, "use_dem_fixed", None)
            dem_fixed_val = dem_fixed_raw if dem_fixed_raw in ("true", "false") else ""
            yield _Row("vary_use_dem",
                       Checkbox("vary use_dem as a Sobol dimension  (u≥0.5 → True)",
                                value=vary_dem, id="vary-use-dem"))
            yield _Row("  use_dem_fixed",
                       Select([("(from template)", ""),
                               ("True", "true"),
                               ("False", "false")],
                              value=dem_fixed_val,
                              id="use-dem-fixed", disabled=vary_dem),
                       "if not varying")
            vary_obj = bool(getattr(d, "vary_use_obj_crop", False))
            obj_fixed_raw = getattr(d, "use_obj_crop_fixed", None)
            obj_fixed_val = obj_fixed_raw if obj_fixed_raw in ("true", "false") else ""
            obj_path_active = vary_obj or obj_fixed_val == "true"
            yield _Row("vary_use_obj_crop",
                       Checkbox("vary OBJ cropping as a Sobol dimension  (u≥0.5 → crop)",
                                value=vary_obj, id="vary-use-obj-crop"))
            yield _Row("  use_obj_crop_fixed",
                       Select([("(from template)", ""),
                               ("True", "true"),
                               ("False", "false")],
                              value=obj_fixed_val,
                              id="use-obj-crop-fixed", disabled=vary_obj),
                       "if not varying")
            yield _Row("  obj_crop_file",
                       Input(str(getattr(d, "obj_crop_file", "") or ""), id="obj-crop-file",
                             disabled=not obj_path_active,
                             placeholder="path/to/apophis.obj  (required when cropping on)"),
                       "OBJ path")
            yield _Row("vary_apophis_only",
                       Checkbox("vary apophis_only (Earth absent when True; CA → NaN)",
                                value=bool(getattr(d, "vary_apophis_only", False)),
                                id="vary-apophis-only"))

            # ── Sinks / post-processing ────────────────────────────────────
            yield Static("Sinks / post-processing", classes="sec")
            yield _Row("sink_earth_id",
                       Input(str(getattr(d, "sink_earth_id", _runner.EARTH_SINK_ID_DEFAULT)),
                             id="sink-earth"),
                       "int")
            yield _Row("sink_apophis_id",
                       Input(str(getattr(d, "sink_apophis_id", _runner.APOPHIS_SINK_ID_DEFAULT)),
                             id="sink-apophis"),
                       "int")

            # ── Batch naming ───────────────────────────────────────────────
            yield Static("Batch naming", classes="sec")
            yield _Row("batch_label",
                       Input(str(getattr(d, "batch_label", "") or ""), id="batch-label",
                             placeholder="(auto slug from varied params)"),
                       "optional")
            yield _Row("batch_slug_max_len",
                       Input(str(getattr(d, "batch_slug_max_len",
                                         _runner._BATCH_SLUG_DEFAULT_MAX_LEN)),
                             id="slug-max-len"),
                       "int ≥ 17")

            # ── Execution ──────────────────────────────────────────────────
            yield Static("Execution", classes="sec")
            yield _Row("dry_run",
                       Checkbox("dry run  (prepare dirs only, skip PHANTOM binary)",
                                value=bool(getattr(d, "dry_run", False)),
                                id="dry-run"))
            yield _Row("jobs",
                       Input(str(getattr(d, "jobs", 1)), id="jobs"),
                       "parallel workers")

        # ── button bar ─────────────────────────────────────────────────────
        with Horizontal(id="bar"):
            yield Button("Run Sweep", id="btn-run",    variant="success")
            yield Button("Dry Run",   id="btn-dryrun", variant="primary")
            yield Button("Quit",      id="btn-quit",   variant="error")
            yield Static("Ready — edit fields above, then press Run Sweep.", id="status")

        yield Footer()

    # ── gate handlers ─────────────────────────────────────────────────────────

    @on(Checkbox.Changed)
    def _checkbox_gate(self, event: Checkbox.Changed) -> None:
        cb_id = event.checkbox.id
        active = event.value

        if cb_id == "vary-mass":
            for fid in _MASS_GATE_IDS:
                self.query_one(f"#{fid}").disabled = not active

        elif cb_id == "vary-use-dem":
            self.query_one("#use-dem-fixed").disabled = active

        elif cb_id == "vary-use-obj-crop":
            self.query_one("#use-obj-crop-fixed").disabled = active
            # path field is needed whenever cropping may be on
            self.query_one("#obj-crop-file").disabled = not active

        else:
            for css, _, _, _ in _SCALE_DIMS:
                if cb_id == f"vary-{css}":
                    self.query_one(f"#{css}-min").disabled = not active
                    self.query_one(f"#{css}-max").disabled = not active
                    break

    @on(Select.Changed)
    def _select_gate(self, event: Select.Changed) -> None:
        if event.select.id == "use-obj-crop-fixed":
            val = "" if event.value is Select.BLANK else str(event.value)
            self.query_one("#obj-crop-file").disabled = (val != "true")

    # ── button / action handlers ───────────────────────────────────────────────

    @on(Button.Pressed)
    def _on_button(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "btn-run":
            self._launch(dry=False)
        elif bid == "btn-dryrun":
            self._launch(dry=True)
        elif bid == "btn-quit":
            self.exit(None)

    def action_run_sweep(self) -> None:
        self._launch(dry=False)

    def action_do_dry_run(self) -> None:
        self._launch(dry=True)

    def action_app_quit(self) -> None:
        self.exit(None)

    # ── internals ─────────────────────────────────────────────────────────────

    def _set_status(self, msg: str, *, error: bool = False) -> None:
        s = self.query_one("#status", Static)
        s.update(msg)
        s.remove_class("err", "ok")
        s.add_class("err" if error else "ok")

    def _launch(self, dry: bool) -> None:
        try:
            argv = self._build_argv(force_dry=dry)
        except ValueError as exc:
            self._set_status(f"Error: {exc}", error=True)
            return
        self.exit(argv)

    # ── query helpers ─────────────────────────────────────────────────────────

    def _iv(self, wid: str) -> str:
        return self.query_one(f"#{wid}", Input).value.strip()

    def _cb(self, wid: str) -> bool:
        return self.query_one(f"#{wid}", Checkbox).value

    def _sel(self, wid: str) -> str:
        val = self.query_one(f"#{wid}", Select).value
        return "" if val is Select.BLANK else str(val)

    # ── argv builder ──────────────────────────────────────────────────────────

    def _build_argv(self, force_dry: bool = False) -> List[str]:  # noqa: C901
        argv: List[str] = []

        def si(wid: str, flag: str) -> None:
            """Emit string flag if non-empty."""
            v = self._iv(wid)
            if v:
                argv.extend([flag, v])

        def ii(wid: str, flag: str) -> None:
            """Emit integer flag if non-empty; raise on bad input."""
            v = self._iv(wid)
            if not v:
                return
            try:
                int(v)
            except ValueError:
                raise ValueError(f"{flag} must be an integer (got '{v}')")
            argv.extend([flag, v])

        def fi(wid: str, flag: str) -> None:
            """Emit float flag if non-empty; raise on bad input."""
            v = self._iv(wid)
            if not v:
                return
            try:
                float(v)
            except ValueError:
                raise ValueError(f"{flag} must be a number (got '{v}')")
            argv.extend([flag, v])

        # Paths
        si("prefix",      "--prefix")
        si("base-dir",    "--base-dir")
        si("eph-cache",   "--ephemeris-cache-dir")
        si("phantom-dir", "--phantom-dir")
        si("output-root", "--output-root")

        # Sampling
        ii("num-samples", "--num-samples")
        ii("seed",        "--seed")
        sn_raw = self._iv("saltelli-n")
        if sn_raw:
            try:
                sn = int(sn_raw)
            except ValueError:
                raise ValueError(f"--saltelli-n must be an integer (got '{sn_raw}')")
            if sn < 2:
                raise ValueError("--saltelli-n must be ≥ 2")
            argv.extend(["--saltelli-n", str(sn)])
        if self._cb("saltelli-2nd"):
            argv.append("--saltelli-calc-second-order")

        # Mass
        if self._cb("vary-mass"):
            fi("mass-min-kg", "--mass-min-kg")
            fi("mass-max-kg", "--mass-max-kg")
            mu = self._sel("mass-unit")
            if mu:
                argv.extend(["--mass-unit", mu])

        # Scale bounds
        for css, attr, _, is_int in _SCALE_DIMS:
            if self._cb(f"vary-{css}"):
                flag_base = "--" + attr.replace("_", "-")
                if is_int:
                    ii(f"{css}-min", f"{flag_base}-min")
                    ii(f"{css}-max", f"{flag_base}-max")
                else:
                    fi(f"{css}-min", f"{flag_base}-min")
                    fi(f"{css}-max", f"{flag_base}-max")

        # Setup toggles
        if self._cb("vary-use-dem"):
            argv.append("--vary-use-dem")
        else:
            df = self._sel("use-dem-fixed")
            if df:
                argv.extend(["--use-dem-fixed", df])
        if self._cb("vary-use-obj-crop"):
            argv.append("--vary-use-obj-crop")
            si("obj-crop-file", "--obj-crop-file")
        else:
            ocf = self._sel("use-obj-crop-fixed")
            if ocf:
                argv.extend(["--use-obj-crop-fixed", ocf])
                if ocf == "true":
                    si("obj-crop-file", "--obj-crop-file")
        if self._cb("vary-apophis-only"):
            argv.append("--vary-apophis-only")

        # Sinks
        ii("sink-earth",   "--sink-earth-id")
        ii("sink-apophis", "--sink-apophis-id")

        # Batch naming
        bl = self._iv("batch-label")
        if bl:
            argv.extend(["--batch-label", bl])
        ii("slug-max-len", "--batch-slug-max-len")

        # Execution
        if force_dry or self._cb("dry-run"):
            argv.append("--dry-run")
        ii("jobs", "--jobs")

        return argv


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    cli_argv = [a for a in sys.argv[1:] if a not in ("--interactive", "-i", "--tui")]
    initial_args = _runner.build_parser().parse_args(cli_argv)

    app = SobolTUIApp(initial_args)
    result = app.run()

    if result is None:
        return  # user quit without running

    runner = Path(__file__).parent / "run_mass_sobol_phantom.py"
    cmd = [sys.executable, str(runner), *result]
    print("Running:", " ".join(cmd), flush=True)
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
