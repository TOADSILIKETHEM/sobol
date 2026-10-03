#!/usr/bin/env python3
"""Stitch one run's render paths side by side to compare them.

tui_sim_render.py renders each ticked path into its own
DEMCSVs/<batch>/<run>_render_<path>/frame_####.png. This module pairs
frames by number across 2+ of those dirs, pastes them left to right with a
label box, and writes <run>_compare/compare_####.png (contiguous, 1-based)
plus an optional animation.mp4 (WSL ffmpeg, libx264). The per-path renders
are never modified.

CLI (already-rendered runs, no sim):
    python3 sobol/render_compare.py DIR DIR [DIR ...] [--output-dir OUT] [--fps 24] [--encode-video]
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

FRAME_RE = re.compile(r"^frame_(\d+)\.png$")
COMPARE_PREFIX = "compare_"
VIDEO_NAME = "animation.mp4"
# libx264 tops out at 4096 px wide; stay under it.
MAX_COMPARE_WIDTH = 3840


class CompareError(RuntimeError):
    pass


def frame_numbers(render_dir: Path) -> Dict[int, Path]:
    """Frame number -> path for every frame_####.png in render_dir ({} if missing)."""
    if not render_dir.is_dir():
        return {}
    out = {}
    for p in render_dir.iterdir():
        m = FRAME_RE.match(p.name)
        if m:
            out[int(m.group(1))] = p
    return out


def label_for(render_dir: Path) -> str:
    """'run_0001_render_composite' -> 'composite'; other names pass through."""
    name = render_dir.name
    return name.split("_render_", 1)[1] if "_render_" in name else name


def default_compare_dir(render_dir: Path) -> Path:
    """<batch>/run_0001_render_composite -> <batch>/run_0001_compare."""
    run_name = render_dir.name.split("_render_", 1)[0]
    return render_dir.parent / f"{run_name}_compare"


@dataclass
class ComparePlan:
    labels: List[str]
    frames: List[Tuple[int, List[Path]]]  # (frame number, one png per label), ascending
    warnings: List[str] = field(default_factory=list)


def plan_compare(render_dirs: Sequence[Tuple[str, Path]]) -> ComparePlan:
    """Pair frames by number across (label, dir) pairs, keeping the given order."""
    if len(render_dirs) < 2:
        raise CompareError(f"compare needs at least 2 render dirs (got {len(render_dirs)})")
    maps = [(label, frame_numbers(d)) for label, d in render_dirs]
    empty = [label for label, m in maps if not m]
    if empty:
        raise CompareError(f"no frame_*.png in: {', '.join(empty)}")
    counts = ", ".join(f"{label} {len(m)}" for label, m in maps)
    common = sorted(set.intersection(*(set(m) for _, m in maps)))
    if not common:
        raise CompareError(f"no frame number shared by every path ({counts})")
    warnings = []
    if any(len(m) != len(common) for _, m in maps):
        warnings.append(f"frame counts differ ({counts}); compared the {len(common)} shared")
    frames = [(n, [m[n] for _, m in maps]) for n in common]
    return ComparePlan([label for label, _ in maps], frames, warnings)


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1 has no size argument
        return ImageFont.load_default()


def stitch_frame(images: Sequence[Path], labels: Sequence[str], out_path: Path,
                 max_width: int = MAX_COMPARE_WIDTH) -> Tuple[int, int]:
    """Paste images left to right at one shared height, label each, save out_path.

    Tiles are scaled (never cropped) to the smallest height, then all of them
    down to max_width if needed. Returns the saved (width, height); both are
    even so yuv420p H.264 accepts them.
    """
    tiles = [Image.open(p).convert("RGB") for p in images]
    height = min(t.height for t in tiles)
    tiles = [
        t if t.height == height else t.resize((max(1, round(t.width * height / t.height)), height))
        for t in tiles
    ]
    total = sum(t.width for t in tiles)
    if total > max_width:
        scale = max_width / total
        height = max(2, int(height * scale))
        tiles = [t.resize((max(1, int(t.width * scale)), height)) for t in tiles]
        total = sum(t.width for t in tiles)

    canvas = Image.new("RGB", (total, height))
    x = 0
    for tile in tiles:
        canvas.paste(tile, (x, 0))
        x += tile.width
    draw = ImageDraw.Draw(canvas)
    font = _font(max(14, height // 30))
    pad = max(4, height // 120)
    x = 0
    for tile, label in zip(tiles, labels):
        left, top, right, bottom = draw.textbbox((x + 2 * pad, 2 * pad), label, font=font)
        draw.rectangle((left - pad, top - pad, right + pad, bottom + pad), fill=(0, 0, 0))
        draw.text((x + 2 * pad, 2 * pad), label, fill=(255, 255, 255), font=font)
        x += tile.width

    even = canvas.crop((0, 0, total - total % 2, height - height % 2))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    even.save(out_path)
    return even.size


def encode_compare_video(frames_dir: Path, fps: int, ffmpeg: Optional[str] = None) -> Path:
    """compare_####.png in frames_dir -> frames_dir/animation.mp4 (H.264, yuv420p)."""
    exe = ffmpeg or shutil.which("ffmpeg")
    if exe is None:
        raise CompareError("ffmpeg not found on PATH (sudo apt install ffmpeg); stitched frames kept")
    video = frames_dir / VIDEO_NAME
    cmd = [
        exe, "-y", "-loglevel", "error",
        "-framerate", str(fps), "-start_number", "1",
        "-i", str(frames_dir / f"{COMPARE_PREFIX}%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        str(video),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    except OSError as exc:
        raise CompareError(f"could not start {exe}: {exc}") from exc
    if result.returncode != 0 or not video.is_file():
        lines = [line for line in (result.stderr or "").splitlines() if line.strip()]
        raise CompareError(f"ffmpeg exited {result.returncode}: {lines[-1] if lines else 'no output'}")
    return video


@dataclass
class CompareResult:
    output_dir: Path
    n_frames: int
    labels: List[str]
    warnings: List[str]
    video: Optional[Path] = None


def run_compare_stage(render_dirs: Sequence[Tuple[str, Path]], output_dir: Path,
                      fps: int = 24, encode_video: bool = False) -> CompareResult:
    """Stitch every shared frame of render_dirs into output_dir. Raises CompareError."""
    plan = plan_compare(render_dirs)
    output_dir.mkdir(parents=True, exist_ok=True)
    # A re-run with fewer frames must not leave old frames in the new video.
    for stale in output_dir.glob(f"{COMPARE_PREFIX}*.png"):
        stale.unlink()
    (output_dir / VIDEO_NAME).unlink(missing_ok=True)
    for i, (frame_no, pngs) in enumerate(plan.frames, start=1):
        try:
            stitch_frame(pngs, plan.labels, output_dir / f"{COMPARE_PREFIX}{i:04d}.png")
        except OSError as exc:  # PIL.UnidentifiedImageError is an OSError
            raise CompareError(f"could not stitch frame {frame_no}: {exc}") from exc
    video = encode_compare_video(output_dir, fps) if encode_video else None
    return CompareResult(output_dir, len(plan.frames), plan.labels, plan.warnings, video)


def format_compare_result(r: CompareResult) -> str:
    text = f"compare: ok, {r.n_frames} frame(s) ({' | '.join(r.labels)}) to {r.output_dir}"
    if r.video is not None:
        text += f", encoded to {r.video}"
    for w in r.warnings:
        text += f"; {w}"
    return text


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Stitch render-path frame dirs side by side.")
    p.add_argument("render_dirs", nargs="+", type=Path,
                   help="2+ <run>_render_<path>/ dirs, in left-to-right order")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="default: <run>_compare/ beside the first dir")
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--encode-video", action="store_true")
    args = p.parse_args(argv)
    out = args.output_dir or default_compare_dir(args.render_dirs[0])
    try:
        result = run_compare_stage(
            [(label_for(d), d) for d in args.render_dirs], out,
            fps=args.fps, encode_video=args.encode_video,
        )
    except CompareError as exc:
        print(f"compare failed: {exc}", file=sys.stderr)
        return 1
    print(format_compare_result(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
