"""The Honours code lives in WSL; data/assets stay in OneDrive."""
import re
from pathlib import Path

from tests._paths import CODE


def _unc_to_wsl(unc: str) -> Path:
    # \\wsl.localhost\Ubuntu\home\... -> /home/...
    m = re.match(r"^\\\\wsl\.localhost\\[^\\]+(\\.*)$", unc)
    assert m, f"not a wsl.localhost UNC path: {unc!r}"
    return Path(m.group(1).replace("\\", "/"))


def test_gui_fallback_dir_points_at_wsl_code():
    # Blender's Scripting tab leaves __file__ unset, so these scripts fall
    # back to _DEFAULT_BLENDER_CONVERT_DIR to import their siblings.
    for name in ("DEMBulkInstanceBlender.py", "DEMBulkVizBlender.py"):
        src = (CODE / "BlenderConvert" / name).read_text(encoding="utf-8")
        m = re.search(r"_DEFAULT_BLENDER_CONVERT_DIR = \(\s*r'([^']+)'\s*\)", src)
        assert m, f"{name}: _DEFAULT_BLENDER_CONVERT_DIR not a raw-string UNC literal"
        fallback = _unc_to_wsl(m.group(1))
        assert fallback == CODE / "BlenderConvert"
        assert (fallback / "DEMCinematicSetup.py").is_file()


def test_no_code_paths_point_at_onedrive():
    # After the move, OneDrive paths may only name DATA (DEMCSVs, .blend,
    # Shapes, assets, csv folders, .venv) -- never a tracked code folder.
    import subprocess

    stale = re.compile(
        r"GC-Max_desktop[/\\]Honours[/\\]Code[/\\]"
        r"(BlenderConvert[/\\][A-Za-z0-9_]+\.py|viz[/\\]|CSVconvert[/\\])"
    )
    roots = [Path("/home/mboyle/Honours/sobol"), CODE]
    hits = []
    for root in roots:
        files = subprocess.run(
            ["git", "-C", str(root), "ls-files", "*.py"],
            capture_output=True, text=True, check=True,
        ).stdout.split()
        for rel in files:
            if not (root / rel).is_file():  # deleted, not yet committed
                continue
            text = (root / rel).read_text(encoding="utf-8", errors="replace")
            for n, line in enumerate(text.splitlines(), 1):
                if stale.search(line):
                    hits.append(f"{root.name}/{rel}:{n}: {line.strip()}")
    assert not hits, "stale code paths:\n" + "\n".join(hits)


def test_gitignore_keeps_envs_worktrees_and_data_out():
    # The context-docs whitelist must not re-include arbitrary dirs: a venv,
    # a Claude worktree or a stray data folder inside Code/ stays ignored.
    import subprocess

    def ignored(rel):
        r = subprocess.run(["git", "-C", str(CODE), "check-ignore", "-q", "--no-index", rel])
        return r.returncode == 0

    for rel in (".venv/lib/python3.12/site-packages/foo/bar.py",
                ".claude/worktrees/x/viz/a.py", "viz/out/README.md", "DEMCSVs/run/notes.md"):
        assert ignored(rel), f"{rel} should be ignored"
    for rel in ("CLAUDE.md", "docs/README.md", "docs/superpowers/plans/x.md",
                ".cursor/rules/code-context.mdc", "viz/viz_preprocess.py", ".gitattributes",
                "Data analysis/FirstData.py", "Data analysis/run_sobol_phantom.sh"):
        assert not ignored(rel), f"{rel} should be tracked"


def test_csvconvert_scripts_are_wsl_native():
    # CSVconvert/ now runs under WSL python3: a \\wsl.localhost input only
    # works from Windows python, and a relative OUTPUT_DIR would write into
    # the git repo instead of the OneDrive data folder.
    src = (CODE / "CSVconvert" / "ToCSVFirst.py").read_text(encoding="utf-8")
    assert "wsl.localhost" not in src
    m = re.search(r'^OUTPUT_DIR\s*=\s*"([^"]+)"', src, re.M)
    assert m and m.group(1).startswith("/mnt/c/Users/22boy/OneDrive/"), m and m.group(1)


def test_generated_p0_report_stays_out_of_git():
    # P0_InstancingSpike.py writes its report + renders into the OneDrive docs
    # folder (Blender output); a tracked WSL copy would silently go stale.
    import subprocess
    r = subprocess.run(["git", "-C", str(CODE), "check-ignore", "-q", "--no-index",
                        "docs/P0_INSTANCING_SPIKE.md"])
    assert r.returncode == 0
