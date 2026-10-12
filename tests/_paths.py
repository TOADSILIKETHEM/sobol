"""Where the Honours code and data live (split on 2026-09-25).

Code (HonoursWin repo) moved into WSL; data and Blender assets stayed in
OneDrive so OneDrive keeps backing them up. Since 2026-10-10 both come from
sobol/honours_code.py + Code/honours_paths.py (defaults = Max's laptop).
"""
from pathlib import Path

import pytest

from honours_code import CODE_DIR, paths as _paths

CODE = CODE_DIR
# Windows programs (blender.exe, the .venv python) reach WSL files via this share.
CODE_WIN = _paths.to_windows_path(CODE)
DATA = Path(_paths.data_dir())
DATA_WIN = _paths.to_windows_path(DATA)

# Blender integration tests drive a Windows blender.exe through WIN_TEMP. A native
# (non-.exe) Blender in the config -> a path that never exists, so they skip.
BLENDER_EXE = (Path(_paths.blender_exe()) if _paths.blender_exe().lower().endswith(".exe")
               else Path("/nonexistent/blender.exe"))
# Tests pinning the WSL -> Windows .exe command layout or Max's own data/asset
# paths; they skip under another machine's Code/local_paths.json.
windows_tools = pytest.mark.skipif(
    not (_paths.blender_exe().lower().endswith(".exe") and _paths.preprocess_python().lower().endswith(".exe")),
    reason="asserts the WSL -> Windows .exe command layout (Max's laptop)",
)
max_laptop = pytest.mark.skipif(
    DATA != Path(_paths._DEFAULTS["data_dir"]), reason="checks Max's laptop data/asset paths",
)
