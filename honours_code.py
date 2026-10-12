"""Find the Code/ repo (HonoursWin) and import its path resolver.

HONOURS_CODE_DIR overrides; otherwise Code/ is this repo's sibling, which is
how ~/Honours is laid out (sobol/ and Code/ side by side).
"""
import os
import sys
from pathlib import Path

CODE_DIR = Path(os.environ.get("HONOURS_CODE_DIR") or Path(__file__).resolve().parent.parent / "Code")
if not (CODE_DIR / "honours_paths.py").is_file():
    sys.exit(f"Code repo not found at {CODE_DIR} (set HONOURS_CODE_DIR to the HonoursWin checkout).")
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import honours_paths as paths  # noqa: E402,F401
