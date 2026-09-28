"""Where the Honours code and data live (split on 2026-09-25).

Code (HonoursWin repo) moved into WSL; data and Blender assets stayed in
OneDrive so OneDrive keeps backing them up.
"""
from pathlib import Path

CODE = Path("/home/mboyle/Honours/Code")
# Windows programs (blender.exe, the .venv python) reach WSL files via this share.
CODE_WIN = r"\\wsl.localhost\Ubuntu\home\mboyle\Honours\Code"
DATA = Path("/mnt/c/Users/22boy/OneDrive/Documents/GC-Max_desktop/Honours/Code")
DATA_WIN = "C:/Users/22boy/OneDrive/Documents/GC-Max_desktop/Honours/Code"
