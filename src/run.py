"""
src/run.py — Forwarding Launcher Shim
======================================
Allows running `python run.py <module>` even when current directory is `src/`.
Forward execution to root run.py.
"""
import os
import sys

# Change directory to root and execute root run.py
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_RUN_PY = os.path.join(ROOT_DIR, "run.py")

if __name__ == "__main__":
    os.chdir(ROOT_DIR)
    with open(ROOT_RUN_PY, "r", encoding="utf-8") as f:
        code = compile(f.read(), ROOT_RUN_PY, "exec")
        exec(code, {"__name__": "__main__", "__file__": ROOT_RUN_PY})
