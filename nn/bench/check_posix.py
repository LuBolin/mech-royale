"""Run kit/check/check.py on Windows without abspath turning the container path /arena into W:\\arena."""
import os
import runpy
import sys

_abspath = os.path.abspath
os.path.abspath = lambda p: p if isinstance(p, str) and p.startswith("/") else _abspath(p)
check = sys.argv[1]
sys.argv = [check] + sys.argv[2:]
runpy.run_path(check, run_name="__main__")
