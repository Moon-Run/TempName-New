"""Locate maintained Phase6 helpers without importing the chunk prototype runtime."""
from pathlib import Path
import sys

HERE7 = Path(__file__).resolve().parent
REPO = HERE7.parents[2]
PHASE6 = REPO / 'tools/h100/phase6'
# Frozen controllers may colocate Phase6 and Phase7 modules. In the maintained
# tree Phase7 only adds uniquely prefixed modules; old helpers remain unchanged.
if PHASE6.is_dir() and str(PHASE6) not in sys.path:
    sys.path.append(str(PHASE6))
