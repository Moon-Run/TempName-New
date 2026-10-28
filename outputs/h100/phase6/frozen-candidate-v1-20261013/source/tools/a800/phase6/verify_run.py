"""Run a campaign's own frozen verifier with its matching frozen routing code."""
from pathlib import Path
import sys

root=Path(sys.argv[1]).resolve()
sys.path[:0]=[str(root/'scripts'),str(root)]
from verify import verify
verify(root)
