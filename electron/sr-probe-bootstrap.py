"""Keep probes isolated from user Python configuration while importing bundled SR."""
import runpy
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
runpy.run_path(str(root / 'check_environment.py'), run_name='__main__')
