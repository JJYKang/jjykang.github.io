"""Problem 2 API; implementations are shared with the report scripts in src/."""
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from rectification import compute_epipole, compute_H, compute_matching_homographies

if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "scripts"))
    runpy.run_path(str(ROOT / "scripts" / "part2.py"), run_name="__main__")
