from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from fundamental import (lls_eight_point_alg, normalized_eight_point_alg,
                         compute_distance_to_epipolar_lines)

if __name__ == "__main__":
    runpy.run_path(str(ROOT / "scripts" / "part1.py"), run_name="__main__")
