from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from triangulation import (estimate_initial_RT, linear_estimate_3d_point,
                           reprojection_error, jacobian, nonlinear_estimate_3d_point,
                           estimate_RT_from_E)

if __name__ == "__main__":
    runpy.run_path(str(ROOT / "scripts" / "part4.py"), run_name="__main__")
    runpy.run_path(str(ROOT / "scripts" / "part4_reconstruction.py"), run_name="__main__")
