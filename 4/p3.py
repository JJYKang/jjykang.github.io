"""Problem 3 API; implementations are shared with the report scripts in src/."""
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from factorization import factorization_method

if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "scripts"))
    runpy.run_path(str(ROOT / "scripts" / "part3.py"), run_name="__main__")
