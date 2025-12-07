import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_ROOT.parent
SRC_PATH = BACKEND_ROOT / "src"

for path in (PROJECT_ROOT, BACKEND_ROOT, SRC_PATH):
    str_path = str(path)
    if str_path not in sys.path:
        sys.path.insert(0, str_path)

