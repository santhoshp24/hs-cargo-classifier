"""Run without installing:  python run.py classify   (same as `hsclass classify` once installed)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from hsclass.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
