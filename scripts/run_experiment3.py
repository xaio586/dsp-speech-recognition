"""Start the experiment 3 live microphone recognition interface."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dsp_speech.experiment3_app import main  # noqa: E402


if __name__ == "__main__":
    main()
