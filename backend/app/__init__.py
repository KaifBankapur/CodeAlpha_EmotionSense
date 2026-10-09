"""
FastAPI backend for the Emotion Recognition from Speech system.

Run from the repository root::

    uvicorn backend.app.main:app --reload --port 8000
"""

from __future__ import annotations

import sys
from pathlib import Path

# The ML package lives at the repository root, one level above ``backend/``.
# Adding it here means ``uvicorn backend.app.main:app`` works from any working
# directory, without requiring an editable install or PYTHONPATH gymnastics.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
