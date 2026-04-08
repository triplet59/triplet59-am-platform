import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
os.environ["APP_MODE"] = "internal"

from scripts import dashboard_streamlit  # noqa: F401
