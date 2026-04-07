import os

os.environ["APP_MODE"] = "internal"

from scripts import dashboard_streamlit  # noqa: F401
