import os

os.environ["APP_MODE"] = "investor"

from scripts import dashboard_streamlit  # noqa: F401
