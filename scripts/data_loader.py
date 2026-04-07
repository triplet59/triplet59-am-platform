from pathlib import Path

import pandas as pd

INDEX_POOL = ["AM100", "AM200", "AM300"]
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_index(name):
    path = PROJECT_ROOT / "output" / f"{name}_total_return.csv"
    df = pd.read_csv(path, parse_dates=["Date"])
    return df.sort_values("Date")


def load_benchmark(path):
    df = pd.read_csv(path, parse_dates=["Date"])
    return df.sort_values("Date")
