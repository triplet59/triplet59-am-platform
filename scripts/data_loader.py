import pandas as pd

INDEX_POOL = ["AM100", "AM200", "AM300"]


def load_index(name):
    path = f"output/{name}_total_return.csv"
    df = pd.read_csv(path, parse_dates=["Date"])
    return df.sort_values("Date")


def load_benchmark(path):
    df = pd.read_csv(path, parse_dates=["Date"])
    return df.sort_values("Date")
