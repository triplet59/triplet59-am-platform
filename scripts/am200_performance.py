import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import am100_performance as base

INDEX_NAME = "AM200"
HISTORY_FILE = "output/AM200_history.xlsx"
OUTPUT_FILE = "output/AM200_total_return.csv"
METRICS_FILE = "output/AM200_metrics.csv"
VALIDATION_FILE = "output/AM200_return_validation_report.csv"


def main():
    prices = base.load_prices()
    dividends = base.load_dividends()

    if not os.path.exists(HISTORY_FILE):
        print(f"⚠️ {INDEX_NAME} history not found: {HISTORY_FILE}")
        return

    history = base.load_history(HISTORY_FILE)
    index_df, validation_df = base.build_total_return_index(prices, history, dividends)

    if len(index_df) == 0:
        print(f"⚠️ No {INDEX_NAME} total return data generated")
        return

    index_df.to_csv(OUTPUT_FILE, index=False)
    total_return = index_df.set_index("Date")["Index Level"]
    base.print_index_debug(INDEX_NAME, total_return)
    metrics = base.compute_metrics(total_return)
    pd.DataFrame([metrics]).to_csv(METRICS_FILE, index=False)
    validation_df["Index"] = INDEX_NAME
    validation_df.to_csv(VALIDATION_FILE, index=False)
    print(f"✅ {INDEX_NAME} total return created -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
