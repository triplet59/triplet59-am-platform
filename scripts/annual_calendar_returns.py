from pathlib import Path

import pandas as pd


OUTPUT_DIR = Path("/Users/derrythornalley/AM100_DATA/output")
INDEX_FILES = {
    "AM100": OUTPUT_DIR / "AM100_total_return.csv",
    "AM200": OUTPUT_DIR / "AM200_total_return.csv",
    "AM300": OUTPUT_DIR / "AM300_total_return.csv",
}
OUTPUT_FILE = OUTPUT_DIR / "AM_annual_returns.csv"
MERGED_OUTPUT_FILE = OUTPUT_DIR / "annual_calendar_returns_daily_panel.csv"
SUMMARY_OUTPUT_FILE = OUTPUT_DIR / "AM_annual_return_summary.csv"


def load_index(file_path: Path, name: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df.columns = [c.strip() for c in df.columns]

    if "Date" not in df.columns or "Index Level" not in df.columns:
        raise ValueError(f"{file_path} must contain 'Date' and 'Index Level'")

    df = df[["Date", "Index Level"]].copy()
    df.columns = ["Date", name]
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").dropna(subset=[name])


def merge_indices(index_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    ordered = [index_frames["AM100"], index_frames["AM200"], index_frames["AM300"]]
    df = ordered[0].merge(ordered[1], on="Date", how="outer").merge(ordered[2], on="Date", how="outer")
    return df.sort_values("Date").reset_index(drop=True)


def compute_annual_returns(df: pd.DataFrame, column: str) -> pd.DataFrame:
    yearly = df[["Date", "Year", column]].dropna().copy()

    results = []

    for year, group in yearly.groupby("Year"):
        if len(group) < 2:
            continue

        start_value = group.iloc[0][column]
        end_value = group.iloc[-1][column]
        annual_return = (end_value / start_value) - 1.0

        results.append({"Year": year, column: annual_return * 100.0})

    return pd.DataFrame(results)


def main() -> None:
    index_frames: dict[str, pd.DataFrame] = {}

    for name, path in INDEX_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing input file: {path}")
        index_frames[name] = load_index(path, name)

    merged = merge_indices(index_frames)
    merged["Year"] = merged["Date"].dt.year

    am100_ret = compute_annual_returns(merged, "AM100")
    am200_ret = compute_annual_returns(merged, "AM200")
    am300_ret = compute_annual_returns(merged, "AM300")

    annual_returns = (
        am100_ret.merge(am200_ret, on="Year", how="outer")
        .merge(am300_ret, on="Year", how="outer")
        .sort_values("Year")
    )
    annual_returns = annual_returns.round(2)

    summary = {}
    for col in ["AM100", "AM200", "AM300"]:
        series = annual_returns[col].dropna()
        summary[col] = {
            "Best Year": series.max(),
            "Worst Year": series.min(),
            "Positive Years (%)": (series > 0).mean() * 100,
        }

    summary_df = pd.DataFrame(summary).T.round(2)

    annual_returns.to_csv(OUTPUT_FILE, index=False)
    merged.to_csv(MERGED_OUTPUT_FILE, index=False)
    summary_df.to_csv(SUMMARY_OUTPUT_FILE)

    print(f"Saved annual calendar returns -> {OUTPUT_FILE}")
    print(f"Saved merged daily panel -> {MERGED_OUTPUT_FILE}")
    print(f"Saved annual return summary -> {SUMMARY_OUTPUT_FILE}")
    print(annual_returns.to_string(index=False))
    print()
    print(summary_df.to_string())


if __name__ == "__main__":
    main()
