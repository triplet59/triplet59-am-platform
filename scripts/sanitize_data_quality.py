from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MASTER_CSV = PROJECT_ROOT / "output" / "master.csv"
MASTER_XLSX = PROJECT_ROOT / "output" / "master.xlsx"
DIVIDEND_DIR = PROJECT_ROOT / "data" / "dividends"
DIVIDEND_LOG = PROJECT_ROOT / "output" / "dividend_sanitization_log.csv"
PRICE_LOG = PROJECT_ROOT / "output" / "price_spike_corrections.csv"

DIVIDEND_IMPACT_CAP = 0.10
PRICE_SPIKE_THRESHOLD = 0.30
ISOLATED_SPIKE_COMBINED_MOVE_CAP = 0.10
POST_SHIFT_STABILITY_CAP = 0.12
SCALE_CANDIDATES = [0.001, 0.01, 0.1, 0.2, 0.25, 0.5, 2.0, 4.0, 5.0, 10.0, 100.0, 1000.0]


def load_master():
    if MASTER_CSV.exists():
        df = pd.read_csv(MASTER_CSV)
    elif MASTER_XLSX.exists():
        df = pd.read_excel(MASTER_XLSX)
    else:
        raise FileNotFoundError("Master dataset not found")

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return df


def save_master(df):
    export = df.copy()
    export["Date"] = pd.to_datetime(export["Date"], errors="coerce")
    export_csv = export.copy()
    export_csv["Date"] = export_csv["Date"].dt.strftime("%d/%m/%Y")
    export_csv.to_csv(MASTER_CSV, index=False)
    export.to_excel(MASTER_XLSX, index=False)


def full_name_from_path(path):
    country = path.parent.name.replace("_", " ").upper()
    company = path.stem.upper().strip()
    company = " ".join(company.split())
    return f"{company} ({country})"


def sanitize_dividends(master):
    price_panel = master.set_index("Date")
    logs = []

    for filepath in sorted(DIVIDEND_DIR.rglob("*.csv")):
        full_name = full_name_from_path(filepath)
        price_col = f"{full_name} Price"
        if price_col not in price_panel.columns:
            continue

        df = pd.read_csv(filepath)
        if not {"Date", "Dividend"}.issubset(df.columns):
            continue

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Dividend"] = pd.to_numeric(df["Dividend"], errors="coerce")
        df = df.dropna(subset=["Date", "Dividend"]).sort_values("Date").reset_index(drop=True)
        if df.empty:
            continue

        price_series = pd.to_numeric(price_panel[price_col], errors="coerce").sort_index()
        prev_price = price_series.shift(1)
        changed = False
        keep_rows = []

        for _, row in df.iterrows():
            date = row["Date"]
            value = row["Dividend"]
            prev = prev_price.reindex([date]).iloc[0]
            action = "keep"
            new_value = value

            if pd.notna(prev) and prev > 0:
                impact = value / prev
                if impact > DIVIDEND_IMPACT_CAP:
                    fixed = False
                    for factor in SCALE_CANDIDATES:
                        candidate = value * factor
                        if candidate / prev <= DIVIDEND_IMPACT_CAP:
                            new_value = candidate
                            action = f"scaled_by_{factor:g}"
                            fixed = True
                            break
                    if not fixed:
                        action = "dropped"
                elif impact < 0:
                    action = "dropped"
            else:
                action = "missing_prev_price"

            logs.append(
                {
                    "Company": full_name,
                    "Date": date,
                    "OriginalDividend": value,
                    "SanitizedDividend": new_value if action != "dropped" else np.nan,
                    "PrevPrice": prev,
                    "OriginalImpact": value / prev if pd.notna(prev) and prev != 0 else np.nan,
                    "Action": action,
                }
            )

            if action != "dropped":
                keep_rows.append({"Date": date, "Dividend": new_value})
                if action != "keep":
                    changed = True
            else:
                changed = True

        if changed:
            clean = pd.DataFrame(keep_rows).drop_duplicates(subset=["Date"], keep="last").sort_values("Date")
            clean.to_csv(filepath, index=False)

    log_df = pd.DataFrame(logs)
    log_df.to_csv(DIVIDEND_LOG, index=False)
    return log_df


def best_scaled_candidate(prev_value, curr_value, next_value):
    best = None
    best_score = None
    for factor in SCALE_CANDIDATES:
        candidate = curr_value * factor
        if candidate <= 0:
            continue
        left = abs(candidate / prev_value - 1)
        right = abs(next_value / candidate - 1)
        score = max(left, right)
        if score <= PRICE_SPIKE_THRESHOLD and (best_score is None or score < best_score):
            best = candidate
            best_score = score
    return best, best_score


def best_forward_regime_factor(prev_value, curr_value):
    best_factor = None
    best_gap = None

    for factor in SCALE_CANDIDATES:
        candidate = curr_value * factor
        if candidate <= 0:
            continue
        gap = abs(candidate / prev_value - 1)
        if gap <= PRICE_SPIKE_THRESHOLD and (best_gap is None or gap < best_gap):
            best_factor = factor
            best_gap = gap

    return best_factor, best_gap


def sanitize_price_series(series):
    values = series.astype(float).dropna().copy()
    logs = []

    for _ in range(4):
        changed = False

        for idx in range(1, len(values) - 1):
            prev_value = values.iloc[idx - 1]
            curr_value = values.iloc[idx]
            next_value = values.iloc[idx + 1]
            date = values.index[idx]

            if any(pd.isna(x) or x <= 0 for x in [prev_value, curr_value, next_value]):
                continue

            r1 = curr_value / prev_value - 1
            r2 = next_value / curr_value - 1
            combined = next_value / prev_value - 1

            if abs(r1) <= PRICE_SPIKE_THRESHOLD:
                continue

            if (
                abs(r2) > PRICE_SPIKE_THRESHOLD
                and np.sign(r1) != np.sign(r2)
                and abs(combined) < ISOLATED_SPIKE_COMBINED_MOVE_CAP
            ):
                new_value = float(np.sqrt(prev_value * next_value))
                logs.append(
                    {
                        "Date": date,
                        "OriginalPrice": curr_value,
                        "SanitizedPrice": new_value,
                        "Action": "isolated_spike_smoothed",
                        "ReturnIn": r1,
                        "ReturnOut": r2,
                    }
                )
                values.iloc[idx] = new_value
                changed = True
                continue

            candidate, score = best_scaled_candidate(prev_value, curr_value, next_value)
            if candidate is not None:
                logs.append(
                    {
                        "Date": date,
                        "OriginalPrice": curr_value,
                        "SanitizedPrice": candidate,
                        "Action": "scaled_price_point",
                        "ReturnIn": r1,
                        "ReturnOut": r2,
                    }
                )
                values.iloc[idx] = candidate
                changed = True
                continue

            if abs(r2) <= POST_SHIFT_STABILITY_CAP:
                factor, gap = best_forward_regime_factor(prev_value, curr_value)
                if factor is not None:
                    original_price = curr_value
                    values.iloc[idx:] = values.iloc[idx:] * factor
                    logs.append(
                        {
                            "Date": date,
                            "OriginalPrice": original_price,
                            "SanitizedPrice": values.iloc[idx],
                            "Action": f"scaled_forward_regime_by_{factor:g}",
                            "ReturnIn": r1,
                            "ReturnOut": r2,
                        }
                    )
                    changed = True
                    break

        if not changed:
            break

    return values, logs


def sanitize_prices(master):
    df = master.copy()
    logs = []
    price_cols = [col for col in df.columns if col.endswith(" Price")]
    dated = df.set_index("Date")

    for price_col in price_cols:
        series = pd.to_numeric(dated[price_col], errors="coerce")
        clean_series, series_logs = sanitize_price_series(series)
        if series_logs:
            df[price_col] = clean_series.reindex(df["Date"]).to_numpy()
            change_col = price_col.replace(" Price", " Change")
            if change_col in df.columns:
                df[change_col] = clean_series.pct_change().reindex(df["Date"]).to_numpy()
            for entry in series_logs:
                entry["Company"] = price_col[:-6]
            logs.extend(series_logs)

    log_df = pd.DataFrame(logs)
    log_df.to_csv(PRICE_LOG, index=False)
    return df, log_df


def main():
    master = load_master()
    clean_master, price_log = sanitize_prices(master)
    dividend_log = sanitize_dividends(clean_master)
    save_master(clean_master)

    print(f"Dividend sanitization actions: {len(dividend_log)}")
    if not dividend_log.empty:
        print(dividend_log["Action"].value_counts().to_string())

    print(f"Price correction actions: {len(price_log)}")
    if not price_log.empty:
        print(price_log["Action"].value_counts().to_string())

    print(f"Dividend log -> {DIVIDEND_LOG}")
    print(f"Price log -> {PRICE_LOG}")


if __name__ == "__main__":
    main()
