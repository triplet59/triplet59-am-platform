from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


OUTPUT_DIR = Path("/Users/derrythornalley/AM100_DATA/output")
INPUT_FILE = OUTPUT_DIR / "AM_annual_returns.csv"
OUTPUT_FILE = OUTPUT_DIR / "AM_annual_returns_chart.png"


def main() -> None:
    annual_returns = pd.read_csv(INPUT_FILE)
    annual_returns = annual_returns.set_index("Year")

    ax = annual_returns.plot(kind="bar", figsize=(12, 6))
    ax.set_title("Annual Calendar Returns (%)")
    ax.set_ylabel("Return (%)")
    ax.set_xlabel("Year")
    ax.axhline(0, color="black", linewidth=0.8)

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved chart -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
