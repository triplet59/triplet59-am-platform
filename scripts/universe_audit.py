import pandas as pd
from datetime import datetime
from pathlib import Path


class UniverseAudit:
    def __init__(self):
        self.records = []

    def log(self, company, country, status, reason):
        self.records.append(
            {
                "timestamp": datetime.utcnow(),
                "company": company,
                "country": country,
                "status": status,
                "reason": reason,
            }
        )

    def to_dataframe(self):
        return pd.DataFrame(self.records)

    def save(self, path="output/universe_audit.csv"):
        df = self.to_dataframe()
        df.to_csv(path, index=False)
        print(f"[AUDIT] Universe audit saved -> {path}")


def main():
    output_path = Path("output/universe_audit.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    audit = UniverseAudit()

    existing_audit = Path("output/audit_log.csv")
    if existing_audit.exists():
        df = pd.read_csv(existing_audit)
        for row in df.fillna("").to_dict(orient="records"):
            audit.log(
                company=row.get("company", ""),
                country=row.get("country", ""),
                status=row.get("status", ""),
                reason=row.get("reason", ""),
            )

    audit.save(output_path)


if __name__ == "__main__":
    main()
