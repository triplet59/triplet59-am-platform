import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

log_filename = LOG_DIR / f"pipeline_{datetime.now().strftime('%Y%m%d')}.log"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def run_script(script_name):
    script_path = BASE_DIR / "scripts" / script_name

    try:
        logging.info(f"START: {script_name}")
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            cwd=BASE_DIR,
        )

        if result.returncode != 0:
            logging.error(f"ERROR in {script_name}: {result.stderr}")
        else:
            logging.info(f"SUCCESS: {script_name}")
            if result.stdout.strip():
                logging.info(result.stdout.strip())
    except Exception as exc:
        logging.error(f"EXCEPTION in {script_name}: {str(exc)}")


def is_rebalance_day():
    today = datetime.today()
    return today.day == 1


def main():
    logging.info("=== PIPELINE START ===")

    run_script("build_dataset.py")

    if is_rebalance_day():
        run_script("am100_rebalance.py")
    else:
        message = "Not rebalance day — skipping."
        print(message)
        logging.info(message)

    run_script("am100_performance.py")

    logging.info("=== PIPELINE END ===")


if __name__ == "__main__":
    main()
