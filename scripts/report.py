"""Generate aggregate reports from completed experiment runs.

Usage:
    python scripts/report.py --runs-dir ./runs [--output-dir ./reports]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from prcl.integritysuite.aggregate import collect_run_cards, export_tables


def main():
    parser = argparse.ArgumentParser(description="Generate aggregate experiment reports")
    parser.add_argument("--runs-dir", type=str, required=True, help="Directory containing run folders")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for tables")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        print(f"Error: runs directory {runs_dir} does not exist")
        sys.exit(1)

    df = collect_run_cards(runs_dir)
    if df.empty:
        print("No run cards found. Run experiments first.")
        sys.exit(0)

    print(f"Found {len(df)} completed runs")
    outputs = export_tables(runs_dir, args.output_dir)

    for name, path in outputs.items():
        print(f"  {name}: {path}")

    print("Report generation complete.")


if __name__ == "__main__":
    main()
