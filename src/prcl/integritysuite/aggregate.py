"""Aggregate results across multiple experiment runs into paper-ready tables."""

import json
from pathlib import Path

import pandas as pd


def collect_run_cards(runs_dir: str | Path) -> pd.DataFrame:
    """Scan run directories and collect all run_card.json files into a DataFrame.

    Each row is one experiment run with columns for all metric fields.
    """
    runs_dir = Path(runs_dir)
    records = []

    for card_json in sorted(runs_dir.rglob("cards/run_card.json")):
        with open(card_json) as f:
            data = json.load(f)

        row = {}
        # Flatten run_metrics
        rm = data.get("run_metrics", {})
        for k, v in rm.items():
            row[k] = v

        # Flatten forensic_metrics
        fm = data.get("forensic_metrics")
        if fm:
            for k, v in fm.items():
                if isinstance(v, dict):
                    for sk, sv in v.items():
                        row[f"forensic_{sk}"] = sv
                else:
                    row[f"forensic_{k}"] = v

        # Attack info
        am = data.get("attack_metadata")
        if am:
            row["attack_type_detail"] = am.get("attack_type", "")

        row["run_path"] = str(card_json.parent.parent)
        records.append(row)

    return pd.DataFrame(records)


def make_main_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create the main results table (Table 1 in paper).

    Columns: Dataset, Backbone, Attack, Poison%, Defense, Clean Acc, ASR
    Grouped by (dataset, attack) pairs.
    """
    cols = [
        "dataset", "backbone", "attack_family", "poison_ratio",
        "defense_mode", "linear_probe_acc", "asr",
    ]
    available = [c for c in cols if c in df.columns]
    table = df[available].copy()

    # Format percentages
    if "linear_probe_acc" in table.columns:
        table["linear_probe_acc"] = table["linear_probe_acc"].apply(
            lambda x: f"{x*100:.1f}" if pd.notna(x) else "—"
        )
    if "asr" in table.columns:
        table["asr"] = table["asr"].apply(
            lambda x: f"{x*100:.1f}" if pd.notna(x) else "—"
        )
    if "poison_ratio" in table.columns:
        table["poison_ratio"] = table["poison_ratio"].apply(
            lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—"
        )

    table.columns = [c.replace("_", " ").title() for c in table.columns]
    return table


def make_forensic_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create the forensic evaluation table.

    Shows ROC-AUC, PR-AUC, top-k recall for each defense+attack combo.
    """
    forensic_cols = [c for c in df.columns if c.startswith("forensic_")]
    if not forensic_cols:
        return pd.DataFrame()

    id_cols = ["dataset", "attack_family", "defense_mode", "poison_ratio"]
    available_id = [c for c in id_cols if c in df.columns]
    table = df[available_id + forensic_cols].copy()
    table.columns = [c.replace("forensic_", "").replace("_", " ").title() for c in table.columns]
    return table


def export_tables(
    runs_dir: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Collect all runs and export paper-ready tables to CSV and LaTeX.

    Returns dict mapping table name to output path.
    """
    runs_dir = Path(runs_dir)
    output_dir = Path(output_dir) if output_dir else runs_dir / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = collect_run_cards(runs_dir)
    if df.empty:
        return {}

    outputs = {}

    # Main results table
    main = make_main_table(df)
    csv_path = output_dir / "main_results.csv"
    main.to_csv(csv_path, index=False)
    outputs["main_csv"] = csv_path

    tex_path = output_dir / "main_results.tex"
    tex_path.write_text(main.to_latex(index=False, escape=True))
    outputs["main_tex"] = tex_path

    # Forensic table
    forensic = make_forensic_table(df)
    if not forensic.empty:
        csv_f = output_dir / "forensic_results.csv"
        forensic.to_csv(csv_f, index=False)
        outputs["forensic_csv"] = csv_f

    # Raw aggregated data
    raw_path = output_dir / "all_runs.csv"
    df.to_csv(raw_path, index=False)
    outputs["raw_csv"] = raw_path

    return outputs
