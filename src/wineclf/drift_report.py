# src/wineclf/drift_report.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import mlflow
import pandas as pd
import evidently

from evidently import Report

try:
    from evidently.metric_preset import DataDriftPreset
except ImportError:
    from evidently.presets import DataDriftPreset


def make_drift_report(
    ref_path: str,
    cur_path: str,
    target_col: Optional[str],
    output_dir: str,
) -> Path:
    print(f"[Evidently] version: {evidently.__version__}")
    print(f"[Evidently] reference data: {ref_path}")
    print(f"[Evidently] current data:   {cur_path}")

    ref = pd.read_csv(ref_path).dropna()
    cur = pd.read_csv(cur_path).dropna()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / "evidently_drift_report.html"

    report = Report(metrics=[DataDriftPreset()])

    eval_result = report.run(reference_data=ref, current_data=cur)
    eval_result.save_html(str(html_path))

    print(f"[Evidently] HTML report saved to: {html_path}")

    with mlflow.start_run(run_name="drift_report"):
        mlflow.log_artifact(str(html_path), artifact_path="evidently")

    return html_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ref-data",
        default="data/raw/WineQT.csv",
        help="Reference dataset CSV (pl. train vagy korai snapshot).",
    )
    parser.add_argument(
        "--cur-data",
        default="data/raw/WineQT.csv",
        help="Current dataset CSV (pl. új batch).",
    )
    parser.add_argument(
        "--target-col",
        default="quality",
        help="Most nem használjuk, de későbbi bővítéshez jól jöhet.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/evidently",
        help="Ide kerül az Evidently HTML riport.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    make_drift_report(
        ref_path=args.ref_data,
        cur_path=args.cur_data,
        target_col=args.target_col,
        output_dir=args.output_dir,
    )
