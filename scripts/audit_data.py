#!/usr/bin/env python3
"""Audit datasets and write docs/dataset_audit.md."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.datasets import (
    audit_dataset,
    checksum_raw_files,
    download_dataset,
    preprocess_graph,
    save_processed,
)
from src.utils.io import ensure_dir, save_json, setup_logging


def render_markdown(reports: list[dict]) -> str:
    lines = [
        "# Dataset Audit",
        "",
        "All methods must use these processed graphs, splits, seeds, and protocols.",
        "",
    ]
    for r in reports:
        lines.append(f"## {r['dataset']}")
        lines.append("")
        lines.append("| Statistic | Value |")
        lines.append("|---|---|")
        for k, v in r.items():
            if k == "dataset":
                continue
            if isinstance(v, dict):
                v = "`" + str(v) + "`"
            lines.append(f"| {k} | {v} |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["facebook", "lastfm", "github"])
    args = parser.parse_args()
    setup_logging()
    reports = []
    for name in args.datasets:
        data = download_dataset(name, raw_dir=ROOT / "data" / "raw")
        data, edge_audit = preprocess_graph(data)
        save_processed(data, name, processed_dir=ROOT / "data" / "processed")
        raw_checksums = checksum_raw_files(ROOT / "data" / "raw" / name)
        report = audit_dataset(
            data,
            edge_audit=edge_audit,
            raw_checksums=raw_checksums,
        )
        reports.append(report)
        save_json(report, ROOT / "results" / "processed" / f"audit_{name}.json")
        print(name, {k: report[k] for k in ("num_nodes", "unique_undirected_edges", "num_classes", "homophily")})

    md = render_markdown(reports)
    ensure_dir(ROOT / "docs")
    (ROOT / "docs" / "dataset_audit.md").write_text(md, encoding="utf-8")
    print("Wrote docs/dataset_audit.md")


if __name__ == "__main__":
    main()
