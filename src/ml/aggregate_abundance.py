import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List


RANK_NAMES: List[str] = [
    "kingdom",
    "supergroup",
    "phylum",
    "clade",
    "class",
    "order",
    "family",
    "genus",
    "species",
]


def aggregate_for_rank(
    rows: List[Dict[str, str]],
    rank: str,
    min_conf: float = 0.0,
):
    """
    Aggregate abundance for a single taxonomic rank.

    - rows: list of CSV rows from predictions
    - rank: e.g. "species"
    - min_conf: ignore predictions below this confidence
    """
    label_col = rank
    conf_col = f"{rank}_confidence"

    counts = defaultdict(float)          # plain read count (1 per read)
    weighted_counts = defaultdict(float) # confidence-weighted count

    total_reads = 0.0
    total_weighted = 0.0

    for row in rows:
        label = row[label_col]
        try:
            conf = float(row[conf_col])
        except ValueError:
            conf = 0.0

        if label == "unknown":
            continue
        if conf < min_conf:
            continue

        counts[label] += 1.0
        weighted_counts[label] += conf

        total_reads += 1.0
        total_weighted += conf

    return counts, weighted_counts, total_reads, total_weighted


def write_abundance_csv(
    out_path: str,
    rank: str,
    counts,
    weighted_counts,
    total_reads: float,
    total_weighted: float,
):
    """
    Write abundance table for a rank to CSV.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                rank,
                "reads",
                "weighted_reads",
                "relative_reads",
                "relative_weighted_reads",
            ]
        )

        # sort by weighted count descending
        items = sorted(
            counts.keys(),
            key=lambda k: weighted_counts.get(k, 0.0),
            reverse=True,
        )

        for taxon in items:
            c = counts[taxon]
            wc = weighted_counts[taxon]

            rel_c = c / total_reads if total_reads > 0 else 0.0
            rel_wc = wc / total_weighted if total_weighted > 0 else 0.0

            writer.writerow([taxon, c, wc, rel_c, rel_wc])


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate abundance from baseline prediction CSV."
    )
    parser.add_argument(
        "--pred-csv",
        required=True,
        help="Path to results/pr2_baseline_predictions.csv",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Directory to write abundance CSVs, e.g. results/abundance",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Ignore predictions below this confidence.",
    )

    args = parser.parse_args()

    # 1) Read prediction CSV
    with open(args.pred_csv, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Loaded {len(rows)} prediction rows from {args.pred_csv}")

    # 2) Aggregate for each rank
    for rank in RANK_NAMES:
        counts, weighted_counts, total_reads, total_weighted = aggregate_for_rank(
            rows, rank, min_conf=args.min-confidence if hasattr(args, "min-confidence") else args.min_confidence
        )

        out_path = os.path.join(args.out_dir, f"abundance_{rank}.csv")
        write_abundance_csv(
            out_path,
            rank,
            counts,
            weighted_counts,
            total_reads,
            total_weighted,
        )

        print(
            f"[{rank}] wrote {len(counts)} taxa to {out_path} "
            f"(total_reads={total_reads}, total_weighted={total_weighted:.2f})"
        )


if __name__ == "__main__":
    main()
