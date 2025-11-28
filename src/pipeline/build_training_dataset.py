# src/pipeline/build_training_dataset.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build training dataset from BLAST outfmt 6 TSV + PR2-style headers."
    )
    parser.add_argument(
        "--blast-tsv",
        required=True,
        help="Path to BLAST outfmt 6 TSV (qseqid, sseqid, pident, ...).",
    )
    parser.add_argument(
        "--out-csv",
        required=True,
        help="Output CSV path for the cleaned training dataset.",
    )
    parser.add_argument(
        "--min-pident",
        type=float,
        default=97.0,
        help="Minimum percent identity to keep a hit (default: 97.0).",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=1000,
        help="Minimum alignment length to keep a hit (default: 1000 bp).",
    )
    return parser.parse_args()


def parse_taxonomy(header: str) -> Dict[str, Optional[str]]:
    """
    Parse a PR2-style FASTA header into taxonomy fields.

    Example:
    GU223777.1.1164_U|18S_rRNA|nucleus|specimen_ARS02062|
    Eukaryota|Archaeplastida|Rhodophyta|Eurhodophytina|
    Florideophyceae|Ceramiales|Rhodomelaceae|Chondria|Chondria_sp.
    """
    parts = header.split("|")

    def get(i: int) -> Optional[str]:
        return parts[i] if len(parts) > i and parts[i] != "" else None

    fields: Dict[str, Optional[str]] = {
        "acc": get(0),
        "gene": get(1),
        "compartment": get(2),
        "specimen": get(3),
        "kingdom": get(4),
        "supergroup": get(5),
        "phylum": get(6),
        "clade": get(7),  # e.g. Eurhodophytina
        "class": get(8),
        "order": get(9),
        "family": get(10),
        "genus": get(11),
        # If there are more fields, last one is usually species-like label
        "species": get(12) if len(parts) > 12 else (parts[-1] if len(parts) > 4 else None),
    }
    return fields


def build_dataset(
    blast_path: Path,
    out_path: Path,
    min_pident: float = 97.0,
    min_length: int = 1000,
) -> None:
    print(f"[INFO] Reading BLAST TSV: {blast_path}")
    cols = [
        "qseqid",
        "sseqid",
        "pident",
        "length",
        "mismatch",
        "gapopen",
        "qstart",
        "qend",
        "sstart",
        "send",
        "evalue",
        "bitscore",
    ]
    df = pd.read_csv(blast_path, sep="\t", header=None, names=cols)

    print(f"[INFO] Total BLAST hits: {len(df):,}")

    # Quality filters
    df = df[(df["pident"] >= min_pident) & (df["length"] >= min_length)].copy()
    print(f"[INFO] After filters (pident>={min_pident}, length>={min_length}): {len(df):,}")

    if df.empty:
        print("[WARN] No hits passed filters. Nothing to write.")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        return

    # Keep best hit (highest bitscore) per query
    df.sort_values(["qseqid", "bitscore"], ascending=[True, False], inplace=True)
    df_best = df.drop_duplicates(subset="qseqid", keep="first").reset_index(drop=True)

    print(f"[INFO] Best hits (one per query): {len(df_best):,}")

    # Parse taxonomy from sseqid headers
    print("[INFO] Parsing taxonomy from sseqid headers...")
    tax_df = df_best["sseqid"].apply(parse_taxonomy).apply(pd.Series)

    out_df = pd.concat([df_best, tax_df], axis=1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"[INFO] Saved training dataset to: {out_path}")
    print(f"[INFO] Final rows: {len(out_df):,}")


def main() -> None:
    args = parse_args()
    build_dataset(
        blast_path=Path(args.blast_tsv),
        out_path=Path(args.out_csv),
        min_pident=args.min_pident,
        min_length=args.min_length,
    )


if __name__ == "__main__":
    main()
