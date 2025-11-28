# src/pipeline/add_sequences_to_training.py

import argparse
from pathlib import Path
import pandas as pd


def load_fasta_as_dict(fasta_path: Path):
    """
    Read a FASTA and return dict: acc -> sequence (uppercase A/C/G/T/...).
    Here 'acc' = text before first '|' in header, e.g.
      >AAAA02046270.109.1957_U|18S_rRNA|...
      acc = 'AAAA02046270.109.1957_U'
    """
    seqs = {}
    header = None
    seq_lines = []

    with fasta_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                # save previous
                if header is not None:
                    acc = header.split("|", 1)[0].split()[0]
                    seqs[acc] = "".join(seq_lines).upper()
                header = line[1:]  # drop ">"
                seq_lines = []
            else:
                seq_lines.append(line)

        # last record
        if header is not None:
            acc = header.split("|", 1)[0].split()[0]
            seqs[acc] = "".join(seq_lines).upper()

    return seqs


def main():
    parser = argparse.ArgumentParser(
        description="Join PR2 FASTA sequences to training CSV using 'acc'."
    )
    parser.add_argument(
        "--training-csv",
        required=True,
        help="Input CSV from build_training_dataset.py",
    )
    parser.add_argument(
        "--fasta",
        required=True,
        help="PR2 FASTA file (pr2_18s_taxo_long.fasta)",
    )
    parser.add_argument(
        "--out-csv",
        required=True,
        help="Output CSV with an extra 'sequence' column",
    )
    args = parser.parse_args()

    training_path = Path(args.training_csv)
    fasta_path = Path(args.fasta)
    out_path = Path(args.out_csv)

    print(f"Reading training CSV: {training_path}")
    df = pd.read_csv(training_path)

    print(f"Loading FASTA sequences from: {fasta_path}")
    acc_to_seq = load_fasta_as_dict(fasta_path)
    print(f"Loaded {len(acc_to_seq):,} sequences from FASTA")

    # Map acc -> sequence
    df["sequence"] = df["acc"].map(acc_to_seq)

    missing = df["sequence"].isna().sum()
    if missing > 0:
        print(f"WARNING: {missing} rows had no matching sequence in FASTA and will be dropped")
        df = df.dropna(subset=["sequence"])

    print(f"Final rows with sequence: {len(df):,}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
