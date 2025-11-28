#!/usr/bin/env python

"""
Extract a small sample of sequences from a BLAST DB.

For PR2 18S DB:
    DB is at: ~/ncbi/custom_dbs/pr2_18s
This script will write: data/pr2_18s_sample.fasta
"""

import subprocess
from pathlib import Path


def extract_sample_from_db(
    db_path: str,
    out_fasta: str,
    n_seqs: int = 200,
) -> None:
    """
    Stream all sequences from the BLAST DB and keep only the first n_seqs.
    No IDs file, no %s/%a confusion.
    """
    db_path = str(db_path)
    out_path = Path(out_fasta)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "blastdbcmd",
        "-db",
        db_path,
        "-entry",
        "all",
        "-outfmt",
        "%f",  # full FASTA for each record
    ]

    print(
        f"Running to fetch FASTA from DB:\n  {' '.join(cmd)}\n"
        f"Will keep first {n_seqs} sequences."
    )

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)

    num = 0
    write = False

    with out_path.open("w") as fout:
        for line in proc.stdout:
            if line.startswith(">"):
                num += 1
                if num > n_seqs:
                    break
                write = True
            if write:
                fout.write(line)

    proc.terminate()
    print(f"✅ Wrote {num} sequences to {out_path}")


if __name__ == "__main__":
    DB = "/Users/saavanthveerumneni/ncbi/custom_dbs/pr2_18s"
    OUT = "data/pr2_18s_sample.fasta"
    extract_sample_from_db(DB, OUT, n_seqs=200)
