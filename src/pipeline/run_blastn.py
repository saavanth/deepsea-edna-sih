import argparse
import subprocess
from pathlib import Path


def run_blastn(
    query_fasta: str,
    db_name: str,
    db_dir: str,
    out_path: str,
    evalue: float = 1e-20,
    max_target_seqs: int = 25,
    num_threads: int = 4,
):
    """
    Run blastn of query_fasta against a local BLAST DB.

    Parameters
    ----------
    query_fasta : str
        Path to query FASTA file (ASVs / OTUs / reads).
    db_name : str
        Name of the BLAST DB (e.g. 'pr2_18s').
    db_dir : str
        Directory containing the BLAST DB files.
    out_path : str
        Path to output TSV.
    evalue : float
        E-value cutoff for reporting hits.
    max_target_seqs : int
        Maximum number of hits per query to keep.
    num_threads : int
        Number of CPU threads for BLAST.
    """
    query_fasta = Path(query_fasta)
    if not query_fasta.exists():
        raise FileNotFoundError(f"Query FASTA not found: {query_fasta}")

    db_path = Path(db_dir) / db_name
    if not (db_path.with_suffix(".nin").exists() or db_path.with_suffix(".ndb").exists()):
        # for version 5 dbs .ndb exists; for version 4 .nin/.nsq
        raise FileNotFoundError(f"BLAST DB not found: {db_path}*")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # BLAST tabular format (outfmt 6)
    # qseqid   : query id
    # sseqid   : subject id (PR2 header)
    # pident   : % identity
    # length   : alignment length
    # mismatch : mismatches
    # gapopen  : gap openings
    # qstart,qend,sstart,send : alignment coords
    # evalue   : e-value
    # bitscore : bitscore
    outfmt = "6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore"

    cmd = [
        "blastn",
        "-query",
        str(query_fasta),
        "-db",
        str(db_path),
        "-out",
        str(out_path),
        "-outfmt",
        outfmt,
        "-evalue",
        str(evalue),
        "-max_target_seqs",
        str(max_target_seqs),
        "-num_threads",
        str(num_threads),
        "-task",
        "blastn",
    ]

    print("\nRunning BLASTN command:\n", " ".join(cmd), "\n")
    subprocess.run(cmd, check=True)
    print(f"BLASTN finished. Results written to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Run blastn against a local PR2 (or other) BLAST DB.")
    parser.add_argument("--query", required=True, help="Path to query FASTA file.")
    parser.add_argument(
        "--db-name", default="pr2_18s", help="BLAST DB name (default: pr2_18s)."
    )
    parser.add_argument(
        "--db-dir",
        default="/Users/saavanthveerumneni/ncbi/custom_dbs",
        help="Directory containing BLAST DB files.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output TSV file path.",
    )
    parser.add_argument(
        "--evalue",
        type=float,
        default=1e-20,
        help="E-value cutoff (default: 1e-20).",
    )
    parser.add_argument(
        "--max-target-seqs",
        type=int,
        default=25,
        help="Max hits per query (default: 25).",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=4,
        help="Number of CPU threads (default: 4).",
    )

    args = parser.parse_args()

    run_blastn(
        query_fasta=args.query,
        db_name=args.db_name,
        db_dir=args.db_dir,
        out_path=args.out,
        evalue=args.evalue,
        max_target_seqs=args.max_target_seqs,
        num_threads=args.num_threads,
    )


if __name__ == "__main__":
    main()
