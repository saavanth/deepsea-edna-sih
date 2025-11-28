# src/ml/dataset_pr2.py

from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from torch.utils.data import Dataset


RANK_COLS: List[str] = [
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


def build_label_encoders(csv_path: str | Path) -> Dict[str, Dict[str, int]]:
    """
    Read the TRAIN CSV and build string->int mappings for each taxonomic rank.
    """
    df = pd.read_csv(csv_path)

    encoders: Dict[str, Dict[str, int]] = {}

    for col in RANK_COLS:
        # Replace missing with 'Unknown'
        values = df[col].fillna("Unknown").astype(str)

        # Unique sorted
        uniq = sorted(set(values))

        # Make sure we have an <UNK> entry at index 0
        vocab = ["<UNK>"] + [v for v in uniq if v != "<UNK>"]

        encoders[col] = {v: i for i, v in enumerate(vocab)}

    return encoders


class PR2Dataset(Dataset):
    """
    PyTorch Dataset for PR2 18S data.

    - Encodes DNA sequence into integer tokens.
    - Returns:
        seq_tensor: LongTensor [max_len]
        labels: dict(rank -> LongTensor scalar)
    """

    def __init__(
        self,
        csv_path: str | Path,
        label_encoders: Dict[str, Dict[str, int]],
        max_len: int = 2000,
    ):
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path)
        self.label_encoders = label_encoders
        self.max_len = max_len

        # DNA vocabulary for simple char-level encoding
        # 0 is PAD
        self.char2idx = {
            "PAD": 0,
            "A": 1,
            "C": 2,
            "G": 3,
            "T": 4,
            "N": 5,   # unknown / other
        }

    def __len__(self) -> int:
        return len(self.df)

    def _encode_sequence(self, seq: str) -> torch.Tensor:
        seq = (seq or "").upper()

        tokens = []
        for base in seq:
            if base in self.char2idx:
                tokens.append(self.char2idx[base])
            else:
                tokens.append(self.char2idx["N"])

        # truncate
        tokens = tokens[: self.max_len]

        # pad
        pad_id = self.char2idx["PAD"]
        if len(tokens) < self.max_len:
            tokens = tokens + [pad_id] * (self.max_len - len(tokens))

        return torch.tensor(tokens, dtype=torch.long)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        seq = row["sequence"]
        x = self._encode_sequence(seq)

        labels: Dict[str, torch.Tensor] = {}

        for col in RANK_COLS:
            enc = self.label_encoders[col]
            raw_val = str(row[col]) if not pd.isna(row[col]) else "Unknown"
            label_id = enc.get(raw_val, enc["<UNK>"])
            labels[col] = torch.tensor(label_id, dtype=torch.long)

        return x, labels
