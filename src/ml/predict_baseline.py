import argparse
import csv
import os
from typing import List, Dict, Any, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


##############################################
# 1. FASTA READER
##############################################

def read_fasta(path: str) -> List[Tuple[str, str]]:
    """
    Very simple FASTA parser.
    Returns list of (id, sequence).
    """
    records = []
    seq_id = None
    seq_chunks = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                # flush previous record
                if seq_id is not None:
                    records.append((seq_id, "".join(seq_chunks).upper()))
                # start new
                seq_id = line[1:].strip()
                seq_chunks = []
            else:
                seq_chunks.append(line)

    # flush last
    if seq_id is not None:
        records.append((seq_id, "".join(seq_chunks).upper()))

    return records


##############################################
# 2. TOKENIZER (must match training!)
##############################################

class SimpleDNATokenizer:
    """
    Simple per-base tokenizer:
      A,C,G,T -> 1..4
      others  -> 5 (N/unknown)
    0 is reserved for padding.
    max_len: sequences longer than this are truncated, shorter are padded.
    """

    def __init__(self, max_len: int = 2000):
        self.max_len = max_len
        self.vocab = {
            "A": 1,
            "C": 2,
            "G": 3,
            "T": 4,
        }
        self.unknown_id = 5
        self.pad_id = 0

    def encode(self, seq: str) -> torch.Tensor:
        ids = []
        for ch in seq.upper():
            ids.append(self.vocab.get(ch, self.unknown_id))
        # truncate / pad
        if len(ids) > self.max_len:
            ids = ids[: self.max_len]
        else:
            ids = ids + [self.pad_id] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)


##############################################
# 3. MODEL (match training architecture)
##############################################

class CNNBaseline(nn.Module):
    """
    CNN-based sequence classifier, consistent with training:
      - embedding on token ids
      - Conv1D -> ReLU -> Conv1D -> ReLU
      - global max pool
      - one classification head per rank
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_filters: int,
        kernel_size: int,
        rank_num_classes: Dict[str, int],
        padding_idx: int = 0,
    ):
        super().__init__()
        self.rank_names = list(rank_num_classes.keys())

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=padding_idx,
        )

        # Two conv layers to match checkpoint keys: conv1, conv2
        self.conv1 = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.conv2 = nn.Conv1d(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)

        # one linear head per rank
        self.heads = nn.ModuleDict()
        for rank, n_classes in rank_num_classes.items():
            self.heads[rank] = nn.Linear(num_filters, n_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x: (batch, seq_len) token ids
        returns: dict rank -> logits (batch, num_classes)
        """
        # (B, L) -> (B, L, E)
        emb = self.embedding(x)
        # (B, L, E) -> (B, E, L)
        emb = emb.transpose(1, 2)

        # conv1 + conv2
        h = self.conv1(emb)
        h = self.activation(h)
        h = self.conv2(h)
        h = self.activation(h)

        # global max pool over L -> (B, F)
        h = torch.max(h, dim=2).values
        h = self.dropout(h)

        outputs = {}
        for rank, head in self.heads.items():
            outputs[rank] = head(h)
        return outputs


##############################################
# 4. DATASET + COLLATE
##############################################

class InferenceDataset(Dataset):
    def __init__(self, records: List[Tuple[str, str]], tokenizer: SimpleDNATokenizer):
        self.records = records
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        seq_id, seq = self.records[idx]
        token_ids = self.tokenizer.encode(seq)
        return {
            "id": seq_id,
            "seq": seq,
            "input_ids": token_ids,
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    ids = [b["id"] for b in batch]
    seqs = [b["seq"] for b in batch]
    input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
    return {
        "ids": ids,
        "seqs": seqs,
        "input_ids": input_ids,
    }


##############################################
# 5. LOAD CHECKPOINT & BUILD MODEL
##############################################

def load_checkpoint_and_model(
    ckpt_path: str,
    device: torch.device,
    rank_names: List[str] = None,
) -> Tuple[CNNBaseline, SimpleDNATokenizer, Dict[str, Dict[int, str]]]:
    """
    Loads:
      - model weights
      - tokenizer config (if stored)
      - label decoders: rank_index_to_label[rank][idx] -> label_str

    Works with 2 formats:
      1) ckpt["rank_index_to_label"] = {rank: {idx: label}}
      2) ckpt["label_encoders"]     = {rank: {label: idx}}  (dict saved from training)
    """
    ckpt = torch.load(ckpt_path, map_location=device)

    # 1) Label decoders
    if "rank_index_to_label" in ckpt:
        # Already stored in the desired format
        rank_index_to_label = ckpt["rank_index_to_label"]

    elif "label_encoders" in ckpt:
        # In training we stored:
        #   label_encoders[rank] = {label_str: idx_int}
        # Here we invert to:
        #   rank_index_to_label[rank][idx_int] = label_str
        encoders = ckpt["label_encoders"]
        rank_index_to_label = {}
        for rank, mapping in encoders.items():
            if isinstance(mapping, dict):
                inv = {int(idx): str(label) for label, idx in mapping.items()}
                rank_index_to_label[rank] = inv
            else:
                raise TypeError(
                    f"Unexpected type for label_encoders['{rank}']: {type(mapping)}. "
                    "Expected a dict mapping label -> index."
                )
    else:
        raise ValueError(
            "Checkpoint missing 'rank_index_to_label' and 'label_encoders'. "
            "Update train_baseline.py to save one of these."
        )

    # 2) Rank -> num_classes
    if "rank_num_classes" in ckpt:
        rank_num_classes = ckpt["rank_num_classes"]
    else:
        # infer from the decoders
        rank_num_classes = {r: len(idx_to_lab) for r, idx_to_lab in rank_index_to_label.items()}

    # If caller provided explicit order, reorder / filter
    if rank_names is None:
        rank_names = list(rank_num_classes.keys())
    rank_num_classes = {r: rank_num_classes[r] for r in rank_names}

    # Also filter decoders to these ranks
    filtered_rank_index_to_label = {}
    for r in rank_names:
        if r not in rank_index_to_label:
            raise KeyError(
                f"Rank '{r}' not found in rank_index_to_label. "
                f"Available ranks: {list(rank_index_to_label.keys())}"
            )
        filtered_rank_index_to_label[r] = rank_index_to_label[r]
    rank_index_to_label = filtered_rank_index_to_label

    # 3) Tokenizer config
    if "tokenizer" in ckpt:
        tok_cfg = ckpt["tokenizer"]
        max_len = int(tok_cfg.get("max_len", 2000))
        tokenizer = SimpleDNATokenizer(max_len=max_len)
        # optional: custom vocab
        if "vocab" in tok_cfg and isinstance(tok_cfg["vocab"], dict):
            tokenizer.vocab = {k: int(v) for k, v in tok_cfg["vocab"].items()}
    else:
        tokenizer = SimpleDNATokenizer(max_len=2000)

    # IMPORTANT: make vocab_size large enough for pad + unknown
    max_token_id = max(
        max(tokenizer.vocab.values()),
        tokenizer.unknown_id,
        tokenizer.pad_id,
    )
    vocab_size = max_token_id + 1  # e.g. 0..5 -> 6

    # Hyperparams (should be stored in ckpt from training)
    embed_dim = ckpt.get("embed_dim", 64)
    num_filters = ckpt.get("num_filters", 256)
    kernel_size = ckpt.get("kernel_size", 9)

    model = CNNBaseline(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_filters=num_filters,
        kernel_size=kernel_size,
        rank_num_classes=rank_num_classes,
        padding_idx=tokenizer.pad_id,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    return model, tokenizer, rank_index_to_label


##############################################
# 6. INFERENCE LOOP
##############################################

@torch.no_grad()
def run_inference(
    fasta_path: str,
    ckpt_path: str,
    out_csv: str,
    device_str: str = "cpu",
    batch_size: int = 32,
    min_confidence: float = 0.0,
):
    device = torch.device(device_str)

    # Rank order (same as training)
    rank_names = [
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

    model, tokenizer, rank_index_to_label = load_checkpoint_and_model(
        ckpt_path=ckpt_path,
        device=device,
        rank_names=rank_names,
    )

    records = read_fasta(fasta_path)
    if not records:
        raise ValueError(f"No sequences found in FASTA: {fasta_path}")

    dataset = InferenceDataset(records, tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    # Build CSV header
    header = ["id", "sequence"]
    for rank in rank_names:
        header.append(rank)
        header.append(f"{rank}_confidence")

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            ids = batch["ids"]
            seqs = batch["seqs"]

            logits_dict = model(input_ids)
            # convert to probabilities
            probs_dict = {r: torch.softmax(logits, dim=1) for r, logits in logits_dict.items()}

            for i in range(len(ids)):
                row = [ids[i], seqs[i]]
                for rank in rank_names:
                    prob_vec = probs_dict[rank][i]  # (num_classes,)
                    conf, idx = prob_vec.max(dim=0)
                    conf_val = float(conf.item())
                    idx_val = int(idx.item())

                    # map index -> label string
                    label_map = rank_index_to_label[rank]
                    label = label_map.get(idx_val, f"UNK_{idx_val}")

                    # optional threshold
                    if conf_val < min_confidence:
                        label = "unknown"

                    row.append(label)
                    row.append(conf_val)

                writer.writerow(row)

    print(f"[OK] Wrote predictions for {len(records)} sequences to: {out_csv}")


##############################################
# 7. CLI
##############################################

def main():
    parser = argparse.ArgumentParser(description="Run baseline taxonomy inference on FASTA.")
    parser.add_argument(
        "--input-fasta",
        required=True,
        help="Path to input FASTA file of eDNA reads / 18S sequences",
    )
    parser.add_argument(
        "--checkpoint",
        default="models/pr2_baseline/epoch_3.pt",
        help="Path to trained baseline checkpoint",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output CSV path for predictions",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="cpu or cuda (if you have GPU)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="If probability below this, label is set to 'unknown'",
    )

    args = parser.parse_args()

    run_inference(
        fasta_path=args.input_fasta,
        ckpt_path=args.checkpoint,
        out_csv=args.out,
        device_str=args.device,
        batch_size=args.batch_size,
        min_confidence=args.min_confidence,
    )


if __name__ == "__main__":
    main()
