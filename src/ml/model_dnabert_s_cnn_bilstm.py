# src/ml/train_dnabert_cnn_bilstm.py

import argparse
import os
import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModel


# ------------- CONFIG ------------- #

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

# same idea as baseline: emphasize fine ranks more
RANK_WEIGHTS: Dict[str, float] = {
    "kingdom": 0.2,
    "supergroup": 0.4,
    "phylum": 0.6,
    "clade": 0.8,
    "class": 1.0,
    "order": 1.5,
    "family": 2.0,
    "genus": 2.5,
    "species": 3.0,
}


# ------------- DATASET & LABEL ENCODING ------------- #

def build_label_encoders(df: pd.DataFrame) -> Dict[str, LabelEncoder]:
    encoders: Dict[str, LabelEncoder] = {}
    for col in RANK_COLS:
        le = LabelEncoder()
        vals = df[col].astype(str).fillna("unknown")
        le.fit(vals.values)
        encoders[col] = le
    return encoders


def encode_labels(df: pd.DataFrame, encoders: Dict[str, LabelEncoder]) -> np.ndarray:
    all_labels = []
    for col in RANK_COLS:
        le = encoders[col]
        vals = df[col].astype(str).fillna("unknown")
        all_labels.append(le.transform(vals.values))
    y = np.stack(all_labels, axis=1)  # (N, num_ranks)
    return y


class Pr2DnaBertDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        encoders: Dict[str, LabelEncoder],
        tokenizer: AutoTokenizer,
        max_len: int = 512,
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.seqs: List[str] = self.df["sequence"].astype(str).tolist()
        self.y = encode_labels(self.df, encoders)  # (N, num_ranks)

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = self.seqs[idx]
        labels = self.y[idx]  # (num_ranks,)

        enc = self.tokenizer(
            seq,
            add_special_tokens=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        item = {
            "input_ids": enc["input_ids"].squeeze(0),      # (L,)
            "attention_mask": enc["attention_mask"].squeeze(0),  # (L,)
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        return item


# ------------- MODEL: DNABERT + CNN + BiLSTM ------------- #

class DnaBertCnnBiLstm(nn.Module):
    def __init__(
        self,
        bert_model_name: str,
        rank_num_classes: Dict[str, int],
        cnn_channels: int = 256,
        cnn_kernel_size: int = 5,
        lstm_hidden: int = 256,
        lstm_layers: int = 1,
        dropout: float = 0.3,
        freeze_bert: bool = True,
    ):
        super().__init__()

        self.rank_names = list(rank_num_classes.keys())

        # Pretrained DNABERT(-S) encoder
        self.bert = AutoModel.from_pretrained(bert_model_name)
        hidden_size = self.bert.config.hidden_size

        # CNN over BERT hidden states
        self.conv1 = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=cnn_channels,
            kernel_size=cnn_kernel_size,
            padding=cnn_kernel_size // 2,
        )
        self.conv2 = nn.Conv1d(
            in_channels=cnn_channels,
            out_channels=cnn_channels,
            kernel_size=cnn_kernel_size,
            padding=cnn_kernel_size // 2,
        )

        # BiLSTM on top of CNN features
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
        )

        feat_dim = lstm_hidden * 2  # bidirectional
        self.dropout = nn.Dropout(dropout)

        # One linear head per taxonomic rank
        self.heads = nn.ModuleDict()
        for rank, n_classes in rank_num_classes.items():
            self.heads[rank] = nn.Linear(feat_dim, n_classes)

        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        input_ids: (B, L)
        attention_mask: (B, L)
        returns: dict rank -> logits (B, num_classes)
        """
        # BERT encoder
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        x = outputs.last_hidden_state  # (B, L, H)

        # CNN expects (B, H, L)
        x = x.transpose(1, 2)  # (B, H, L)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.transpose(1, 2)  # (B, L, C)

        # BiLSTM
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(x)  # (B, L, 2 * lstm_hidden)

        # Masked global max pooling over L
        mask = attention_mask.unsqueeze(-1).bool()  # (B, L, 1)
        lstm_out_masked = lstm_out.masked_fill(~mask, -1e9)
        pooled, _ = torch.max(lstm_out_masked, dim=1)  # (B, 2 * lstm_hidden)

        pooled = self.dropout(pooled)

        logits: Dict[str, torch.Tensor] = {}
        for rank, head in self.heads.items():
            logits[rank] = head(pooled)

        return logits


# ------------- TRAIN / EVAL LOOPS ------------- #

def compute_loss(
    logits: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    criterion: nn.Module,
    rank_names: List[str],
) -> torch.Tensor:
    """
    labels: (B, num_ranks)
    """
    total_loss = 0.0
    for i, rank in enumerate(rank_names):
        weight = RANK_WEIGHTS[rank]
        loss_rank = criterion(logits[rank], labels[:, i])
        total_loss = total_loss + weight * loss_rank
    return total_loss


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    rank_names: List[str],
) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_batches = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)  # (B, num_ranks)

        optimizer.zero_grad()
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = compute_loss(logits, labels, criterion, rank_names)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    return total_loss / max(total_batches, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    rank_names: List[str],
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_batches = 0

    correct = {r: 0 for r in rank_names}
    total = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)  # (B, num_ranks)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = compute_loss(logits, labels, criterion, rank_names)

        total_loss += loss.item()
        total_batches += 1

        # accuracies
        batch_size = labels.size(0)
        total += batch_size
        for i, rank in enumerate(rank_names):
            preds = logits[rank].argmax(dim=1)  # (B,)
            correct[rank] += (preds == labels[:, i]).sum().item()

    avg_loss = total_loss / max(total_batches, 1)
    acc = {r: correct[r] / max(total, 1) for r in rank_names}
    return avg_loss, acc


# ------------- MAIN ------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Train DNABERT-S + CNN + BiLSTM multi-rank taxonomy model on PR2."
    )
    parser.add_argument("--train-csv", default="data/pr2_train.csv")
    parser.add_argument("--val-csv", default="data/pr2_val.csv")
    parser.add_argument("--bert-model-name", required=True,
                        help="Hugging Face model id for DNA-BERT(-S), e.g. 'your-dnabert-s-model-id'")
    parser.add_argument("--max-len", type=int, default=512,
                        help="Max token length for BERT input")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--out-dir", default="models/pr2_dnabert")
    parser.add_argument("--freeze-bert", action="store_true",
                        help="Freeze DNABERT encoder parameters")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1) Load data
    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)

    # 2) Build encoders on TRAIN only
    label_encoders = build_label_encoders(train_df)
    rank_num_classes = {
        rank: len(le.classes_) for rank, le in label_encoders.items()
    }

    print("Classes per rank:")
    for r, n in rank_num_classes.items():
        print(f"  {r:10s}: {n}")

    # 3) Tokenizer & datasets
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model_name)

    train_ds = Pr2DnaBertDataset(
        train_df,
        encoders=label_encoders,
        tokenizer=tokenizer,
        max_len=args.max_len,
    )
    val_ds = Pr2DnaBertDataset(
        val_df,
        encoders=label_encoders,
        tokenizer=tokenizer,
        max_len=args.max_len,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
    )

    # 4) Model
    model = DnaBertCnnBiLstm(
        bert_model_name=args.bert_model_name,
        rank_num_classes=rank_num_classes,
        freeze_bert=args.freeze_bert,
    )
    model.to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )

    rank_names = RANK_COLS

    # 5) Training loop
    best_species_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, rank_names
        )
        val_loss, val_acc = evaluate(
            model, val_loader, device, rank_names
        )

        print(f"  Train loss: {train_loss:.4f}")
        print(f"  Val   loss: {val_loss:.4f}")
        print("  Val acc per rank:")
        for r in rank_names:
            print(f"    {r:10s}: {val_acc[r]:.4f}")

        # Track by species accuracy
        species_acc = val_acc["species"]
        if species_acc > best_species_acc:
            best_species_acc = species_acc

            # Build index->label mapping for convenient inference later
            rank_index_to_label = {
                rank: {int(i): str(lbl) for i, lbl in enumerate(le.classes_)}
                for rank, le in label_encoders.items()
            }

            ckpt = {
                "bert_model_name": args.bert_model_name,
                "max_len": args.max_len,
                "model_state_dict": model.state_dict(),
                "rank_num_classes": rank_num_classes,
                "label_encoders": label_encoders,
                "rank_index_to_label": rank_index_to_label,
            }
            ckpt_path = os.path.join(args.out_dir, f"epoch_{epoch}.pt")
            torch.save(ckpt, ckpt_path)
            print(f"  Saved checkpoint to: {ckpt_path}")


if __name__ == "__main__":
    main()
