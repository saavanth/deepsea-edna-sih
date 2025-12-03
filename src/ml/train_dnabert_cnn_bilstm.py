import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from .model_dnabert_s_cnn_bilstm import DnabertSCNNBiLSTM

# Taxonomic ranks we predict (column names in the CSV)
RANKS = [
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

# Special label bucket for anything unknown / unseen / missing
UNK_LABEL = "<UNK>"

# Rank weights (we emphasize fine-grained ranks more)
RANK_WEIGHTS = {
    "kingdom": 0.2,
    "supergroup": 0.4,
    "phylum": 0.6,
    "clade": 0.8,
    "class": 1.0,
    "order": 1.2,
    "family": 1.5,
    "genus": 2.0,
    "species": 3.0,
}
WEIGHT_SUM = sum(RANK_WEIGHTS[r] for r in RANKS)


class PR2DnabertDataset(Dataset):
    """
    Dataset for PR2-style eukaryote taxonomy with DNABERT tokenization.

    Expects a DataFrame with columns:
      - "sequence"
      - one column per rank in RANKS

    Unseen or missing labels are mapped to UNK_LABEL.
    """

    def __init__(self, df, tokenizer, max_length, ranks, rank_label_to_id):
        self.ranks = list(ranks)
        self.rank_label_to_id = rank_label_to_id

        # Ensure a clean index
        df = df.reset_index(drop=True)

        # Pre-tokenize all sequences for speed
        sequences = df["sequence"].astype(str).tolist()
        encodings = tokenizer(
            sequences,
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

        self.input_ids = torch.tensor(encodings["input_ids"], dtype=torch.long)
        self.attention_mask = torch.tensor(encodings["attention_mask"], dtype=torch.long)

        # Encode labels, mapping unseen / NaN to UNK, with hard fallback using dict.get
        label_rows = []
        for _, row in df.iterrows():
            label_vec = []
            for rank in self.ranks:
                label_str = row[rank]

                # NaN -> UNK
                if pd.isna(label_str):
                    label_str = UNK_LABEL
                else:
                    # Ensure it's a string (some pandas dtypes can be weird)
                    label_str = str(label_str)

                label_to_id = self.rank_label_to_id[rank]
                # If label is unknown, fall back to UNK_LABEL
                if label_str not in label_to_id:
                    label_str = UNK_LABEL

                # FINAL: always use .get with UNK fallback (this guarantees no KeyError)
                label_id = label_to_id.get(label_str, label_to_id[UNK_LABEL])
                label_vec.append(label_id)

            label_rows.append(label_vec)

        self.labels = torch.tensor(label_rows, dtype=torch.long)

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_label_encoders(df, ranks):
    """
    Build label encoders from the TRAINING dataframe only.

    For each rank:
      - Collect all distinct labels from train
      - Add an UNK_LABEL bucket
      - Create label_to_id and id_to_label dicts
    """
    rank_label_to_id = {}
    rank_id_to_label = {}
    rank_num_classes = {}

    for rank in ranks:
        if rank not in df.columns:
            raise ValueError(f"Required column '{rank}' not found in training CSV.")

        # All distinct non-NaN labels in this rank
        labels = df[rank].dropna().astype(str).unique().tolist()
        labels = sorted(labels)

        # Always include an UNK label for unseen / missing values
        if UNK_LABEL not in labels:
            labels.append(UNK_LABEL)

        label_to_id = {label: idx for idx, label in enumerate(labels)}
        id_to_label = {idx: label for label, idx in label_to_id.items()}

        rank_label_to_id[rank] = label_to_id
        rank_id_to_label[rank] = id_to_label
        rank_num_classes[rank] = len(labels)

    return rank_label_to_id, rank_id_to_label, rank_num_classes


def train_one_epoch(model, dataloader, optimizer, device, ranks, criterion):
    model.train()
    running_loss = 0.0

    correct = {rank: 0 for rank in ranks}
    total = {rank: 0 for rank in ranks}

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)  # (B, num_ranks)

        optimizer.zero_grad()

        logits_dict = model(input_ids=input_ids, attention_mask=attention_mask)

        loss = 0.0
        for i, rank in enumerate(ranks):
            logits = logits_dict[rank]  # (B, num_classes)
            target = labels[:, i]
            rank_loss = criterion(logits, target)
            loss += RANK_WEIGHTS[rank] * rank_loss

            preds = torch.argmax(logits, dim=1)
            correct[rank] += (preds == target).sum().item()
            total[rank] += target.size(0)

        loss = loss / WEIGHT_SUM
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * input_ids.size(0)

    avg_loss = running_loss / len(dataloader.dataset)
    acc = {rank: (correct[rank] / total[rank]) if total[rank] > 0 else 0.0 for rank in ranks}
    return avg_loss, acc


def eval_one_epoch(model, dataloader, device, ranks, criterion):
    model.eval()
    running_loss = 0.0

    correct = {rank: 0 for rank in ranks}
    total = {rank: 0 for rank in ranks}

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits_dict = model(input_ids=input_ids, attention_mask=attention_mask)

            loss = 0.0
            for i, rank in enumerate(ranks):
                logits = logits_dict[rank]
                target = labels[:, i]
                rank_loss = criterion(logits, target)
                loss += RANK_WEIGHTS[rank] * rank_loss

                preds = torch.argmax(logits, dim=1)
                correct[rank] += (preds == target).sum().item()
                total[rank] += target.size(0)

            loss = loss / WEIGHT_SUM
            running_loss += loss.item() * input_ids.size(0)

    avg_loss = running_loss / len(dataloader.dataset)
    acc = {rank: (correct[rank] / total[rank]) if total[rank] > 0 else 0.0 for rank in ranks}
    return avg_loss, acc


def save_checkpoint(
    path,
    model,
    encoder_name,
    max_len,
    ranks,
    rank_num_classes,
    rank_label_to_id,
    rank_id_to_label,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "encoder_name": encoder_name,
        "max_len": max_len,
        "ranks": ranks,
        "rank_num_classes": rank_num_classes,
        "rank_label_to_id": rank_label_to_id,
        "rank_id_to_label": rank_id_to_label,
        "model_hyperparams": {
            "cnn_channels": model.cnn_channels,
            "lstm_hidden_size": model.lstm_hidden_size,
            "lstm_layers": model.lstm_layers,
            "dropout": model.dropout,
            "freeze_encoder": model.freeze_encoder,
        },
        "unk_label": UNK_LABEL,
    }
    torch.save(ckpt, path)
    print(f"✅ Saved checkpoint to {path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DNABERT-2 + CNN + BiLSTM multi-rank taxonomy model on PR2 data."
    )
    parser.add_argument(
        "--train-csv",
        type=str,
        default="data/pr2_train.csv",
        help="Path to training CSV.",
    )
    parser.add_argument(
        "--val-csv",
        type=str,
        default="data/pr2_val.csv",
        help="Path to validation CSV.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=256,
        help="Max sequence length for DNABERT tokenizer.",
    )
    parser.add_argument(
        "--encoder-name",
        type=str,
        default="zhihan1996/DNABERT-2-117M",
        help="HuggingFace model name for DNABERT encoder.",
    )
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="If set, freeze DNABERT encoder weights.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="checkpoints/dnabert_cnn_bilstm",
        help="Output directory for checkpoints.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print(f"Loading training data from {args.train_csv}")
    train_df = pd.read_csv(args.train_csv)
    print(f"Loading validation data from {args.val_csv}")
    val_df = pd.read_csv(args.val_csv)

    # Build label encoders on training data
    rank_label_to_id, rank_id_to_label, rank_num_classes = build_label_encoders(
        train_df, RANKS
    )

    # Tokenizer
    print(f"Loading tokenizer: {args.encoder_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.encoder_name,
        trust_remote_code=True,
    )

    # Datasets
    train_dataset = PR2DnabertDataset(
        train_df,
        tokenizer=tokenizer,
        max_length=args.max_len,
        ranks=RANKS,
        rank_label_to_id=rank_label_to_id,
    )
    val_dataset = PR2DnabertDataset(
        val_df,
        tokenizer=tokenizer,
        max_length=args.max_len,
        ranks=RANKS,
        rank_label_to_id=rank_label_to_id,
    )

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Model
    print("Initializing DNABERT-2 + CNN + BiLSTM model...")
    model = DnabertSCNNBiLSTM(
        encoder_name=args.encoder_name,
        ranks=RANKS,
        num_classes_per_rank=rank_num_classes,
        cnn_channels=256,
        lstm_hidden_size=256,
        lstm_layers=1,
        dropout=0.3,
        freeze_encoder=args.freeze_encoder,
    )
    model.to(device)

    # Optimizer & loss
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )
    criterion = nn.CrossEntropyLoss()

    best_species_acc = 0.0
    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"\n========== Epoch {epoch}/{args.epochs} ==========")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device, RANKS, criterion
        )
        print(f"Train loss: {train_loss:.4f}")
        for rank in RANKS:
            print(f"  Train {rank:10s} acc: {train_acc[rank]*100:6.2f}%")

        val_loss, val_acc = eval_one_epoch(
            model, val_loader, device, RANKS, criterion
        )
        print(f"Val   loss: {val_loss:.4f}")
        for rank in RANKS:
            print(f"  Val   {rank:10s} acc: {val_acc[rank]*100:6.2f}%")

        species_acc = val_acc["species"]
        if species_acc > best_species_acc:
            best_species_acc = species_acc
            best_path = os.path.join(args.out_dir, "best_model.pt")
            save_checkpoint(
                best_path,
                model,
                encoder_name=args.encoder_name,
                max_len=args.max_len,
                ranks=RANKS,
                rank_num_classes=rank_num_classes,
                rank_label_to_id=rank_label_to_id,
                rank_id_to_label=rank_id_to_label,
            )
            print(
                f"🌟 New best species accuracy: {best_species_acc*100:.2f}% (checkpoint updated)"
            )

    # Save final checkpoint
    final_path = os.path.join(args.out_dir, "last_model.pt")
    save_checkpoint(
        final_path,
        model,
        encoder_name=args.encoder_name,
        max_len=args.max_len,
        ranks=RANKS,
        rank_num_classes=rank_num_classes,
        rank_label_to_id=rank_label_to_id,
        rank_id_to_label=rank_id_to_label,
    )
    print("Training complete.")


if __name__ == "__main__":
    main()
