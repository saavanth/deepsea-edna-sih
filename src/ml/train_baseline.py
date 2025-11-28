# src/ml/train_baseline.py

import argparse
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset_pr2 import (
    PR2Dataset,
    build_label_encoders,
    RANK_COLS,
)
from .model_baseline import DNAClassifier


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", type=str, default="data/pr2_train.csv")
    ap.add_argument("--val-csv", type=str, default="data/pr2_val.csv")
    ap.add_argument("--max-len", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out-dir", type=str, default="models/pr2_baseline")
    return ap.parse_args()


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return correct / total if total > 0 else 0.0


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Label encoders from TRAIN set only ----
    print(f"Building label encoders from {args.train_csv} ...")
    label_encoders = build_label_encoders(args.train_csv)

    # Num classes per rank
    num_classes_per_rank: Dict[str, int] = {
        rank: len(enc) for rank, enc in label_encoders.items()
    }

    print("Classes per rank:")
    for r, n in num_classes_per_rank.items():
        print(f"  {r}: {n}")

    # ---- Datasets & loaders ----
    train_ds = PR2Dataset(
        args.train_csv, label_encoders=label_encoders, max_len=args.max_len
    )
    val_ds = PR2Dataset(
        args.val_csv, label_encoders=label_encoders, max_len=args.max_len
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # ---- Model ----
    vocab_size = 6  # PAD + A C G T N
    model = DNAClassifier(
        vocab_size=vocab_size,
        emb_dim=64,
        hidden_dim=256,
        num_classes_per_rank=num_classes_per_rank,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # ---- Training loop ----
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_steps = 0

        for x, labels in train_loader:
            x = x.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}

            optimizer.zero_grad()
            logits = model(x)

            loss = 0.0
            for rank in RANK_COLS:
                loss = loss + criterion(logits[rank], labels[rank])
            loss = loss / len(RANK_COLS)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_steps += 1

        avg_train_loss = total_loss / max(total_steps, 1)

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        val_steps = 0
        acc_per_rank = {r: 0.0 for r in RANK_COLS}
        count_per_rank = {r: 0 for r in RANK_COLS}

        with torch.no_grad():
            for x, labels in val_loader:
                x = x.to(device)
                labels = {k: v.to(device) for k, v in labels.items()}

                logits = model(x)

                loss = 0.0
                for rank in RANK_COLS:
                    loss = loss + criterion(logits[rank], labels[rank])
                loss = loss / len(RANK_COLS)

                val_loss += loss.item()
                val_steps += 1

                # accuracy per rank
                for rank in RANK_COLS:
                    acc = compute_accuracy(logits[rank], labels[rank])
                    acc_per_rank[rank] += acc
                    count_per_rank[rank] += 1

        avg_val_loss = val_loss / max(val_steps, 1)
        avg_acc_per_rank = {
            r: (acc_per_rank[r] / max(count_per_rank[r], 1)) for r in RANK_COLS
        }

        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train loss: {avg_train_loss:.4f}")
        print(f"  Val   loss: {avg_val_loss:.4f}")
        print("  Val acc per rank:")
        for r in RANK_COLS:
            print(f"    {r:10s}: {avg_acc_per_rank[r]:.4f}")

        # Save checkpoint each epoch
        ckpt_path = out_dir / f"epoch_{epoch}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "label_encoders": label_encoders,
                "num_classes_per_rank": num_classes_per_rank,
                "args": vars(args),
            },
            ckpt_path,
        )
        print(f"  Saved checkpoint to: {ckpt_path}")


if __name__ == "__main__":
    main()
