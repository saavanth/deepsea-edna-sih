# src/ml/model_baseline.py

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataset_pr2 import RANK_COLS


class DNAClassifier(nn.Module):
    """
    Simple baseline:
      - Embedding over DNA tokens
      - 1D CNN + global max pool
      - Separate softmax head per taxonomic rank
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        hidden_dim: int,
        num_classes_per_rank: Dict[str, int],
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.emb_dim = emb_dim

        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)

        self.conv1 = nn.Conv1d(emb_dim, hidden_dim, kernel_size=9, padding=4)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=9, padding=4)

        # One head per rank
        self.heads = nn.ModuleDict()
        for rank, n_classes in num_classes_per_rank.items():
            self.heads[rank] = nn.Linear(hidden_dim, n_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x: LongTensor [B, L]
        returns: dict(rank -> logits [B, num_classes])
        """
        emb = self.embedding(x)      # [B, L, E]
        emb = emb.transpose(1, 2)    # [B, E, L]

        h = F.relu(self.conv1(emb))
        h = F.relu(self.conv2(h))

        # Global max-pooling over sequence length
        h = F.adaptive_max_pool1d(h, 1).squeeze(-1)  # [B, hidden_dim]

        out: Dict[str, torch.Tensor] = {}
        for rank in RANK_COLS:
            out[rank] = self.heads[rank](h)

        return out
