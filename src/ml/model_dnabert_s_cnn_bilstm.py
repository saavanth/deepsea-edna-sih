import torch
from torch import nn
from transformers import AutoModel


class DnabertSCNNBiLSTM(nn.Module):
    """
    DNABERT-2 encoder + CNN + BiLSTM + multi-head taxonomy classifier.

    Expects:
      - input_ids: (batch, seq_len)
      - attention_mask: (batch, seq_len)
    Returns:
      - dict(rank -> logits tensor of shape (batch, num_classes_for_rank))
    """

    def __init__(
        self,
        encoder_name: str,
        ranks,
        num_classes_per_rank,
        cnn_channels: int = 256,
        lstm_hidden_size: int = 256,
        lstm_layers: int = 1,
        dropout: float = 0.3,
        freeze_encoder: bool = False,
    ):
        super().__init__()

        self.encoder_name = encoder_name
        self.ranks = list(ranks)
        self.num_classes_per_rank = dict(num_classes_per_rank)
        self.cnn_channels = cnn_channels
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers
        self.dropout = dropout
        self.freeze_encoder = freeze_encoder

        # Load DNABERT-2 (or compatible) encoder
        self.encoder = AutoModel.from_pretrained(
            encoder_name,
            trust_remote_code=True,
            attn_implementation="eager",  # safer on Kaggle (avoids some flash-attn/triton issues)
        )
        hidden_size = self.encoder.config.hidden_size

        # CNN branch
        self.conv1 = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=cnn_channels,
            kernel_size=3,
            padding=1,
        )
        self.conv_activation = nn.GELU()
        self.conv_dropout = nn.Dropout(dropout)

        # BiLSTM branch
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm_dropout = nn.Dropout(dropout)

        feature_dim = cnn_channels + 2 * lstm_hidden_size

        # One classification head per taxonomic rank
        self.classifiers = nn.ModuleDict()
        for rank in self.ranks:
            n_classes = self.num_classes_per_rank[rank]
            self.classifiers[rank] = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(feature_dim, n_classes),
            )

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass.

        Returns:
            logits_dict: {rank: (batch, num_classes)}
        """
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = encoder_outputs.last_hidden_state  # (B, L, H)

        if attention_mask is not None:
            # Zero-out padded positions so CNN/LSTM don't see junk
            mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)  # (B, L, 1)
            hidden_states = hidden_states * mask

        # CNN branch: (B, L, H) -> (B, H, L) -> (B, C, L) -> pooled (B, C)
        x_cnn = hidden_states.permute(0, 2, 1)  # (B, H, L)
        x_cnn = self.conv1(x_cnn)
        x_cnn = self.conv_activation(x_cnn)
        x_cnn = self.conv_dropout(x_cnn)
        cnn_pooled = torch.max(x_cnn, dim=2).values  # (B, C)

        # BiLSTM branch: (B, L, H) -> (B, L, 2*H_lstm) -> pooled (B, 2*H_lstm)
        x_lstm, _ = self.lstm(hidden_states)  # (B, L, 2*H_lstm)

        if attention_mask is not None:
            # Mask out padding before max-pooling
            mask = attention_mask.unsqueeze(-1).to(x_lstm.dtype)  # (B, L, 1)
            x_lstm = x_lstm * mask + (1.0 - mask) * (-1e9)

        lstm_pooled = torch.max(x_lstm, dim=1).values  # (B, 2*H_lstm)
        lstm_pooled = self.lstm_dropout(lstm_pooled)

        # Final shared feature vector
        features = torch.cat([cnn_pooled, lstm_pooled], dim=1)  # (B, feature_dim)

        # Multi-head logits
        logits_dict = {
            rank: head(features)
            for rank, head in self.classifiers.items()
        }
        return logits_dict
