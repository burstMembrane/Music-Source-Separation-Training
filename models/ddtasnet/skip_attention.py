import torch.nn as nn
import torch


class SkipAttention(nn.Module):
    def __init__(self, feature_size):
        super().__init__()
        self.query = nn.Linear(feature_size, feature_size)
        self.key = nn.Linear(feature_size, feature_size)
        self.value = nn.Linear(feature_size, feature_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """x shape: [B, T, H]"""
        Q = self.query(x)  # Query
        K = self.key(x)  # Key
        V = self.value(x)  # Value

        attention = self.softmax(
            torch.bmm(Q, K.transpose(1, 2)) / (x.shape[-1] ** 0.5)
        )  # Attention weights
        out = torch.bmm(attention, V)  # Apply attention
        return out + x  # Residual connection
