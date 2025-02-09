import torch.nn as nn


class CBAM(nn.Module):
    def __init__(self, feature_size):
        super().__init__()

        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(feature_size, feature_size // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(feature_size // 4, feature_size, kernel_size=1),
            nn.Sigmoid(),
        )

        self.spatial_att = nn.Sequential(
            # Conv1d expects (N, C, L)
            nn.Conv1d(1, 1, kernel_size=7, padding=3),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        x shape: [B, T, H] or [B, T, F]
        """
        # Ensure input shape is [B, H, T] for channel attention
        x = x.permute(0, 2, 1)  # Swap to [B, H, T]

        # Channel Attention
        channel_att = self.channel_att(x)  # [B, H, 1]
        channel_att = channel_att * x  # Apply attention

        # Spatial Attention: Reduce across channels
        spatial_input = channel_att.mean(dim=1, keepdim=True)  # [B, 1, T]
        spatial_att = self.spatial_att(spatial_input)  # [B, 1, T]
        out = spatial_att * channel_att  # Apply spatial attention

        return out.permute(0, 2, 1)  # Swap back to [B, T, H]
