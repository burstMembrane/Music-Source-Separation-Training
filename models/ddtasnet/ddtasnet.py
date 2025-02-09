import torch
import torch.nn as nn

from .cbam import CBAM
from .memory import Memory
from .skip_attention import SkipAttention
from .spec_codec import SpecDecoder, SpecEncoder
from .time_codec import TimeDecoder, TimeEncoder


class DDTasNet(nn.Module):
    def __init__(
        self,
        num_sources,
        num_channels,
        time_win_size=1024,
        time_hop_size=512,
        time_ftr_size=1000,
        spec_win_size=1024,
        spec_hop_size=512,
        spec_fft_size=1024,
        rnn_hidden_size=1000,
        rnn_num_layers=1,
        device=torch.device("cpu"),
    ):
        super().__init__()

        self.num_sources = num_sources
        self.num_channels = num_channels
        self.time_win_size = time_win_size
        self.time_hop_size = time_hop_size
        self.time_ftr_size = time_ftr_size
        self.spec_win_size = spec_win_size
        self.spec_hop_size = spec_hop_size
        self.spec_fft_size = spec_fft_size
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers

        time_feature_size = time_ftr_size
        spec_feature_size = (spec_win_size // 2) + 1
        self.time_feature_size = time_feature_size
        self.spec_feature_size = spec_feature_size

        assert time_win_size == spec_win_size, "Time and spec window sizes must match."
        assert time_hop_size == spec_hop_size, "Time and spec hop sizes must match."

        # Time-domain encoder & decoder
        self.time_encoder = TimeEncoder(
            N=time_win_size, O=time_hop_size, M=time_ftr_size, device=device
        )
        self.time_decoder = TimeDecoder(
            N=time_win_size, O=time_hop_size, M=time_ftr_size, device=device
        )

        # Time-domain RNN + attention
        self.time_rnn_in = Memory(
            input_size=num_channels * time_feature_size,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            device=device,
        )

        self.time_skip_attention = SkipAttention(rnn_hidden_size)
        # https://www.sciencedirect.com/science/article/pii/S2772941924000784?via%3Dihub
        self.time_cbam = CBAM(rnn_hidden_size)

        self.time_skip_fc = nn.Linear(
            in_features=num_channels * time_feature_size,
            out_features=rnn_hidden_size,
            device=device,
        )
        self.time_rnn_out = Memory(
            input_size=rnn_hidden_size,
            hidden_size=rnn_hidden_size,
            num_layers=1,
            device=device,
        )
        self.time_mask_fc = nn.Linear(
            in_features=rnn_hidden_size,
            out_features=num_channels * num_sources * time_feature_size,
            device=device,
        )

        # Spectrogram encoder & decoder
        self.spec_encoder = SpecEncoder(
            n_win=spec_win_size,
            n_hop=spec_hop_size,
            n_fft=spec_fft_size,
            window="hamming",
            device=device,
        )
        self.spec_decoder = SpecDecoder(
            n_win=spec_win_size,
            n_hop=spec_hop_size,
            n_fft=spec_fft_size,
            window="hamming",
            device=device,
        )

        # Frequency-domain RNN + attention
        self.spec_rnn_in = Memory(
            input_size=num_channels * spec_feature_size,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            device=device,
        )
        self.spec_skip_attention = SkipAttention(rnn_hidden_size)
        self.spec_cbam = CBAM(rnn_hidden_size)

        self.spec_skip_fc = nn.Linear(
            in_features=num_channels * spec_feature_size,
            out_features=rnn_hidden_size,
            device=device,
        )
        self.spec_rnn_out = Memory(
            input_size=rnn_hidden_size,
            hidden_size=rnn_hidden_size,
            num_layers=1,
            device=device,
        )
        self.spec_mask_fc = nn.Linear(
            in_features=rnn_hidden_size,
            out_features=num_channels * num_sources * spec_feature_size,
            device=device,
        )

        # Hybrid RNN to merge time & frequency feature streams
        self.hybrid_rnn = Memory(
            input_size=3 * rnn_hidden_size,
            hidden_size=3 * rnn_hidden_size,
            num_layers=rnn_num_layers,
            device=device,
        )

    def forward(self, waveform, length=None):
        B, C, L = waveform.size()
        x = waveform.view(B * C, L)

        # Encode in time domain
        x_time, x_norm = self.time_encoder(x)

        # Reshape back to (B, C, T, M)
        x_time = x_time.view(B, C, *x_time.shape[1:])
        x_norm = x_norm.view(B, C, *x_norm.shape[1:])

        B, C, T, M = x_time.size()
        x_time = x_time.view(B, C, T, M).permute(0, 2, 1, 3).reshape(B, T, C * M)

        # Time-domain RNN + SkipAttention + CBAM
        y_time = self.time_rnn_in(x_time)
        y_time = self.time_skip_attention(y_time)
        y_time = self.time_cbam(y_time)

        # Encode in frequency domain
        x_spec_magn, x_spec_phase = self.spec_encoder(x)

        BC, T, F = x_spec_magn.size()
        x_spec_magn = (
            x_spec_magn.view(B, C, T, F).permute(0, 2, 1, 3).reshape(B, T, C * F)
        )
        x_spec_phase = (
            x_spec_phase.view(B, C, T, F).permute(0, 2, 1, 3).reshape(B, T, C * F)
        )

        # Frequency-domain RNN + SkipAttention + CBAM
        y_spec_magn = self.spec_rnn_in(x_spec_magn)
        y_spec_magn = self.spec_skip_attention(y_spec_magn)
        y_spec_magn = self.spec_cbam(y_spec_magn)

        y_spec_phase = self.spec_rnn_in(x_spec_phase)
        y_spec_phase = self.spec_skip_attention(y_spec_phase)
        y_spec_phase = self.spec_cbam(y_spec_phase)

        # Merge time & frequency features
        y = torch.cat((y_time, y_spec_magn, y_spec_phase), dim=2)
        y = self.hybrid_rnn(y)

        # Split back into time, magnitude, and phase
        H = self.rnn_hidden_size
        y_time, y_spec_magn, y_spec_phase = torch.split(y, H, dim=2)

        # Apply skip-connection
        y_time = self.time_rnn_out(y_time) + self.time_skip_fc(x_time)
        y_spec_magn = self.spec_rnn_out(y_spec_magn) + self.spec_skip_fc(x_spec_magn)
        y_spec_phase = self.spec_rnn_out(y_spec_phase) + self.spec_skip_fc(x_spec_phase)

        # Generate masks
        m_time = (
            self.time_mask_fc(y_time)
            .view(B, T, self.num_sources, C, M)
            .permute(0, 2, 3, 1, 4)
        )
        m_spec_magn = (
            self.spec_mask_fc(y_spec_magn)
            .view(B, T, self.num_sources, C, F)
            .permute(0, 2, 3, 1, 4)
        )
        m_spec_phase = (
            self.spec_mask_fc(y_spec_phase)
            .view(B, T, self.num_sources, C, F)
            .permute(0, 2, 3, 1, 4)
        )

        # Apply masks
        y_time = m_time * x_time.view(B, 1, C, T, M)
        y_spec_magn = m_spec_magn * x_spec_magn.view(B, 1, C, T, F)
        y_spec_phase = m_spec_phase * x_spec_phase.reshape(B, 1, C, T, F)

        # Decode
        x_norm_expanded = x_norm.repeat(1, self.num_sources, 1, 1, 1)

        z_time = self.time_decoder(
            y_time.reshape(B * self.num_sources * C, T, M),
            x_norm_expanded.reshape(B * self.num_sources * C, T, 1),
        )

        z_spec = self.spec_decoder(
            y_spec_magn.reshape(B * self.num_sources * C, T, F),
            y_spec_phase.reshape(B * self.num_sources * C, T, F),
        )

        output = (z_time + z_spec).view(B, self.num_sources, C, -1).contiguous()

        # Add padding to match input length
        if output.size(-1) < L:
            padding_size = L - output.size(-1)
            output = torch.nn.functional.pad(output, (0, padding_size))
        elif output.size(-1) > L:
            output = output[..., :L]

        return output


if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Example input parameters
    B, C, L, S = 4, 2, 160000, 4  # Batch=4, Stereo=2, Length=160k, Sources=4

    # Create a random input waveform (simulated audio signal)
    input_waveform = torch.randn(B, C, L, device=DEVICE)

    # Should be (B, C, L)
    print(f"Input waveform shape: {input_waveform.shape}")

    # Instantiate the model
    model = DDTasNet(
        num_sources=S,
        num_channels=C,
        time_win_size=1024,
        time_hop_size=512,
        time_ftr_size=200,
        spec_win_size=1024,
        spec_hop_size=512,
        spec_fft_size=1024,
        rnn_hidden_size=500,
        rnn_num_layers=1,
        device=DEVICE,
    ).to(DEVICE)

    # Run a forward pass
    output_waveform = model(input_waveform, length=L)

    print(f"Output waveform shape: {output_waveform.shape}")
    # Expected: (B, S, C, L) → Batch, Sources, Channels, Length
