import torch  # Main PyTorch library for tensors
import torch.nn as nn  # Neural network layers (e.g. Linear, LSTM, etc.)
import torch.nn.functional as ff  # Functional interface for common operations
from typing import Optional
from .spec_codec import (
    SpecEncoder,
    SpecDecoder,
)  # Modules for STFT-based encoding/decoding
from .time_codec import (
    TimeEncoder,
    TimeDecoder,
)  # Modules for learned convolutional basis
from .memory import Memory  # Custom LSTM-based "Memory" module


class HSTasNet(nn.Module):
    def __init__(
        self,
        num_sources: int,
        num_channels: int,
        time_win_size: int = 1024,
        time_hop_size: int = 512,
        time_ftr_size: int = 1000,
        spec_win_size: int = 1024,
        spec_hop_size: int = 512,
        spec_fft_size: int = 1024,
        rnn_hidden_size: int = 1000,
        rnn_num_layers: int = 1,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()  # Init the superclass (nn.Module)
        # if it isn't torchscript, use the args

        if not torch.jit.is_scripting():
            self.args = locals()  # Store all arguments for serialization
        self.num_sources: int = (
            num_sources  # Number of target sources (vocals, drums, etc.)
        )
        self.num_channels: int = (
            num_channels  # Number of input channels (e.g. stereo=2)
        )
        self.time_win_size: int = time_win_size  # Window size for time-domain encoder
        self.time_hop_size: int = time_hop_size  # Hop size for time-domain encoder
        self.time_ftr_size: int = (
            time_ftr_size  # Dimensionality of time-domain features
        )
        self.spec_win_size: int = spec_win_size  # Window size for frequency-domain STFT
        self.spec_hop_size: int = spec_hop_size  # Hop size for frequency-domain STFT
        self.spec_fft_size: int = spec_fft_size  # FFT size (e.g., 1024 points)
        self.rnn_hidden_size: int = rnn_hidden_size  # Hidden units for LSTMs
        self.rnn_num_layers: int = (
            rnn_num_layers  # Number of LSTM layers in each RNN block
        )

        # Calculate feature sizes for time and spec. This is the dimensionality after encoder.
        time_feature_size = time_ftr_size
        spec_feature_size = (
            spec_win_size // 2
        ) + 1  # For real-valued STFT with n_fft=win_size
        self.time_feature_size: int = time_feature_size
        self.spec_feature_size: int = spec_feature_size

        # Assertions ensure time/spec windows/hops match for hybrid approach (Section 3.3).
        assert (time_win_size % time_hop_size) == 0, (
            f"time_win_size ({time_win_size}) must be a multiple of time_hop_size ({time_hop_size})"
        )
        assert (spec_win_size % spec_hop_size) == 0, (
            f"spec_win_size ({spec_win_size}) must be a multiple of spec_hop_size ({spec_hop_size})"
        )
        assert time_win_size == spec_win_size, (
            f"time_win_size ({time_win_size}) must equal spec_win_size ({spec_win_size})"
        )
        assert time_hop_size == spec_hop_size, (
            f"time_hop_size ({time_hop_size}) must equal spec_hop_size ({spec_hop_size})"
        )
        if torch.jit.is_scripting():
            # Initialize all modules directly without tracing
            self.time_encoder = TimeEncoder(
                N=time_win_size, O=time_hop_size, M=time_ftr_size, device=device
            )
            self.time_decoder = TimeDecoder(
                N=time_win_size, O=time_hop_size, M=time_ftr_size, device=device
            )
            self.time_rnn_in = Memory(
                input_size=num_channels * time_feature_size,
                hidden_size=rnn_hidden_size,
                num_layers=rnn_num_layers,
                device=device,
            )
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
            # Initialize frequency domain components
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
            self.spec_rnn_in = Memory(
                input_size=num_channels * spec_feature_size,
                hidden_size=rnn_hidden_size,
                num_layers=rnn_num_layers,
                device=device,
            )
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
            self.hybrid_rnn = Memory(
                input_size=2 * rnn_hidden_size,
                hidden_size=2 * rnn_hidden_size,
                num_layers=rnn_num_layers,
                device=device,
            )
        else:
            # Non-scripting initialization
            self.time_encoder = TimeEncoder(
                N=time_win_size, O=time_hop_size, M=time_ftr_size, device=device
            )
            self.time_decoder = TimeDecoder(
                N=time_win_size, O=time_hop_size, M=time_ftr_size, device=device
            )
            self.time_rnn_in = Memory(
                input_size=num_channels * time_feature_size,
                hidden_size=rnn_hidden_size,
                num_layers=rnn_num_layers,
                device=device,
            )
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

        # Initialize time domain components
        self.time_encoder = TimeEncoder(
            N=time_win_size, O=time_hop_size, M=time_ftr_size, device=device
        )
        self.time_decoder = TimeDecoder(
            N=time_win_size, O=time_hop_size, M=time_ftr_size, device=device
        )
        self.time_rnn_in = Memory(
            input_size=num_channels * time_feature_size,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            device=device,
        )
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

        # Initialize frequency domain components
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
        self.spec_rnn_in = Memory(
            input_size=num_channels * spec_feature_size,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            device=device,
        )
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
        self.hybrid_rnn = Memory(
            input_size=2 * rnn_hidden_size,
            hidden_size=2 * rnn_hidden_size,
            num_layers=rnn_num_layers,
            device=device,
        )

    def forward(
        self, waveform: torch.Tensor, length: Optional[int] = None
    ) -> torch.Tensor:
        """Regular forward pass for non-JIT execution"""
        if torch.jit.is_scripting():
            return self.forward_jit(waveform, 0)
        return self._forward_impl(waveform, length)

    def forward_jit(self, waveform: torch.Tensor, length: int = 0) -> torch.Tensor:
        """Optimized forward pass for TorchScript compilation"""
        B, C, L = waveform.size()
        x: torch.Tensor = waveform.reshape(B * C, L)

        # Parallel encoding
        x_time, x_norm = self.time_encoder(x)
        x_spec, x_angl = self.spec_encoder(x)

        # Efficient reshaping
        BC, T, M = x_time.size()
        BC, T, F = x_spec.size()

        # Vectorized feature processing
        x_time = x_time.view(B, C, T, M)
        s_time = x_time.permute(0, 2, 1, 3).reshape(B, T, C * M)

        x_spec = x_spec.view(B, C, T, F)
        s_spec = x_spec.permute(0, 2, 1, 3).reshape(B, T, C * F)

        # Parallel RNN processing
        y_time = self.time_rnn_in(s_time)
        y_spec = self.spec_rnn_in(s_spec)

        # Hybrid processing
        y = torch.cat((y_time, y_spec), dim=2)
        y = self.hybrid_rnn(y)

        # Efficient splitting and processing
        H: int = self.rnn_hidden_size
        y_time, y_spec = y.chunk(2, dim=2)

        # Skip connections
        s_time_fc = self.time_skip_fc(s_time)
        s_spec_fc = self.spec_skip_fc(s_spec)

        y_time = self.time_rnn_out(y_time) + s_time_fc
        y_spec = self.spec_rnn_out(y_spec) + s_spec_fc

        # Mask generation
        S: int = self.num_sources
        m_time = self.time_mask_fc(y_time).view(B, T, S, C, M)
        m_spec = self.spec_mask_fc(y_spec).view(B, T, S, C, F)

        m_time = m_time.permute(0, 2, 3, 1, 4)
        m_spec = m_spec.permute(0, 2, 3, 1, 4)

        # Memory-efficient expansion
        x_time_exp = x_time.view(B, 1, C, T, M).expand(B, S, C, T, M)
        x_spec_exp = x_spec.view(B, 1, C, T, F).expand(B, S, C, T, F)

        # Mask application and reshape
        y_time = (m_time * x_time_exp).reshape(B * S * C, T, M)
        y_spec = (m_spec * x_spec_exp).reshape(B * S * C, T, F)

        # Expand norms for decoding
        x_norm_exp = (
            x_norm.view(B, 1, C, T, 1).expand(B, S, C, T, 1).reshape(B * S * C, T, 1)
        )
        x_angl_exp = (
            x_angl.view(B, 1, C, T, F).expand(B, S, C, T, F).reshape(B * S * C, T, F)
        )

        # Parallel decoding
        z_time = self.time_decoder(y_time, x_norm_exp, length)
        z_spec = self.spec_decoder(y_spec, x_angl_exp)

        # Final reshape and combination
        out = z_time.view(B, S, C, -1) + z_spec.view(B, S, C, -1)

        # Handle padding without conditionals
        L_out = out.size(-1)
        pad_amount = max(0, length - L_out)
        if pad_amount > 0:
            out = torch.nn.functional.pad(out, (0, pad_amount))

        return out.contiguous()

    @torch.jit.ignore
    def _forward_impl(
        self, waveform: torch.Tensor, length: Optional[int] = None
    ) -> torch.Tensor:
        """Original forward implementation for non-JIT execution"""
        # Add shape validation
        assert waveform.dim() == 3, f"Expected 3D input tensor, got {waveform.dim()}D"
        B, C, L = waveform.size()

        # Ensure types are explicit for TorchScript
        x: torch.Tensor = waveform.view(B * C, L)

        # Encode in the time domain -> x_time has shape [(B*C), T, M]

        x_time, x_norm = self.time_encoder(x)

        # Reshape time features to feed RNN: [B, C, T, M] -> [B, T, C*M]

        BC, T, M = x_time.size()
        x_time = x_time.view(B, C, T, M)
        s_time: torch.Tensor = x_time.permute(0, 2, 1, 3).reshape(B, T, C * M)

        # Forward pass through the time-domain RNN block
        y_time: torch.Tensor = self.time_rnn_in(s_time)

        # Encode in the frequency domain -> x_spec has shape [(B*C), T, F]

        x_spec, x_angl = self.spec_encoder(x)

        # # Reshape frequency features to feed RNN: [B, C, T, F] -> [B, T, C*F]

        BC, T, F = x_spec.size()
        x_spec = x_spec.view(B, C, T, F)
        s_spec: torch.Tensor = x_spec.permute(0, 2, 1, 3).reshape(B, T, C * F)

        # Forward pass through the frequency-domain RNN block
        y_spec: torch.Tensor = self.spec_rnn_in(s_spec)

        # Concatenate time & frequency features -> shape [B, T, 2*H]
        y: torch.Tensor = torch.cat((y_time, y_spec), dim=2)

        # The "hybrid RNN" merges the two feature streams (Section 3.3)
        y = self.hybrid_rnn(y)

        # Split back into time & frequency hidden states, each shape [B, T, H]
        H: int = self.rnn_hidden_size
        y_time, y_spec = torch.split(y, H, dim=2)

        # Additional RNN + skip-connection in time domain
        y_time = self.time_rnn_out(y_time)  # [B, T, H]
        s_time_fc: torch.Tensor = self.time_skip_fc(s_time)  # [B, T, H] skip
        y_time = y_time + s_time_fc  # Combine skip and RNN output

        # Additional RNN + skip-connection in frequency domain
        y_spec = self.spec_rnn_out(y_spec)  # [B, T, H]
        s_spec_fc: torch.Tensor = self.spec_skip_fc(s_spec)  # [B, T, H] skip
        y_spec = y_spec + s_spec_fc  # Combine skip and RNN output

        # Time-domain mask estimation -> shape [B, T, S*C*M], then rearrange
        m_time: torch.Tensor = self.time_mask_fc(y_time)
        S: int = self.num_sources
        m_time = m_time.view(B, T, S, C, M)  # [B, T, S, C, M]
        m_time = m_time.permute(0, 2, 3, 1, 4)  # [B, S, C, T, M]

        # Expand x_time for all sources, shape [B, S, C, T, M]
        x_time = x_time.view(B, 1, C, T, M).expand(B, S, C, T, M)
        # Apply time masks
        y_time = m_time * x_time

        # Frequency-domain mask estimation -> shape [B, T, S*C*F], then rearrange
        m_spec = self.spec_mask_fc(y_spec)
        S, C, F = self.num_sources, self.num_channels, self.spec_feature_size
        m_spec = m_spec.view(B, T, S, C, F)  # [B, T, S, C, F]
        m_spec = m_spec.permute(0, 2, 3, 1, 4)  # [B, S, C, T, F]

        # Expand x_spec for all sources, shape [B, S, C, T, F]
        x_spec = x_spec.view(B, 1, C, T, F).expand(B, S, C, T, F)
        # Apply frequency masks
        y_spec = m_spec * x_spec

        # # Decode time-domain frames -> shape [(B*S*C), T, M] -> [B*S*C, L]
        y_time = y_time.reshape(B * S * C, T, M)
        x_norm = (
            x_norm.view(B, 1, C, T, 1).expand(B, S, C, T, 1).reshape(B * S * C, T, 1)
        )
        z_time = self.time_decoder(y_time, x_norm)  # [B*S*C, L]
        z_time = z_time.view(B, S, C, -1)  # [B, S, C, L]

        # # Decode frequency-domain frames -> shape [(B*S*C), T, F] -> [B*S*C, L]
        y_spec = y_spec.reshape(B * S * C, T, F)
        x_angl = (
            x_angl.view(B, 1, C, T, F).expand(B, S, C, T, F).reshape(B * S * C, T, F)
        )
        z_spec = self.spec_decoder(y_spec, x_angl)  # [B*S*C, L]
        z_spec = z_spec.view(B, S, C, -1)  # [B, S, C, L]

        # # Sum time + freq domain outputs (Section 3.3: see Fig. 1 in the paper)
        out = z_time + z_spec  # [B, S, C, L]

        # # Optional zero-padding to match 'length' if specified
        if not torch.jit.is_scripting():
            if length:
                L_out = out.size(-1)
                out = ff.pad(out, (0, length - L_out), "constant")
        return out.contiguous()  # Final shape [B, S, C, L]

    @torch.jit.ignore
    def _init_args_kwargs(self) -> tuple[list[int], dict[str, int]]:
        # Return model constructor args/kwargs for easy serialization
        args = [self.num_sources, self.num_channels]
        kwargs = {
            "time_win_size": self.time_win_size,
            "time_hop_size": self.time_hop_size,
            "time_ftr_size": self.time_ftr_size,
            "spec_win_size": self.spec_win_size,
            "spec_hop_size": self.spec_hop_size,
            "spec_fft_size": self.spec_fft_size,
            "rnn_hidden_size": self.rnn_hidden_size,
        }
        return args, kwargs

    @torch.jit.ignore
    def get_model_args(self) -> tuple[list[int], dict[str, int]]:
        """Returns the model arguments for easy serialization."""
        args, kwargs = self._init_args_kwargs()
        return args, kwargs

    @torch.jit.ignore
    def serialize(self) -> dict:
        """Packages model class + arguments + weights.
        Refer Section 5.2 in the paper (Implementation details)."""
        import pytorch_lightning as pl  # Not used in torch.hub

        klass = self.__class__
        args, kwargs = self._init_args_kwargs()
        state_dict = self.state_dict()
        model_conf = {
            "model_name": klass.__name__,
            "klass": klass,
            "model_args": args,
            "model_kwargs": kwargs,
            "state_dict": state_dict,
        }
        # Additional infos
        infos = dict()
        infos["software_versions"] = dict(
            torch_version=torch.__version__,
            pytorch_lightning_version=pl.__version__,
        )
        model_conf["infos"] = infos
        return model_conf

    @torch.jit.ignore
    def save_to_path(self, model_path: str) -> str:
        """Saves the entire model to disk.
        The paper references storing model weights for runtime usage."""
        model_package = self.serialize()
        torch.save(model_package, model_path)
        return model_path


if __name__ == "__main__":
    DEVICE: torch.device = torch.device("cpu")  # Using CPU in this example
    B: int = 10
    C: int = 2
    L: int = 100000
    S: int = 4
    x: torch.Tensor = torch.randn(B, C, L, device=DEVICE)  # Random input
    print(f"{x.size() = }")  # Debug print

    # Instantiate the model, as described in Section 3 for real-time separation
    model = HSTasNet(
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
    )

    y = model(x, length=L)  # Forward pass with optional length
    print(f"{y.size() = }")  # Outputs [B, S, C, L]
