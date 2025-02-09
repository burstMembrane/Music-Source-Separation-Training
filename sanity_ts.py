import torch
import torchaudio
import soundfile as sf
import argparse
import logging
from pathlib import Path
from utils import get_model_from_config  # Ensure this is consistent with `convert.py`

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def sanity_check_torchscript(model_path, model_type, config_path, input_wav, output_dir):
    """Loads a TorchScript model, runs an input audio file, and saves separated sources."""

    logger.info(f"Loading model config from {config_path}...")
    model, config = get_model_from_config(model_type, config_path)

    # Load model
    logger.info(f"Loading TorchScript model from {model_path}...")
    ts_model = torch.jit.load(model_path)
    ts_model.eval()

    # Load input audio
    waveform, sample_rate = torchaudio.load(input_wav)
    logger.info(f"Loaded input audio: {input_wav} with shape {waveform.shape} and sample rate {sample_rate}")

    # Ensure mono/stereo compatibility
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)  # Convert mono to (1, samples)

    num_channels, total_samples = waveform.shape
    chunk_size = config.audio.chunk_size

    # Pad if necessary to fit chunk size
    pad_amount = (chunk_size - (total_samples % chunk_size)) % chunk_size
    if pad_amount > 0:
        pad_tensor = torch.zeros(num_channels, pad_amount)
        waveform = torch.cat([waveform, pad_tensor], dim=-1)

    # Reshape to match expected input format (batch, channels, samples)
    waveform = waveform.unsqueeze(0)  # Add batch dimension → [1, channels, samples]

    logger.info(f"Reshaped input waveform to {waveform.shape}")

    # Run inference
    with torch.no_grad():
        output = ts_model(waveform)

    # Output shape: [n_instruments, n_channels, samples]
    logger.info(f"Model output shape: {output.shape}")

    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    instruments = getattr(
        config.training, "instruments", ["bass", "drums", "other", "vocals"]
    )

    for i, source in enumerate(instruments):
        # Try to copy ground truth if it exists
        output_path = output_dir / f"{source}_pred.wav"
        sf.write(output_path, output[0, i].cpu().numpy().T, sample_rate)
        logger.info(f"Saved {source} to {output_path}")


    logger.info("Sanity check complete. Check the output folder for separated sources.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sanity check TorchScript model with an input WAV file.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to TorchScript model (.pt)")
    parser.add_argument("--model_type", type=str, required=True, help="Type of model to load")
    parser.add_argument("--config_path", type=str, required=True, help="Path to model config")
    parser.add_argument("--input_wav", type=str, required=True, help="Path to input WAV file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save separated sources")

    args = parser.parse_args()

    sanity_check_torchscript(args.model_path, args.model_type, args.config_path, args.input_wav, args.output_dir)

