import argparse
import logging
import shutil
import time
from pathlib import Path
from contextlib import nullcontext

import librosa
import soundfile as sf
import torch
import torch.nn.functional as F
from torchaudio.transforms import Fade
from tqdm import tqdm
from utils import get_model_from_config
import torch.cuda.amp as amp

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_device_settings(device: str):
    """Configure device-specific settings and optimizations."""
    if device == "cuda" and torch.cuda.is_available():
        # Set optimal CUDA settings
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return "cuda"
    elif device == "mps" and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def optimize_model(model, device):
    """Apply various optimizations to the model."""
    model = model.to(device)
    model.eval()

    # Enable fusion optimizations if available
    if hasattr(torch, "compile") and device != "cpu":
        try:
            model = torch.compile(model, mode="reduce-overhead")
            logger.info("Applied torch.compile optimization")
        except Exception as e:
            logger.warning(f"Could not apply torch.compile: {e}")

    # Optimize memory format for CUDA
    if device == "cuda":
        model = model.to(memory_format=torch.channels_last)
        logger.info("Enabled channels_last memory format")

    return model


def overlap_add_separation(
    model,
    mix,
    sample_rate,
    chunk,
    overlap,
    device=None,
    num_sources=4,
    batch_size=None,
    enable_amp=True,
):
    start = time.perf_counter()
    chunk_len = int(sample_rate * chunk)
    overlap_len = int(sample_rate * overlap)
    step_size = chunk_len - overlap_len

    # Move data to device and optimize memory format
    if device == "cuda":
        mix = mix.to(device, memory_format=torch.channels_last)
    else:
        mix = mix.to(device)

    input_batch_size, num_channels, total_len = mix.shape
    output = torch.zeros(
        input_batch_size, num_sources, num_channels, total_len, device=device
    )

    # Create fade on device
    fade = Fade(overlap_len, overlap_len, "linear")(
        torch.ones(1, chunk_len, device=device)
    )
    fade = fade.unsqueeze(0).unsqueeze(0)[..., :chunk_len]

    # Set up mixed precision context
    amp_context = (
        amp.autocast(device_type=device, dtype=torch.float16)
        if enable_amp and device != "cpu"
        else nullcontext()
    )

    # Pre-allocate memory for chunks
    max_chunks = (total_len + step_size - 1) // step_size
    chunks = []
    chunk_idxs = []

    # 1) Gather all chunks with memory optimization
    for start_pos in tqdm(
        range(0, total_len, step_size), desc="Chunking audio", total=max_chunks
    ):
        end_pos = min(start_pos + chunk_len, total_len)
        chunk_raw = mix[:, :, start_pos:end_pos]
        if (end_pos - start_pos) < chunk_len:
            chunk_raw = F.pad(chunk_raw, (0, chunk_len - (end_pos - start_pos)))
        chunks.append(chunk_raw)
        chunk_idxs.append((start_pos, end_pos))

    # Batch all chunks at once to minimize GPU transfers
    chunks_tensor = torch.cat(chunks, dim=0).to(device)
    num_chunks = chunks_tensor.shape[0]

    logger.info(
        f"Split into {num_chunks} chunks of length {chunk_len} ({chunk_len / sample_rate:.2f} seconds)"
    )

    # 2) Batched forward pass with optimizations
    with torch.inference_mode(), amp_context:
        if batch_size is None:
            processed_batch = model(chunks_tensor)
        else:
            processed_chunks = []
            for i in tqdm(range(0, num_chunks, batch_size), desc="Processing batches"):
                batch = chunks_tensor[i : i + batch_size]
                # Clear cache periodically to prevent OOM
                if device == "cuda" and i > 0 and i % (batch_size * 10) == 0:
                    torch.cuda.empty_cache()
                processed_chunks.append(model(batch))
            processed_batch = torch.cat(processed_chunks, dim=0)

    # 3) Reassemble with optimized memory access
    n = 0
    for start_pos, end_pos in tqdm(chunk_idxs, desc="Reassembling"):
        current_len = end_pos - start_pos

        if processed_batch.size(-1) < chunk_len:
            processed_batch = F.pad(
                processed_batch, (0, chunk_len - processed_batch.size(-1))
            )

        # Process chunk
        with torch.inference_mode():
            processed_chunk = processed_batch[n : n + input_batch_size] * fade
            output[:, :, :, start_pos:end_pos] += processed_chunk[:, :, :, :current_len]
        n += input_batch_size

    end = time.perf_counter()
    logger.info(f"Processing took {end - start:.2f} seconds")

    # Clear CUDA cache after processing
    if device == "cuda":
        torch.cuda.empty_cache()

    return output


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type", type=str, required=True, help="Type of model to load"
    )
    parser.add_argument(
        "--config_path", type=str, required=True, help="Path to model config"
    )
    parser.add_argument(
        "--model_path", type=Path, required=True, help="Path to the model weights"
    )
    parser.add_argument(
        "--input_audio", type=Path, required=True, help="Path to the input audio sample"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to run the model on"
    )

    parser.add_argument(
        "--output_dir", type=Path, default="output", help="Output directory"
    )
    parser.add_argument(
        "--overlap", type=float, default=0.1, help="Overlap length in seconds"
    )
    parser.add_argument(
        "--chunk_length", type=float, default=2.0, help="Chunk length in seconds"
    )
    parser.add_argument(
        "--num_sources", type=int, default=4, help="Number of sources to separate"
    )
    parser.add_argument(
        "--batch_size", type=int, default=None, help="Batch size for model processing"
    )
    parser.add_argument(
        "--disable_amp", action="store_true", help="Disable automatic mixed precision"
    )

    return parser.parse_args()


def main():
    args = get_args()

    # Configure device and optimization settings
    device = get_device_settings(args.device)
    logger.info(f"Using device: {device}")

    # Load and optimize model
    model, config = get_model_from_config(args.model_type, args.config_path)
    model = optimize_model(model, device)

    # Load weights with optimization
    if args.model_path:
        state_dict = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state_dict)

    sr_cfg = getattr(config.audio, "sample_rate", None)
    mix, sr = librosa.load(args.input_audio, sr=None, mono=False)

    # Debug print waveform stats
    logger.info(
        f"Input stats - max: {mix.max():.4f}, min: {mix.min():.4f}, mean: {mix.mean():.4f}"
    )

    # Optimize tensor creation and transfer
    mix = torch.tensor(mix, dtype=torch.float32, device=device)
    mix = mix.unsqueeze(0)

    # Process with optimizations
    output = overlap_add_separation(
        model=model,
        mix=mix,
        sample_rate=sr,
        chunk=args.chunk_length,
        overlap=args.overlap,
        device=device,
        num_sources=args.num_sources,
        batch_size=args.batch_size,
        enable_amp=not args.disable_amp,
    )

    # Save outputs
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    input_stem = args.input_audio.parent.stem
    output_dir = Path(args.output_dir) / input_stem
    output_dir.mkdir(parents=True, exist_ok=True)

    instruments = getattr(
        config.training, "instruments", ["bass", "drums", "other", "vocals"]
    )

    # Optimize output saving
    output_cpu = output.cpu()
    for i, source in enumerate(instruments):
        # Copy ground truth if available
        gt_path = args.input_audio.parent / f"{source}.wav"
        if gt_path.exists():
            shutil.copy(gt_path, output_dir / f"{source}_gt.wav")

        output_path = output_dir / f"{source}_pred.wav"
        sf.write(output_path, output_cpu[0, i].numpy().T, sr)
        logger.info(f"Saved {source} to {output_path}")


if __name__ == "__main__":
    main()
