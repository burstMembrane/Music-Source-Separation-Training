import argparse
import logging
import shutil
import time
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from torchaudio.transforms import Fade
from tqdm import tqdm

from utils import get_model_from_config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def overlap_add_separation(
    model, mix, sample_rate, chunk, overlap, device=None, num_sources=4, batch_size=None
):
    start = time.perf_counter()
    chunk_len = int(sample_rate * chunk)
    overlap_len = int(sample_rate * overlap)
    step_size = chunk_len - overlap_len

    input_batch_size, num_channels, total_len = mix.shape
    output = torch.zeros(
        input_batch_size, num_sources, num_channels, total_len, device=device
    )
    fade = Fade(overlap_len, overlap_len, "linear")(
        torch.ones(1, chunk_len, device=device)
    )
    fade = fade.unsqueeze(0).unsqueeze(0)[..., :chunk_len]

    # 1) Gather all chunks
    chunks = []
    chunk_idxs = []
    total_chunks = len(range(0, total_len, step_size))
    for start_pos in tqdm(
        range(0, total_len, step_size), desc="Chunking audio", total=total_chunks
    ):
        end_pos = min(start_pos + chunk_len, total_len)
        chunk_raw = mix[:, :, start_pos:end_pos]
        if (end_pos - start_pos) < chunk_len:
            chunk_raw = F.pad(chunk_raw, (0, chunk_len - (end_pos - start_pos)))

        chunks.append(chunk_raw)
        chunk_idxs.append((start_pos, end_pos))

    # 2) Batched forward pass
    chunks_tensor = torch.cat(chunks, dim=0).to(
        device
    )  # [n_chunks * batch_size, channels, chunk_len]
    num_chunks = chunks_tensor.shape[0]

    logger.info(
        f"Split into {num_chunks} chunks of length {chunk_len} ({chunk_len / sample_rate:.2f} seconds)"
    )

    with torch.inference_mode():
        if batch_size is None:
            processed_batch = model(chunks_tensor)
        else:
            processed_batch = []
            for i in tqdm(range(0, num_chunks, batch_size), desc="Processing batches"):
                batch = chunks_tensor[i : i + batch_size]
                processed_batch.append(model(batch))
            processed_batch = torch.cat(processed_batch, dim=0)

    # 3) Reassemble
    n = 0
    for start_pos, end_pos in chunk_idxs:
        current_len = end_pos - start_pos

        # pad the processed chunk if needed
        if processed_batch.size(-1) < chunk_len:
            processed_batch = F.pad(
                processed_batch, (0, chunk_len - processed_batch.size(-1))
            )
        processed_chunk = processed_batch[n : n + input_batch_size] * fade
        output[:, :, :, start_pos:end_pos] += processed_chunk[:, :, :, :current_len]
        n += input_batch_size

    end = time.perf_counter()
    logger.info(f"Processing took {end - start:.2f} seconds")
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
        "--use_accelerate",
        action="store_true",
        help="Use Hugging Face Accelerate for inference",
    )

    return parser.parse_args()


def main():
    args = get_args()

    # Load model using utils function
    model, config = get_model_from_config(args.model_type, args.config_path)

    # Load weights
    if args.model_path:
        # load the weights
        model.load_state_dict(torch.load(args.model_path, map_location=args.device))

        model = model.to(args.device)
        model.eval()

    # Rest of processing remains the same
    mix, sr = torchaudio.load(str(args.input_audio))
    mix = mix.to(args.device)
    mix = mix.unsqueeze(0)

    output = overlap_add_separation(
        model=model,
        mix=mix,
        sample_rate=sr,
        chunk=args.chunk_length,
        overlap=args.overlap,
        device=args.device,
        num_sources=args.num_sources,
        batch_size=args.batch_size,
    )

    # Update output saving to use config instruments if available
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    input_stem = args.input_audio.parent.stem
    output_dir = Path(args.output_dir) / input_stem
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get instruments from config if available, otherwise use default
    instruments = getattr(
        config.training, "instruments", ["bass", "drums", "other", "vocals"]
    )

    for i, source in enumerate(instruments):
        # Try to copy ground truth if it exists
        gt_path = args.input_audio.parent / f"{source}.wav"
        if gt_path.exists():
            shutil.copy(
                gt_path,
                output_dir / f"{source}_gt.wav",
            )

        output_path = output_dir / f"{source}_pred.wav"
        sf.write(output_path, output[0, i].cpu().numpy().T, sr)
        logger.info(f"Saved {source} to {output_path}")


if __name__ == "__main__":
    main()
