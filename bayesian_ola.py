import logging
import time
import warnings
import argparse
from pathlib import Path

import torch
import torchaudio
from skopt import gp_minimize
from skopt.space import Integer
from skopt.utils import use_named_args

warnings.filterwarnings("ignore")
from batch_ola import get_args, overlap_add_separation
from utils import get_model_from_config, load_not_compatible_weights

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def optimisation_objective(
    chunk,
    batch_size,
    model,
    mix,
    sample_rate,
    device,
    num_sources,
):
    """Objective function for Bayesian optimisation."""
    try:
        logger.info(f"Running with chunk_length={chunk}, batch_size={batch_size}")
        torch.cuda.synchronize()  # Ensure all previous operations are completed
        start = time.perf_counter()

        overlap_add_separation(
            model=model,
            mix=mix,
            sample_rate=sample_rate,
            chunk=chunk,
            overlap=0.1,  # Keep overlap fixed
            device=device,
            num_sources=num_sources,
            batch_size=batch_size,
        )

        torch.cuda.synchronize()  # Ensure all GPU operations are completed
        end = time.perf_counter()
        elapsed_time = end - start
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            logger.warning(
                f"CUDA OOM encountered with chunk_length={chunk}, batch_size={batch_size}."
            )
            torch.cuda.empty_cache()  # Free memory
            elapsed_time = 1000.0  # Penalize configuration
        else:
            logger.error(f"Error during processing: {e}")
            elapsed_time = 1000.0  # Penalize configuration
    return elapsed_time


def bayesian_optimisation(
    model, mix, sample_rate, device, num_sources, num_calls, random_state
):
    # TODO: make these parameters configurable
    space = [
        Integer(1, 2, name="chunk"),  # Chunk size in seconds
        Integer(32, 86, name="batch_size"),  # Batch size
    ]

    @use_named_args(space)
    def objective(
        chunk,
        batch_size,
    ):
        return optimisation_objective(
            chunk, batch_size, model, mix, sample_rate, device, num_sources
        )

    # Run Bayesian optimisation
    return gp_minimize(
        func=objective,
        dimensions=space,
        n_calls=num_calls,  # Number of optimisation iterations
        random_state=random_state,
    )


def get_base_args():
    """Create base argument parser shared between batch_ola and bayesian_ola."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=Path, required=True, help="Path to the model weights"
    )
    parser.add_argument(
        "--num_calls",
        type=int,
        default=100,
        help="Number of runs to complete in the optimization objective",
    )
    parser.add_argument("--random_state", type=int, default=42, help="Random state")

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
        "--model_type", type=str, required=True, help="Type of model to load"
    )
    parser.add_argument(
        "--config_path", type=str, required=True, help="Path to model config"
    )
    return parser


def get_args():
    """Get arguments specific to bayesian optimization."""
    parser = get_base_args()
    # Add any bayesian-specific arguments here if needed
    return parser.parse_args()


def main():
    args = get_args()

    # Load model using utils function
    model, config = get_model_from_config(args.model_type, args.config_path)

    # Load weights
    if args.model_path:
        load_not_compatible_weights(model, args.model_path)
        model = model.to(args.device)
        model.eval()

    mix, sr = torchaudio.load(args.input_audio)
    mix = mix.to(args.device)
    mix = mix.unsqueeze(0)

    # Get number of sources from config if available
    num_sources = len(
        getattr(config.training, "instruments", ["bass", "drums", "other", "vocals"])
    )

    # Run Bayesian optimisation
    logger.info(
        "Starting Bayesian optimisation for chunk and batch size configuration."
    )
    result = bayesian_optimisation(
        model=model,
        mix=mix,
        sample_rate=sr,
        device=args.device,
        num_sources=num_sources,
        num_calls=args.num_calls,
        random_state=args.random_state,
    )

    optimal_chunk, optimal_batch_size = result.x
    logger.info(
        f"Optimal parameters found: chunk_length={optimal_chunk}, batch_size={optimal_batch_size}"
    )
    logger.info(f"Minimum processing time: {result.fun:.2f} seconds")


if __name__ == "__main__":
    main()
