import argparse
import logging
import time

import torch
from tqdm import tqdm

from utils import get_model_from_config, load_config

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Argument parser
parser = argparse.ArgumentParser(description="Benchmark model latency")
parser.add_argument(
    "--config", type=str, default="config.yaml", help="Path to config file"
)
parser.add_argument("--model", type=str, default="resnet18", help="Model to benchmark")
parser.add_argument(
    "--device", type=str, default="cpu", help="Device to run the model on"
)
parser.add_argument(
    "--runs", type=int, default=20, help="Number of inference runs per test"
)
args = parser.parse_args()

# Load model configuration
model, config = get_model_from_config(args.model, config_path=args.config)
model_args = config.get("model", {})
SAMPLE_RATE = config.get("sample_rate", 44100)
DURATIONS = [1, 10, 30] if "sample_rate" in config else [1]

logger.info(f"Benchmarking model: {args.model}")
logger.info(f"Config file: {args.config}")
logger.info(f"Device: {args.device}")

# Warm-up runs to reduce startup overhead
logger.info("Running warm-up passes...")
for _ in range(5):
    _ = model(torch.randn(1, 2, SAMPLE_RATE, device=args.device))

# Run latency tests for different durations
results = {}

for duration in DURATIONS:
    L = duration * SAMPLE_RATE  # Convert duration to samples
    x = torch.randn(1, 2, L, device=args.device)  # Create input tensor

    latencies = []
    for _ in tqdm(range(args.runs), desc=f"Testing {duration}s Audio"):
        start_time = time.perf_counter()
        _ = model(x)  # Forward pass
        end_time = time.perf_counter()
        latencies.append((end_time - start_time) * 1000)  # Convert to ms

    # Compute latency statistics
    avg_latency = sum(latencies) / args.runs
    min_latency = min(latencies)
    max_latency = max(latencies)

    results[duration] = {
        "avg_latency": avg_latency,
        "min_latency": min_latency,
        "max_latency": max_latency,
    }

# Log final results
logger.info("\nFinal Latency Results:")
logger.info(f"Device: {args.device}")
for duration, stats in results.items():
    logger.info(f"\nLatency for {duration} sec of audio:")
    logger.info(f"  Avg Latency: {stats['avg_latency']:.2f} ms")
    logger.info(f"  Min Latency: {stats['min_latency']:.2f} ms")
    logger.info(f"  Max Latency: {stats['max_latency']:.2f} ms")
