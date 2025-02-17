import argparse
import json
import logging
import time
from datetime import datetime

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
    "--device",
    type=str,
    default="all",
    choices=["all", "cpu", "cuda"],
    help="Device to run the model on ('all' to test both CPU and CUDA if available)",
)
parser.add_argument(
    "--runs", type=int, default=20, help="Number of inference runs per test"
)
parser.add_argument(
    "--output",
    type=str,
    default="benchmark_results.json",
    help="Path to output JSON file",
)
args = parser.parse_args()

# Determine which devices to test
devices_to_test = []
if args.device == "all":
    devices_to_test.append("cpu")
    if torch.cuda.is_available():
        devices_to_test.append("cuda")
elif args.device == "cuda" and not torch.cuda.is_available():
    logger.warning("CUDA requested but not available. Falling back to CPU.")
    devices_to_test.append("cpu")
else:
    devices_to_test.append(args.device)

all_results = {}

for device in devices_to_test:
    logger.info(f"\nBenchmarking on {device.upper()}...")

    # Load model configuration and move to device
    model, config = get_model_from_config(args.model, config_path=args.config)
    model = model.to(device)
    model.eval()

    model_args = config.get("model", {})
    SAMPLE_RATE = config.get("sample_rate", 44100)
    DURATIONS = [1, 10, 30, 360] if "sample_rate" in config else [1]

    # Warm-up runs
    logger.info("Running warm-up passes...")
    for _ in range(1):
        _ = model(torch.randn(1, 2, SAMPLE_RATE, device=device))

    # Run latency tests
    results = {}

    for duration in DURATIONS:
        L = duration * SAMPLE_RATE
        x = torch.randn(1, 2, L, device=device)

        # Handle input reshaping based on model framework
        if hasattr(model, "input_shape"):  # TensorFlow/Keras model
            expected_shape = model.input_shape[1:]  # Skip batch dimension
            x = x.reshape(-1, *expected_shape)
        elif hasattr(
            model, "get_expected_input_shape"
        ):  # PyTorch model with custom method
            expected_shape = model.get_expected_input_shape(x)
            x = x.reshape(expected_shape)

        latencies = []
        for _ in tqdm(
            range(args.runs), desc=f"Testing {duration}s Audio on {device.upper()}"
        ):
            torch.cuda.synchronize() if device == "cuda" else None
            start_time = time.perf_counter()
            with torch.no_grad():
                _ = model(x)
            torch.cuda.synchronize() if device == "cuda" else None
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)

        results[duration] = {
            "avg_latency": sum(latencies) / args.runs,
            "min_latency": min(latencies),
            "max_latency": max(latencies),
            "std_latency": torch.tensor(latencies).std().item(),
        }

    all_results[device] = results

# Prepare results dictionary with metadata
benchmark_data = {
    "metadata": {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "config_file": args.config,
        "tested_devices": devices_to_test,
        "runs": args.runs,
        "sample_rate": SAMPLE_RATE,
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
    },
    "results": all_results,
}

# Save results to JSON file
with open(args.output, "w") as f:
    json.dump(benchmark_data, f, indent=2)

logger.info(f"\nResults saved to: {args.output}")

# Log final results
logger.info("\nFinal Latency Results:")
for device, results in all_results.items():
    logger.info(f"\nDevice: {device.upper()}")
    for duration, stats in results.items():
        logger.info(f"\nLatency for {duration} sec of audio:")
        logger.info(f"  Avg Latency: {stats['avg_latency']:.2f} ms")
        logger.info(f"  Min Latency: {stats['min_latency']:.2f} ms")
        logger.info(f"  Max Latency: {stats['max_latency']:.2f} ms")
        logger.info(f"  Std Latency: {stats['std_latency']:.2f} ms")
