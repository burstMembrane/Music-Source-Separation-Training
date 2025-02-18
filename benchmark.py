#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
import argparse
import json
import logging
import time
from datetime import datetime

import torch
from tqdm import tqdm
import argcomplete
from utils import get_model_from_config

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Argument parser
parser = argparse.ArgumentParser(description="Benchmark model latency")
parser.add_argument(
    "--config", type=str, help="Path to config file"
)
parser.add_argument("--model", type=str, help="Model to benchmark")
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
    "--durations", type=list, default=[1, 10, 30], help="Durations to run the benchmark on"
)
parser.add_argument(
    "--output",
    type=str,
    default="benchmark_results.json",
    help="Path to output JSON file",
)
parser.add_argument(
    "--model_type",
    type=str,
    help="Type of the model (e.g., 'mdx23c' for MDX23C model)",
)
argcomplete.autocomplete(parser)
args = parser.parse_args()

def process_benchmark(model, args):
    """Run benchmark tests and return results dictionary."""
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

        # Move model to device
        model = model.to(device)
        model.eval()

        # Get model configuration
        config = model.config if hasattr(model, 'config') else {}
        SAMPLE_RATE = config.get("sample_rate", 44100)
        DURATIONS = args.durations
        # Create input based on model type
        if args.model_type == "mdx23c":
            # MDX23C expects raw audio input (batch, channels, samples)
        
            input_shape = lambda duration: (1, 2, duration * SAMPLE_RATE)
        else:
            # Other models might expect different shapes
          
            input_shape = lambda duration: (1, 2, duration * SAMPLE_RATE )
     
        # Warm-up runs
        logger.info("Running warm-up passes...")
      
        with torch.inference_mode():
            for _ in range(1):
                x = torch.randn(input_shape(1), device=device)
                logger.info(x.size())
                _ = model(x)

        # Run latency tests
        results = {}

        # clear cache
        torch.cuda.empty_cache()

        for duration in DURATIONS:
            x = torch.randn(input_shape(duration), device=device)

            latencies = []
            for _ in tqdm(
                range(args.runs), desc=f"Testing {duration}s Audio on {device.upper()}"
            ):
                torch.cuda.synchronize() if device == "cuda" else None
                start_time = time.perf_counter()
                with torch.inference_mode():

                    try:
                        _ = model(x)
                    except Exception as e:
                        logger.error(f"Error running model: {e}")
                        continue
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

    # Save results to JSON file if output path is provided
    if hasattr(args, 'output') and args.output:
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

    return benchmark_data

if __name__ == "__main__":
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    
    # Load model and run benchmark
    model, config = get_model_from_config(args.model, config_path=args.config)
    process_benchmark(model, args)
