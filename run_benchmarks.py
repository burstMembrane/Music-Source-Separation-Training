#!/usr/bin/env python
import yaml
import json
import pandas as pd
from datetime import datetime

from pathlib import Path
import argparse
from typing import Dict, Any, List

import torch
from utils import get_model_from_config
from benchmark import process_benchmark
from valid import valid
# Setup logging
import logging
import time

def load_config(config_path: str) -> Dict[str, Any]:
    """Load and parse the benchmark YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_benchmarks_and_validation(config_path: str):
    """
    Run benchmarks and validation for all models specified in the config.
    
    Args:
        config_path: Path to benchmark.yml
        subset_size: Number of validation samples to use (None for all)
    """
    config = load_config(config_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create results directory
    results_dir = Path(config['benchmark']['out_dir'])
    results_dir.mkdir(exist_ok=True)

    # Clear any existing handlers
    logging.getLogger().handlers.clear()
    
    log_file = results_dir / f"benchmark_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        force=True,  # Force reconfiguration of the root logger
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # This will maintain console output
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Starting benchmark run at {timestamp}")
    logger.info(f"Results will be saved to {results_dir}")
    
    
    all_results = []
    
    # Prepare validation args
 
    # first validate all the model configs, checkpoints, etc.
    for model_config in config['models']:
        model_name = model_config['name']
        valid_args = {
        'valid_path': [config['validation']['path']],
        'metrics': config['validation']['metrics'],
        'sample_rate': config['validation']['sample_rate'],
        'chunk_length': config['validation']['chunk_length'],
        'store_dir': Path.absolute(Path(__file__).parent / config["validation"]["store_dir"] / model_name ),
        'extension': 'wav',
        'use_tta': False,
        'draw_spectro': True
    }
        logger.info(f"Storing validation results in {valid_args['store_dir']}")
        if not Path(model_config['config']).exists():
            logger.error(f"Config file for {model_name} does not exist")
            continue
        if not Path(model_config['checkpoint']).exists():
            logger.error(f"Checkpoint file for {model_name} does not exist")
            continue
        
    for model_config in config['models']:
        logger.info(f"Model config: {model_config}")
        model_name = model_config['name']
        logger.info(f"\nProcessing model: {model_name}")
        
    
        model, train_config = get_model_from_config(
            model_type=model_name,
            config_path=model_config['config']
        )
    
        checkpoint = torch.load(model_config['checkpoint'], map_location='cpu')
        model.load_state_dict(checkpoint)


        result_dict = {
            'model_name': model_name,
            'timestamp': timestamp,
            'config_path': model_config['config'],
            'checkpoint_path': model_config['checkpoint']
        }

        # Run latency benchmark
        for device in config['benchmark']['devices']:
    
            benchmark_args = argparse.Namespace(
                model=model_name,
                config=model_config['config'],
                device=device,
                runs=config['benchmark']['benchmark_runs'],
                output=str(results_dir / f"{model_name}_{device}_benchmark_{timestamp}.json"),
                model_type=model_name,
                durations=config['benchmark']['durations']
            )
            
            benchmark_results = process_benchmark(model, benchmark_args)
            result_dict.update({
                f'{device}_latency_{k}_{duration}s': v 
                for duration, metrics in benchmark_results['results'][device].items()
                for k, v in metrics.items()
            })
    
        valid_args["model_type"] = model_name
        logger.info("Validating with subset ratio: {}".format(config['validation']['subset_ratio']))
        valid_kwargs = {
            "model": model,
            "args": argparse.Namespace(**valid_args),
            "config": train_config,
            "verbose": False,
            "subset_ratio": config['validation']['subset_ratio'],
            "device": 'cuda' if torch.cuda.is_available() and 'cuda' in config['benchmark']['devices'] else 'cpu'
        }

        try:
            valid_results = valid(**valid_kwargs)
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"Error validating {model_name}: {e}")
            valid_kwargs["device"] = 'cpu'
            valid_results = valid(**valid_kwargs)
        
        result_dict.update({
            f'validation_{k}': v for k, v in valid_results.items()
        })
    
        all_results.append(result_dict)
        
        # Save individual result
        with open(results_dir / f"{model_name}_results_{timestamp}.json", 'w') as f:
            json.dump(result_dict, f, indent=2)

    # Create final DataFrame and save to CSV
    df = pd.DataFrame(all_results)
    csv_path = results_dir / f"benchmark_results_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"\nAll results saved to {csv_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run benchmarks and validation for multiple models")
    parser.add_argument("--config", type=str, default="benchmark.yml", help="Path to benchmark config file")
   
    args = parser.parse_args()
    run_benchmarks_and_validation(args.config) 