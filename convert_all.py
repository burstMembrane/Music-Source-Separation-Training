import argparse
import logging
import yaml
from pathlib import Path
import subprocess
import sys

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def load_yaml_config(yaml_path):
    """Load YAML configuration file."""
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

def convert_model(model_config, device):
    """Convert a single model using convert.py."""
    convert_script = Path(__file__).parent / "convert.py"
    # just make a subprocess call to convert.py
    cmd = [
        sys.executable,
        str(convert_script),
        "--model", model_config["name"],
        "--config", model_config["config"],
        "--checkpoint", model_config["checkpoint"],
        "--device", device,
       "--trace" if model_config["trace"] else "",
        "--out_dir", model_config["out_dir"]
            ]
    
    logger.info(f"Converting model: {model_config['name']}")
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        logger.info(f"Successfully converted {model_config['name']}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to convert {model_config['name']}: {e}")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Convert multiple models using a YAML config.")
    parser.add_argument(
        "--config", 
        type=str, 
        default="convert.yaml",
        help="Path to YAML config file containing model definitions."
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cpu",
        help="Device to run conversion on (cpu/cuda)."
    )
    
    args = parser.parse_args()
    
    # Load YAML config
    try:
        config = load_yaml_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load config file {args.config}: {e}")
        return
    
    # Process each model
    success_count = 0
    total_models = len(config["models"])
    
    for idx, model_config in enumerate(config["models"], 1):
        logger.info(f"\nProcessing model {idx}/{total_models}")
        if convert_model(model_config, args.device):
            success_count += 1
    
    logger.info(f"\nConversion complete: {success_count}/{total_models} models converted successfully")

if __name__ == "__main__":
    main() 
