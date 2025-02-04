from doctest import Example
from pathlib import Path
import argparse
from utils import get_model_from_config
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    model, config = get_model_from_config(args.model, args.config)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    example_input = torch.randn(1, 2, 44100)

    # export to onnx
    torch.onnx.export(model, example_input, args.output, opset_version=18, dynamo=True)


if __name__ == "__main__":
    main()
