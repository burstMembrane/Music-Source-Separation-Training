import argparse
import logging
import json
import torch

from utils import get_model_from_config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(description="Converts a  model to torchscript.")

    parser.add_argument(
        "--model", type=str, required=True, help="Path to the model ckpt."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to model's config YAML.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="model.pt",
        help="Path to the output TorchScript model.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on.")

    args = parser.parse_args()
    device = torch.device(args.device)

    # Initialize model from config
    model, config = get_model_from_config("hstasnet", args.config)

    logger.info(f"Model loaded from {args.model} to {device}")

    # Verify all layers and parameters are moved correctly
    for name, param in model.named_parameters():
        if param.device != device:
            logger.warning(f"Parameter {name} is on {param.device}, expected {device}")
            param.data = param.data.to(device)

    # Ensure all buffers are on the correct device
    for name, buffer in model.named_buffers():
        if buffer.device != device:
            logger.warning(f"Buffer {name} is on {buffer.device}, expected {device}")
            buffer.data = buffer.data.to(device)

    # Ensure all parameters are float32
    for param in model.parameters():
        if param.dtype != torch.float32:
            logger.warning(f"Converting {param.dtype} to float32")
            param.data = param.data.to(torch.float32)

    for buffer in model.buffers():
        if buffer.dtype != torch.float32:
            logger.warning(f"Converting buffer from {buffer.dtype} to float32")
            buffer.data = buffer.data.to(torch.float32)

    # Final consistency check
    assert all(
        p.device == device for p in model.parameters()
    ), "Some parameters are still on incorrect device!"
    assert all(
        b.device == device for b in model.buffers()
    ), "Some buffers are still on incorrect device!"

    # Example input
    B, C, L, S = 1, 2, 100000, 4  # batch=1, 2-channels, length=100k, 4 sources
    example_input = torch.randn(B, C, L, device=device, dtype=torch.float32)

    # Convert model to TorchScript
    model.eval()
    scripted_model = torch.jit.trace(model, example_inputs=example_input)
    scripted_model.save(args.output)

    logger.info(f"Scripted model saved to {args.output}")


if __name__ == "__main__":
    main()
