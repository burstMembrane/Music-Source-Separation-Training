import argparse
import logging
from halo import Halo
import torch

from utils import get_model_from_config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def validate_model_layers(model, device):
    """Validate that all layers in the model are on the correct device."""
    logger.info(f"Validating model layers on {device}")
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
    assert all(p.device == device for p in model.parameters()), (
        "Some parameters are still on incorrect device!"
    )
    assert all(b.device == device for b in model.buffers()), (
        "Some buffers are still on incorrect device!"
    )

    return model


def main():
    parser = argparse.ArgumentParser(description="Converts a model to torchscript.")

    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to the model ckpt."
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on.")

    args = parser.parse_args()
    device = torch.device(args.device)

    # Initialize model from config
    model, config = get_model_from_config(args.model, args.config)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)
    model.eval()
    logger.info(f"Model loaded from {args.model} to {device}")
    model = validate_model_layers(model, device)

    # Example input
    B, C, L = (
        config.inference.batch_size,
        config.audio.num_channels,
        config.audio.chunk_size,
    )
    example_inputs = torch.randn(B, C, L, device=device, dtype=torch.float32).to(device)

    # Convert model to TorchScript using script instead of trace
    model.eval()
    with Halo(text="Converting to TorchScript...", spinner="dots"):
        # Use script instead of trace for better optimization

        scripted_model = torch.jit.script(model, example_inputs)

    # Verify the model works with example inputs
    with torch.no_grad():
        scripted_model(example_inputs)  # Warm-up run

    out_name = f"{args.model}_cs_{config.audio.chunk_size}.pt"
    # Save the TorchScript model with optimization_level=3
    scripted_model.save(out_name, _extra_files={"model_config": str(config)})

    logger.info(f"Scripted model saved to {out_name}")


if __name__ == "__main__":
    main()
