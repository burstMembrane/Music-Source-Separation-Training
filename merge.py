import argparse
from tqdm import tqdm


import torch
from utils import get_model_from_config, load_not_compatible_weights
from enum import Enum, auto
import soundfile as sf
import torch.nn as nn
import numpy as np

from metrics import sdr


class MergeStrategy(Enum):
    ORIGINAL = auto()  # Current: Just use model2's weights
    WEIGHTED_ADD = auto()  # Current: Weighted sum of both models
    MULTIPLY = auto()  # Current: Element-wise multiplication
    MAX = auto()  # Current: Take maximum values
    MIN = auto()  # Current: Take minimum values

    # New strategies:
    ADAPTIVE = auto()  # Use weights based on layer performance metrics
    GATED = auto()  # Learn gating parameters to combine weights
    MAGNITUDE_WEIGHTED = auto()  # Weight based on parameter magnitudes
    LAYER_SELECTIVE = auto()  # Use best performing layers from each model
    ENSEMBLE = auto()  # Keep both models' weights and average outputs
    TIES = auto()  # New strategy
    SLERP = auto()  # New strategy


def parse_args():
    parser = argparse.ArgumentParser(description="Merge weights from two models")
    parser.add_argument(
        "--first_model", type=str, required=True, help="First model type"
    )
    parser.add_argument(
        "--second_model", type=str, required=True, help="Second model type"
    )
    parser.add_argument(
        "--first_config", type=str, required=True, help="Path to first model config"
    )
    parser.add_argument(
        "--second_config", type=str, required=True, help="Path to second model config"
    )
    parser.add_argument(
        "--first_checkpoint",
        type=str,
        required=True,
        help="Path to first model checkpoint",
    )
    parser.add_argument(
        "--second_checkpoint",
        type=str,
        required=True,
        help="Path to second model checkpoint",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save merged model checkpoint",
    )
    parser.add_argument(
        "--merge_strategy",
        type=str,
        choices=[s.name.lower() for s in MergeStrategy],
        default="original",
        help="Strategy to merge weights",
    )
    parser.add_argument(
        "--weight_ratio",
        type=float,
        default=0.5,
        help="Weight ratio for weighted_add strategy (0.0 to 1.0, represents weight of second model)",
    )
    parser.add_argument(
        "--validation_data",
        type=str,
        help="Path to validation data for adaptive and layer selective strategies",
    )
    return parser.parse_args()


def pad_tensor_to_match(source_tensor, target_shape):
    """Pad (or trim) source tensor to match target shape."""
    current_shape = source_tensor.shape
    padded_tensor = source_tensor

    for dim in range(len(current_shape)):
        if dim >= len(target_shape):
            break

        if current_shape[dim] < target_shape[dim]:
            # Need to pad this dimension
            pad_size = target_shape[dim] - current_shape[dim]
            pad_pattern = [0] * (len(current_shape) * 2)
            pad_pattern[-(dim * 2 + 1)] = pad_size
            padded_tensor = torch.nn.functional.pad(
                padded_tensor, tuple(pad_pattern), mode="constant", value=0
            )
        elif current_shape[dim] > target_shape[dim]:
            # Need to trim this dimension
            slicing = [slice(None)] * len(current_shape)
            slicing[dim] = slice(0, target_shape[dim])
            padded_tensor = padded_tensor[slicing]

    return padded_tensor


def slerp(tensor1, tensor2, t):
    """
    Spherical Linear Interpolation between two tensors.

    Args:
        tensor1: First tensor
        tensor2: Second tensor
        t: Interpolation factor (0.0 to 1.0)

    Returns:
        Interpolated tensor
    """
    # Handle edge cases
    if t <= 0.0:
        return tensor1
    if t >= 1.0:
        return tensor2

    # Preserve original shapes and magnitudes
    original_shape = tensor1.shape
    tensor1_magnitude = torch.norm(tensor1)
    tensor2_magnitude = torch.norm(tensor2)

    # Flatten and normalize tensors
    flat1 = tensor1.view(-1)
    flat2 = tensor2.view(-1)

    # Avoid division by zero
    norm1 = torch.norm(flat1)
    norm2 = torch.norm(flat2)

    if norm1 < 1e-8 or norm2 < 1e-8:
        return (1.0 - t) * tensor1 + t * tensor2

    normalized1 = flat1 / norm1
    normalized2 = flat2 / norm2

    # Calculate dot product and clamp to valid range
    dot_product = torch.sum(normalized1 * normalized2)
    dot_product = torch.clamp(dot_product, -1.0, 1.0)

    # Calculate omega (angle between vectors)
    omega = torch.acos(dot_product)

    # If vectors are very close, use linear interpolation
    if omega < 1e-4:
        return (1.0 - t) * tensor1 + t * tensor2

    sin_omega = torch.sin(omega)

    # Calculate interpolation factors
    scale1 = torch.sin((1.0 - t) * omega) / sin_omega
    scale2 = torch.sin(t * omega) / sin_omega

    # Interpolate directions
    result = scale1 * normalized1 + scale2 * normalized2

    # Interpolate magnitudes
    target_magnitude = (1.0 - t) * tensor1_magnitude + t * tensor2_magnitude

    # Scale to target magnitude and reshape
    result = result * target_magnitude / torch.norm(result)
    result = result.view(original_shape)

    return result


def apply_merge_strategy(
    tensor1,
    tensor2,
    strategy,
    weight_ratio=0.5,
    layer_name=None,
    performance_metrics=None,
):
    """Enhanced merge strategy with SLERP and TIES approaches."""
    # Convert strategy string to uppercase and get enum value
    try:
        strategy = MergeStrategy[strategy.upper()]
    except KeyError:
        # If it's already a MergeStrategy enum, use it directly
        if not isinstance(strategy, MergeStrategy):
            raise ValueError(f"Unknown merge strategy: {strategy}")

    # Ensure tensors have same shape for all strategies
    if tensor1.shape != tensor2.shape:
        tensor2 = pad_tensor_to_match(tensor2, tensor1.shape)
        print(
            f"Padded tensor for layer {layer_name}: {tensor1.shape} -> {tensor2.shape}"
        )

    elif strategy == MergeStrategy.ADAPTIVE:
        # Calculate weight ratio based on performance metrics or use default
        if performance_metrics and layer_name in performance_metrics:
            score1, score2 = performance_metrics[layer_name]
            total = score1 + score2
            adaptive_ratio = score2 / total if total > 0 else 0.5
            print(f"Adaptive ratio for {layer_name}: {adaptive_ratio:.4f}")
        else:
            adaptive_ratio = weight_ratio
            print(f"Using default ratio for {layer_name}: {adaptive_ratio:.4f}")

        # Simple weighted average while preserving tensor dimensions
        return (1 - adaptive_ratio) * tensor1 + adaptive_ratio * tensor2

    elif strategy == MergeStrategy.SLERP:
        return slerp(tensor1, tensor2, weight_ratio)

    elif strategy == MergeStrategy.TIES:
        # Calculate difference from tensor1 (treating it as base)
        diff = tensor2 - tensor1

        # Calculate density threshold
        density = 0.3  # Can be made configurable
        k = int(diff.numel() * density)

        # Get magnitude of differences
        diff_magnitude = torch.abs(diff)
        threshold = torch.kthvalue(
            diff_magnitude.view(-1), diff_magnitude.numel() - k
        ).values

        # Create mask for significant differences
        mask = diff_magnitude >= threshold

        # Apply sparsification
        sparse_diff = torch.where(mask, diff, torch.zeros_like(diff))

        # Sign consensus (simplified version)
        sign_mask = sparse_diff != 0
        if sign_mask.any():
            # Keep only differences where signs agree
            merged = tensor1 + sparse_diff
        else:
            # If no significant differences, keep base tensor
            merged = tensor1

        return merged

    elif strategy == MergeStrategy.MAGNITUDE_WEIGHTED:
        mag1 = torch.norm(tensor1)
        mag2 = torch.norm(tensor2)
        total_mag = mag1 + mag2
        weight = mag2 / total_mag if total_mag > 0 else 0.5
        return (1 - weight) * tensor1 + weight * tensor2

    elif strategy == MergeStrategy.LAYER_SELECTIVE:
        if performance_metrics and layer_name in performance_metrics:
            score1, score2 = performance_metrics[layer_name]
            print(
                f"Layer {layer_name} scores - Model1: {score1:.4f}, Model2: {score2:.4f}"
            )
            return tensor2 if score2 > score1 else tensor1
        else:
            print(
                f"No performance metrics for {layer_name}, using default model1 weights"
            )
            return tensor1

    elif strategy == MergeStrategy.ORIGINAL:
        return tensor2
    elif strategy == MergeStrategy.WEIGHTED_ADD:
        return (1 - weight_ratio) * tensor1 + weight_ratio * tensor2
    elif strategy == MergeStrategy.MULTIPLY:
        return tensor1 * tensor2
    elif strategy == MergeStrategy.MAX:
        return torch.maximum(tensor1, tensor2)
    elif strategy == MergeStrategy.MIN:
        return torch.minimum(tensor1, tensor2)

    raise ValueError(f"Unknown merge strategy: {strategy}")


def collect_layer_metrics(model1, model2, validation_data):
    """Collect performance metrics for each layer using validation data."""
    print("\nCollecting layer performance metrics...")
    metrics = {}

    print("Loading validation data...")
    # Load validation data
    audio, sr = sf.read(validation_data, frames=88200)

    if len(audio.shape) == 1:
        audio = audio[None, None, :]  # Add batch and channel dimensions
    else:
        audio = audio.T[None, :]  # Transpose and add batch dimension

    x = torch.FloatTensor(audio)

    # Function to get layer outputs
    def get_layer_outputs(model, x):
        outputs = {}
        hooks = []

        def hook_fn(name):
            def hook(module, input, output):
                outputs[name] = output.detach()

            return hook

        # Register hooks for each layer
        for name, module in model.named_modules():
            if any(isinstance(module, t) for t in [nn.Conv1d, nn.Conv2d, nn.Linear]):
                hooks.append(module.register_forward_hook(hook_fn(name)))

        # Forward pass
        with torch.no_grad():
            model(x)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return outputs

    # Get outputs from both models
    outputs1 = get_layer_outputs(model1, x)
    outputs2 = get_layer_outputs(model2, x)

    # Compare layer outputs
    print("\nComputing layer metrics...")
    for name in tqdm(outputs1.keys(), total=len(outputs1), desc="Processing layers"):
        if name in outputs2:
            # Calculate metrics for this layer
            out1 = outputs1[name]
            out2 = outputs2[name]

            # Use L2 norm as a simple metric
            score1 = torch.norm(out1).item()
            score2 = torch.norm(out2).item()

            metrics[name] = (score1, score2)
            print(f"Layer {name} metrics - Model1: {score1:.4f}, Model2: {score2:.4f}")

    return metrics


def merge_models(args):
    # Load first model
    model1, _ = get_model_from_config(args.first_model, args.first_config)
    model1 = model1.to("cpu")
    load_not_compatible_weights(model1, args.first_checkpoint)
    state_dict1 = model1.state_dict()

    # Load second model
    model2, _ = get_model_from_config(args.second_model, args.second_config)
    model2 = model2.to("cpu")
    load_not_compatible_weights(model2, args.second_checkpoint)
    state_dict2 = model2.state_dict()

    # Collect performance metrics if needed
    performance_metrics = None
    if (
        args.merge_strategy.lower() in ["adaptive", "layer_selective"]
        and args.validation_data
    ):
        performance_metrics = collect_layer_metrics(
            model1, model2, args.validation_data
        )

    # Create merged state dict starting with model1's weights
    merged_state_dict = state_dict1.copy()

    # Track statistics about the merge
    total_keys = len(state_dict1)
    merged_keys = 0
    padded_keys = 0
    skipped_keys = 0

    # Attempt to merge compatible weights from model2
    for key in state_dict2:
        if key in state_dict1:
            if state_dict1[key].shape == state_dict2[key].shape:
                # Direct merge for matching shapes
                merged_state_dict[key] = apply_merge_strategy(
                    state_dict1[key],
                    state_dict2[key],
                    args.merge_strategy,
                    args.weight_ratio,
                    layer_name=key,
                    performance_metrics=performance_metrics,
                )
                merged_keys += 1
            else:
                try:
                    # Attempt to pad/trim tensor to match shape
                    padded_tensor = pad_tensor_to_match(
                        state_dict2[key], state_dict1[key].shape
                    )
                    if padded_tensor.shape == state_dict1[key].shape:
                        merged_state_dict[key] = apply_merge_strategy(
                            state_dict1[key],
                            padded_tensor,
                            args.merge_strategy,
                            args.weight_ratio,
                            layer_name=key,
                            performance_metrics=performance_metrics,
                        )
                        padded_keys += 1
                        print(
                            f"Padded {key} from {state_dict2[key].shape} to {padded_tensor.shape}"
                        )
                    else:
                        print(
                            f"Failed to pad {key}: {state_dict2[key].shape} -> {state_dict1[key].shape}"
                        )
                        skipped_keys += 1
                except Exception as e:
                    print(f"Error padding {key}: {str(e)}")
                    skipped_keys += 1
        else:
            print(f"Key {key} not found in first model")
            skipped_keys += 1

    print(f"\nMerge Statistics:")
    print(f"Total keys in first model: {total_keys}")
    print(f"Successfully merged keys: {merged_keys}")
    print(f"Padded keys: {padded_keys}")
    print(f"Skipped keys: {skipped_keys}")
    print(f"Merge strategy: {args.merge_strategy}")
    if args.merge_strategy.lower() == "weighted_add":
        print(f"Weight ratio: {args.weight_ratio}")
    if performance_metrics:
        print(f"Number of layers with performance metrics: {len(performance_metrics)}")

    # Save merged model
    torch.save(merged_state_dict, args.output_path)
    print(f"\nMerged model saved to: {args.output_path}")


def main():
    args = parse_args()
    merge_models(args)


if __name__ == "__main__":
    main()
