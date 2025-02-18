import argparse


import json
import numpy as np
import soundfile as sf
from pathlib import Path
import torch
from metrics import sdr


def load_audio(path):
    """Load audio file and return as numpy array."""
    print(f"Loading audio from: {path}")
    audio, sr = sf.read(path)
    print(f"Original shape: {audio.shape}, Sample rate: {sr}")

    # Convert to (channels, samples) format if mono
    if len(audio.shape) == 1:
        audio = np.expand_dims(audio, axis=0)
        print(f"Expanded mono to shape: {audio.shape}")
    elif len(audio.shape) == 2:
        # Convert from (samples, channels) to (channels, samples)
        audio = audio.T
        print(f"Transposed stereo to shape: {audio.shape}")
    return audio


def calculate_sdr_for_stems(pred_dir, gt_dir):
    """Calculate SDR between predicted and ground truth stems."""
    print(f"\nCalculating SDR between:")
    print(f"Predicted dir: {pred_dir}")
    print(f"Ground truth dir: {gt_dir}")

    stems = ["vocals", "drums", "bass", "other"]
    results = {}

    all_references = []
    all_estimates = []

    for stem in stems:
        print(f"\nProcessing {stem}...")
        pred_path = Path(pred_dir) / f"{stem}_pred.wav"
        gt_path = Path(gt_dir) / f"{stem}.wav"

        print(f"Pred path exists: {pred_path.exists()}")
        print(f"GT path exists: {gt_path.exists()}")

        if pred_path.exists() and gt_path.exists():
            # Load and prepare audio
            pred_audio = load_audio(str(pred_path))
            gt_audio = load_audio(str(gt_path))

            # Ensure same length
            min_len = min(pred_audio.shape[1], gt_audio.shape[1])
            pred_audio = pred_audio[..., :min_len]
            gt_audio = gt_audio[..., :min_len]
            print(f"Trimmed to length: {min_len}")

            # Add to collection
            all_references.append(gt_audio)
            all_estimates.append(pred_audio)

    if all_references:
        # Stack all stems
        references = np.stack(all_references, axis=0)
        estimates = np.stack(all_estimates, axis=0)
        print(
            f"Final shapes - References: {references.shape}, Estimates: {estimates.shape}"
        )

        # Calculate SDR for all stems at once
        sdr_values = sdr(references, estimates)

        # Store results
        for stem, sdr_val in zip(stems, sdr_values):
            results[stem] = float(sdr_val)
            print(f"SDR for {stem}: {sdr_val:.2f} dB")

        # Calculate average
        results["average"] = float(np.mean(sdr_values))
        print(f"\nAverage SDR: {results['average']:.2f} dB")
    else:
        print("\nNo valid stems found for SDR calculation")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", type=str, required=True)
    parser.add_argument("--gt_dir", type=str, required=True)
    parser.add_argument("--strategy", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    print("\nStarting evaluation with arguments:")
    print(f"Pred dir: {args.pred_dir}")
    print(f"GT dir: {args.gt_dir}")
    print(f"Strategy: {args.strategy}")
    print(f"Output file: {args.output_file}")

    # Calculate SDR for all stems
    results = calculate_sdr_for_stems(args.pred_dir, args.gt_dir)

    # Prepare output
    output = {"strategy": args.strategy, "sdr_scores": results}

    # Save to file
    print(f"\nSaving results to: {args.output_file}")
    with open(args.output_file, "w") as f:
        json.dump(output, f, indent=4)

    # Print to console
    print(f"\nFinal SDR Results for {args.strategy}:")
    for stem, score in results.items():
        print(f"{stem}: {score:.2f} dB")


if __name__ == "__main__":
    main()
