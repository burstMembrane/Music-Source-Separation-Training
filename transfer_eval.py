import argparse
import glob
import json
import os

import librosa
import numpy as np
import torch
from tqdm import tqdm

from inference import parse_args, proc_folder
from metrics import get_metrics
from utils import get_model_from_config


def load_models(
    hstasnet_config,
    pretrained_config,
    hstasnet_ckpt,
    pretrained_ckpt,
    device,
    transfer_layers=None,
):
    """Load both models and attempt weight transfer"""

    # Load HSTasNet model
    hst_model, _ = get_model_from_config("hstasnet", hstasnet_config)
    hst_model = hst_model.to(device)

    pt_model, _ = get_model_from_config("bs_roformer", pretrained_config["path"])
    pt_model = pt_model.to(device)

    hstasnet_ckpt = os.path.join(os.path.dirname(__file__), hstasnet_ckpt)
    pretrained_ckpt = os.path.join(os.path.dirname(__file__), pretrained_ckpt)

    # load the weights
    hst_model.load_state_dict(torch.load(hstasnet_ckpt, map_location=device))
    pt_model.load_state_dict(torch.load(pretrained_ckpt, map_location=device))

    # Weight transfer logic
    full_transfer_map = {
        # Time domain components
        "time_encoder.conv": "encoder.conv",
        "time_encoder.gate": "encoder.gate",
        "time_rnn_in": "temporal.rnn",
        # "time_mask_fc": "mask_generator.linear",
        # Frequency domain components
        # "spec_encoder.transform": "stft_layer",
        "spec_rnn_in": "spectral.rnn",
        "spec_mask_fc": "mask_generator.freq_linear",
        # Shared components
        "hybrid_rnn": "fusion.rnn",
    }

    # Use only specified layers if provided
    transfer_map = {
        k: v
        for k, v in full_transfer_map.items()
        if transfer_layers is None or k in transfer_layers
    }

    transferred = []
    skipped = []
    for hst_param, pt_param in transfer_map.items():
        try:
            hst_model.state_dict()[hst_param].copy_(pt_model.state_dict()[pt_param])
            transferred.append(f"{pt_param} -> {hst_param}")
        except (KeyError, RuntimeError) as e:
            skipped.append(f"{pt_param}: {str(e)}")

    return hst_model, transferred, skipped


def evaluate_separation(results_dir, reference_dir, device):
    """Calculate metrics for separated tracks"""
    metrics = {"sdr": [], "si_sdr": [], "bleedless": [], "aura_mrstft": []}

    for track_dir in tqdm(glob.glob(os.path.join(results_dir, "*"))):
        track_name = os.path.basename(track_dir)

        # Skip the results.json file if it exists
        if track_name == "results.json":
            continue

        # Load references and estimates
        refs = {}
        ests = {}
        mix = None

        for stem in ["vocals", "drums", "bass", "other"]:
            ref_path = os.path.join(reference_dir, track_name, f"{stem}.wav")
            est_path = os.path.join(track_dir, f"{stem}.wav")

            if os.path.exists(ref_path) and os.path.exists(est_path):
                ref, _ = librosa.load(ref_path, sr=44100, mono=False)
                est, _ = librosa.load(est_path, sr=44100, mono=False)

                # Ensure same length
                min_len = min(ref.shape[-1], est.shape[-1])
                ref = ref[..., :min_len]
                est = est[..., :min_len]

                # Convert to (channels, samples)
                refs[stem] = ref.T if ref.ndim == 2 else ref[np.newaxis]
                ests[stem] = est.T if est.ndim == 2 else est[np.newaxis]

                # Get mixture from first stem
                if mix is None:
                    mix_path = os.path.join(results_dir, track_name, "mixture.wav")
                    mix, _ = librosa.load(mix_path, sr=44100, mono=False)
                    mix = mix.T if mix.ndim == 2 else mix[np.newaxis]
                    mix = mix[..., :min_len]  # Ensure same length as stems

        # Calculate metrics per stem
        for stem in refs.keys():
            stem_metrics = get_metrics(
                metrics=["sdr", "si_sdr", "bleedless", "aura_mrstft"],
                reference=refs[stem],
                estimate=ests[stem],
                mix=mix,
                device=device,
            )

            for k, v in stem_metrics.items():
                if not np.isnan(v):  # Only append valid metrics
                    metrics[k].append(v)

    # Aggregate results
    return {k: np.mean(v) if v else float("nan") for k, v in metrics.items()}


def main():
    parser = argparse.ArgumentParser(
        description="Weight Transfer and Evaluation Pipeline"
    )
    parser.add_argument("--hst-config", required=True, help="HSTasNet config path")
    parser.add_argument("--hst-ckpt", required=True, help="HSTasNet checkpoint path")
    parser.add_argument(
        "--pt-config", required=True, help="Pretrained model config path"
    )
    parser.add_argument(
        "--pt-ckpt", required=True, help="Pretrained model checkpoint path"
    )
    parser.add_argument("--input-dir", required=True, help="Input audio directory")
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for separated tracks"
    )
    parser.add_argument(
        "--reference-dir", required=True, help="Directory with reference stems"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for processing",
    )
    parser.add_argument(
        "--transfer-layers",
        nargs="+",
        help="Specific layers to transfer. If not specified, transfers all layers",
    )
    parser.add_argument(
        "--experiment-mode",
        action="store_true",
        help="Run experiments with different layer combinations",
    )

    args = parser.parse_args()

    if args.experiment_mode:
        # Define different layer combinations to try
        layer_combinations = [
            ["time_encoder.conv", "time_encoder.gate"],  # Just time encoder
            ["spec_encoder.transform"],  # Just STFT layer
            ["time_rnn_in", "spec_rnn_in"],  # Just RNN layers
            ["time_mask_fc", "spec_mask_fc"],  # Just mask generators
            ["hybrid_rnn"],  # Just fusion layer
            [
                "time_encoder.conv",
                "time_encoder.gate",
                "time_rnn_in",
            ],  # All time domain
            ["spec_encoder.transform", "spec_rnn_in"],  # All frequency domain
        ]

        results = {}
        for layers in layer_combinations:
            print(f"\nTrying layer combination: {layers}")

            # Create a subdirectory for this experiment
            exp_output_dir = os.path.join(
                args.output_dir, "_".join(layer.split(".")[-1] for layer in layers)
            )
            os.makedirs(exp_output_dir, exist_ok=True)

            # Load model with specific layer transfer
            model, transferred, skipped = load_models(
                args.hst_config,
                {"path": args.pt_config},
                args.hst_ckpt,
                args.pt_ckpt,
                args.device,
                transfer_layers=layers,
            )

            # Run inference
            infer_args = {
                "input_folder": args.input_dir,
                "store_dir": exp_output_dir,
                "model_type": "hstasnet",
                "config_path": args.hst_config,
                "device_ids": [0],
                "force_cpu": args.device == "cpu",
            }
            proc_folder(infer_args)

            # Evaluate results
            metrics = evaluate_separation(
                exp_output_dir, args.reference_dir, args.device
            )
            results["+".join(layers)] = {
                "metrics": metrics,
                "transferred": transferred,
                "skipped": skipped,
            }

        # Save all results
        with open(os.path.join(args.output_dir, "experiment_results.json"), "w") as f:
            json.dump(results, f, indent=2)

        # Print summary
        print("\nExperiment Results:")
        for combo, result in results.items():
            print(f"\nLayer combination: {combo}")
            for k, v in result["metrics"].items():
                print(f"{k.upper()}: {v:.2f} dB")

    else:
        # 1. Load and transfer weights
        print("Loading models and transferring weights...")
        model, transferred, skipped = load_models(
            args.hst_config,
            {"path": args.pt_config},
            args.hst_ckpt,
            args.pt_ckpt,
            args.device,
            transfer_layers=args.transfer_layers,
        )

        print(f"Successfully transferred {len(transferred)} parameters:")
        print("\n".join(transferred))
        print(f"\nSkipped {len(skipped)} parameters:")
        print("\n".join(skipped))

        # 2. Run inference
        print("\nRunning inference...")
        infer_args = {
            "input_folder": args.input_dir,
            "store_dir": args.output_dir,
            "model_type": "hstasnet",
            "config_path": args.hst_config,
            "device_ids": [0],
            "force_cpu": args.device == "cpu",
        }
        proc_folder(infer_args)

        # 3. Evaluate results
        print("\nEvaluating results...")
        metrics = evaluate_separation(args.output_dir, args.reference_dir, args.device)

        print("\nFinal Metrics:")
        for k, v in metrics.items():
            print(f"{k.upper()}: {v:.2f} dB")

        # Save results
        with open(os.path.join(args.output_dir, "results.json"), "w") as f:
            json.dump(
                {
                    "transferred_params": transferred,
                    "skipped_params": skipped,
                    "metrics": metrics,
                },
                f,
                indent=2,
            )


if __name__ == "__main__":
    main()
