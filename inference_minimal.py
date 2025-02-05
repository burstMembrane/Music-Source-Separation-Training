import argparse
import os

import librosa
import soundfile as sf
import torch

from utils import get_model_from_config, load_start_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_audio",
        type=str,
        required=True,
        help="Path to input audio file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save output files",
    )
    parser.add_argument(
        "--output_audio",
        type=str,
        default="output.wav",
        help="Output audio file name (if model returns a single tensor)",
    )
    parser.add_argument(
        "--lora_checkpoint",
        type=str,
        default="",
        help="Path to LoRA checkpoint",
    )
    parser.add_argument(
        "--model_type", type=str, default="mdx23c", help="Model type to load"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to model config file",
    )
    parser.add_argument(
        "--start_check_point",
        type=str,
        default="",
        help="Path to model checkpoint (if any)",
    )
    parser.add_argument(
        "--force_cpu",
        action="store_true",
        help="Force CPU usage even if CUDA is available",
    )
    args = parser.parse_args()

    device = "cpu" if args.force_cpu or not torch.cuda.is_available() else "cuda"
    model, config = get_model_from_config(args.model_type, args.config_path)
    load_start_checkpoint(args, model, type_="inference")
    model = model.to(device).eval()
    sr_cfg = getattr(config.audio, "sample_rate", None)
    mix, sr = librosa.load(args.input_audio, sr=sr_cfg, mono=False)
    inp = torch.tensor(mix, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.inference_mode():
        out = model(inp)

    # squeeze batch dimension
    out = out.squeeze(0)
    print(out.shape)

    # np.nan_to_num(out, copy=True, nan=0.0)
    os.makedirs(args.output_dir, exist_ok=True)
    instruments = config.training.instruments
    print("Saving estimates for:", instruments)
    for i, instr in enumerate(instruments):
        print(f"Output shape for {instr}: {out[i].shape}")
        estimates = out[i]
        output_path = os.path.join(args.output_dir, f"{instr}.wav")
        sf.write(output_path, estimates.T, sr, subtype="FLOAT")


if __name__ == "__main__":
    main()
