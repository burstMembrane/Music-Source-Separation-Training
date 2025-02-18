#!/usr/bin/env python3
"""
This script performs transfer learning from a pretrained HTDemucs model
(on models/demucs4ht.py) to the HSTasNet model (on models/hstasnet/hstasnet.py).
It copies the weights from:
  - The HTDemucs "spectral" encoder (i.e. the first HEncLayer in the frequency branch)
    → to HSTasNet's SpecEncoder (assumed to contain a convolutional filter bank as `conv`).
  - The HTDemucs time-(waveform) encoder (the first element of tencoder)
    → to HSTasNet's TimeEncoder (assumed to contain a `conv` attribute).
  - The HTDemucs time-(waveform) decoder (the first element of tdecoder)
    → to HSTasNet's TimeDecoder (assumed to contain a `conv` attribute).
  - Optionally, the HTDemucs spectral decoder (the first element of decoder)
    → to HSTasNet's SpecDecoder.

Usage:
    python transfer_weights.py --demucs_checkpoint path/to/htdemucs.ckpt --output_path path/to/hstasnet_transferred.ckpt
"""

import argparse
import torch

# Import the two models
from models.demucs4ht import HTDemucs
from models.hstasnet.hstasnet import HSTasNet


def transfer_conv_weights(src_conv: torch.nn.Module, tgt_conv: torch.nn.Module) -> None:
    """
    Transfers weight (and bias) parameters from a source convolutional layer to a target one.
    If shapes don't match, attempts to expand/interpolate the weights.
    """
    # Print shape information for debugging
    print(f"Source conv weight shape: {src_conv.weight.shape}")
    print(f"Target conv weight shape: {tgt_conv.weight.shape}")

    if src_conv.weight.shape != tgt_conv.weight.shape:
        # For this specific case, expanding from [48, 2, 8] to [1024, 1024]
        # First reshape and repeat the source weights
        src_weights = src_conv.weight

        # Reshape to 2D matrix first
        expanded = src_weights.view(48 * 2, 8)  # [96, 8]

        # Repeat and interpolate to reach target size
        expanded = torch.nn.functional.interpolate(
            expanded.unsqueeze(0),  # Add batch dimension
            size=(1024,),  # Target width
            mode="linear",
        ).squeeze(0)  # Remove batch dimension

        # Repeat along first dimension to reach 1024
        repeats_needed = 1024 // expanded.size(0) + 1
        expanded = expanded.repeat(repeats_needed, 1)

        # Trim to exact size
        expanded = expanded[:1024, :1024]

        # Copy to target
        tgt_conv.weight.data.copy_(expanded)
        print(
            f"Expanded weights from {src_conv.weight.shape} to {tgt_conv.weight.shape}"
        )
    else:
        tgt_conv.weight.data.copy_(src_conv.weight.data)

    # Handle bias if present
    if (src_conv.bias is not None) and (tgt_conv.bias is not None):
        if src_conv.bias.shape != tgt_conv.bias.shape:
            # Interpolate bias similarly
            src_bias = src_conv.bias
            expanded_bias = torch.nn.functional.interpolate(
                src_bias.unsqueeze(0).unsqueeze(0), size=(1024,), mode="linear"
            ).squeeze()
            tgt_conv.bias.data.copy_(expanded_bias)
            print(f"Expanded bias from {src_conv.bias.shape} to {tgt_conv.bias.shape}")
        else:
            tgt_conv.bias.data.copy_(src_conv.bias.data)


def main(args):
    # Load pretrained HTDemucs checkpoint.
    # Here we assume the checkpoint is saved with a "state_dict" key.
    print(f"Loading pretrained HTDemucs checkpoint from {args.demucs_checkpoint} ...")
    checkpoint = torch.load(
        args.demucs_checkpoint, map_location="cpu", weights_only=False
    )

    if "state" in checkpoint:
        demucs_state = checkpoint["state"]
    else:
        demucs_state = checkpoint  # if state_dict is saved at the top-level

    # Instantiate HTDemucs with correct architecture
    demucs_model = HTDemucs(*checkpoint["args"], **checkpoint["kwargs"])

    # Load with strict=False to handle residual architecture differences
    demucs_model.load_state_dict(demucs_state, strict=False)
    demucs_model.eval()  # set to eval mode

    # Instantiate HSTasNet with matching hyperparameters from the pretrained model
    hstnet_model = HSTasNet(
        num_sources=4,
        num_channels=2,
        time_win_size=1024,
        time_hop_size=512,
        time_ftr_size=1024,  # Match the encoder/decoder sizes
        spec_win_size=1024,
        spec_hop_size=512,
        spec_fft_size=1024,
        rnn_hidden_size=1000,  # Match the RNN hidden sizes (was 500)
        rnn_num_layers=2,  # Match the number of RNN layers
        device=torch.device("cpu"),
    )

    # --- Transfer waveform (time) branch weights ---
    # We assume that the HTDemucs time encoder is stored as a list (tencoder).
    if len(demucs_model.tencoder) > 0:
        # Transfer from the first time branch encoder layer into HSTasNet's time encoder.
        # (Assumes the submodule we want has a "conv" attribute.)
        print("Transferring waveform encoder weights...")
        src_tenc = demucs_model.tencoder[0]
        # Assuming TimeEncoder in HSTasNet has an attribute "conv"
        transfer_conv_weights(src_tenc.conv, hstnet_model.time_encoder.conv)
    else:
        print("Warning: No time encoder found in the pretrained HTDemucs model.")

    if len(demucs_model.tdecoder) > 0:
        # Transfer from the first (or appropriate) time branch decoder layer to TimeDecoder.
        print("Transferring waveform decoder weights...")
        src_tdec = demucs_model.tdecoder[0]
        # For HDecLayer, the conv is likely inside the deconv attribute
        if hasattr(src_tdec, "deconv"):
            transfer_conv_weights(src_tdec.deconv, hstnet_model.time_decoder.conv)
        else:
            print("Warning: Could not find deconv in time decoder layer")
    else:
        print("Warning: No time decoder found in the pretrained HTDemucs model.")

    # --- Transfer spectral branch weights ---
    if len(demucs_model.encoder) > 0:
        print("Transferring spectral encoder weights...")
        src_spec_enc = demucs_model.encoder[0]
        # Check if the encoder has a transform attribute
        if hasattr(src_spec_enc, "transform"):
            transfer_conv_weights(
                src_spec_enc.transform, hstnet_model.spec_encoder.transform
            )
        else:
            print("Warning: Could not find transform in spectral encoder")
    else:
        print("Warning: No spectral encoder found in the pretrained HTDemucs model.")

    if len(demucs_model.decoder) > 0:
        print("Transferring spectral decoder weights...")
        src_spec_dec = demucs_model.decoder[0]
        # Check if the decoder has a transform attribute
        if hasattr(src_spec_dec, "transform"):
            transfer_conv_weights(
                src_spec_dec.transform, hstnet_model.spec_decoder.transform
            )
        else:
            print("Warning: Could not find transform in spectral decoder")
    else:
        print("Warning: No spectral decoder found in the pretrained HTDemucs model.")

    # (Optionally, you could add more mapping if you want to transfer additional weights.)

    # Save the updated HSTasNet state_dict.
    torch.save(hstnet_model.state_dict(), args.output_path)
    print(f"Transferred HSTasNet weights saved to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transfer weights from a pretrained HTDemucs model to HSTasNet."
    )
    parser.add_argument(
        "--demucs_checkpoint",
        type=str,
        required=True,
        help="Path to the pretrained HTDemucs checkpoint.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Filename to save the HSTasNet checkpoint with transferred weights.",
    )
    args = parser.parse_args()
    main(args)
