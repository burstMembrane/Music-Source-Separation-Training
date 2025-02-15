# coding: utf-8
__author__ = "Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/"
__version__ = "1.0.3"

# Read more here:
# https://huggingface.co/docs/accelerate/index

import argparse
import glob
import os
import time
import warnings
import sys
import yaml

import auraloss
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from torch.optim import SGD, Adam, AdamW, RAdam, RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import wandb
from dataset import MSSDataset
from metrics import sdr
from train import manual_seed, masked_loss
from utils import (
    demix,
    get_model_from_config,
    load_not_compatible_weights,
    prefer_target_instrument,
)

warnings.filterwarnings("ignore")


class MSSValidationDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        all_mixtures_path = []
        for valid_path in args.valid_path:
            part = sorted(glob.glob(valid_path + "/*/mixture.wav"))
            if len(part) == 0:
                print("No validation data found in: {}".format(valid_path))
            all_mixtures_path += part

        self.list_of_files = all_mixtures_path

    def __len__(self):
        return len(self.list_of_files)

    def __getitem__(self, index):
        return self.list_of_files[index]


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        type=str,
        default="mdx23c",
        help="One of mdx23c, htdemucs, segm_models, mel_band_roformer, bs_roformer, swin_upernet, bandit",
    )
    parser.add_argument("--config_path", type=str, help="path to config file")
    parser.add_argument(
        "--start_check_point",
        type=str,
        default="",
        help="Initial checkpoint to start training",
    )
    parser.add_argument(
        "--results_path",
        type=str,
        help="path to folder where results will be stored (weights, metadata)",
    )
    parser.add_argument(
        "--data_path",
        nargs="+",
        type=str,
        help="Dataset data paths. You can provide several folders.",
    )
    parser.add_argument(
        "--dataset_type",
        type=int,
        default=1,
        help="Dataset type. Must be one of: 1, 2, 3 or 4. Details here: https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/dataset_types.md",
    )
    parser.add_argument(
        "--valid_path",
        nargs="+",
        type=str,
        help="validation data paths. You can provide several folders.",
    )
    parser.add_argument(
        "--num_workers", type=int, default=0, help="dataloader num_workers"
    )
    parser.add_argument(
        "--pin_memory", type=bool, default=False, help="dataloader pin_memory"
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--device_ids", nargs="+", type=int, default=[0], help="list of gpu ids"
    )
    parser.add_argument(
        "--use_multistft_loss",
        action="store_true",
        help="Use MultiSTFT Loss (from auraloss package)",
    )
    parser.add_argument(
        "--use_mse_loss", action="store_true", help="Use default MSE loss"
    )
    parser.add_argument("--use_l1_loss", action="store_true", help="Use L1 loss")
    parser.add_argument("--wandb_key", type=str, default="", help="wandb API Key")
    parser.add_argument(
        "--pre_valid", action="store_true", help="Run validation before training"
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        type=str,
        default=["sdr"],
        choices=[
            "sdr",
            "l1_freq",
            "si_sdr",
            "neg_log_wmse",
            "aura_stft",
            "aura_mrstft",
            "bleedless",
            "fullness",
        ],
        help="List of metrics to use.",
    )
    parser.add_argument(
        "--metric_for_scheduler",
        default="sdr",
        choices=[
            "sdr",
            "l1_freq",
            "si_sdr",
            "neg_log_wmse",
            "aura_stft",
            "aura_mrstft",
            "bleedless",
            "fullness",
        ],
        help="Metric which will be used for scheduler.",
    )
    parser.add_argument(
        "--sweep_config",
        type=str,
        default="",
        help="Path to wandb sweep configuration YAML file",
    )
    parser.add_argument(
        "--sweep_run",
        action="store_true",
        help="Indicates this is a sweep run - will override some args with wandb config values",
    )

    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    # If this is a sweep run, override relevant args with wandb config
    if args.sweep_run:
        sweep_config = wandb.config
        # Override training parameters from sweep
        if hasattr(sweep_config, "learning_rate"):
            args.learning_rate = sweep_config.learning_rate
        if hasattr(sweep_config, "optimizer"):
            args.optimizer = sweep_config.optimizer
        # Add other sweep parameters as needed

    # Ensure metric_for_scheduler is included in metrics list
    if args.metric_for_scheduler not in args.metrics:
        args.metrics += [args.metric_for_scheduler]

    return args


def validate(model, valid_loader, args, config, device, verbose=False):
    instruments = prefer_target_instrument(config)

    # Initialize metrics dict for each instrument and metric
    metrics = {metric: {instr: [] for instr in instruments} for metric in args.metrics}

    all_mixtures_path = valid_loader
    if verbose:
        all_mixtures_path = tqdm(valid_loader)

    pbar_dict = {}
    for path_list in all_mixtures_path:
        path = path_list[0]
        mix, sr = sf.read(path)
        folder = os.path.dirname(path)
        res = demix(config, model, mix.T, device, model_type=args.model_type)

        for instr in instruments:
            if instr != "other" or config.training.other_fix is False:
                track, sr1 = sf.read(folder + "/{}.wav".format(instr))
            else:
                track, sr1 = sf.read(folder + "/{}.wav".format("vocals"))
                track = mix - track

            references = np.expand_dims(track, axis=0)
            estimates = np.expand_dims(res[instr].T, axis=0)

            # Calculate all requested metrics
            for metric_name in args.metrics:
                if metric_name == "sdr":
                    metric_val = sdr(references, estimates)[0]
                # Add other metric calculations here
                # elif metric_name == "si_sdr":
                #     metric_val = calculate_si_sdr(references, estimates)
                # etc...

                single_val = torch.from_numpy(np.array([metric_val])).to(device)
                metrics[metric_name][instr].append(single_val)
                pbar_dict[f"{metric_name}_{instr}"] = metric_val

        if verbose:
            all_mixtures_path.set_postfix(pbar_dict)

    return metrics


def setup_wandb(accelerator, args, config, device_ids, batch_size):
    if (
        accelerator.is_main_process
        and args.wandb_key is not None
        and args.wandb_key.strip() != ""
    ):
        wandb.login(key=args.wandb_key)

        if args.sweep_run:
            # Sweep run - wandb.init() is handled by the sweep controller
            pass
        else:
            # Normal run - initialize as before
            wandb.init(
                project="msst-accelerate",
                config={
                    "config": config,
                    "args": args,
                    "device_ids": device_ids,
                    "batch_size": batch_size,
                },
            )
    else:
        wandb.init(mode="disabled")


def setup_optimizer(model, config, accelerator):
    optim_params = dict()
    if "optimizer" in config:
        optim_params = dict(config["optimizer"])
        accelerator.print("Optimizer params from config:\n{}".format(optim_params))

    optimizers = {
        "adam": Adam,
        "adamw": AdamW,
        "radam": RAdam,
        "rmsprop": RMSprop,
        "sgd": SGD,
    }

    if config.training.optimizer in optimizers:
        optimizer = optimizers[config.training.optimizer](
            model.parameters(), lr=config.training.lr, **optim_params
        )
    elif config.training.optimizer == "prodigy":
        from prodigyopt import Prodigy

        optimizer = Prodigy(model.parameters(), lr=config.training.lr, **optim_params)
    elif config.training.optimizer == "adamw8bit":
        import bitsandbytes as bnb

        optimizer = bnb.optim.AdamW8bit(
            model.parameters(), lr=config.training.lr, **optim_params
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.training.optimizer}")

    return optimizer


def compute_loss(model, x, y, args, config, loss_multistft=None):
    if args.model_type in ["mel_band_roformer", "bs_roformer"]:
        return model(x, y)

    y_ = model(x)
    # Ensure matching shapes
    if y_.shape[3] > y.shape[3]:
        y_ = y_[:, :, :, : y.shape[3]]
    elif y_.shape[3] < y.shape[3]:
        y = y[:, :, :, : y_.shape[3]]

    if args.use_multistft_loss:
        y1_ = torch.reshape(y_, (y_.shape[0], y_.shape[1] * y_.shape[2], y_.shape[3]))
        y1 = torch.reshape(y, (y.shape[0], y.shape[1] * y.shape[2], y.shape[3]))
        loss = loss_multistft(y1_, y1)
        if args.use_mse_loss:
            loss += 1000 * nn.MSELoss()(y1_, y1)
        if args.use_l1_loss:
            loss += 1000 * F.l1_loss(y1_, y1)
    elif args.use_mse_loss:
        loss = nn.MSELoss()(y_, y)
    elif args.use_l1_loss:
        loss = F.l1_loss(y_, y)
    else:
        loss = masked_loss(
            y_, y, q=config.training.q, coarse=config.training.coarse_loss_clip
        )
    return loss


def train_epoch(
    model,
    train_loader,
    optimizer,
    args,
    config,
    accelerator,
    device,
    loss_multistft=None,
):
    model.train().to(device)
    loss_val = 0.0
    total = 0

    pbar = tqdm(train_loader, disable=not accelerator.is_main_process)
    for i, (batch, mixes) in enumerate(pbar):
        loss = compute_loss(model, mixes, batch, args, config, loss_multistft)

        accelerator.backward(loss)
        if config.training.grad_clip:
            accelerator.clip_grad_norm_(model.parameters(), config.training.grad_clip)

        optimizer.step()
        optimizer.zero_grad()

        li = loss.item()
        loss_val += li
        total += 1

        if accelerator.is_main_process:
            wandb.log(
                {
                    "loss": 100 * li,
                    "avg_loss": 100 * loss_val / (i + 1),
                    "total": total,
                    "loss_val": loss_val,
                    "i": i,
                }
            )
            pbar.set_postfix({"loss": 100 * li, "avg_loss": 100 * loss_val / (i + 1)})

    return loss_val / total


def save_checkpoint(model, accelerator, path):
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        accelerator.save(unwrapped_model.state_dict(), path)


def train_model(args):
    # Setup
    accelerator = Accelerator()
    device = accelerator.device
    args = parse_args(args)
    manual_seed(args.seed + int(time.time()))
    torch.backends.cudnn.deterministic = False
    torch.multiprocessing.set_start_method("spawn")

    # Model and config setup
    model, config = get_model_from_config(args.model_type, args.config_path)
    accelerator.print("Instruments: {}".format(config.training.instruments))
    os.makedirs(args.results_path, exist_ok=True)

    # Initialize wandb
    setup_wandb(accelerator, args, config, args.device_ids, config.training.batch_size)

    # Fix for num of steps
    config.training.num_steps *= accelerator.num_processes

    trainset = MSSDataset(
        config,
        args.data_path,
        batch_size=config.training.batch_size,
        metadata_path=os.path.join(
            args.results_path, "metadata_{}.pkl".format(args.dataset_type)
        ),
        dataset_type=args.dataset_type,
        verbose=accelerator.is_main_process,
    )

    train_loader = DataLoader(
        trainset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    validset = MSSValidationDataset(args)
    valid_dataset_length = len(validset)

    valid_loader = DataLoader(
        validset,
        batch_size=1,
        shuffle=False,
    )

    valid_loader = accelerator.prepare(valid_loader)

    if args.start_check_point != "":
        accelerator.print("Start from checkpoint: {}".format(args.start_check_point))
        if 1:
            load_not_compatible_weights(model, args.start_check_point, verbose=False)
        else:
            model.load_state_dict(torch.load(args.start_check_point))

    optimizer = setup_optimizer(model, config, accelerator)

    if accelerator.is_main_process:
        print("Processes GPU: {}".format(accelerator.num_processes))
        print(
            "Patience: {} Reduce factor: {} Batch size: {} Optimizer: {}".format(
                config.training.patience,
                config.training.reduce_factor,
                config.training.batch_size,
                config.training.optimizer,
            )
        )
    # Reduce LR if no SDR improvements for several epochs
    scheduler = ReduceLROnPlateau(
        optimizer,
        "max",
        patience=config.training.patience,
        factor=config.training.reduce_factor,
    )

    if args.use_multistft_loss:
        try:
            loss_options = dict(config.loss_multistft)
        except AttributeError:
            loss_options = dict()

        accelerator.print("Loss options: {}".format(loss_options))
        loss_multistft = auraloss.freq.MultiResolutionSTFTLoss(**loss_options)

    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )

    if args.pre_valid:
        metrics_list = validate(
            model,
            valid_loader,
            args,
            config,
            device,
            verbose=accelerator.is_main_process,
        )
        metrics_list = accelerator.gather(metrics_list)
        accelerator.wait_for_everyone()

        metrics_avg = {}
        instruments = prefer_target_instrument(config)

        for metric_name in args.metrics:
            metric_avg = 0.0
            for instr in instruments:
                metric_data = (
                    torch.cat(metrics_list[metric_name][instr], dim=0).cpu().numpy()
                )
                metric_val = metric_data[:valid_dataset_length].mean()
                if accelerator.is_main_process:
                    print(f"Instr {metric_name} {instr}: {metric_val:.4f}")
                    wandb.log({f"{instr}_{metric_name}": metric_val})
                metric_avg += metric_val
            metric_avg /= len(instruments)
            metrics_avg[metric_name] = metric_avg

            if accelerator.is_main_process:
                print(f"{metric_name} Avg: {metric_avg:.4f}")
                wandb.log({f"{metric_name}_avg": metric_avg})

        # Use metric_for_scheduler for model saving and scheduler
        scheduler_metric = metrics_avg[args.metric_for_scheduler]
        if accelerator.is_main_process:
            # Initialize best_metric if not already set
            if "best_metric" not in locals():
                best_metric = -100

            if scheduler_metric > best_metric:
                store_path = (
                    args.results_path
                    + f"/model_{args.model_type}_ep_0_{args.metric_for_scheduler}_{scheduler_metric:.4f}.ckpt"
                )
                print("Store weights: {}".format(store_path))
                unwrapped_model = accelerator.unwrap_model(model)
                accelerator.save(unwrapped_model.state_dict(), store_path)
                best_metric = scheduler_metric
                wandb.log({"best_metric": best_metric})

            scheduler.step(scheduler_metric)

        metrics_list = None
        accelerator.wait_for_everyone()

    accelerator.print("Train for: {}".format(config.training.num_epochs))
    best_metric = -100

    # main training loop
    for epoch in range(config.training.num_epochs):
        accelerator.print(
            f"Train epoch: {epoch} Learning rate: {optimizer.param_groups[0]['lr']}"
        )

        # Train one epoch
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            args,
            config,
            accelerator,
            device,
            loss_multistft,
        )

        if accelerator.is_main_process:
            print(f"Training loss: {train_loss:.6f}")
            wandb.log({"train_loss": train_loss, "epoch": epoch})

        # Save last checkpoint
        save_checkpoint(
            model, accelerator, f"{args.results_path}/last_{args.model_type}.ckpt"
        )

        # Validation and metric computation
        metrics_list = validate(
            model,
            valid_loader,
            args,
            config,
            device,
            verbose=accelerator.is_main_process,
        )
        metrics_list = accelerator.gather(metrics_list)
        accelerator.wait_for_everyone()

        metrics_avg = {}
        instruments = prefer_target_instrument(config)

        for metric_name in args.metrics:
            metric_avg = 0.0
            for instr in instruments:
                metric_data = (
                    torch.cat(metrics_list[metric_name][instr], dim=0).cpu().numpy()
                )
                metric_val = metric_data[:valid_dataset_length].mean()
                if accelerator.is_main_process:
                    print(f"Instr {metric_name} {instr}: {metric_val:.4f}")
                    wandb.log({f"{instr}_{metric_name}": metric_val})
                metric_avg += metric_val
            metric_avg /= len(instruments)
            metrics_avg[metric_name] = metric_avg

            if accelerator.is_main_process:
                print(f"{metric_name} Avg: {metric_avg:.4f}")
                wandb.log({f"{metric_name}_avg": metric_avg})

        # Use metric_for_scheduler for model saving and scheduler
        scheduler_metric = metrics_avg[args.metric_for_scheduler]
        if accelerator.is_main_process:
            if scheduler_metric > best_metric:
                store_path = (
                    args.results_path
                    + f"/model_{args.model_type}_ep_{epoch}_{args.metric_for_scheduler}_{scheduler_metric:.4f}.ckpt"
                )
                print("Store weights: {}".format(store_path))
                unwrapped_model = accelerator.unwrap_model(model)
                accelerator.save(unwrapped_model.state_dict(), store_path)
                best_metric = scheduler_metric
                wandb.log({"best_metric": best_metric})

            scheduler.step(scheduler_metric)

        metrics_list = None
        accelerator.wait_for_everyone()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "sweep":
        # Run in sweep mode
        with open(sys.argv[2], "r") as f:
            sweep_configuration = yaml.safe_load(f)
        sweep_id = wandb.sweep(sweep_configuration, project="msst-accelerate")
        wandb.agent(sweep_id, function=lambda: train_model(["--sweep_run"]))
    else:
        # Normal training run
        train_model(None)
