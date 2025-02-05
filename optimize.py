# coding: utf-8
__author__ = "Your Name"
__version__ = "0.1"

import random
import argparse
from tqdm.auto import tqdm
import os
import sys
import torch
import wandb
import numpy as np
import auraloss
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD, RAdam, RMSprop
from torch.utils.data import DataLoader
from torch.cuda.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ml_collections import ConfigDict
import torch.nn.functional as F
from typing import List, Tuple, Dict, Union, Callable

import optuna
import yaml

from dataset import MSSDataset
from utils import get_model_from_config
from valid import valid_multi_gpu, valid
from utils import bind_lora_to_model, load_start_checkpoint
import loralib as lora

import warnings

warnings.filterwarnings("ignore")


def parse_args(dict_args: Union[Dict, None]) -> argparse.Namespace:
    """
    Parse command-line arguments for configuring the model, dataset, and training parameters.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        type=str,
        default="mdx23c",
        help="Options: mdx23c, htdemucs, segm_models, mel_band_roformer, bs_roformer, swin_upernet, bandit",
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
        help="Folder where results will be stored (weights, metadata)",
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
        help="Dataset type. Must be one of: 1, 2, 3 or 4.",
    )
    parser.add_argument(
        "--valid_path",
        nargs="+",
        type=str,
        help="Validation data paths. You can provide several folders.",
    )
    parser.add_argument(
        "--num_workers", type=int, default=0, help="dataloader num_workers"
    )
    parser.add_argument(
        "--pin_memory", action="store_true", help="dataloader pin_memory"
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
    parser.add_argument("--use_mse_loss", action="store_true", help="Use MSE loss")
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
        help="Metric used for scheduler.",
    )
    parser.add_argument("--train_lora", action="store_true", help="Train with LoRA")
    parser.add_argument(
        "--lora_checkpoint",
        type=str,
        default="",
        help="Initial checkpoint for LoRA weights",
    )

    if dict_args is not None:
        args = parser.parse_args([])
        args_dict = vars(args)
        args_dict.update(dict_args)
        args = argparse.Namespace(**args_dict)
    else:
        args = parser.parse_args()

    if args.metric_for_scheduler not in args.metrics:
        args.metrics += [args.metric_for_scheduler]

    return args


def manual_seed(seed: int) -> None:
    """
    Set the random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def initialize_environment(seed: int, results_path: str) -> None:
    """
    Initialize the environment: set the seed, and create results directory.
    """
    manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    try:
        torch.multiprocessing.set_start_method("spawn")
    except Exception:
        pass
    os.makedirs(results_path, exist_ok=True)


def wandb_init(
    args: argparse.Namespace, config: Dict, device_ids: List[int], batch_size: int
) -> None:
    """
    Initialize the Weights & Biases (wandb) logging.
    """
    if args.wandb_key is None or args.wandb_key.strip() == "":
        wandb.init(mode="disabled")
    else:
        wandb.login(key=args.wandb_key)
        wandb.init(
            project="msst_optimization",
            config={
                "config": config,
                "args": args,
                "device_ids": device_ids,
                "batch_size": batch_size,
            },
        )


def prepare_data(config: Dict, args: argparse.Namespace, batch_size: int) -> DataLoader:
    """
    Prepare the training dataset and dataloader.
    """
    trainset = MSSDataset(
        config,
        args.data_path,
        batch_size=batch_size,
        metadata_path=os.path.join(
            args.results_path, f"metadata_{args.dataset_type}.pkl"
        ),
        dataset_type=args.dataset_type,
    )

    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    return train_loader


def initialize_model_and_device(
    model: torch.nn.Module, device_ids: List[int]
) -> Tuple[Union[torch.device, str], torch.nn.Module]:
    """
    Place the model on GPU(s) if available.
    """
    if torch.cuda.is_available():
        if len(device_ids) <= 1:
            device = torch.device(f"cuda:{device_ids[0]}")
            model = model.to(device)
        else:
            device = torch.device(f"cuda:{device_ids[0]}")
            model = nn.DataParallel(model, device_ids=device_ids).to(device)
    else:
        device = "cpu"
        model = model.to(device)
        print("CUDA is not available. Running on CPU.")

    return device, model


def get_optimizer(config: ConfigDict, model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    Initialize the optimizer as defined in config.
    """
    optim_params = dict()
    if "optimizer" in config:
        optim_params = dict(config["optimizer"])
        print(f"Optimizer params from config:\n{optim_params}")

    name_optimizer = getattr(config.training, "optimizer", "No optimizer in config")

    if name_optimizer == "adam":
        optimizer = Adam(model.parameters(), lr=config.training.lr, **optim_params)
    elif name_optimizer == "adamw":
        optimizer = AdamW(model.parameters(), lr=config.training.lr, **optim_params)
    elif name_optimizer == "radam":
        optimizer = RAdam(model.parameters(), lr=config.training.lr, **optim_params)
    elif name_optimizer == "rmsprop":
        optimizer = RMSprop(model.parameters(), lr=config.training.lr, **optim_params)
    elif name_optimizer == "sgd":
        print("Use SGD optimizer")
        optimizer = SGD(model.parameters(), lr=config.training.lr, **optim_params)
    else:
        print(f"Unknown optimizer: {name_optimizer}")
        exit()
    return optimizer


def masked_loss(
    y_: torch.Tensor, y: torch.Tensor, q: float, coarse: bool = True
) -> torch.Tensor:
    """
    Compute masked loss using a quantile mask.
    """
    loss = torch.nn.MSELoss(reduction="none")(y_, y).transpose(0, 1)
    if coarse:
        loss = torch.mean(loss, dim=(-1, -2))
    loss = loss.reshape(loss.shape[0], -1)
    L = loss.detach()
    quantile = torch.quantile(L, q, interpolation="linear", dim=1, keepdim=True)
    mask = L < quantile
    return (loss * mask).mean()


def multistft_loss(
    y: torch.Tensor,
    y_: torch.Tensor,
    loss_multistft: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """
    Compute multi-STFT loss between prediction and ground truth.
    """
    if len(y_.shape) == 4:
        y1_ = torch.reshape(y_, (y_.shape[0], y_.shape[1] * y_.shape[2], y_.shape[3]))
        y1 = torch.reshape(y, (y.shape[0], y.shape[1] * y.shape[2], y.shape[3]))
    elif len(y_.shape) == 3:
        y1_ = y_
        y1 = y
    else:
        raise ValueError(
            f"Invalid shape for predicted tensor: {y_.shape}. Expected 3 or 4 dimensions."
        )

    return loss_multistft(y1_, y1)


def choice_loss(
    args: argparse.Namespace, config: ConfigDict
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Select the loss function based on configuration and args.
    """
    if args.use_multistft_loss:
        loss_options = dict(getattr(config, "loss_multistft", {}))
        print(f"Loss options: {loss_options}")
        loss_multistft = auraloss.freq.MultiResolutionSTFTLoss(**loss_options)

        if args.use_mse_loss and args.use_l1_loss:

            def multi_loss(y_, y):
                return (
                    (multistft_loss(y_, y, loss_multistft) / 1000)
                    + nn.MSELoss()(y_, y)
                    + F.l1_loss(y_, y)
                )

        elif args.use_mse_loss:

            def multi_loss(y_, y):
                return (multistft_loss(y_, y, loss_multistft) / 1000) + nn.MSELoss()(
                    y_, y
                )

        elif args.use_l1_loss:

            def multi_loss(y_, y):
                return (multistft_loss(y_, y, loss_multistft) / 1000) + F.l1_loss(y_, y)

        else:

            def multi_loss(y_, y):
                return multistft_loss(y_, y, loss_multistft) / 1000

    elif args.use_mse_loss:
        if args.use_l1_loss:

            def multi_loss(y_, y):
                return nn.MSELoss()(y_, y) + F.l1_loss(y_, y)

        else:
            multi_loss = nn.MSELoss()
    elif args.use_l1_loss:
        multi_loss = F.l1_loss
    else:

        def multi_loss(y_, y):
            return masked_loss(
                y_, y, q=config.training.q, coarse=config.training.coarse_loss_clip
            )

    return multi_loss


def train_one_epoch(
    model: torch.nn.Module,
    config: ConfigDict,
    args: argparse.Namespace,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    device_ids: List[int],
    epoch: int,
    use_amp: bool,
    scaler: torch.cuda.amp.GradScaler,
    gradient_accumulation_steps: int,
    train_loader: torch.utils.data.DataLoader,
    multi_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> None:
    """
    Train the model for one epoch.
    """
    model.train().to(device)
    print(f"Train epoch: {epoch} Learning rate: {optimizer.param_groups[0]['lr']}")
    loss_val = 0.0
    total = 0

    normalize = getattr(config.training, "normalize", False)
    pbar = tqdm(train_loader)
    for i, (batch, mixes) in enumerate(pbar):
        x = mixes.to(device)
        y = batch.to(device)

        with torch.cuda.amp.autocast(enabled=use_amp):
            if args.model_type in ["mel_band_roformer", "bs_roformer"]:
                loss = model(x, y)
                if isinstance(device_ids, (list, tuple)):
                    loss = loss.mean()
            else:
                y_ = model(x)
                if y_.shape[-1] != y.shape[-1]:
                    target_length = max(y_.shape[-1], y.shape[-1])
                    if y_.shape[-1] < target_length:
                        y_ = F.pad(y_, (0, target_length - y_.shape[-1]))
                    if y.shape[-1] < target_length:
                        y = F.pad(y, (0, target_length - y.shape[-1]))
                y = y[: y_.shape[0]]
                loss = multi_loss(y_, y)

        loss /= gradient_accumulation_steps
        scaler.scale(loss).backward()
        if config.training.grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)

        if ((i + 1) % gradient_accumulation_steps == 0) or (i == len(train_loader) - 1):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        li = loss.item() * gradient_accumulation_steps
        loss_val += li
        total += 1
        pbar.set_postfix({"loss": 100 * li, "avg_loss": 100 * loss_val / (i + 1)})
        wandb.log({"loss": 100 * li, "avg_loss": 100 * loss_val / (i + 1), "i": i})
        loss.detach()

    print(f"Training loss: {loss_val / total}")
    wandb.log(
        {
            "train_loss": loss_val / total,
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
    )


def compute_epoch_metrics(
    model: torch.nn.Module,
    args: argparse.Namespace,
    config: ConfigDict,
    device: torch.device,
    device_ids: List[int],
    best_metric: float,
    epoch: int,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
) -> float:
    """
    Compute and log metrics for the current epoch.
    """
    if torch.cuda.is_available() and len(device_ids) > 1:
        metrics_avg = valid_multi_gpu(
            model, args, config, args.device_ids, verbose=False
        )
    else:
        metrics_avg = valid(model, args, config, device, verbose=False)
    metric_avg = metrics_avg[args.metric_for_scheduler]
    if metric_avg > best_metric:
        store_path = f"{args.results_path}/model_{args.model_type}_ep_{epoch}_{args.metric_for_scheduler}_{metric_avg:.4f}.ckpt"
        print(f"Store weights: {store_path}")
        train_lora = args.train_lora
        if train_lora:
            torch.save(lora.lora_state_dict(model), store_path)
        else:
            state_dict = (
                model.state_dict()
                if len(device_ids) <= 1
                else model.module.state_dict()
            )
            torch.save(state_dict, store_path)
        best_metric = metric_avg
    scheduler.step(metric_avg)
    wandb.log({"metric_main": metric_avg, "best_metric": best_metric})
    for metric_name in metrics_avg:
        wandb.log({f"metric_{metric_name}": metrics_avg[metric_name]})
    return best_metric


def objective(trial: optuna.trial.Trial) -> float:
    """
    Objective function for Optuna.
    Sets hyperparameters, runs training, and returns the optimized metric.
    """
    # Parse normal training args (remaining command-line args)
    args = parse_args(None)

    # Suggest hyperparameters
    lr = trial.suggest_loguniform("lr", 1e-6, 1e-2)
    optimizer_choice = trial.suggest_categorical("optimizer", ["adam", "adamw", "sgd"])
    grad_accum_steps = trial.suggest_int("gradient_accumulation_steps", 1, 4)

    # Get model and config from provided config_path and model type.
    model, config = get_model_from_config(args.model_type, args.config_path)

    # Update training configuration with trial hyperparameters.
    config.training.lr = lr
    config.training.optimizer = optimizer_choice
    config.training.gradient_accumulation_steps = grad_accum_steps

    # Optionally, for tuning you may want to reduce the number of epochs.
    num_epochs = getattr(config.training, "num_epochs", 10)
    num_epochs = min(num_epochs, 1)

    # Initialize environment, seed, and results directory.
    initialize_environment(args.seed, args.results_path)

    wandb_init(
        args, config, args.device_ids, config.training.batch_size * len(args.device_ids)
    )
    train_loader = prepare_data(
        config, args, config.training.batch_size * len(args.device_ids)
    )

    if args.start_check_point:
        load_start_checkpoint(args, model, type_="train")

    if args.train_lora:
        model = bind_lora_to_model(config, model)
        lora.mark_only_lora_as_trainable(model)

    device, model = initialize_model_and_device(model, args.device_ids)

    optimizer = get_optimizer(config, model)
    scheduler = ReduceLROnPlateau(
        optimizer,
        "max",
        patience=config.training.patience,
        factor=config.training.reduce_factor,
    )
    multi_loss = choice_loss(args, config)
    scaler = GradScaler()

    best_metric = float("-inf")
    for epoch in range(num_epochs):
        train_one_epoch(
            model,
            config,
            args,
            optimizer,
            device,
            args.device_ids,
            epoch,
            getattr(config.training, "use_amp", True),
            scaler,
            config.training.gradient_accumulation_steps,
            train_loader,
            multi_loss,
        )
        best_metric = compute_epoch_metrics(
            model, args, config, device, args.device_ids, best_metric, epoch, scheduler
        )
        trial.report(best_metric, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_metric


def output_merged_config(config_path: str, best_params: dict, output_path: str) -> None:
    """
    Load the original YAML config file at config_path, merge in the best trial parameters
    (for example, in the 'training' section), and write out the merged configuration file.
    """
    # Load the original configuration from YAML.
    with open(config_path, "r") as f:
        orig_config = yaml.safe_load(f)

    # Ensure a 'training' section exists.
    if "training" not in orig_config:
        orig_config["training"] = {}

    # Merge best trial parameters into the training section.
    orig_config["training"]["lr"] = best_params.get(
        "lr", orig_config["training"].get("lr")
    )
    orig_config["training"]["optimizer"] = best_params.get(
        "optimizer", orig_config["training"].get("optimizer")
    )
    orig_config["training"]["gradient_accumulation_steps"] = best_params.get(
        "gradient_accumulation_steps",
        orig_config["training"].get("gradient_accumulation_steps", 1),
    )

    # Write out the merged config to the specified output path.
    with open(output_path, "w") as f:
        yaml.dump(orig_config, f)
    print(f"Merged config file saved to: {output_path}")


if __name__ == "__main__":
    # Parse Optuna-specific arguments and remove them from sys.argv before training arg parsing.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_trials",
        type=int,
        default=10,
        help="Number of trials for Optuna optimization.",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL for persistence (optional).",
    )
    opt_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    study = optuna.create_study(
        direction="maximize",
        study_name="hyper_optimization",
        storage=opt_args.storage,
        load_if_exists=bool(opt_args.storage),
    )
    study.optimize(objective, n_trials=opt_args.n_trials)

    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Retrieve the original train args to get the config file path.
    args = parse_args(None)
    # Merge best trial parameters with the original YAML configuration.
    output_merged_config(args.config_path, trial.params, "merged_config.yaml")
