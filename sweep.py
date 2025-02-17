import argparse
import os
import time
import warnings
import sys
import yaml
import auraloss
import torch
from accelerate import Accelerator
import torch.nn.functional as F
from torch.optim import SGD, Adam, AdamW, RAdam, RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb
from dataset import MSSDataset
from train import manual_seed
from train_accelerate import (
    MSSValidationDataset,
    valid,
)
from utils import (
    get_model_from_config,
    load_not_compatible_weights,
    prefer_target_instrument,


)
import random

warnings.filterwarnings("ignore")




def save_checkpoint(model, accelerator, path):
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        accelerator.save(unwrapped_model.state_dict(), path)


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

def sample_subset(dataset, subset_fraction):
    """
    Randomly samples a subset of the dataset.
    """
    subset_size = int(len(dataset) * subset_fraction)
    indices = random.sample(range(len(dataset)), subset_size)
    return torch.utils.data.Subset(dataset, indices)

def train_model():
    # Initialize WandB
    wandb.init()
    sweep_config = wandb.config

    # Setup
    accelerator = Accelerator()
    device = accelerator.device
    manual_seed(sweep_config.seed + int(time.time()))
    torch.backends.cudnn.deterministic = False
    # torch.multiprocessing.set_start_method("spawn")

    # Model and config setup
    model, config = get_model_from_config(
        sweep_config.model_type, sweep_config.config_path
    )
    accelerator.print("Instruments: {}".format(config.training.instruments))
    os.makedirs(sweep_config.results_path, exist_ok=True)

    # Override config with sweep values
    config.training.lr = sweep_config.learning_rate
    config.training.optimizer = sweep_config.optimizer
    config.training.batch_size = sweep_config.batch_size
    config.training.num_epochs = sweep_config.num_epochs

    # Fix for num of steps
    config.training.num_steps *= accelerator.num_processes

    trainset = MSSDataset(
        config,
        sweep_config.data_path,
        batch_size=config.training.batch_size,
        metadata_path=os.path.join(
            sweep_config.results_path,
            "metadata_{}.pkl".format(sweep_config.dataset_type),
        ),
        dataset_type=sweep_config.dataset_type,
        verbose=accelerator.is_main_process,
    )
    # Restrict dataset size for quick iterations
    if sweep_config.train_subset < 1.0:
        trainset = sample_subset(trainset, sweep_config.train_subset)

    train_loader = DataLoader(
        trainset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=sweep_config.num_workers,
        pin_memory=sweep_config.pin_memory,
    )

    validset = MSSValidationDataset(sweep_config)


        
    if sweep_config.valid_subset < 1.0:
        validset = sample_subset(validset, sweep_config.valid_subset)
        valid_dataset_length = len(validset)

    valid_loader = DataLoader(
        validset,
        batch_size=1,
        shuffle=False,
    )

    valid_loader = accelerator.prepare(valid_loader)

    if sweep_config.start_check_point:
        accelerator.print(
            "Start from checkpoint: {}".format(sweep_config.start_check_point)
        )
        load_not_compatible_weights(
            model, sweep_config.start_check_point, verbose=False
        )

    # Setup optimizer
    optimizers = {
        "adam": Adam,
        "adamw": AdamW,
        "radam": RAdam,
        "rmsprop": RMSprop,
        "sgd": SGD,
    }

    optimizer = optimizers[config.training.optimizer](
        model.parameters(), lr=config.training.lr
    )

    # Reduce LR if no SDR improvements for several epochs
    scheduler = ReduceLROnPlateau(
        optimizer,
        "max",
        patience=config.training.patience,
        factor=config.training.reduce_factor,
    )

    try:
        loss_options = dict(config.loss_multistft)
    except AttributeError:
        loss_options = dict()

    loss_multistft = auraloss.freq.MultiResolutionSTFTLoss(**loss_options)

    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )

    # Training loop
    best_metric = -100
    for epoch in range(config.training.num_epochs):
        accelerator.print(
            f"Train epoch: {epoch} Learning rate: {optimizer.param_groups[0]['lr']}"
        )

        # Train one epoch
        model.train()
        loss_val = 0.0
        total = 0

        pbar = tqdm(train_loader, disable=not accelerator.is_main_process)
        for i, (batch, mixes) in enumerate(pbar):
            loss = compute_loss(
                model, mixes, batch, sweep_config, config, loss_multistft
            )

            accelerator.backward(loss)
            if config.training.grad_clip:
                accelerator.clip_grad_norm_(
                    model.parameters(), config.training.grad_clip
                )

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
                    }
                )
                pbar.set_postfix(
                    {"loss": 100 * li, "avg_loss": 100 * loss_val / (i + 1)}
                )

        train_loss = loss_val / total
        if accelerator.is_main_process:
            wandb.log({"train_loss": train_loss, "epoch": epoch})

        # Validation
        metrics_list = valid(
            model,
            valid_loader,
            sweep_config,
            config,
            device,
            verbose=accelerator.is_main_process,
        )
        metrics_list = accelerator.gather(metrics_list)
        accelerator.wait_for_everyone()

        metrics_avg = {}
        instruments = prefer_target_instrument(config)

        for metric_name in sweep_config.metrics:
            metric_avg = 0.0
            for instr in instruments:
                metric_data = (
                    torch.cat(metrics_list[metric_name][instr], dim=0).cpu().numpy()
                )
                metric_val = metric_data[:valid_dataset_length].mean()
                if accelerator.is_main_process:
                    wandb.log({f"{instr}_{metric_name}": metric_val})
                metric_avg += metric_val
            metric_avg /= len(instruments)
            metrics_avg[metric_name] = metric_avg

            if accelerator.is_main_process:
                wandb.log({f"{metric_name}_avg": metric_avg})

        # Use metric_for_scheduler for model saving and scheduler
        scheduler_metric = metrics_avg[sweep_config.metric_for_scheduler]
        if accelerator.is_main_process:
            if scheduler_metric > best_metric:
                store_path = (
                    sweep_config.results_path
                    + f"/model_{sweep_config.model_type}_ep_{epoch}_{sweep_config.metric_for_scheduler}_{scheduler_metric:.4f}.ckpt"
                )
                unwrapped_model = accelerator.unwrap_model(model)
                accelerator.save(unwrapped_model.state_dict(), store_path)
                best_metric = scheduler_metric
                wandb.log({"best_metric": best_metric})

            scheduler.step(scheduler_metric)

        metrics_list = None
        accelerator.wait_for_everyone()


if __name__ == "__main__":
    with open(sys.argv[1], "r") as f:
        sweep_configuration = yaml.safe_load(f)
    sweep_id = wandb.sweep(sweep_configuration, project="msst-sweep")
    wandb.agent(sweep_id, function=train_model)

