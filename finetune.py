from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR, StepLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data import build_loaders, get_dataset_meta
from utils import build_torchvision_model, count_parameters, infer_backbone_family


# -----------------------------
# utils
# -----------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def maybe_amp(enabled: bool, device: torch.device):
    return torch.cuda.amp.autocast(enabled=enabled and device.type == "cuda")


def setup_logger(run_dir: Path) -> logging.Logger:
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(run_dir.name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(run_dir / "train.log")
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False
    return logger


def save_json(path: Path, payload: Dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str))


def get_optimizer(name: str, params: Iterable[torch.nn.Parameter], lr: float, weight_decay: float, momentum: float):
    name = name.lower()
    if name == "adamw":
        return AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return SGD(params, lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=True)
    raise ValueError(f"Unsupported optimizer: {name}")


def get_scheduler(name: str, optimizer, epochs: int, step_size: int, gamma: float, milestones: str):
    name = name.lower()
    if name == "none":
        return None
    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs)
    if name == "step":
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    if name == "multistep":
        ms = [int(x) for x in milestones.split(",") if x.strip()]
        return MultiStepLR(optimizer, milestones=ms, gamma=gamma)
    raise ValueError(f"Unsupported scheduler: {name}")


def build_model(args, num_classes: int) -> nn.Module:
    weights = None if args.weights == "none" else ("DEFAULT" if args.weights == "default" else args.weights)
    model = build_torchvision_model(
        args.model_name,
        num_classes=num_classes,
        weights=weights,
        pretrained=(args.weights != "none"),
        strict_head=False,
    )
    if args.freeze_backbone:
        for name, p in model.named_parameters():
            if not any(k in name.lower() for k in ["fc", "classifier", "heads"]):
                p.requires_grad = False
    return model


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    device: torch.device,
    criterion,
    scaler,
    amp: bool,
    logger: logging.Logger,
    epoch: int,
    grad_clip: float,
    phase: str = "train",
):
    is_train = phase == "train"
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(loader, desc=f"{phase} e{epoch}", leave=False)
    for step, (images, targets) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with maybe_amp(amp, device):
            logits = model(images)
            loss = criterion(logits, targets)

        if is_train:
            if amp and device.type == "cuda":
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        bs = images.size(0)
        total_loss += loss.item() * bs
        total_correct += (logits.argmax(1) == targets).sum().item()
        total_samples += bs
        pbar.set_postfix(loss=total_loss / total_samples, acc=total_correct / total_samples)

        logger.info(
            json.dumps(
                {
                    "epoch": epoch,
                    "phase": phase,
                    "step": step,
                    "loss": float(loss.item()),
                    "acc": float((logits.argmax(1) == targets).float().mean().item()),
                }
            )
        )

    return {"loss": total_loss / total_samples, "acc": total_correct / total_samples}


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(loader, desc="val", leave=False)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        bs = images.size(0)
        total_loss += loss.item() * bs
        total_correct += (logits.argmax(1) == targets).sum().item()
        total_samples += bs
        pbar.set_postfix(loss=total_loss / total_samples, acc=total_correct / total_samples)

    return {"loss": total_loss / total_samples, "acc": total_correct / total_samples}


def print_model_info(model: nn.Module, args, dataset_meta: Dict[str, object], logger: logging.Logger) -> None:
    family = infer_backbone_family(model)
    logger.info("Dataset info: %s", json.dumps(dataset_meta))
    logger.info("Model family: %s", family)
    logger.info("Model name: %s", args.model_name)
    logger.info("Trainable params: %.3fM", count_parameters(model, trainable_only=True) / 1e6)
    logger.info("Total params: %.3fM", count_parameters(model) / 1e6)
    logger.info("Optimizer: %s | Scheduler: %s", args.optimizer, args.scheduler)
    logger.info("Image size: %s | Color jitter: %s", args.image_size, args.color_jitter)
    logger.info("Freeze backbone: %s | AMP: %s", args.freeze_backbone, args.amp)


def parse_args():
    p = argparse.ArgumentParser("Fine-tune a torchvision teacher or student backbone.")
    p.add_argument("--dataset", type=str, required=True, choices=["cifar100", "food101", "imagenet100", "cifar-100", "food-101", "imagenet-100", "imagenet_subset", "imagenet-subset"])
    p.add_argument("--dataset-root", type=str, required=True)
    p.add_argument("--model-name", type=str, required=True, help="e.g. resnet18, vgg16, vit_b_16")
    p.add_argument("--weights", type=str, default="default", help="default|none|torchvision weight enum name")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--color-jitter", type=str, default="none", choices=["paper", "none"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd"])
    p.add_argument("--scheduler", type=str, default="cosine", choices=["none", "cosine", "step", "multistep"])
    p.add_argument("--step-size", type=int, default=10)
    p.add_argument("--gamma", type=float, default=0.1)
    p.add_argument("--milestones", type=str, default="20,40")
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--freeze-backbone", action="store_true")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--grad-clip", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=str, default="runs/finetune")
    p.add_argument("--run-name", type=str, default="exp")
    p.add_argument("--resume", type=str, default="")
    return p.parse_args()


def main(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    meta = get_dataset_meta(args.dataset)
    train_loader, val_loader = build_loaders(
        args.dataset,
        root=args.dataset_root,
        batch_size=args.batch_size,
        image_size=args.image_size,
        color_jitter=(args.color_jitter == "paper"),
        num_workers=args.num_workers,
    )

    model = build_model(args, num_classes=int(meta["num_classes"]))
    model.to(device)

    run_dir = Path(args.out_dir) / args.run_name
    logger = setup_logger(run_dir)
    save_json(run_dir / "args.json", vars(args))

    logger.info("========== Dataset ==========")
    logger.info(json.dumps(meta, indent=2))
    logger.info("Train size: %d | Val size: %d", len(train_loader.dataset), len(val_loader.dataset) if val_loader is not None else -1)
    logger.info("========== Model ==========")
    print_model_info(model, args, meta, logger)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = get_optimizer(args.optimizer, [p for p in model.parameters() if p.requires_grad], args.lr, args.weight_decay, args.momentum)
    scheduler = get_scheduler(args.scheduler, optimizer, args.epochs, args.step_size, args.gamma, args.milestones)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp and device.type == "cuda")

    start_epoch = 1
    best_val_acc = -math.inf
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if ckpt.get("scaler_state_dict") is not None:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_val_acc = float(ckpt.get("best_val_acc", -math.inf))
        logger.info("Resumed from %s at epoch %d", args.resume, start_epoch)

    logger.info("========== Training ==========")
    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, criterion, scaler, args.amp, logger, epoch, args.grad_clip, phase="train")
        val_metrics = evaluate(model, val_loader, device, criterion) if val_loader is not None else {"loss": float("nan"), "acc": float("nan")}

        if scheduler is not None:
            scheduler.step()

        logger.info(
            "Epoch %d | train_loss=%.6f train_acc=%.4f | val_loss=%.6f val_acc=%.4f",
            epoch,
            train_metrics["loss"],
            train_metrics["acc"],
            val_metrics["loss"],
            val_metrics["acc"],
        )

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "best_val_acc": best_val_acc,
            "args": vars(args),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }
        torch.save(ckpt, run_dir / "last.pt")

        if not math.isnan(val_metrics["acc"]) and val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            ckpt["best_val_acc"] = best_val_acc
            torch.save(ckpt, run_dir / "best.pt")
            logger.info("Saved new best checkpoint with val_acc=%.4f", best_val_acc)

    logger.info("Training finished. Best val acc: %.4f", best_val_acc)


if __name__ == "__main__":
    args = parse_args()
    main(args)
