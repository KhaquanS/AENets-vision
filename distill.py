from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR, StepLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data import build_loaders, get_dataset_meta
from sae_injection import SAEInjection
from logit_kd import LogitKD
from dkd import DKD
from fitnets import FitNetsDistiller
from utils import build_torchvision_model, count_parameters, infer_backbone_family


SCHEMES = ("sae_injection", "logit_kd", "dkd", "fitnets")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def setup_logger(run_dir: Path, filename: str = "distill.log") -> logging.Logger:
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(run_dir.as_posix())
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(run_dir / filename)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


def csv_log_epoch(run_dir: Path, row: Dict) -> None:
    csv_path = run_dir / "metrics.csv"
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_json(path: Path, payload: Dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str))


def parse_csv(value: Optional[str]) -> List[str]:
    if value is None:
        return []
    items = [v.strip() for v in value.split(",")]
    return [v for v in items if v]


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == targets).float().mean().item()


def maybe_amp(enabled: bool, device: torch.device):
    return torch.cuda.amp.autocast(enabled=enabled and device.type == "cuda")


def get_optimizer(params: Iterable[torch.nn.Parameter], lr: float, weight_decay: float):
    return AdamW(params, lr=lr, weight_decay=weight_decay)


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


def build_models(args, num_classes: int):
    teacher_weights = None if args.teacher_weights == "none" else ("DEFAULT" if args.teacher_weights == "default" else args.teacher_weights)
    student_weights = None if args.student_weights == "none" else ("DEFAULT" if args.student_weights == "default" else args.student_weights)

    teacher = build_torchvision_model(
        args.teacher_name,
        num_classes=num_classes,
        weights=teacher_weights,
        pretrained=(args.teacher_weights != "none"),
        strict_head=False,
    )
    student = build_torchvision_model(
        args.student_name,
        num_classes=num_classes,
        weights=student_weights,
        pretrained=(args.student_weights != "none"),
        strict_head=False,
    )
    return teacher, student


def build_distiller(args, teacher: nn.Module, student: nn.Module):
    if args.scheme == "sae_injection":
        return SAEInjection(
            teacher=teacher,
            student=student,
            teacher_cut=args.teacher_cut or None,
            student_start=args.student_start or None,
            adapter_bottleneck_ratio=args.adapter_bottleneck_ratio,
            adapter_sparsity=args.adapter_sparsity,
            freeze_teacher=args.freeze_teacher,
            freeze_student=args.freeze_student,
        )

    if args.scheme == "logit_kd":
        return LogitKD(teacher, student, freeze_teacher=args.freeze_teacher)

    if args.scheme == "dkd":
        return DKD(
            teacher,
            student,
            alpha=args.alpha,
            beta=args.beta,
            temperature=args.temperature,
            freeze_teacher=args.freeze_teacher,
        )

    if args.scheme == "fitnets":
        teacher_hints = parse_csv(args.teacher_hints) or [args.teacher_cut or ""]
        student_hints = parse_csv(args.student_hints) or [args.student_start or ""]
        teacher_hints = [x for x in teacher_hints if x]
        student_hints = [x for x in student_hints if x]
        if not teacher_hints or not student_hints:
            raise ValueError("FitNets requires teacher_hints and student_hints, or teacher_cut/student_start defaults.")
        return FitNetsDistiller(
            teacher,
            student,
            teacher_hints=teacher_hints,
            student_hints=student_hints,
            reverse=args.reverse,
            bottleneck_ratio=args.fitnet_bottleneck_ratio,
            freeze_teacher=args.freeze_teacher,
        )

    raise ValueError(f"Unsupported scheme: {args.scheme}")


def _forward_batch(model, scheme: str, images: torch.Tensor, targets: torch.Tensor, args):
    if scheme == "sae_injection":
        teacher_features = model.teacher_prefix.module(images)
        reconstruction, latent, sae_loss, recon_loss, sparsity_loss = model.adapter(teacher_features)
        logits = model.student_suffix.module(latent)
        ce_loss = F.cross_entropy(logits, targets, label_smoothing=args.label_smoothing)
        total = ce_loss + args.sae_weight * sae_loss
        return {
            "logits": logits,
            "teacher_logits": None,
            "ce_loss": ce_loss,
            "distill_loss": sae_loss,
            "reconstruction_loss": recon_loss,
            "sparsity_loss": sparsity_loss,
            "total_loss": total,
        }

    if scheme == "logit_kd":
        teacher_logits, student_logits = model(images)
        ce_loss = F.cross_entropy(student_logits, targets, label_smoothing=args.label_smoothing)
        kd_loss = F.kl_div(
            F.log_softmax(student_logits / args.temperature, dim=1),
            F.softmax(teacher_logits / args.temperature, dim=1),
            reduction="batchmean",
        ) * (args.temperature ** 2)
        total = ce_loss + args.distill_weight * kd_loss
        return {
            "logits": student_logits,
            "teacher_logits": teacher_logits,
            "ce_loss": ce_loss,
            "distill_loss": kd_loss,
            "total_loss": total,
        }

    if scheme == "dkd":
        teacher_logits, student_logits, kd_loss = model(images, targets)
        ce_loss = F.cross_entropy(student_logits, targets, label_smoothing=args.label_smoothing)
        total = ce_loss + args.distill_weight * kd_loss
        return {
            "logits": student_logits,
            "teacher_logits": teacher_logits,
            "ce_loss": ce_loss,
            "distill_loss": kd_loss,
            "total_loss": total,
        }

    if scheme == "fitnets":
        teacher_logits, student_logits, hint_loss = model(images)
        ce_loss = F.cross_entropy(student_logits, targets, label_smoothing=args.label_smoothing)
        total = ce_loss + args.distill_weight * hint_loss
        return {
            "logits": student_logits,
            "teacher_logits": teacher_logits,
            "ce_loss": ce_loss,
            "distill_loss": hint_loss,
            "total_loss": total,
        }

    raise ValueError(f"Unsupported scheme: {scheme}")


def train_one_epoch(
    model,
    scheme: str,
    loader: DataLoader,
    optimizer,
    device: torch.device,
    scaler,
    amp: bool,
    epoch: int,
    args,
):
    model.train()
    if scheme in {"sae_injection", "logit_kd", "dkd", "fitnets"} and hasattr(model, "teacher"):
        model.teacher.eval()

    totals = {"loss": 0.0, "ce": 0.0, "distill": 0.0, "acc": 0.0, "n": 0}
    extra_totals = {"recon": 0.0, "sparse": 0.0}

    pbar = tqdm(loader, desc=f"train e{epoch}", leave=False)
    for step, (images, targets) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with maybe_amp(amp, device):
            out = _forward_batch(model, scheme, images, targets, args)
            loss = out["total_loss"]

        if amp and device.type == "cuda":
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        bs = images.size(0)
        acc = accuracy(out["logits"], targets)
        totals["loss"] += loss.item() * bs
        totals["ce"] += out["ce_loss"].item() * bs
        totals["distill"] += out["distill_loss"].item() * bs
        totals["acc"] += acc * bs
        totals["n"] += bs

        if scheme == "sae_injection":
            extra_totals["recon"] += out["reconstruction_loss"].item() * bs
            extra_totals["sparse"] += out["sparsity_loss"].item() * bs

        postfix = {
            "loss": totals["loss"] / totals["n"],
            "acc": totals["acc"] / totals["n"],
            "ce": totals["ce"] / totals["n"],
            "kd": totals["distill"] / totals["n"],
        }
        if scheme == "sae_injection":
            postfix["recon"] = extra_totals["recon"] / totals["n"]
            postfix["sparse"] = extra_totals["sparse"] / totals["n"]
        pbar.set_postfix(postfix)

    metrics = {
        "loss": totals["loss"] / max(totals["n"], 1),
        "ce_loss": totals["ce"] / max(totals["n"], 1),
        "distill_loss": totals["distill"] / max(totals["n"], 1),
        "acc": totals["acc"] / max(totals["n"], 1),
    }
    if scheme == "sae_injection":
        metrics["reconstruction_loss"] = extra_totals["recon"] / max(totals["n"], 1)
        metrics["sparsity_loss"] = extra_totals["sparse"] / max(totals["n"], 1)
    return metrics


@torch.no_grad()
def evaluate(model, scheme: str, loader: DataLoader, device: torch.device, args):
    model.eval()
    totals = {"loss": 0.0, "ce": 0.0, "distill": 0.0, "acc": 0.0, "n": 0}
    extra_totals = {"recon": 0.0, "sparse": 0.0}

    pbar = tqdm(loader, desc="val", leave=False)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        out = _forward_batch(model, scheme, images, targets, args)
        bs = images.size(0)
        acc = accuracy(out["logits"], targets)

        totals["loss"] += out["total_loss"].item() * bs
        totals["ce"] += out["ce_loss"].item() * bs
        totals["distill"] += out["distill_loss"].item() * bs
        totals["acc"] += acc * bs
        totals["n"] += bs

        if scheme == "sae_injection":
            extra_totals["recon"] += out["reconstruction_loss"].item() * bs
            extra_totals["sparse"] += out["sparsity_loss"].item() * bs

        postfix = {
            "loss": totals["loss"] / totals["n"],
            "acc": totals["acc"] / totals["n"],
        }
        pbar.set_postfix(postfix)

    metrics = {
        "loss": totals["loss"] / max(totals["n"], 1),
        "ce_loss": totals["ce"] / max(totals["n"], 1),
        "distill_loss": totals["distill"] / max(totals["n"], 1),
        "acc": totals["acc"] / max(totals["n"], 1),
    }
    if scheme == "sae_injection":
        metrics["reconstruction_loss"] = extra_totals["recon"] / max(totals["n"], 1)
        metrics["sparsity_loss"] = extra_totals["sparse"] / max(totals["n"], 1)
    return metrics


def describe_run(model, args, dataset_meta: Dict[str, object], logger: logging.Logger) -> None:
    logger.info("========== Dataset ==========")
    logger.info(json.dumps(dataset_meta, indent=2))
    logger.info("dataset=%s | root=%s | image_size=%d | color_jitter=%s", args.dataset, args.dataset_root, args.image_size, args.color_jitter)
    logger.info("batch_size=%d | num_workers=%d", args.batch_size, args.num_workers)

    logger.info("========== Teacher ==========")
    logger.info("name=%s | family=%s | weights=%s", args.teacher_name, infer_backbone_family(model.teacher), args.teacher_weights)
    logger.info("params=%.3fM | trainable=%.3fM", count_parameters(model.teacher) / 1e6, count_parameters(model.teacher, trainable_only=True) / 1e6)

    logger.info("========== Student ==========")
    logger.info("name=%s | family=%s | weights=%s", args.student_name, infer_backbone_family(model.student), args.student_weights)
    logger.info("params=%.3fM | trainable=%.3fM", count_parameters(model.student) / 1e6, count_parameters(model.student, trainable_only=True) / 1e6)

    logger.info("========== Distillation ==========")
    logger.info("scheme=%s", args.scheme)
    if args.scheme == "sae_injection":
        logger.info(
            "teacher_cut=%s | student_start=%s | adapter_bottleneck_ratio=%.4f | adapter_sparsity=%.6f | freeze_teacher=%s | freeze_student=%s",
            args.teacher_cut,
            args.student_start,
            args.adapter_bottleneck_ratio,
            args.adapter_sparsity,
            args.freeze_teacher,
            args.freeze_student,
        )
        if hasattr(model, "parameter_summary"):
            logger.info(json.dumps(model.parameter_summary(), indent=2))
    elif args.scheme in {"logit_kd", "dkd"}:
        logger.info("temperature=%.4f | distill_weight=%.4f", args.temperature, args.distill_weight)
        if args.scheme == "dkd":
            logger.info("alpha=%.4f | beta=%.4f", args.alpha, args.beta)
    elif args.scheme == "fitnets":
        logger.info(
            "teacher_hints=%s | student_hints=%s | reverse=%s | distill_weight=%.4f | fitnet_bottleneck_ratio=%.4f",
            args.teacher_hints,
            args.student_hints,
            args.reverse,
            args.distill_weight,
            args.fitnet_bottleneck_ratio,
        )
        if hasattr(model, "parameter_summary"):
            logger.info(json.dumps(model.parameter_summary(), indent=2))


def parse_args():
    p = argparse.ArgumentParser(
        "Train distillation baselines or SAE injection with a single unified CLI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    model = p.add_argument_group("model")
    model.add_argument("--scheme", type=str, default="sae_injection", choices=SCHEMES)
    model.add_argument("--teacher-name", type=str, required=True)
    model.add_argument("--student-name", type=str, required=True)
    model.add_argument("--teacher-weights", type=str, default="default", help="default|none|torchvision weight enum name")
    model.add_argument("--student-weights", type=str, default="default", help="default|none|torchvision weight enum name")

    data = p.add_argument_group("data")
    data.add_argument("--dataset", type=str, required=True, choices=["cifar100", "cifar-100", "food101", "food-101", "imagenet100", "imagenet-100", "imagenet_subset", "imagenet-subset"])
    data.add_argument("--dataset-root", type=str, required=True)
    data.add_argument("--image-size", type=int, default=224)
    data.add_argument("--color-jitter", type=str, default="paper", choices=["paper", "none"])
    data.add_argument("--batch-size", type=int, default=64)
    data.add_argument("--num-workers", type=int, default=4)

    distill = p.add_argument_group("distillation")
    distill.add_argument("--teacher-cut", type=str, default="")
    distill.add_argument("--student-start", type=str, default="")
    distill.add_argument("--teacher-hints", type=str, default="")
    distill.add_argument("--student-hints", type=str, default="")
    distill.add_argument("--reverse", action="store_true")
    distill.add_argument("--adapter-bottleneck-ratio", type=float, default=0.5)
    distill.add_argument("--adapter-sparsity", type=float, default=1e-4)
    distill.add_argument("--fitnet-bottleneck-ratio", type=float, default=0.5)
    distill.add_argument("--distill-weight", type=float, default=1.0)
    distill.add_argument("--sae-weight", type=float, default=1.0)
    distill.add_argument("--temperature", type=float, default=4.0)
    distill.add_argument("--alpha", type=float, default=1.0)
    distill.add_argument("--beta", type=float, default=1.0)
    distill.add_argument("--freeze-teacher", action="store_true")
    distill.add_argument("--freeze-student", action="store_true")

    train = p.add_argument_group("training")
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--lr", type=float, default=3e-4)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--scheduler", type=str, default="cosine", choices=["none", "cosine", "step", "multistep"])
    train.add_argument("--step-size", type=int, default=10)
    train.add_argument("--gamma", type=float, default=0.1)
    train.add_argument("--milestones", type=str, default="20,40")
    train.add_argument("--label-smoothing", type=float, default=0.0)
    train.add_argument("--amp", action="store_true")
    train.add_argument("--grad-clip", type=float, default=0.0)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--out-dir", type=str, default="runs/distill")
    train.add_argument("--run-name", type=str, default="exp")
    train.add_argument("--resume", type=str, default="")

    return p.parse_args()


def main(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_meta = get_dataset_meta(args.dataset)
    train_loader, val_loader = build_loaders(
        args.dataset,
        root=args.dataset_root,
        batch_size=args.batch_size,
        image_size=args.image_size,
        color_jitter=(args.color_jitter == "paper"),
        num_workers=args.num_workers,
    )

    teacher, student = build_models(args, num_classes=int(dataset_meta["num_classes"]))
    model = build_distiller(args, teacher, student).to(device)

    if args.scheme == "fitnets":
        sample_images, _ = next(iter(train_loader))
        sample_images = sample_images.to(device)
        model.initialize(sample_images)
        model = model.to(device)

    run_dir = Path(args.out_dir) / args.run_name
    logger = setup_logger(run_dir)
    save_json(run_dir / "args.json", vars(args))

    describe_run(model, args, dataset_meta, logger)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = get_optimizer([p for p in model.parameters() if p.requires_grad], args.lr, args.weight_decay)
    scheduler = get_scheduler(args.scheduler, optimizer, args.epochs, args.step_size, args.gamma, args.milestones)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

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
        train_metrics = train_one_epoch(model, args.scheme, train_loader, optimizer, device, scaler, args.amp, epoch, args)
        val_metrics = evaluate(model, args.scheme, val_loader, device, args) if val_loader is not None else {"loss": float("nan"), "ce_loss": float("nan"), "distill_loss": float("nan"), "acc": float("nan")}

        if scheduler is not None:
            scheduler.step()

        logger.info(
            "epoch=%d | train_loss=%.6f | train_acc=%.4f | val_loss=%.6f | val_acc=%.4f",
            epoch,
            train_metrics["loss"],
            train_metrics["acc"],
            val_metrics["loss"],
            val_metrics["acc"],
        )

        if args.scheme == "sae_injection":
            logger.info(
                "epoch=%d | train_ce=%.6f | train_sae=%.6f | train_recon=%.6f | train_sparse=%.6f | val_ce=%.6f | val_sae=%.6f | val_recon=%.6f | val_sparse=%.6f",
                epoch,
                train_metrics["ce_loss"],
                train_metrics["distill_loss"],
                train_metrics.get("reconstruction_loss", float("nan")),
                train_metrics.get("sparsity_loss", float("nan")),
                val_metrics["ce_loss"],
                val_metrics["distill_loss"],
                val_metrics.get("reconstruction_loss", float("nan")),
                val_metrics.get("sparsity_loss", float("nan")),
            )
            csv_log_epoch(run_dir, {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["acc"],
                "train_ce_loss": train_metrics["ce_loss"],
                "train_distill_loss": train_metrics["distill_loss"],
                "train_reconstruction_loss": train_metrics.get("reconstruction_loss", float("nan")),
                "train_sparsity_loss": train_metrics.get("sparsity_loss", float("nan")),
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["acc"],
                "val_ce_loss": val_metrics["ce_loss"],
                "val_distill_loss": val_metrics["distill_loss"],
                "val_reconstruction_loss": val_metrics.get("reconstruction_loss", float("nan")),
                "val_sparsity_loss": val_metrics.get("sparsity_loss", float("nan")),
            })
        else:
            logger.info(
                "epoch=%d | train_ce=%.6f | train_distill=%.6f | val_ce=%.6f | val_distill=%.6f",
                epoch,
                train_metrics["ce_loss"],
                train_metrics["distill_loss"],
                val_metrics["ce_loss"],
                val_metrics["distill_loss"],
            )
            csv_log_epoch(run_dir, {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["acc"],
                "train_ce_loss": train_metrics["ce_loss"],
                "train_distill_loss": train_metrics["distill_loss"],
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["acc"],
                "val_ce_loss": val_metrics["ce_loss"],
                "val_distill_loss": val_metrics["distill_loss"],
            })

        checkpoint = {
            "epoch": epoch,
            "scheme": args.scheme,
            "model_state_dict": model.state_dict(),
            "teacher_state_dict": model.teacher.state_dict() if hasattr(model, "teacher") else None,
            "student_state_dict": model.student.state_dict() if hasattr(model, "student") else None,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "best_val_acc": best_val_acc,
            "args": vars(args),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }
        torch.save(checkpoint, run_dir / "last.pt")

        if not math.isnan(val_metrics["acc"]) and val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            checkpoint["best_val_acc"] = best_val_acc
            torch.save(checkpoint, run_dir / "best.pt")
            logger.info("Saved best checkpoint with val_acc=%.4f", best_val_acc)

    logger.info("Training finished. Best val acc: %.4f", best_val_acc)


if __name__ == "__main__":
    main(parse_args())
