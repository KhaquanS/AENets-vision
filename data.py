from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Union

from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Configuration
# ============================================================

@dataclass(frozen=True)
class DataConfig:
    dataset: str

    # torchvision datasets root
    root: str = "./data"

    # huggingface cache directory
    hf_cache_dir: Optional[str] = "./hf_cache"

    image_size: int = 224

    batch_size: int = 64
    num_workers: int = 4

    pin_memory: bool = True
    drop_last: bool = False

    # False | True | "paper"
    color_jitter: Union[bool, str] = False


# ============================================================
# HuggingFace Wrapper
# ============================================================

class HFDatasetWrapper(Dataset):
    """
    Converts a HuggingFace image dataset into a standard
    PyTorch Dataset.

    Expected format:
        sample["image"]
        sample["label"]
    """

    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        image = sample["image"].convert("RGB")
        label = sample["label"]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


# ============================================================
# Normalization
# ============================================================

def _imagenet_norm():
    from torchvision import transforms

    return transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )


def _cifar_norm():
    from torchvision import transforms

    return transforms.Normalize(
        mean=(0.5071, 0.4867, 0.4408),
        std=(0.2675, 0.2565, 0.2761),
    )


# ============================================================
# Color Jitter
# ============================================================

def _paper_color_jitter(color_jitter):
    """
    Transformation invariance setting from:

    Understanding Knowledge Distillation
    arXiv:2205.16004

    brightness = 0.4
    contrast   = 0.4
    saturation = 0.4
    hue        = 0.2
    """

    from torchvision import transforms

    if color_jitter in {False, None}:
        return None

    if color_jitter is True or color_jitter == "paper":
        return transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.2,
        )

    if isinstance(color_jitter, (tuple, list)) and len(color_jitter) == 4:
        return transforms.ColorJitter(*color_jitter)

    raise ValueError(
        "color_jitter must be False, True, 'paper', or a tuple/list of length 4."
    )


# ============================================================
# Transforms
# ============================================================

def build_transforms(
    dataset: str,
    *,
    train: bool,
    image_size: int = 224,
    color_jitter: Union[bool, str] = False,
):
    from torchvision import transforms

    dataset = dataset.lower()

    jitter = (
        _paper_color_jitter(color_jitter)
        if train
        else None
    )

    # --------------------------------------------------------
    # CIFAR100
    # --------------------------------------------------------

    if dataset in {"cifar100", "cifar-100"}:

        ops = []

        if train:
            ops.extend([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
            ])
        else:
            ops.append(
                transforms.Resize((image_size, image_size))
            )

        if jitter is not None:
            ops.append(jitter)

        ops.extend([
            transforms.ToTensor(),
            _cifar_norm(),
        ])

        return transforms.Compose(ops)

    # --------------------------------------------------------
    # Food101 / ImageNet100
    # --------------------------------------------------------

    ops = []

    if train:
        ops.extend([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
        ])
    else:
        ops.extend([
            transforms.Resize(int(image_size * 256 / 224)),
            transforms.CenterCrop(image_size),
        ])

    if jitter is not None:
        ops.append(jitter)

    ops.extend([
        transforms.ToTensor(),
        _imagenet_norm(),
    ])

    return transforms.Compose(ops)


# ============================================================
# Dataset Builder
# ============================================================

def build_dataset(
    dataset: str,
    *,
    root: str,
    split: str = "train",
    image_size: int = 224,
    color_jitter: Union[bool, str] = False,
    hf_cache_dir: Optional[str] = None,
):
    dataset = dataset.lower()

    transform = build_transforms(
        dataset,
        train=(split == "train"),
        image_size=image_size,
        color_jitter=color_jitter,
    )

    # --------------------------------------------------------
    # torchvision datasets
    # --------------------------------------------------------

    from torchvision import datasets

    if dataset in {"cifar100", "cifar-100"}:
        return datasets.CIFAR100(
            root=root,
            train=(split == "train"),
            download=True,
            transform=transform,
        )

    if dataset in {"food101", "food-101"}:
        return datasets.Food101(
            root=root,
            split=split,
            download=True,
            transform=transform,
        )

    # --------------------------------------------------------
    # HuggingFace ImageNet100
    # --------------------------------------------------------

    if dataset in {"imagenet100", "imagenet-100"}:

        split_map = {
            "train": "train",
            "val": "validation",
            "validation": "validation",
        }

        hf_ds = load_dataset(
            "clane9/imagenet-100",
            split=split_map[split],
            cache_dir=hf_cache_dir,
        )

        return HFDatasetWrapper(
            hf_ds,
            transform=transform,
        )

    raise ValueError(f"Unsupported dataset: {dataset}")


# ============================================================
# Loader Builder
# ============================================================

def build_loaders(
    dataset: str,
    *,
    root: str = "./data",
    hf_cache_dir: Optional[str] = "./hf_cache",
    batch_size: int = 64,
    image_size: int = 224,
    color_jitter: Union[bool, str] = False,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
):
    dataset_lower = dataset.lower()

    train_ds = build_dataset(
        dataset=dataset,
        root=root,
        split="train",
        image_size=image_size,
        color_jitter=color_jitter,
        hf_cache_dir=hf_cache_dir,
    )

    # --------------------------------------------------------
    # Validation split naming
    # --------------------------------------------------------

    if dataset_lower in {"food101", "food-101"}:
        val_split = "test"

    elif dataset_lower in {"imagenet100", "imagenet-100"}:
        val_split = "validation"

    else:
        val_split = "val"

    val_ds = build_dataset(
        dataset=dataset,
        root=root,
        split=val_split,
        image_size=image_size,
        color_jitter=False,
        hf_cache_dir=hf_cache_dir,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader


# ============================================================
# Dataset Metadata
# ============================================================

def get_dataset_meta(dataset: str) -> Dict[str, object]:
    dataset = dataset.lower()

    if dataset in {"cifar100", "cifar-100"}:
        return {
            "num_classes": 100,
            "image_size": 224,
            "channels": 3,
        }

    if dataset in {"food101", "food-101"}:
        return {
            "num_classes": 101,
            "image_size": 224,
            "channels": 3,
        }

    if dataset in {"imagenet100", "imagenet-100"}:
        return {
            "num_classes": 100,
            "image_size": 224,
            "channels": 3,
            "hf_id": "clane9/imagenet-100",
        }

    raise ValueError(f"Unknown dataset: {dataset}")