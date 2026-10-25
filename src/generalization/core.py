import random
from typing import ClassVar

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


class FashionMNIST(datasets.FashionMNIST):
    # Official upstream GitHub mirror; torchvision still verifies published MD5 hashes.
    mirrors: ClassVar[list[str]] = [
        "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
    ]


class CNN(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        if not 0 <= dropout < 1:
            raise ValueError("Dropout must be in [0,1)")
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)


def loaders(root, seed=42, limit=1000, augment=False, synthetic=False, batch_size=64):
    if limit < 20 or limit > 48000:
        raise ValueError("Training limit must be 20..48000")
    base = [transforms.ToTensor(), transforms.Normalize((0.286,), (0.353,))]
    eval_tf = transforms.Compose(base)
    train_tf = transforms.Compose(
        (
            [transforms.RandomHorizontalFlip(), transforms.RandomAffine(10, translate=(0.05, 0.05))]
            if augment
            else []
        )
        + base
    )
    if synthetic:
        size = limit + max(20, limit // 4)
        train = datasets.FakeData(
            size=size, image_size=(1, 28, 28), num_classes=10, transform=train_tf, random_offset=0
        )
        clean = datasets.FakeData(
            size=size, image_size=(1, 28, 28), num_classes=10, transform=eval_tf, random_offset=0
        )
        test = datasets.FakeData(
            size=max(20, limit // 4),
            image_size=(1, 28, 28),
            num_classes=10,
            transform=eval_tf,
            random_offset=100000,
        )
    else:
        train = FashionMNIST(root, train=True, download=True, transform=train_tf)
        clean = FashionMNIST(root, train=True, download=False, transform=eval_tf)
        test = FashionMNIST(root, train=False, download=True, transform=eval_tf)
    # Fixed split seed independent of optimization seed; comparable across runs/treatments.
    order = np.random.default_rng(2026).permutation(len(train))
    ids = order[:limit]
    val_ids = order[limit : limit + min(len(order) - limit, max(20, limit // 4))]
    test_ids = np.random.default_rng(2027).permutation(len(test))[
        : min(len(test), max(20, limit // 4))
    ]
    generator = torch.Generator().manual_seed(seed)
    return (
        DataLoader(Subset(train, ids), batch_size=batch_size, shuffle=True, generator=generator),
        DataLoader(Subset(clean, ids), batch_size=batch_size),
        DataLoader(Subset(clean, val_ids), batch_size=batch_size),
        DataLoader(Subset(test, test_ids), batch_size=batch_size),
    ), {"train": ids.tolist(), "validation": val_ids.tolist(), "test": test_ids.tolist()}


def epoch(
    model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer | None = None
) -> dict[str, float]:
    model.train(optimizer is not None)
    total = correct = 0
    loss_sum = 0.0
    for x, y in loader:
        with torch.set_grad_enabled(optimizer is not None):
            pred = model(x)
            loss = nn.functional.cross_entropy(pred, y)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total += len(y)
        correct += (pred.argmax(1) == y).sum().item()
        loss_sum += loss.item() * len(y)
    return {"loss": loss_sum / total, "accuracy": correct / total}
