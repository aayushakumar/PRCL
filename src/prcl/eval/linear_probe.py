"""Linear probe evaluation — standard downstream evaluation for SSL encoders."""

import logging
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


class LinearProbe(nn.Module):
    """Frozen encoder + trainable linear classifier."""

    def __init__(self, encoder: nn.Module, feat_dim: int, num_classes: int = 10):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(feat_dim, num_classes)
        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.encoder(x)
        return self.classifier(features)


def train_linear_probe(
    encoder: nn.Module,
    feat_dim: int,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_classes: int = 10,
    epochs: int = 100,
    lr: float = 0.1,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> dict:
    """Train and evaluate a linear probe on frozen encoder features.

    Returns dict with train_acc, test_acc, and timing info.
    """
    if device is None:
        device = torch.device("cpu")
    probe = LinearProbe(encoder, feat_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        probe.classifier.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    logger.info(f"Linear probe: {epochs} epochs, lr={lr}, {num_classes} classes")

    start_time = time.time()
    best_test_acc = 0.0

    for epoch in range(epochs):
        # Train
        probe.train()
        correct, total = 0, 0
        for images, labels in tqdm(train_loader, desc=f"Probe {epoch+1}/{epochs}", leave=False):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            logits = probe(images)
            loss = criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        scheduler.step()

        # Evaluate every 10 epochs or at the end
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            test_acc = evaluate_accuracy(probe, test_loader, device)
            best_test_acc = max(best_test_acc, test_acc)
            logger.info(
                f"  Probe epoch {epoch+1}/{epochs}: "
                f"train_acc={train_acc:.4f}  test_acc={test_acc:.4f}"
            )

    total_time = time.time() - start_time
    final_test_acc = evaluate_accuracy(probe, test_loader, device)

    results = {
        "linear_probe_acc": round(final_test_acc, 4),
        "best_test_acc": round(best_test_acc, 4),
        "final_train_acc": round(train_acc, 4),
        "probe_time": round(total_time, 2),
    }
    logger.info(f"Linear probe done: test_acc={final_test_acc:.4f} (best={best_test_acc:.4f})")
    return results


def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Evaluate classification accuracy."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            logits = model(images)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0
