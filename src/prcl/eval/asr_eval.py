"""ASR (Attack Success Rate) evaluation for backdoor attacks."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class TriggeredDataset(Dataset):
    """Applies a trigger function to test images for ASR evaluation.

    Only applies to the target class's images to measure whether the
    encoder has learned the backdoor association.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        trigger_fn,
        target_class: int,
        transform=None,
    ):
        self.base_dataset = base_dataset
        self.trigger_fn = trigger_fn
        self.target_class = target_class
        self.transform = transform

        # Index non-target samples (ASR = fraction of non-target triggered → target)
        self.non_target_indices = [
            i for i in range(len(base_dataset))
            if base_dataset[i][1] != target_class
        ]

    def __len__(self):
        return len(self.non_target_indices)

    def __getitem__(self, idx):
        real_idx = self.non_target_indices[idx]
        img, label = self.base_dataset[real_idx]

        # Apply trigger
        img = self.trigger_fn(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, self.target_class  # "label" is target for ASR measurement


def evaluate_asr(
    encoder: nn.Module,
    classifier_head: nn.Module,
    triggered_loader: DataLoader,
    target_class: int,
    device: torch.device,
) -> dict:
    """Evaluate attack success rate.

    ASR = fraction of triggered non-target images classified as target_class
    by the downstream classifier (encoder + linear head).

    Returns dict with asr, total triggered samples, successful attacks.
    """
    encoder.eval()
    classifier_head.eval()

    correct_attacks = 0
    total = 0

    with torch.no_grad():
        for images, _ in triggered_loader:
            images = images.to(device)
            features = encoder(images)
            logits = classifier_head(features)
            preds = logits.argmax(dim=1)
            correct_attacks += (preds == target_class).sum().item()
            total += images.size(0)

    asr = correct_attacks / total if total > 0 else 0.0
    return {
        "asr": asr,
        "triggered_total": total,
        "successful_attacks": correct_attacks,
        "target_class": target_class,
    }
