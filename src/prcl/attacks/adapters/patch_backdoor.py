"""Patch-based backdoor attack adapter (BadNets-style).

Applies a small solid-color patch to a corner of the image.
This is the simplest and most widely studied trigger family.
"""

import numpy as np
from PIL import Image

from .base import AttackAdapter


class PatchBackdoorAdapter(AttackAdapter):
    """BadNets-style fixed-position patch trigger.

    Trigger: small white patch placed at a fixed image location.
    The patch overwrites pixel values in the selected region.
    """

    def __init__(
        self,
        patch_size: int = 4,
        patch_position: str = "bottom_right",
        target_class: int = 0,
        patch_color: tuple[int, int, int] = (255, 255, 255),
    ):
        self.patch_size = patch_size
        self.patch_position = patch_position
        self.target_class = target_class
        self.patch_color = patch_color

    def select_poison_indices(
        self, dataset_size: int, poison_ratio: float, seed: int = 42
    ) -> np.ndarray:
        rng = np.random.RandomState(seed)
        n_poison = max(1, int(dataset_size * poison_ratio))
        indices = rng.choice(dataset_size, size=n_poison, replace=False)
        return np.sort(indices)

    def apply_trigger(self, image: Image.Image) -> Image.Image:
        img = image.copy()
        w, h = img.size
        ps = self.patch_size

        if self.patch_position == "bottom_right":
            x0, y0 = w - ps, h - ps
        elif self.patch_position == "top_left":
            x0, y0 = 0, 0
        elif self.patch_position == "top_right":
            x0, y0 = w - ps, 0
        elif self.patch_position == "bottom_left":
            x0, y0 = 0, h - ps
        elif self.patch_position == "center":
            x0, y0 = (w - ps) // 2, (h - ps) // 2
        else:
            raise ValueError(f"Unknown patch_position: {self.patch_position}")

        pixels = img.load()
        for dx in range(ps):
            for dy in range(ps):
                px, py = x0 + dx, y0 + dy
                if 0 <= px < w and 0 <= py < h:
                    pixels[px, py] = self.patch_color

        return img

    def get_metadata(self) -> dict:
        return {
            "attack_type": "patch_backdoor",
            "patch_size": self.patch_size,
            "patch_position": self.patch_position,
            "target_class": self.target_class,
            "patch_color": list(self.patch_color),
        }
