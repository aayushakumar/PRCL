"""Blend-based backdoor attack adapter.

Blends a trigger pattern into the image with an alpha coefficient.
Structurally different from patch attacks — affects the entire image.
"""

import numpy as np
from PIL import Image

from .base import AttackAdapter


class BlendBackdoorAdapter(AttackAdapter):
    """Blend-style global trigger overlay.

    Trigger: random fixed pattern blended across the full image.
    Formula: img_out = (1 - alpha) * img + alpha * trigger_pattern
    """

    def __init__(
        self,
        blend_alpha: float = 0.1,
        target_class: int = 0,
        trigger_seed: int = 0,
    ):
        self.blend_alpha = blend_alpha
        self.target_class = target_class
        self.trigger_seed = trigger_seed
        self._trigger_cache: dict[tuple[int, int], np.ndarray] = {}

    def _get_trigger_pattern(self, width: int, height: int) -> np.ndarray:
        key = (width, height)
        if key not in self._trigger_cache:
            rng = np.random.RandomState(self.trigger_seed)
            pattern = rng.randint(0, 256, (height, width, 3), dtype=np.uint8)
            self._trigger_cache[key] = pattern
        return self._trigger_cache[key]

    def select_poison_indices(
        self, dataset_size: int, poison_ratio: float, seed: int = 42
    ) -> np.ndarray:
        rng = np.random.RandomState(seed)
        n_poison = max(1, int(dataset_size * poison_ratio))
        indices = rng.choice(dataset_size, size=n_poison, replace=False)
        return np.sort(indices)

    def apply_trigger(self, image: Image.Image) -> Image.Image:
        img_arr = np.array(image, dtype=np.float32)
        w, h = image.size
        trigger = self._get_trigger_pattern(w, h).astype(np.float32)

        blended = (1.0 - self.blend_alpha) * img_arr + self.blend_alpha * trigger
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        return Image.fromarray(blended)

    def get_metadata(self) -> dict:
        return {
            "attack_type": "blend_backdoor",
            "blend_alpha": self.blend_alpha,
            "target_class": self.target_class,
            "trigger_seed": self.trigger_seed,
        }
