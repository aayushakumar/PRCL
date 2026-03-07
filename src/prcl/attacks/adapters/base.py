"""Abstract base class for attack adapters."""

from abc import ABC, abstractmethod

import numpy as np
from PIL import Image


class AttackAdapter(ABC):
    """Base class for poisoning attack adapters.

    An adapter generates:
    1. A set of poison indices (which samples to poison)
    2. A poison function (how to apply the trigger to a PIL image)
    3. Metadata for reproducibility and evaluation
    """

    @abstractmethod
    def select_poison_indices(
        self, dataset_size: int, poison_ratio: float, seed: int = 42
    ) -> np.ndarray:
        """Select which samples to poison.

        Returns sorted array of integer indices.
        """

    @abstractmethod
    def apply_trigger(self, image: Image.Image) -> Image.Image:
        """Apply the trigger pattern to a single PIL image.

        Input and output are PIL Images (before any tensor transform).
        """

    @abstractmethod
    def get_metadata(self) -> dict:
        """Return a JSON-serializable dict describing the attack configuration."""
