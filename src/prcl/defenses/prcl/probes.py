"""Probe transform registry — cheap pixel-level perturbations for PCF forensics.

These transforms are intentionally different from SSL augmentations. They target
the kinds of artifacts that backdoor triggers create (high-frequency patches,
color anomalies) to test whether a sample's embedding is unusually sensitive
to specific perturbation types.
"""

import torch
from torchvision import transforms

_PROBE_REGISTRY: dict[str, type] = {}


def register_probe(name: str):
    """Decorator to register a probe transform class."""
    def decorator(cls):
        _PROBE_REGISTRY[name] = cls
        return cls
    return decorator


def get_probe_transforms(names: list[str]) -> list:
    """Instantiate probe transforms by name."""
    probes = []
    for name in names:
        if name not in _PROBE_REGISTRY:
            raise ValueError(f"Unknown probe '{name}'. Available: {list(_PROBE_REGISTRY)}")
        probes.append(_PROBE_REGISTRY[name]())
    return probes


@register_probe("blur")
class BlurProbe:
    """Strong Gaussian blur — attenuates high-frequency trigger patterns.

    Uses σ=2.0 and kernel_size=7, which is notably stronger than the mild
    blur in standard SimCLR augmentation (σ~0.1-2.0 with p=0.5).
    """

    def __init__(self, kernel_size: int = 7, sigma: float = 2.0):
        self.transform = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply blur to a single image tensor (C, H, W) or batch (B, C, H, W)."""
        return self.transform(x)


@register_probe("occlusion")
class OcclusionProbe:
    """Random patch occlusion — masks ~25% of the image area.

    Replaces a random rectangular patch with zeros. Backdoor triggers are
    often localized, so occluding different regions can reveal whether the
    embedding depends on a specific local patch.
    """

    def __init__(self, occlusion_ratio: float = 0.25):
        self.occlusion_ratio = occlusion_ratio

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        result = x.clone()
        if result.ndim == 3:
            _, h, w = result.shape
            patch_h = int(h * self.occlusion_ratio ** 0.5)
            patch_w = int(w * self.occlusion_ratio ** 0.5)
            top = torch.randint(0, max(1, h - patch_h), (1,)).item()
            left = torch.randint(0, max(1, w - patch_w), (1,)).item()
            result[:, top:top + patch_h, left:left + patch_w] = 0
        elif result.ndim == 4:
            b, _, h, w = result.shape
            patch_h = int(h * self.occlusion_ratio ** 0.5)
            patch_w = int(w * self.occlusion_ratio ** 0.5)
            for i in range(b):
                top = torch.randint(0, max(1, h - patch_h), (1,)).item()
                left = torch.randint(0, max(1, w - patch_w), (1,)).item()
                result[i, :, top:top + patch_h, left:left + patch_w] = 0
        return result


@register_probe("freq_lowpass")
class FreqLowpassProbe:
    """Frequency-domain low-pass filter — removes high-frequency components.

    Backdoor triggers often contain sharp, high-frequency patterns. A low-pass
    filter in the frequency domain attenuates these while preserving natural
    image structure.
    """

    def __init__(self, cutoff_ratio: float = 0.3):
        self.cutoff_ratio = cutoff_ratio

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            return self._filter_single(x)
        elif x.ndim == 4:
            return torch.stack([self._filter_single(xi) for xi in x])
        return x

    def _filter_single(self, x: torch.Tensor) -> torch.Tensor:
        """Apply low-pass filter to a single (C, H, W) tensor."""
        c, h, w = x.shape
        # Create circular low-pass mask in frequency domain
        cy, cx = h // 2, w // 2
        cutoff_h = int(h * self.cutoff_ratio)
        cutoff_w = int(w * self.cutoff_ratio)

        y_coords = torch.arange(h, device=x.device).float() - cy
        x_coords = torch.arange(w, device=x.device).float() - cx
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
        dist = torch.sqrt((yy / max(cutoff_h, 1)) ** 2 + (xx / max(cutoff_w, 1)) ** 2)
        mask = (dist <= 1.0).float()

        result = torch.zeros_like(x)
        for ch in range(c):
            freq = torch.fft.fftshift(torch.fft.fft2(x[ch]))
            freq_filtered = freq * mask
            result[ch] = torch.fft.ifft2(torch.fft.ifftshift(freq_filtered)).real

        return result


@register_probe("desaturation")
class DesaturationProbe:
    """Desaturation — converts to grayscale, removing color-based triggers.

    Some backdoor triggers rely on specific color patterns. Full desaturation
    removes all color information while preserving luminance structure.
    """

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            # ITU-R 601-2 luma transform
            gray = 0.2989 * x[0] + 0.5870 * x[1] + 0.1140 * x[2]
            return gray.unsqueeze(0).expand_as(x)
        elif x.ndim == 4:
            gray = 0.2989 * x[:, 0] + 0.5870 * x[:, 1] + 0.1140 * x[:, 2]
            return gray.unsqueeze(1).expand_as(x)
        return x
